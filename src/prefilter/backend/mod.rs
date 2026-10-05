use core::fmt::Debug;

use super::algo::Prefilter;

mod block;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(test, scalar_fallback))]
mod scalar;
#[cfg(target_arch = "x86_64")]
mod sse;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;

// Low-level SIMD backends re-exported so other modules (e.g. the literal
// matcher) can build on the raw byte-search primitives (`splat`/`occ`/`load`)
// directly.
#[cfg(target_arch = "x86_64")]
pub(crate) use avx::PrefilterAVXBackend;
#[cfg(target_arch = "x86_64")]
pub(crate) use avx512::PrefilterAVX512Backend;
#[cfg(target_arch = "aarch64")]
pub(crate) use neon::PrefilterNEONBackend;
#[cfg(any(test, scalar_fallback))]
pub(crate) use scalar::PrefilterScalarBackend;
#[cfg(target_arch = "x86_64")]
pub(crate) use sse::PrefilterSSEBackend;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm::PrefilterWasmBackend;

pub(crate) use block::{
    BlockMask, ChunkBlock, Narrow, copy_partial, eq_block_chunks, load_block_chunks,
};

#[cfg(target_arch = "x86_64")]
pub use avx::PrefilterAVX;

#[cfg(target_arch = "x86_64")]
pub type PrefilterAVX512 = Prefilter<avx512::PrefilterAVX512Backend>;
#[cfg(target_arch = "aarch64")]
pub type PrefilterNEON = Prefilter<neon::PrefilterNEONBackend>;
#[cfg(any(test, scalar_fallback))]
pub type PrefilterScalar = Prefilter<scalar::PrefilterScalarBackend>;
#[cfg(target_arch = "x86_64")]
pub type PrefilterSSE = Prefilter<sse::PrefilterSSEBackend>;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub type PrefilterWasm = Prefilter<wasm::PrefilterWasmBackend>;

/// A chunk's compare result. `Into<u64>` gives one bit per byte, which is how
/// chunk masks are assembled into a [`BlockMask`].
pub(crate) trait Mask:
    Copy + Debug + BitMaskOps + Into<u64> + Send + Sync + 'static
{
}

impl<T> Mask for T where T: Copy + Debug + BitMaskOps + Into<u64> + Send + Sync + 'static {}

pub(crate) trait BitMaskOps {
    fn all() -> Self;
    fn first_n(n: usize) -> Self;
    fn is_zero(&self) -> bool;
    fn trailing_zeros(self) -> usize;
    fn leading_zeros(self) -> usize;
    fn and(self, other: Self) -> Self;
    fn clear_through_lowest(self, hit: Self) -> Self;
}

macro_rules! impl_mask {
    ($ty:ty) => {
        impl BitMaskOps for $ty {
            #[inline(always)]
            fn all() -> Self {
                <$ty>::MAX
            }

            #[inline(always)]
            fn first_n(n: usize) -> Self {
                if n >= <$ty>::BITS as usize {
                    <$ty>::MAX
                } else {
                    ((1 as $ty) << n) - 1
                }
            }

            #[inline(always)]
            fn is_zero(&self) -> bool {
                *self == 0
            }

            #[inline(always)]
            fn trailing_zeros(self) -> usize {
                <$ty>::trailing_zeros(self) as usize
            }

            #[inline(always)]
            fn leading_zeros(self) -> usize {
                <$ty>::leading_zeros(self) as usize
            }

            #[inline(always)]
            fn and(self, other: Self) -> Self {
                self & other
            }

            #[inline(always)]
            fn clear_through_lowest(self, hit: Self) -> Self {
                self & !(hit ^ hit.wrapping_sub(1))
            }
        }
    };
}

impl_mask!(u16);
impl_mask!(u32);
impl_mask!(u64);
impl_mask!(u128);

/// Closures (`array::map`/`from_fn`) around intrinsics lose the caller's
/// `#[target_feature]`: with default codegen units LLVM may not inline them,
/// so every intrinsic becomes a call. The backends and walks spell such loops
/// out (marked "no closures").
pub(crate) trait Backend: Sized + Debug + Clone + 'static {
    const LANES: usize;

    type Chunk: Copy + Debug;
    type Mask: Mask;
    /// Same as `Mask` but always one bit per byte
    type Block: BlockMask;
    /// The chunks of one block, `Block::BITS / LANES` of them
    type Chunks: Copy;

    fn is_available() -> bool;

    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn splat(c: u8) -> Self::Chunk;

    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask;

    /// # Safety
    /// `ptr` must point to at least `LANES` readable bytes, and the backend's
    /// target features must be enabled.
    unsafe fn load(ptr: *const u8) -> Self::Chunk;

    /// # Safety
    /// `ptr` must point to `remaining` readable bytes, `remaining < LANES`, and
    /// the backend's target features must be enabled.
    #[inline(always)]
    unsafe fn load_partial(ptr: *const u8, remaining: usize, _mask: Self::Mask) -> Self::Chunk {
        unsafe { Self::load(copy_partial::<64>(ptr, remaining).as_ptr()) }
    }

    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask;

    /// # Safety
    /// The backend's target features must be enabled and `hit` must be nonzero.
    #[inline(always)]
    unsafe fn first_hit_pos(hit: Self::Mask) -> usize {
        hit.trailing_zeros()
    }

    /// Whether [`Backend::load_block`] fills the bytes past the haystack with
    /// [`BLOCK_PAD`](block::BLOCK_PAD), so occurrence masks need no masking
    const PADDED_BLOCKS: bool = false;

    /// Loads the chunks of the block at the start of `haystack`, with the
    /// mask of the bytes inside it. Bytes past the haystack are
    /// [`BLOCK_PAD`](block::BLOCK_PAD) when [`Backend::PADDED_BLOCKS`], unspecified otherwise.
    ///
    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn load_block(haystack: &[u8]) -> (Self::Chunks, Self::Block);

    /// The bytes of a block equal to `needle`
    ///
    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn eq_block(chunks: &Self::Chunks, needle: Self::Chunk) -> Self::Block;

    /// Lowercases the ASCII letters of a block in place
    ///
    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn fold_block(chunks: &mut Self::Chunks);
}
