use core::{
    fmt::Debug,
    ops::{BitAnd, BitOr, Not, Shl, Shr},
};

use super::algo::{Prefilter, load_window};

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(test, not(all(target_arch = "wasm32", target_feature = "simd128"))))]
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
#[cfg(any(test, not(all(target_arch = "wasm32", target_feature = "simd128"))))]
pub(crate) use scalar::PrefilterScalarBackend;
#[cfg(target_arch = "x86_64")]
pub(crate) use sse::PrefilterSSEBackend;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm::PrefilterWasmBackend;

#[cfg(target_arch = "x86_64")]
pub use avx::PrefilterAVX;

#[cfg(target_arch = "x86_64")]
pub type PrefilterAVX512 = Prefilter<avx512::PrefilterAVX512Backend>;
#[cfg(target_arch = "aarch64")]
pub type PrefilterNEON = Prefilter<neon::PrefilterNEONBackend>;
#[cfg(any(test, not(all(target_arch = "wasm32", target_feature = "simd128"))))]
pub type PrefilterScalar = Prefilter<scalar::PrefilterScalarBackend>;
#[cfg(target_arch = "x86_64")]
pub type PrefilterSSE = Prefilter<sse::PrefilterSSEBackend>;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub type PrefilterWasm = Prefilter<wasm::PrefilterWasmBackend>;

/// A chunk's compare result. `Into<u64>` gives one bit per byte, which is how
/// chunk masks are assembled into a [`BlockMask`].
pub(crate) trait Mask:
    Copy + Debug + PartialOrd + BitMaskOps + Into<u64> + Send + Sync + 'static
{
}

impl<T> Mask for T where
    T: Copy + Debug + PartialOrd + BitMaskOps + Into<u64> + Send + Sync + 'static
{
}

pub(crate) trait BitMaskOps {
    fn all() -> Self;
    fn first_n(n: usize) -> Self;
    fn is_zero(&self) -> bool;
    fn trailing_zeros(self) -> usize;
    fn leading_zeros(self) -> usize;
    fn or(self, other: Self) -> Self;
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
            fn or(self, other: Self) -> Self {
                self | other
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

/// A bit per byte of a block of haystack bytes for [`super::algo::block`]
pub(crate) trait BlockMask:
    BitMaskOps
    + Copy
    + Debug
    + Send
    + Sync
    + 'static
    + From<u64>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Not<Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
{
    /// Bytes per block
    const BITS: usize;
    const ZERO: Self;
    const ALL: Self;
    /// Every bit above the lowest set bit (none when zero)
    fn above_lowest(self) -> Self;
}

macro_rules! impl_block {
    ($ty:ty) => {
        impl BlockMask for $ty {
            const BITS: usize = <$ty>::BITS as usize;
            const ZERO: Self = 0;
            const ALL: Self = <$ty>::MAX;

            #[inline(always)]
            fn above_lowest(self) -> Self {
                !(self ^ self.wrapping_sub(1))
            }
        }
    };
}

impl_block!(u64);
impl_block!(u128);

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
        unsafe { load_partial_copy::<Self>(ptr, remaining) }
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
    /// [`BLOCK_PAD`] (through masked loads), so occurrence masks need no
    /// masking
    const PADDED_BLOCKS: bool = false;

    /// Loads the chunks of the block at the start of `haystack`, with the
    /// mask of the bytes inside it. Bytes past the haystack are
    /// [`BLOCK_PAD`] when [`Backend::PADDED_BLOCKS`], unspecified otherwise.
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

/// The byte padded blocks hold past the haystack's end: it never occurs in
/// UTF-8, so no needle row matches it
pub(crate) const BLOCK_PAD: u8 = 0xFF;

/// A block as `N` chunks, of which the first `count` hold haystack bytes:
/// the per-chunk compares stop there, so a short haystack does not pay for
/// the whole block (the count is the same for every needle byte of a
/// haystack, which keeps the loop predictable)
#[derive(Clone, Copy)]
pub(crate) struct ChunkBlock<C: Copy, const N: usize> {
    pub(crate) chunks: [C; N],
    pub(crate) count: usize,
}

/// [`Backend::load_block`] for backends without masked loads: chunks past
/// the end are skipped (their mask bits are clear)
#[inline(always)]
pub(crate) unsafe fn load_block_chunks<B: Backend, const N: usize>(
    haystack: &[u8],
) -> (ChunkBlock<B::Chunk, N>, B::Block) {
    unsafe {
        let mut chunks = [B::splat(0); N];
        let count = haystack.len().div_ceil(B::LANES).min(N);
        for (c, chunk) in chunks.iter_mut().enumerate().take(count) {
            *chunk = load_window::<B>(haystack, c * B::LANES, haystack.len()).0;
        }
        (
            ChunkBlock { chunks, count },
            B::Block::first_n(haystack.len()),
        )
    }
}

/// [`Backend::eq_block`] from per-chunk compares
#[inline(always)]
pub(crate) unsafe fn eq_block_chunks<B: Backend, const N: usize>(
    block: &ChunkBlock<B::Chunk, N>,
    needle: B::Chunk,
) -> B::Block {
    let mut bits = B::Block::ZERO;
    for (c, &chunk) in block.chunks.iter().enumerate().take(block.count) {
        let eq = unsafe { B::eq(chunk, needle) };
        bits = bits | (B::Block::from(eq.into()) << (c * B::LANES));
    }
    bits
}

#[cold]
#[inline(never)]
unsafe fn load_partial_copy<B: Backend>(ptr: *const u8, remaining: usize) -> B::Chunk {
    unsafe {
        let mut data = [0u8; 64];
        core::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), remaining);
        B::load(data.as_ptr())
    }
}
