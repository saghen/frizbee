use core::marker::PhantomData;

use super::block::PAD_FROM;
use super::{Backend, ChunkBlock, eq_block_chunks, load_block_chunks};

/// A backend of 16 byte registers that [`PairBackend`] builds on
pub(crate) trait Register: Backend {
    /// The bytes of `a` equal to `b`, as a register
    ///
    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn cmpeq(a: Self::Chunk, b: Self::Chunk) -> Self::Chunk;

    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn or(a: Self::Chunk, b: Self::Chunk) -> Self::Chunk;

    /// One bit per byte of two compare results, `lo`'s in the low half
    ///
    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn movemask_pair(lo: Self::Chunk, hi: Self::Chunk) -> u32;

    /// Lowercases the ASCII letters of a register
    ///
    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn fold(chunk: Self::Chunk) -> Self::Chunk;

    /// [`Backend::eq_block`] of the pair backend
    ///
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    unsafe fn eq_block_pair(chunks: &PairChunks<Self>, needle: Self::Chunk) -> u128 {
        unsafe { eq_block_chunks::<PairBackend<Self>, 4>(chunks, [needle; 2]) }
    }
}

/// The blocks of a [`PairBackend`]: four chunks of two registers
pub(crate) type PairChunks<R> = ChunkBlock<[<R as Backend>::Chunk; 2], 4>;

/// The prefilter's backend over 16 byte registers: chunks of two registers,
/// so the chunk walks branch half as often. Needle chunks are splats, so only
/// their first register is read
#[derive(Debug, Clone, Copy)]
pub(crate) struct PairBackend<R>(PhantomData<R>);

impl<R: Register> Backend for PairBackend<R> {
    const LANES: usize = 2 * R::LANES;

    type Chunk = [R::Chunk; 2];
    type Mask = u32;
    type Block = u128;
    type Chunks = PairChunks<R>;

    fn is_available() -> bool {
        R::is_available()
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> Self::Chunk {
        unsafe { [R::splat(c); 2] }
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        unsafe { R::movemask_pair(R::cmpeq(a[0], b[0]), R::cmpeq(a[1], b[0])) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { [R::load(ptr), R::load(ptr.add(R::LANES))] }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        unsafe {
            let (a, b) = (needle.0[0], needle.1[0]);
            let lo = R::or(R::cmpeq(a, chunk[0]), R::cmpeq(b, chunk[0]));
            let hi = R::or(R::cmpeq(a, chunk[1]), R::cmpeq(b, chunk[1]));
            R::movemask_pair(lo, hi)
        }
    }

    // padding costs once per haystack; masking costs per row. Not on NEON
    // (register pressure)
    const PADDED_BLOCKS: bool = !cfg!(target_arch = "aarch64");

    #[inline(always)]
    unsafe fn pad(chunk: Self::Chunk, len: usize) -> Self::Chunk {
        unsafe {
            let pad = PAD_FROM.as_ptr().add(64 - len);
            [
                R::or(chunk[0], R::load(pad)),
                R::or(chunk[1], R::load(pad.add(R::LANES))),
            ]
        }
    }

    #[inline(always)]
    unsafe fn load_block(haystack: &[u8]) -> (Self::Chunks, Self::Block) {
        unsafe { load_block_chunks::<Self, 4>(haystack) }
    }

    #[inline(always)]
    unsafe fn fold_block(chunks: &mut Self::Chunks) {
        unsafe {
            for chunk in chunks.chunks.iter_mut().take(chunks.count) {
                // (no closures: target features)
                *chunk = [R::fold(chunk[0]), R::fold(chunk[1])];
            }
        }
    }

    #[inline(always)]
    unsafe fn eq_block(chunks: &Self::Chunks, needle: Self::Chunk) -> Self::Block {
        unsafe { R::eq_block_pair(chunks, needle[0]) }
    }
}
