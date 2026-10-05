//! The blocks of haystack bytes the prefilter's walks read: their masks
//! ([`BlockMask`]), and loads and compares from per-chunk ones

use core::{
    fmt::Debug,
    ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr},
};

use super::{Backend, BitMaskOps};
use crate::prefilter::algo::load_window;

/// A bit per byte of a block of haystack bytes for the prefilter walks
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
    + BitXor<Output = Self>
    + Not<Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
{
    /// Bytes per block
    const BITS: usize;
    /// `self + rhs + carry` and the carry out
    fn carrying_add(self, rhs: Self, carry: bool) -> (Self, bool);
    /// Every bit when `cond`, none otherwise
    fn all_if(cond: bool) -> Self;
}

macro_rules! impl_block {
    ($ty:ty) => {
        impl BlockMask for $ty {
            const BITS: usize = <$ty>::BITS as usize;

            #[inline(always)]
            fn carrying_add(self, rhs: Self, carry: bool) -> (Self, bool) {
                let (sum, c1) = self.overflowing_add(rhs);
                let (sum, c2) = sum.overflowing_add(carry as $ty);
                (sum, c1 | c2)
            }

            #[inline(always)]
            fn all_if(cond: bool) -> Self {
                (cond as $ty).wrapping_neg()
            }
        }
    };
}

impl_block!(u64);
impl_block!(u128);

/// The byte padded blocks hold past the haystack's end: it never occurs in
/// UTF-8, so no needle row matches it
#[cfg(target_arch = "x86_64")]
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
    let mut bits = B::Block::from(0);
    for (c, &chunk) in block.chunks.iter().enumerate().take(block.count) {
        let eq = unsafe { B::eq(chunk, needle) };
        bits = bits | (B::Block::from(eq.into()) << (c * B::LANES));
    }
    bits
}
