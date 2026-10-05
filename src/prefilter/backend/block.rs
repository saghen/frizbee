//! The blocks of haystack bytes the prefilter's walks read: their masks
//! ([`BlockMask`]), and loads and compares from per-chunk ones

use core::{
    fmt::Debug,
    ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr},
};

use super::{Backend, BitMaskOps};
use crate::prefilter::algo::load_window;

/// A block mask truncated to its low bytes
pub(crate) trait Narrow<M> {
    fn narrow(self) -> M;
}

impl<M: BlockMask> Narrow<M> for M {
    #[inline(always)]
    fn narrow(self) -> M {
        self
    }
}

impl Narrow<u64> for u128 {
    #[inline(always)]
    fn narrow(self) -> u64 {
        self as u64
    }
}

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
    + Narrow<u64>
{
    /// Bytes per block
    const BITS: usize;
    /// `self + rhs + carry` and the carry out
    fn carrying_add(self, rhs: Self, carry: bool) -> (Self, bool);
    /// Every bit when `cond`, none otherwise
    fn all_if(cond: bool) -> Self;
    /// Every bit above the lowest set bit (none when zero)
    fn above_lowest(self) -> Self;
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

            #[inline(always)]
            fn above_lowest(self) -> Self {
                !(self ^ self.wrapping_sub(1))
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

/// A block as `N` chunks, of which the first `count` hold haystack bytes: the
/// per-chunk compares stop there
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
        debug_assert!(!haystack.is_empty());
        let mut chunks = [B::splat(0); N];
        let count = haystack.len().div_ceil(B::LANES).min(N);
        // only the last chunk may pass the haystack's end
        let last = count - 1;
        let tail = load_window::<B>(haystack, last * B::LANES, haystack.len()).0;
        for (c, chunk) in chunks.iter_mut().enumerate().take(count) {
            *chunk = if c == last {
                tail
            } else {
                B::load(haystack.as_ptr().add(c * B::LANES))
            };
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
    // Unrolled, so the shifts are constant (a variable u128 shift is slow)
    for (c, &chunk) in block.chunks.iter().enumerate() {
        if c < block.count {
            let eq = unsafe { B::eq(chunk, needle) };
            bits = bits | (B::Block::from(eq.into()) << (c * B::LANES));
        }
    }
    bits
}

/// The `remaining` bytes at `ptr`, zero padded. Only the copy is outlined, so
/// the caller loads it with its target features
///
/// # Safety
/// `ptr` must point to `remaining <= N` readable bytes.
#[cold]
#[inline(never)]
pub(crate) unsafe fn copy_partial<const N: usize>(ptr: *const u8, remaining: usize) -> [u8; N] {
    let mut data = [0u8; N];
    unsafe { core::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), remaining) };
    data
}
