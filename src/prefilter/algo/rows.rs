//! The needle rows of the typo walk: where each row (a byte, or a whole
//! scalar for Unicode needles) occurs in a block of haystack bytes, and the
//! walk within one block.

use super::Prefilter;
use super::typos::{lcs_row, walk_bytes};
use crate::prefilter::{
    UnicodeChar, Window,
    backend::{Backend, BlockMask, Narrow},
};

/// Overlap between chunks due to unicode chars being up 4 bytes long
/// This ensures that one of the chunks will see the full multi-byte char
const UNICODE_OVERLAP: usize = 3;

/// A needle scalar splatted for the block walk
#[derive(Debug, Clone)]
pub(crate) struct ScalarRow<B: Backend> {
    bytes: [B::Chunk; 4],
    flipped: [B::Chunk; 4],
    char: UnicodeChar,
}

impl<B: Backend> ScalarRow<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub(super) unsafe fn new(c: &UnicodeChar) -> Self {
        // (no closures: target features)
        unsafe {
            let mut row = Self {
                bytes: [B::splat(0); 4],
                flipped: [B::splat(0); 4],
                char: *c,
            };
            for i in 0..4 {
                row.bytes[i] = B::splat(c.chars[i]);
                row.flipped[i] = B::splat(c.flipped_chars[i]);
            }
            row
        }
    }
}

/// A block of haystack bytes with the mask of the bytes inside the haystack
pub(super) type Block<B> = (<B as Backend>::Chunks, <B as Backend>::Block);

/// Positions of the block equal to `needle`, among the low `M::BITS` bytes
#[inline(always)]
unsafe fn byte_occ<B: Backend, M: BlockMask>((chunks, valid): &Block<B>, needle: B::Chunk) -> M
where
    B::Block: Narrow<M>,
{
    let occ = unsafe { B::eq_block(chunks, needle) };
    // padding never equals a needle byte
    if B::PADDED_BLOCKS {
        occ.narrow()
    } else {
        (occ & *valid).narrow()
    }
}

/// Positions of the block where the `len` bytes of a scalar end
#[inline(always)]
unsafe fn scalar_occ<B: Backend, M: BlockMask>(
    block: &Block<B>,
    bytes: &[B::Chunk; 4],
    len: usize,
) -> M
where
    B::Block: Narrow<M>,
{
    unsafe {
        let mut occ = byte_occ::<B, M>(block, bytes[len - 1]);
        for back in 1..len {
            occ = occ & (byte_occ::<B, M>(block, bytes[len - 1 - back]) << back);
        }
        occ
    }
}

impl<B: Backend> Prefilter<B> {
    #[inline(always)]
    pub(super) fn rows<const UNICODE: bool>(&self) -> usize {
        if UNICODE {
            self.needle_scalars.len()
        } else {
            self.needle_ascii.len()
        }
    }

    /// Loads the block at the start of `haystack`, lowercased for a case
    /// insensitive ASCII needle
    #[inline(always)]
    pub(super) unsafe fn load_block<const UNICODE: bool>(&self, haystack: &[u8]) -> Block<B> {
        unsafe {
            let (mut chunks, valid) = B::load_block(haystack);
            if !UNICODE && !self.case_sensitive {
                B::fold_block(&mut chunks);
            }
            (chunks, valid)
        }
    }

    /// Positions among `unseen` where needle row `i` ends, and where those
    /// occurrences start
    #[inline(always)]
    pub(super) unsafe fn row_occ<const UNICODE: bool, M: BlockMask>(
        &self,
        block: &Block<B>,
        unseen: M,
        i: usize,
    ) -> (M, M)
    where
        B::Block: Narrow<M>,
    {
        unsafe {
            if !UNICODE {
                let occ = byte_occ::<B, M>(block, self.needle_ascii.get_unchecked(i).0) & unseen;
                return (occ, occ);
            }
            let row = self.needle_scalars.get_unchecked(i);
            let mut occ = scalar_occ::<B, M>(block, &row.bytes, row.char.len);
            if row.char.has_flip {
                occ = occ | scalar_occ::<B, M>(block, &row.flipped, row.char.len);
            }
            let occ = occ & unseen;
            (occ, occ >> (row.char.len - 1))
        }
    }

    #[inline(always)]
    pub(super) unsafe fn match_single_block<const UNICODE: bool, M: BlockMask>(
        &self,
        haystack: &[u8],
        max_typos: usize,
    ) -> Window
    where
        B::Block: Narrow<M>,
    {
        let rows = BlockRows::<B, UNICODE> {
            prefilter: self,
            block: unsafe { self.load_block::<UNICODE>(haystack) },
        };
        unsafe { walk_single::<M, _>(&rows, self.rows::<UNICODE>(), max_typos) }
    }
}

/// Occurrences of needle row `i` within one block: where it ends,
/// and where those occurrences start
pub(super) trait SingleRows<M: BlockMask> {
    /// Whether a row is one compare (ASCII), cheap enough to read at several
    /// sites
    const ASCII: bool;

    unsafe fn occ(&self, i: usize) -> (M, M);
}

struct BlockRows<'a, B: Backend, const UNICODE: bool> {
    prefilter: &'a Prefilter<B>,
    block: Block<B>,
}

impl<B: Backend, M: BlockMask, const UNICODE: bool> SingleRows<M> for BlockRows<'_, B, UNICODE>
where
    B::Block: Narrow<M>,
{
    const ASCII: bool = !UNICODE;

    #[inline(always)]
    unsafe fn occ(&self, i: usize) -> (M, M) {
        unsafe {
            self.prefilter
                .row_occ::<UNICODE, M>(&self.block, M::all(), i)
        }
    }
}

/// Bytes shared by consecutive blocks of the multi-block walk, so that every
/// scalar lies within one of them
#[inline(always)]
pub(super) const fn overlap<const UNICODE: bool>() -> usize {
    if UNICODE { UNICODE_OVERLAP } else { 0 }
}

/// The LCS walk over a haystack within one block of `M::BITS` bytes
#[inline(always)]
unsafe fn walk_single<M: BlockMask, R: SingleRows<M>>(
    rows: &R,
    n: usize,
    max_typos: usize,
) -> Window {
    // Row 0 is read unchecked: empty needles never reach the prefilter
    // (`Matcher::compile` skips them)
    debug_assert!(n > 0);
    // Only the first and last `max_typos + 1` rows feed the window: ASCII
    // rows are peeled off the middle loop, Unicode rows (many compares each)
    // are read at one site and masked
    let (lcs, head, tail) = if R::ASCII {
        match max_typos {
            1 => unsafe { walk_bytes(rows, n, 1) },
            2 => unsafe { walk_bytes(rows, n, 2) },
            _ => unsafe { walk_bytes(rows, n, max_typos) },
        }
    } else {
        let mut v = M::all();
        let mut lcs = 0usize;
        let mut head = M::from(0);
        let mut tail = M::from(0);
        let tail_from = n - max_typos - 1;
        for i in 0..n {
            let (occ, occ_start) = unsafe { lcs_row(rows, i, &mut v, &mut lcs) };
            head = head | (occ_start & M::all_if(i <= max_typos));
            tail = tail | (occ & M::all_if(i >= tail_from));
        }
        (lcs, head, tail)
    };

    // no early return on a miss: the branch would mispredict, and the
    // batched callers only read the window of a hit
    (
        lcs + max_typos >= n,
        head.trailing_zeros(),
        M::BITS - tail.leading_zeros(),
    )
}
