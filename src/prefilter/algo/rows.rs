//! The needle rows shared by the 0-typo and typo walks: where each row (a
//! byte, or a whole scalar for Unicode needles) occurs in a block, chunk or
//! two chunks of haystack bytes, and the walk within one block.

use super::typos::{lcs_row, walk_bytes};
use super::{Prefilter, load_window};
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

/// Positions of `chunk` where a needle scalar (either case) ends
#[inline(always)]
unsafe fn row_ends<B: Backend>(chunk: B::Chunk, row: &ScalarRow<B>) -> u64 {
    unsafe {
        let mut ends = scalar_ends::<B>(chunk, &row.bytes, row.char.len);
        if row.char.has_flip {
            ends |= scalar_ends::<B>(chunk, &row.flipped, row.char.len);
        }
        ends
    }
}

/// Positions of `chunk` where the `len` (1 to 4) bytes of a scalar end, one
/// bit per byte: the compares of its bytes shifted into place and ANDed
#[inline(always)]
unsafe fn scalar_ends<B: Backend>(chunk: B::Chunk, bytes: &[B::Chunk; 4], len: usize) -> u64 {
    unsafe {
        // unrolled per width: a loop over `len` costs the 0-typo walk
        match len {
            1 => eq_bits::<B>(chunk, bytes[0]),
            2 => eq_bits::<B>(chunk, bytes[1]) & (eq_bits::<B>(chunk, bytes[0]) << 1),
            3 => {
                eq_bits::<B>(chunk, bytes[2])
                    & (eq_bits::<B>(chunk, bytes[1]) << 1)
                    & (eq_bits::<B>(chunk, bytes[0]) << 2)
            }
            _ => {
                eq_bits::<B>(chunk, bytes[3])
                    & (eq_bits::<B>(chunk, bytes[2]) << 1)
                    & (eq_bits::<B>(chunk, bytes[1]) << 2)
                    & (eq_bits::<B>(chunk, bytes[0]) << 3)
            }
        }
    }
}

#[inline(always)]
unsafe fn eq_bits<B: Backend>(chunk: B::Chunk, needle: B::Chunk) -> u64 {
    unsafe { B::eq(chunk, needle) }.into()
}

impl<B: Backend> Prefilter<B> {
    /// Positions of `chunk` where needle row `i` ends, and the row's length
    /// less one (from an end back to its start)
    #[inline(always)]
    pub(super) unsafe fn chunk_occ<const UNICODE: bool>(
        &self,
        chunk: B::Chunk,
        i: usize,
    ) -> (u64, usize) {
        unsafe {
            if !UNICODE {
                let needle = *self.needle_ascii.get_unchecked(i);
                return (B::occ(chunk, needle).into(), 0);
            }
            let row = self.needle_scalars.get_unchecked(i);
            (row_ends::<B>(chunk, row), row.char.len - 1)
        }
    }

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
        unsafe { walk_single::<M, _, false>(&rows, self.rows::<UNICODE>(), max_typos) }
    }

    /// [`Self::match_single_block`] for haystacks of at most one chunk, from
    /// that chunk's compares (`valid` masks the haystack's bytes)
    #[inline(always)]
    pub(super) unsafe fn match_single_chunk<const UNICODE: bool, const GREEDY: bool>(
        &self,
        chunk: B::Chunk,
        valid: u64,
        max_typos: usize,
    ) -> Window {
        let rows = ChunkRows::<B, UNICODE> {
            prefilter: self,
            chunk,
            valid,
        };
        unsafe { walk_single::<u64, _, GREEDY>(&rows, self.rows::<UNICODE>(), max_typos) }
    }

    /// The greedy walk of [`Self::match_single_chunk`] for ASCII haystacks of
    /// one to two chunks, over the first chunk and the in-bounds one at the
    /// end when both fit a u64 (AVX2, SSE, wasm), else (AVX-512) the bounded
    /// one after the first
    #[inline(always)]
    pub(super) unsafe fn match_two_chunks(&self, haystack: &[u8]) -> Window {
        let len = haystack.len();
        debug_assert!(len > B::LANES && len <= 2 * B::LANES);
        let head = unsafe { B::load(haystack.as_ptr()) };
        let n = self.rows::<false>();
        if 2 * B::LANES <= 64 {
            let shift = len - B::LANES;
            let tail = unsafe { B::load(haystack.as_ptr().add(shift)) };
            let rows = TwoChunkRows::<B> {
                prefilter: self,
                head,
                tail,
                valid: u64::MAX,
                shift,
            };
            return unsafe { walk_single::<u64, _, true>(&rows, n, 0) };
        }
        let (tail, valid) = unsafe { load_window::<B>(haystack, B::LANES, len) };
        let rows = TwoChunkRows::<B> {
            prefilter: self,
            head,
            tail,
            valid: valid.into(),
            shift: B::LANES,
        };
        unsafe { walk_single::<B::Block, _, true>(&rows, n, 0) }
    }
}

/// Occurrences of needle row `i` within one block or chunk: where it ends,
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

/// Bytes shared by consecutive chunks of the chunk walks, so that every
/// scalar lies within one of them
#[inline(always)]
pub(super) const fn overlap<const UNICODE: bool>() -> usize {
    if UNICODE { UNICODE_OVERLAP } else { 0 }
}

struct ChunkRows<'a, B: Backend, const UNICODE: bool> {
    prefilter: &'a Prefilter<B>,
    chunk: B::Chunk,
    valid: u64,
}

impl<B: Backend, const UNICODE: bool> SingleRows<u64> for ChunkRows<'_, B, UNICODE> {
    const ASCII: bool = !UNICODE;

    #[inline(always)]
    unsafe fn occ(&self, i: usize) -> (u64, u64) {
        unsafe {
            let (occ, back) = self.prefilter.chunk_occ::<UNICODE>(self.chunk, i);
            let occ = occ & self.valid;
            (occ, occ >> back)
        }
    }
}

/// The ASCII rows of [`Prefilter::match_two_chunks`]: `tail` (its haystack
/// bytes `valid`) starts at `shift`
struct TwoChunkRows<'a, B: Backend> {
    prefilter: &'a Prefilter<B>,
    head: B::Chunk,
    tail: B::Chunk,
    valid: u64,
    shift: usize,
}

impl<B: Backend, M: BlockMask> SingleRows<M> for TwoChunkRows<'_, B> {
    const ASCII: bool = true;

    #[inline(always)]
    unsafe fn occ(&self, i: usize) -> (M, M) {
        unsafe {
            let (head, _) = self.prefilter.chunk_occ::<false>(self.head, i);
            let (tail, _) = self.prefilter.chunk_occ::<false>(self.tail, i);
            let occ = M::from(head) | (M::from(tail & self.valid) << self.shift);
            (occ, occ)
        }
    }
}

/// The walk over a haystack within one block (or chunk) of `M::BITS` bytes:
/// greedy for 0 typos, the LCS otherwise
#[inline(always)]
unsafe fn walk_single<M: BlockMask, R: SingleRows<M>, const GREEDY: bool>(
    rows: &R,
    n: usize,
    max_typos: usize,
) -> Window {
    // Row 0 is read unchecked: empty needles never reach the prefilter
    // (`Matcher::compile` skips them)
    debug_assert!(n > 0);
    if GREEDY {
        debug_assert_eq!(max_typos, 0);
        // The greedy walk is shorter: each row keeps its occurrences past the
        // lowest one of the previous row
        let (first, head) = unsafe { rows.occ(0) };
        let mut reached = first;
        let mut tail = first;
        for i in 1..n {
            (tail, _) = unsafe { rows.occ(i) };
            reached = tail & reached.above_lowest();
        }
        if reached.is_zero() {
            return (false, 0, 0);
        }
        return (true, head.trailing_zeros(), M::BITS - tail.leading_zeros());
    }

    // Only the first and last `max_typos + 1` rows feed the window: ASCII
    // rows are peeled off the middle loop, Unicode rows (many compares each)
    // are read at one site and masked
    let (lcs, head, tail) = if R::ASCII {
        match max_typos {
            // (not on wasm, where the copies cost size without a gain)
            #[cfg(not(target_arch = "wasm32"))]
            1 => unsafe { walk_bytes(rows, n, 1) },
            #[cfg(not(target_arch = "wasm32"))]
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
