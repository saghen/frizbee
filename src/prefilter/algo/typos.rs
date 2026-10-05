//! The typo walk: bit-parallel LCS over blocks of haystack bytes.
//!
//! The haystack is read in blocks of [`BlockMask::BITS`] bytes. Each needle row
//! (a byte, or a whole scalar for Unicode needles) has an occurrence mask
//! `O[i]` with a HI bit wherever the row ends in the block. The needle matches
//! with at most `k` typos (deleted rows) when the longest common subsequence
//! of the rows and the haystack is at least `n - k`, which the bit-vector LCS
//! of Crochemore et al. (2001) computes in one add per row, independent of `k`:
//!
//! ```text
//! V = ALL
//! V = (V + (V & O[i])) | (V & !O[i])   for each row i
//! ```
//!
//! The zeros of `V` mark where the LCS of the rows so far grows. Every bit
//! above the last zero is set, so a row extends the LCS exactly when its add
//! carries out of the top: the LCS is the number of carries out. The add
//! carries towards later haystack positions, so haystacks longer than a block
//! carry one bit per row across blocks.
//!
//! The returned window starts at the first occurrence of any of the first
//! `k + 1` rows and ends after the last occurrence of any of the last `k + 1`.
//!

use super::Prefilter;
use super::rows::{SingleRows, overlap};
use crate::prefilter::{
    Window,
    backend::{Backend, BitMaskOps, BlockMask},
};

impl<B: Backend> Prefilter<B> {
    /// Matches with at most `max_typos` deleted rows
    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub(crate) unsafe fn match_haystack_typos<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        max_typos: usize,
    ) -> Window {
        let len = haystack.len();
        // every row can be skipped
        if self.rows::<UNICODE>() <= max_typos {
            return (true, 0, len);
        }
        if len == 0 {
            return (false, 0, 0);
        }
        if len <= B::Block::BITS {
            unsafe { self.match_single_block::<UNICODE>(haystack, max_typos) }
        } else {
            unsafe { self.match_multi_block::<UNICODE>(haystack, max_typos) }
        }
    }

    #[inline(always)]
    unsafe fn match_multi_block<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        max_typos: usize,
    ) -> Window {
        let len = haystack.len();
        let n = self.rows::<UNICODE>();
        let overlap = overlap::<UNICODE>();

        // carry of each row's add into the next block
        self.carries.clear();
        self.carries.resize(n, 0);

        let mut match_start = usize::MAX;
        let mut match_end = 0usize;
        let mut start = 0usize;

        while start < len {
            let block = unsafe { self.load_block::<UNICODE>(&haystack[start..]) };
            // scalars ending inside the overlap were seen by the previous block
            let unseen = if start > 0 {
                !B::Block::first_n(overlap)
            } else {
                B::Block::all()
            };
            let mut v = B::Block::all();
            let mut head = B::Block::from(0);
            let mut tail = B::Block::from(0);

            for i in 0..n {
                let (occ, occ_start) = unsafe { self.row_occ::<UNICODE>(&block, unseen, i) };
                if i <= max_typos {
                    head = head | occ_start;
                }
                if i + max_typos + 1 >= n {
                    tail = tail | occ;
                }
                let carry = unsafe { self.carries.get_unchecked_mut(i) };
                let carried;
                (v, carried) = step::<B::Block>(v, occ, *carry != 0);
                *carry = carried as u64;
            }

            if match_start == usize::MAX && !head.is_zero() {
                match_start = start + head.trailing_zeros();
            }
            if !tail.is_zero() {
                match_end = start + B::Block::BITS - tail.leading_zeros();
            }
            start += B::Block::BITS - overlap;
        }
        let lcs = self.carries.iter().sum::<u64>() as usize;

        if lcs + max_typos < n {
            return (false, 0, len);
        }
        (true, match_start, match_end)
    }
}

/// Advances `v` over a row with occurrences `occ`, with the carry of the
/// same row from the previous block
#[inline(always)]
fn step<M: BlockMask>(v: M, occ: M, carry: bool) -> (M, bool) {
    let matched = v & occ;
    let (sum, carry) = v.carrying_add(matched, carry);
    // `v ^ matched` is `v & !occ`
    (sum | (v ^ matched), carry)
}

/// Reads row `i` and advances the LCS over it, returning its occurrences
#[inline(always)]
pub(super) unsafe fn lcs_row<M: BlockMask, R: SingleRows<M>>(
    rows: &R,
    i: usize,
    v: &mut M,
    lcs: &mut usize,
) -> (M, M) {
    let row = unsafe { rows.occ(i) };
    let carry;
    (*v, carry) = step(*v, row.0, false);
    *lcs += carry as usize;
    row
}

/// The LCS walk of `walk_single` over ASCII rows, with the rows that feed
/// the window peeled off the middle loop. Returns the LCS length and the
/// window's head and tail occurrences
#[inline(always)]
pub(super) unsafe fn walk_bytes<M: BlockMask, R: SingleRows<M>>(
    rows: &R,
    n: usize,
    max_typos: usize,
) -> (usize, M, M) {
    let mut v = M::all();
    let mut lcs = 0usize;
    let mut head = M::from(0);
    let mut tail = M::from(0);
    let tail_from = n - max_typos - 1;
    if tail_from > max_typos {
        // disjoint: both peeled loops run `max_typos + 1` times
        for i in 0..max_typos + 1 {
            let (_, occ_start) = unsafe { lcs_row(rows, i, &mut v, &mut lcs) };
            head = head | occ_start;
        }
        for i in max_typos + 1..tail_from {
            unsafe { lcs_row(rows, i, &mut v, &mut lcs) };
        }
        for i in 0..max_typos + 1 {
            let (occ, _) = unsafe { lcs_row(rows, tail_from + i, &mut v, &mut lcs) };
            tail = tail | occ;
        }
    } else {
        for i in 0..max_typos + 1 {
            let (occ, occ_start) = unsafe { lcs_row(rows, i, &mut v, &mut lcs) };
            head = head | occ_start;
            if i >= tail_from {
                tail = tail | occ;
            }
        }
        for i in max_typos + 1..n {
            let (occ, _) = unsafe { lcs_row(rows, i, &mut v, &mut lcs) };
            tail = tail | occ;
        }
    }
    (lcs, head, tail)
}
