//! Typo-tolerant prefilter: a bit-parallel greedy walk over blocks of haystack
//! bytes.
//!
//! The haystack is read in blocks of [`BlockMask::BITS`] bytes. Each needle row
//! (a byte, or a whole scalar for Unicode needles) has an occurrence mask
//! `O[i]` with a HI bit wherever the row ends in the block. `R[i][t]` holds the
//! positions where `needle[..=i]` can end after skipping at most `t` rows:
//!
//! ```text
//! R[i][t] = (O[i] & above(R[i-1][t])) | R[i-1][t-1]
//! ```
//!
//! `above(x)` is every position past the lowest set bit of `x`, since the
//! leftmost end is always the best one to continue from. The second term skips
//! row `i`. The needle matches with at most `k` typos when `R[n-1][k]` is
//! non-empty.
//!
//! Haystacks longer than a block only carry one bit per state across blocks:
//! whether it was reached in an earlier block, in which case every position of
//! the current block is past it.
//!
//! The returned window starts at the first occurrence of any of the first
//! `k + 1` rows and ends after the last occurrence of any of the last `k + 1`.

use super::Prefilter;
use crate::prefilter::{
    UnicodeChar, Window,
    backend::{Backend, BitMaskOps, BlockMask},
};

/// Overlap between chunks due to unicode chars being up 4 bytes long
/// This ensures that one of the chunks will see the full multi-byte char
const UNICODE_OVERLAP: usize = 3;

/// A needle scalar splatted for the block walk
#[derive(Debug, Clone)]
pub(crate) struct ScalarRow<B: Backend> {
    bytes: [B::Chunk; 4],
    flipped: [B::Chunk; 4],
    pub(super) char: UnicodeChar,
}

impl<B: Backend> ScalarRow<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    pub(crate) unsafe fn new(c: &UnicodeChar) -> Self {
        Self {
            bytes: c.chars.map(|b| unsafe { B::splat(b) }),
            flipped: c.flipped_chars.map(|b| unsafe { B::splat(b) }),
            char: *c,
        }
    }
}

/// A block of haystack bytes with the mask of the bytes inside the haystack
type Block<B> = (<B as Backend>::Chunks, <B as Backend>::Block);

/// Advances `state` from `R[i-1]` to `R[i]` given row `i`'s occurrences. Bit
/// `t` of `ended` says `R[i-1][t]` was reached in an earlier block.
#[inline(always)]
fn step<B: Backend>(state: &mut [B::Block], i: usize, occ: B::Block, ended: u64) {
    for t in (0..state.len()).rev() {
        // the first `t` rows can all be skipped, so the walk may start anywhere
        let above = if t >= i || (ended >> t) & 1 != 0 {
            B::Block::ALL
        } else {
            state[t].above_lowest()
        };
        let skipped = if t > 0 { state[t - 1] } else { B::Block::ZERO };
        state[t] = (occ & above) | skipped;
    }
}

/// Positions of the block equal to `needle`
#[inline(always)]
unsafe fn byte_occ<B: Backend>((chunks, valid): &Block<B>, needle: B::Chunk) -> B::Block {
    let occ = unsafe { B::eq_block(chunks, needle) };
    // padding never equals a needle byte
    if B::PADDED_BLOCKS { occ } else { occ & *valid }
}

/// Positions of the block where the `len` bytes of a scalar end
#[inline(always)]
unsafe fn scalar_occ<B: Backend>(block: &Block<B>, bytes: &[B::Chunk; 4], len: usize) -> B::Block {
    unsafe {
        let mut occ = byte_occ::<B>(block, bytes[len - 1]);
        for back in 1..len {
            occ = occ & (byte_occ::<B>(block, bytes[len - 1 - back]) << back);
        }
        occ
    }
}

impl<B: Backend> Prefilter<B> {
    /// Matches with at most `K` typos (`K <= 2`)
    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub(crate) unsafe fn match_haystack_typos<const K: usize, const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
    ) -> Window {
        let mut state = [B::Block::ZERO; 3];
        unsafe { self.match_haystack_block::<UNICODE>(haystack, &mut state[..=K]) }
    }

    /// Matches with a runtime typo budget
    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub(crate) unsafe fn match_haystack_typos_dyn<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        max_typos: usize,
    ) -> Window {
        let mut state = core::mem::take(&mut self.state);
        state.clear();
        state.resize(max_typos + 1, B::Block::ZERO);
        let window = unsafe { self.match_haystack_block::<UNICODE>(haystack, &mut state) };
        self.state = state;
        window
    }

    /// `state` holds one zeroed entry per typo count
    #[inline(always)]
    unsafe fn match_haystack_block<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        state: &mut [B::Block],
    ) -> Window {
        let len = haystack.len();
        // every row can be skipped, or the budget exceeds the bits of `ended`
        if self.rows::<UNICODE>() < state.len() || state.len() > u64::BITS as usize {
            return (true, 0, len);
        }
        if len == 0 {
            return (false, 0, 0);
        }
        if len <= B::Block::BITS {
            unsafe { self.match_single_block::<UNICODE>(haystack, state) }
        } else {
            unsafe { self.match_multi_block::<UNICODE>(haystack, state) }
        }
    }

    #[inline(always)]
    fn rows<const UNICODE: bool>(&self) -> usize {
        if UNICODE {
            self.needle_scalars.len()
        } else {
            self.needle_folded.len()
        }
    }

    /// Loads the block at the start of `haystack`, lowercased for a case
    /// insensitive ASCII needle
    #[inline(always)]
    unsafe fn load_block<const UNICODE: bool>(&self, haystack: &[u8]) -> Block<B> {
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
    unsafe fn row_occ<const UNICODE: bool>(
        &self,
        block: &Block<B>,
        unseen: B::Block,
        i: usize,
    ) -> (B::Block, B::Block) {
        unsafe {
            if !UNICODE {
                let occ = byte_occ::<B>(block, *self.needle_folded.get_unchecked(i)) & unseen;
                return (occ, occ);
            }
            let row = self.needle_scalars.get_unchecked(i);
            let mut occ = scalar_occ::<B>(block, &row.bytes, row.char.len);
            if row.char.has_flip {
                occ = occ | scalar_occ::<B>(block, &row.flipped, row.char.len);
            }
            let occ = occ & unseen;
            (occ, occ >> (row.char.len - 1))
        }
    }

    #[inline(always)]
    unsafe fn match_single_block<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        state: &mut [B::Block],
    ) -> Window {
        let n = self.rows::<UNICODE>();
        let max_typos = state.len() - 1;
        let block = unsafe { self.load_block::<UNICODE>(haystack) };
        let mut head = B::Block::ZERO;
        let mut tail = B::Block::ZERO;

        // Only the first and last `max_typos + 1` rows feed the window, so
        // they are peeled off the plain middle loop
        let tail_from = n - max_typos - 1;
        for i in 0..n.min(max_typos + 1) {
            let (occ, occ_start) = unsafe { self.row_occ::<UNICODE>(&block, B::Block::ALL, i) };
            head = head | occ_start;
            if i >= tail_from {
                tail = tail | occ;
            }
            step::<B>(state, i, occ, 0);
        }
        for i in max_typos + 1..tail_from {
            let (occ, _) = unsafe { self.row_occ::<UNICODE>(&block, B::Block::ALL, i) };
            step::<B>(state, i, occ, 0);
        }
        for i in tail_from.max(max_typos + 1)..n {
            let (occ, _) = unsafe { self.row_occ::<UNICODE>(&block, B::Block::ALL, i) };
            tail = tail | occ;
            step::<B>(state, i, occ, 0);
        }

        if state[max_typos].is_zero() {
            return (false, 0, haystack.len());
        }
        (
            true,
            head.trailing_zeros(),
            B::Block::BITS - tail.leading_zeros(),
        )
    }

    #[inline(always)]
    unsafe fn match_multi_block<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        state: &mut [B::Block],
    ) -> Window {
        let len = haystack.len();
        let n = self.rows::<UNICODE>();
        let max_typos = state.len() - 1;
        let overlap = if UNICODE { UNICODE_OVERLAP } else { 0 };

        // bit `t` of `ended[i]` says `R[i][t]` was reached in an earlier block
        self.ended.clear();
        self.ended.resize(n, 0u64);

        let mut match_start = usize::MAX;
        let mut match_end = 0usize;
        let mut start = 0usize;

        while start < len {
            let block = unsafe { self.load_block::<UNICODE>(&haystack[start..]) };
            // scalars ending inside the overlap were seen by the previous block
            let unseen = if start > 0 {
                !B::Block::first_n(overlap)
            } else {
                B::Block::ALL
            };
            let mut head = B::Block::ZERO;
            let mut tail = B::Block::ZERO;
            let mut prev_ended = 0u64;
            state.fill(B::Block::ZERO);

            for i in 0..n {
                let (occ, occ_start) = unsafe { self.row_occ::<UNICODE>(&block, unseen, i) };
                if i <= max_typos {
                    head = head | occ_start;
                }
                if i + max_typos + 1 >= n {
                    tail = tail | occ;
                }
                step::<B>(state, i, occ, prev_ended);

                // skipping row `i` keeps whatever the previous row had reached
                let mut now_ended = self.ended[i] | (prev_ended << 1);
                for (t, r) in state.iter().enumerate() {
                    now_ended |= (!r.is_zero() as u64) << t;
                }
                prev_ended = core::mem::replace(&mut self.ended[i], now_ended);
            }

            if match_start == usize::MAX && !head.is_zero() {
                match_start = start + head.trailing_zeros();
            }
            if !tail.is_zero() {
                match_end = start + B::Block::BITS - tail.leading_zeros();
            }
            start += B::Block::BITS - overlap;
        }

        if (self.ended[n - 1] >> max_typos) & 1 == 0 {
            return (false, 0, len);
        }
        (true, match_start, match_end)
    }
}
