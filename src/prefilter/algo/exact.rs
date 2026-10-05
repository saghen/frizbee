//! The 0-typo walk: each row keeps its occurrences past the lowest one of the
//! previous row, within one or two chunks, or chunk by chunk

use super::rows::overlap;
use super::{Prefilter, load_window};
use crate::prefilter::{
    Window,
    backend::{Backend, BitMaskOps, BlockMask},
};

impl<B: Backend> Prefilter<B> {
    /// Matches with 0 typos
    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub(crate) unsafe fn match_haystack<const UNICODE: bool>(&self, haystack: &[u8]) -> Window {
        let len = haystack.len();
        if len == 0 {
            return (false, 0, 0);
        }

        // Branch-free walk over one chunk
        if len <= B::LANES {
            let (chunk, valid) = unsafe { load_window::<B>(haystack, 0, len) };
            return unsafe { self.match_single_chunk::<UNICODE, true>(chunk, valid.into(), 0) };
        }
        // ASCII: two chunks, branch-free too (not on NEON, where the chunk
        // walk is faster)
        if !UNICODE && len <= 2 * B::LANES && !cfg!(target_arch = "aarch64") {
            return unsafe { self.match_two_chunks(haystack) };
        }
        unsafe { self.match_chunks::<UNICODE>(haystack) }
    }

    /// The greedy 0-typo walk chunk by chunk, stopping at the first chunk
    /// without the next row; once every row matched, scans back from the
    /// haystack's end for the window's end. The first row's walk is apart so
    /// each loop keeps few values live
    #[inline(always)]
    unsafe fn match_chunks<const UNICODE: bool>(&self, haystack: &[u8]) -> Window {
        let len = haystack.len();
        let step = B::LANES - overlap::<UNICODE>();
        let mut start = 0;
        let (mut chunk, valid) = unsafe { load_window::<B>(haystack, 0, len) };
        let mut unseen: u64 = valid.into();
        let (mut ends, back) = unsafe { self.chunk_occ::<UNICODE>(chunk, 0) };
        ends &= unseen;
        while ends == 0 {
            start += step;
            if start >= len {
                return (false, 0, len);
            }
            (chunk, unseen) = unsafe { self.load_unseen::<UNICODE>(haystack, start) };
            ends = unsafe { self.chunk_occ::<UNICODE>(chunk, 0) }.0 & unseen;
        }
        let match_start = start + ends.trailing_zeros() as usize - back;

        for i in 1..self.rows::<UNICODE>() {
            unseen &= ends.above_lowest();
            ends = unsafe { self.chunk_occ::<UNICODE>(chunk, i) }.0 & unseen;
            while ends == 0 {
                start += step;
                if start >= len {
                    return (false, match_start, len);
                }
                (chunk, unseen) = unsafe { self.load_unseen::<UNICODE>(haystack, start) };
                ends = unsafe { self.chunk_occ::<UNICODE>(chunk, i) }.0 & unseen;
            }
        }

        let end = if start + B::LANES >= len {
            start + u64::BITS as usize - ends.leading_zeros() as usize
        } else {
            unsafe { self.last_end::<UNICODE>(haystack) }
        };
        (true, match_start, end)
    }

    /// The chunk of the chunk walk at `start` (past the first) and its
    /// haystack bytes, less the overlap seen by the previous chunk
    #[inline(always)]
    unsafe fn load_unseen<const UNICODE: bool>(
        &self,
        haystack: &[u8],
        start: usize,
    ) -> (B::Chunk, u64) {
        let (chunk, valid) = unsafe { load_window::<B>(haystack, start, haystack.len()) };
        (chunk, valid.into() & !u64::first_n(overlap::<UNICODE>()))
    }

    /// End of the last occurrence of the last row in a haystack longer than
    /// a chunk, which has one
    #[inline(always)]
    unsafe fn last_end<const UNICODE: bool>(&self, haystack: &[u8]) -> usize {
        let last = self.rows::<UNICODE>() - 1;
        let mut start = haystack.len() - B::LANES;
        loop {
            let chunk = unsafe { B::load(haystack.as_ptr().add(start)) };
            let (ends, _) = unsafe { self.chunk_occ::<UNICODE>(chunk, last) };
            if ends != 0 {
                return start + u64::BITS as usize - ends.leading_zeros() as usize;
            }
            start = start.saturating_sub(B::LANES - overlap::<UNICODE>());
        }
    }
}
