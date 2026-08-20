//! Single-byte pre-rejection for 0-typo prefilter for rare needle bytes
//!
//! A prefilter match with 0-typos requires that every needle byte be present,
//! in order. This module samples the first [`SAMPLE`] haystacks to see if any
//! of the needle bytes are consistently absent (<20% presence). If so, future
//! calls will immediately reject the match by scanning for the single rare
//! needle char.

use super::load_window;
use crate::prefilter::backend::{Backend, BitMaskOps};
use alloc::vec::Vec;

/// Number of haystacks sampled before choosing a byte
const SAMPLE: u16 = 256;
/// Maximum percent of samples that may contain the chosen byte for it to be
/// considered rare
const MAX_PRESENCE_PERCENT: u32 = 20;

#[derive(Debug, Clone)]
pub(crate) struct RareByte<B: Backend> {
    /// `(original, case-flipped)` needle bytes
    needle: Vec<(B::Chunk, B::Chunk)>,
    /// Number of times each needle char has been absent from the sampled
    /// haystacks
    absent_counts: Vec<u16>,
    /// Number of remaining haystacks to sample
    remaining: u16,
    /// Chosen candidate, if the pre-reject is enabled
    chosen_needle_char: Option<(B::Chunk, B::Chunk)>,
}

impl<B: Backend> RareByte<B> {
    pub(crate) fn new(needle: Vec<(B::Chunk, B::Chunk)>) -> Self {
        let len = needle.len();
        Self {
            needle,
            absent_counts: alloc::vec![0; len],
            // ignore for single-byte needles
            remaining: if len >= 2 { SAMPLE } else { 0 },
            chosen_needle_char: None,
        }
    }

    /// Returns true when the haystack cannot match. Always returns false while
    /// sampling haystacks initially.
    ///
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub(crate) unsafe fn rejects(&mut self, haystack: &[u8]) -> bool {
        if let Some(chosen) = self.chosen_needle_char {
            return unsafe { !contains::<B>(haystack, chosen) };
        }
        if self.remaining > 0 {
            unsafe { self.sample(haystack) };
        }
        false
    }

    #[cold]
    #[inline(never)]
    unsafe fn sample(&mut self, haystack: &[u8]) {
        for (idx, candidate) in self.needle.iter().enumerate() {
            if unsafe { !contains::<B>(haystack, *candidate) } {
                self.absent_counts[idx] += 1;
            }
        }
        self.remaining -= 1;
        if self.remaining == 0 {
            let (best_idx, &best) = self
                .absent_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, absent)| **absent)
                .unwrap();
            if best as u32 * 100 >= SAMPLE as u32 * MAX_PRESENCE_PERCENT {
                self.chosen_needle_char = Some(self.needle[best_idx]);
            }
        }
    }
}

/// Whether either byte of `needle` occurs anywhere in the haystack
#[inline(always)]
unsafe fn contains<B: Backend>(haystack: &[u8], needle: (B::Chunk, B::Chunk)) -> bool {
    let len = haystack.len();
    let mut start = 0usize;
    while start < len {
        let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
        if !unsafe { B::occ(chunk, needle) }.and(chunk_mask).is_zero() {
            return true;
        }
        start += B::LANES;
    }
    false
}
