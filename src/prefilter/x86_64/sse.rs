use std::arch::x86_64::*;

use super::overlapping_load;
use crate::prefilter::{case_needle, scalar};

#[derive(Debug, Clone)]
pub struct PrefilterSSE {
    needle: Vec<(u8, u8)>,
}

impl PrefilterSSE {
    /// # Safety
    /// Caller must ensure that SSE2 is available at runtime
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn new(needle: &[u8]) -> Self {
        Self {
            needle: case_needle(needle),
        }
    }

    pub fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_feature_info()
            .is_some_and(|info| info.has_sse2())
    }

    /// Checks if the needle is wholly contained in the haystack, ignoring the exact order of the
    /// bytes. For example, if the needle is "test", the haystack "tset" will return true. However,
    /// the order does matter across 16 byte boundaries. The needle chars must include both the
    /// uppercase and lowercase variants of the character.
    ///
    /// # Safety
    /// The caller must ensure that SSE2 is available.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize) {
        let len = haystack.len();

        match len {
            0 => return (true, 0),
            1..=7 => {
                return (scalar::match_haystack(&self.needle, haystack), 0);
            }
            _ => {}
        };

        let mut can_skip_chunks = true;
        let mut skipped_chunks = 0;

        let mut needle_iter = self
            .needle
            .iter()
            .map(|&(c1, c2)| (_mm_set1_epi8(c1 as i8), _mm_set1_epi8(c2 as i8)));
        let mut needle_char = needle_iter.next().unwrap();

        for start in (0..len).step_by(16) {
            let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };

            loop {
                let mask = _mm_movemask_epi8(_mm_or_si128(
                    _mm_cmpeq_epi8(needle_char.1, haystack_chunk),
                    _mm_cmpeq_epi8(needle_char.0, haystack_chunk),
                ));
                if mask == 0 {
                    // No match, advance to next chunk
                    break;
                }

                // Progress to next needle char, if available
                if let Some(next_needle_char) = needle_iter.next() {
                    if can_skip_chunks {
                        skipped_chunks = start / 16;
                    }
                    can_skip_chunks = false;
                    needle_char = next_needle_char;
                } else {
                    return (true, skipped_chunks);
                }
            }
        }

        (false, skipped_chunks)
    }

    /// Checks if the needle is wholly contained in the haystack, ignoring the exact order of the
    /// bytes. For example, if the needle is "test", the haystack "tset" will return true. However,
    /// the order does matter across 16 byte boundaries. The needle chars must include both the
    /// uppercase and lowercase variants of the character.
    ///
    /// # Safety
    /// The caller must ensure that SSE2 is available.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn match_haystack_typos(&self, haystack: &[u8], max_typos: u16) -> (bool, usize) {
        let len = haystack.len();

        match len {
            0 => return (true, 0),
            1..=7 => {
                return (
                    scalar::match_haystack_typos(&self.needle, haystack, max_typos),
                    0,
                );
            }
            _ => {}
        };

        if max_typos >= 3 {
            return (true, 0);
        }

        let mut needle_iter = self
            .needle
            .iter()
            .map(|&(c1, c2)| (_mm_set1_epi8(c1 as i8), _mm_set1_epi8(c2 as i8)));
        let mut needle_char = needle_iter.next().unwrap();

        let mut typos = 0;
        loop {
            // TODO: this is slightly incorrect, because if we match on the third chunk,
            // we would only scan from the third chunk onwards for the next needle. Technically,
            // we should scan from the beginning of the haystack instead, but I believe the
            // previous memchr implementation had the same bug.
            for start in (0..len).step_by(16) {
                let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };

                loop {
                    let mask = _mm_movemask_epi8(_mm_or_si128(
                        _mm_cmpeq_epi8(needle_char.1, haystack_chunk),
                        _mm_cmpeq_epi8(needle_char.0, haystack_chunk),
                    ));
                    if mask == 0 {
                        // No match, advance to next chunk
                        break;
                    }

                    // Progress to next needle char, if available
                    if let Some(next_needle_char) = needle_iter.next() {
                        needle_char = next_needle_char;
                    } else {
                        return (true, 0);
                    }
                }
            }

            typos += 1;
            if typos > max_typos as usize {
                return (false, 0);
            }

            if let Some(next_needle_char) = needle_iter.next() {
                needle_char = next_needle_char;
            } else {
                return (true, 0);
            }
        }
    }
}
