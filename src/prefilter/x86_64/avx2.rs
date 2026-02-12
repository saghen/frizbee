use crate::prefilter::{case_needle, scalar};

use super::overlapping_load;
use std::arch::x86_64::*;

#[derive(Debug, Clone)]
pub struct PrefilterAVX2 {
    cased_needle: Vec<(u8, u8)>,
    needle: Vec<__m256i>,
}

impl PrefilterAVX2 {
    /// Creates a new prefilter algorithm for AVX2
    ///
    /// # Safety
    /// Caller must ensure that AVX2 is available at runtime
    #[inline]
    #[target_feature(enable = "avx")]
    pub unsafe fn new(needle: &[u8]) -> Self {
        let cased_needle = case_needle(needle);
        let needle = unsafe { cased_needle_to_avx2(&cased_needle) };
        Self {
            cased_needle,
            needle,
        }
    }

    pub fn is_available() -> bool {
        // TODO: these get converted to constants if compiling with -C target-cpu=x86-64-v3
        is_x86_feature_detected!("sse2")
            && is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("avx2")
    }

    /// Checks if the needle is wholly contained in the haystack, ignoring the exact order of the
    /// bytes. For example, if the needle is "test", the haystack "tset" will return true. However,
    /// the order does matter across 16 byte boundaries. The needle chars must include both the
    /// uppercase and lowercase variants of the character.
    ///
    /// # Safety
    /// The caller must ensure that AVX2 is available.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize) {
        let len = haystack.len();

        match len {
            0 => return (true, 0),
            1..=7 => {
                return (scalar::match_haystack(&self.cased_needle, haystack), 0);
            }
            _ => {}
        };

        let mut skipped_chunks = 0;
        let mut can_skip_chunks = true;

        let mut needle_iter = self.needle.iter();
        let mut needle_char = *needle_iter.next().unwrap();

        for start in (0..len).step_by(16) {
            let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };
            let haystack_chunk = _mm256_broadcastsi128_si256(haystack_chunk);
            loop {
                if _mm256_movemask_epi8(_mm256_cmpeq_epi8(needle_char, haystack_chunk)) == 0 {
                    // No match, advance to next chunk
                    break;
                }

                // Progress to next needle char, if available
                if let Some(next_needle_char) = needle_iter.next() {
                    if can_skip_chunks {
                        skipped_chunks = start / 16;
                    }
                    can_skip_chunks = false;
                    needle_char = *next_needle_char;
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
    /// The caller must ensure that the minimum length of the haystack is >= 8.
    /// The caller must ensure the needle.len() > 0 and that AVX2 is available.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn match_haystack_typos(&self, haystack: &[u8], max_typos: u16) -> (bool, usize) {
        let len = haystack.len();

        match len {
            0 => return (true, 0),
            1..=7 => {
                return (
                    scalar::match_haystack_typos(&self.cased_needle, haystack, max_typos),
                    0,
                );
            }
            _ => {}
        };

        let mut needle_iter = self.needle.iter();
        let mut needle_char = *needle_iter.next().unwrap();

        let mut typos = 0;
        loop {
            let mut skipped_chunks = 0;
            let mut can_skip_chunks = true;

            if typos > max_typos as usize {
                return (false, 0);
            }

            // TODO: this is slightly incorrect, because if we match on the third chunk,
            // we would only scan from the third chunk onwards for the next needle. Technically,
            // we should scan from the beginning of the haystack instead, but I believe the
            // previous memchr implementation had the same bug.
            for start in (0..len).step_by(16) {
                let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };
                let haystack_chunk = _mm256_broadcastsi128_si256(haystack_chunk);

                // For AVX2, we store the uppercase in the first 16 bytes, and the lowercase in the
                // last 16 bytes. This allows us to compare the uppercase and lowercase versions of
                // the needle char in the same comparison.
                loop {
                    if _mm256_movemask_epi8(_mm256_cmpeq_epi8(needle_char, haystack_chunk)) == 0 {
                        // No match, advance to next chunk
                        break;
                    }

                    // Progress to next needle char, if available
                    if let Some(next_needle_char) = needle_iter.next() {
                        if can_skip_chunks {
                            skipped_chunks = start / 16;
                        }
                        can_skip_chunks = false;

                        needle_char = *next_needle_char;
                    } else {
                        return (true, skipped_chunks);
                    }
                }
            }

            typos += 1;
            if typos > max_typos as usize {
                return (false, 0);
            }

            if let Some(next_needle_char) = needle_iter.next() {
                needle_char = *next_needle_char;
            } else {
                return (true, skipped_chunks);
            }
        }
    }
}

/// Loads the cased needle into a __m256i vector, where the first 16 bytes are the uppercase
/// and the last 16 bytes are the lowercase version of the needle.
///
/// # Safety
/// Caller must ensure that AVX2 is available at runtime
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn cased_needle_to_avx2(needle_cased: &[(u8, u8)]) -> Vec<std::arch::x86_64::__m256i> {
    needle_cased
        .iter()
        .map(|&(c1, c2)| unsafe {
            _mm256_loadu2_m128i(&_mm_set1_epi8(c1 as i8), &_mm_set1_epi8(c2 as i8))
        })
        .collect::<Vec<_>>()
}
