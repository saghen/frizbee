use crate::prefilter::{case_needle, scalar};

use super::overlapping_load;
use std::arch::x86_64::*;

#[derive(Debug, Clone)]
pub struct PrefilterAVX {
    needle_scalar: Vec<(u8, u8)>,
    /// Lowercase in low 128-bits, uppercase in high 128-bits
    needle_simd: Vec<__m256i>,
}

impl PrefilterAVX {
    /// # Safety
    /// Caller must ensure that AVX2 is available at runtime
    #[inline]
    #[target_feature(enable = "avx")]
    pub unsafe fn new(needle: &[u8]) -> Self {
        let needle_scalar = case_needle(needle);
        let needle_simd = unsafe { cased_needle_to_avx2(&needle_scalar) };
        Self {
            needle_scalar,
            needle_simd,
        }
    }

    pub fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_extended_feature_info()
            .is_some_and(|info| info.has_avx2())
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
                return (scalar::match_haystack(&self.needle_scalar, haystack), 0);
            }
            _ => {}
        };

        let mut skipped_chunks = 0;
        let mut can_skip_chunks = true;

        let mut needle_iter = self.needle_simd.iter();
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
    /// The caller must ensure that AVX2 is available.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn match_haystack_typos(&self, haystack: &[u8], max_typos: u16) -> (bool, usize) {
        let len = haystack.len();

        match len {
            0 => return (true, 0),
            1..=7 => {
                return (
                    scalar::match_haystack_typos(&self.needle_scalar, haystack, max_typos),
                    0,
                );
            }
            _ => {}
        };

        if max_typos >= 3 {
            return (true, 0);
        }

        let mut needle_iter = self.needle_simd.iter();
        let mut needle_char = *needle_iter.next().unwrap();

        let mut typos = 0;
        loop {
            let mut skipped_chunks = 0;
            let mut can_skip_chunks = true;

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
