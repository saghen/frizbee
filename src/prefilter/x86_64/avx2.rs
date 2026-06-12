use crate::prefilter::{case_needle, scalar};

use std::arch::x86_64::*;

#[derive(Debug, Clone)]
pub struct PrefilterAVX {
    needle_scalar: Vec<(u8, u8)>,
    /// (case_a, case_b) broadcast to 256 bits per needle char
    /// len % 2 == 0 && len >= 2
    needle_simd: Vec<(__m256i, __m256i)>,
}

impl PrefilterAVX {
    /// # Safety
    /// Caller must ensure that AVX2 is available at runtime
    #[inline]
    #[target_feature(enable = "avx")]
    pub unsafe fn new(needle: &[u8]) -> Self {
        assert!(!needle.is_empty(), "needle must not be empty");

        let needle_scalar = case_needle(needle);
        let mut needle_simd = needle_scalar
            .iter()
            .map(|&(c1, c2)| (_mm256_set1_epi8(c1 as i8), _mm256_set1_epi8(c2 as i8)))
            .collect::<Vec<_>>();

        // ensure we always have an even number of elements to simplify matching
        if needle_simd.len() & 1 != 0 {
            needle_simd.push((_mm256_setzero_si256(), _mm256_setzero_si256()));
        }

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

    /// Checks if the needle is wholly contained in the haystack.
    ///
    /// Returns `(matched, start_pos, end_pos)` where `end_pos` is the
    /// exclusive byte offset just past the rightmost occurrence of the final
    /// needle char in `haystack[start_pos..end_pos]`.
    ///
    /// # Safety
    /// The caller must ensure that AVX2 is available.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        match len {
            0 => return (true, 0, len),
            1..=7 => {
                return (
                    scalar::match_haystack(&self.needle_scalar, haystack),
                    0,
                    len,
                );
            }
            _ => {}
        };

        let mut match_start_pos = 0;
        let mut can_skip_chunks = true;

        let mut needle_idx = 0;
        let needle_len = self.needle_scalar.len();
        // The needle_simd will always have an even number of elements as per .new()
        // and will always contain atleast two elements so this is safe
        let mut needle_char = self.needle_simd[needle_idx];
        let mut second_needle_char = self.needle_simd[needle_idx + 1];

        for start in (0..len).step_by(32) {
            let (haystack_chunk, mut haystack_chunk_mask) =
                unsafe { load_window(haystack, start, len) };
            // TODO: for pure letter haystacks, we could instead do
            // haystack_chunk |= 0x20
            // and do a single compare against the lowercase needle
            // ~2% faster overall while using 2 cmpeqs, without widening to 4 cmpeqs
            // to fill all ports (possibly 3-4% faster if implemented?)

            loop {
                // Compute two masks simultaneously since AVX2 cpus can typically do 4 simultaneous
                // compares per cycle
                let first_mask = _mm256_or_si256(
                    _mm256_cmpeq_epi8(needle_char.0, haystack_chunk),
                    _mm256_cmpeq_epi8(needle_char.1, haystack_chunk),
                );
                let second_mask = _mm256_or_si256(
                    _mm256_cmpeq_epi8(second_needle_char.0, haystack_chunk),
                    _mm256_cmpeq_epi8(second_needle_char.1, haystack_chunk),
                );

                // Ensure first needle matches
                let first_mask = _mm256_movemask_epi8(first_mask) as u32 & haystack_chunk_mask;
                if first_mask == 0 {
                    break;
                }

                // Mask out bits <= min_bit for sequential fuzzy matching
                let min_bit = first_mask.trailing_zeros();
                haystack_chunk_mask &= !1u32 << min_bit;

                if can_skip_chunks {
                    match_start_pos = start + min_bit as usize;
                    can_skip_chunks = false;
                }

                // If last needle char, return result
                if needle_idx + 1 >= needle_len {
                    if start + 32 >= len {
                        return (
                            true,
                            match_start_pos,
                            start + 32 - (first_mask.leading_zeros() as usize),
                        );
                    } else {
                        let offset = start + min_bit as usize;
                        let end_pos = offset
                            + unsafe { find_last_char_pos(needle_char, &haystack[offset..]) };
                        return (true, match_start_pos, end_pos);
                    }
                }

                // Ensure second needle matches
                let second_mask = _mm256_movemask_epi8(second_mask) as u32 & haystack_chunk_mask;
                if second_mask == 0 {
                    break;
                }
                haystack_chunk_mask &= !1u32 << second_mask.trailing_zeros();

                // Progress to next two needle chars, if available
                needle_idx += 2;
                // TODO: it's slightly faster (~2% overall) to have a separate path for odd vs even
                // needle lengths
                if needle_idx < needle_len {
                    needle_char = self.needle_simd[needle_idx];
                    second_needle_char = self.needle_simd[needle_idx + 1];
                } else {
                    if start + 32 >= len {
                        return (
                            true,
                            match_start_pos,
                            start + 32 - (second_mask.leading_zeros() as usize),
                        );
                    } else {
                        let offset = start + min_bit as usize;
                        let end_pos = offset
                            + unsafe { find_last_char_pos(needle_char, &haystack[offset..]) };
                        return (true, match_start_pos, end_pos);
                    }
                }
            }
        }

        (false, match_start_pos, len)
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
        todo!();

        // if max_typos >= 3 {
        //     return (true, 0);
        // }
        //
        // let mut needle_iter = self.needle_simd.iter();
        // let mut needle_char = *needle_iter.next().unwrap();
        //
        // let mut typos = 0;
        // loop {
        //     // TODO: this is slightly incorrect, because if we match on the third chunk,
        //     // we would only scan from the third chunk onwards for the next needle. Technically,
        //     // we should scan from the beginning of the haystack instead, but I believe the
        //     // previous memchr implementation had the same bug.
        //     for start in (0..len).step_by(16) {
        //         let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };
        //         let haystack_chunk = _mm256_broadcastsi128_si256(haystack_chunk);
        //
        //         // For AVX2, we store the uppercase in the first 16 bytes, and the lowercase in the
        //         // last 16 bytes. This allows us to compare the uppercase and lowercase versions of
        //         // the needle char in the same comparison.
        //         loop {
        //             if _mm256_movemask_epi8(_mm256_cmpeq_epi8(needle_char, haystack_chunk)) == 0 {
        //                 // No match, advance to next chunk
        //                 break;
        //             }
        //
        //             // Progress to next needle char, if available
        //             if let Some(next_needle_char) = needle_iter.next() {
        //                 needle_char = *next_needle_char;
        //             } else {
        //                 return (true, 0);
        //             }
        //         }
        //     }
        //
        //     typos += 1;
        //     if typos > max_typos as usize {
        //         return (false, 0);
        //     }
        //
        //     if let Some(next_needle_char) = needle_iter.next() {
        //         needle_char = *next_needle_char;
        //     } else {
        //         return (true, 0);
        //     }
        // }
    }
}

/// Scans `haystack[min_pos..]` backwards for the rightmost occurrence of
/// `last_needle_char` (case-insensitive via the broadcast pair) and returns
/// the exclusive end offset (`pos + 1`) into the original `haystack`.
///
/// The forward pass already proved an ordered alignment exists in
/// `haystack[min_pos..]` ending at some position `p`. Since the last char of
/// that alignment must equal `last_needle_char` and the rightmost occurrence
/// `q >= p`, the slice `haystack[min_pos..=q]` still contains that alignment
/// (and any later candidate). Falls back to `haystack.len()` if nothing is
/// found, which can't happen when the forward pass succeeded.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512BW availability and `min_pos <= haystack.len()`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn find_last_char_pos(last_needle_char: (__m256i, __m256i), haystack: &[u8]) -> usize {
    let len = haystack.len();
    let mut start = len.saturating_sub(32);
    loop {
        let (haystack_chunk, haystack_chunk_mask) = unsafe { load_window(haystack, start, len) };
        let haystack_chunk = _mm256_or_si256(haystack_chunk, _mm256_set1_epi8(0x20));
        let mask = _mm256_or_si256(
            _mm256_cmpeq_epi8(last_needle_char.0, haystack_chunk),
            _mm256_cmpeq_epi8(last_needle_char.1, haystack_chunk),
        );
        let mask = _mm256_movemask_epi8(mask) as u32 & haystack_chunk_mask;
        if mask != 0 {
            return start + 32 - (mask.leading_zeros() as usize);
        }
        // looping infinitely is safe, since the caller already ensured that
        // the needle char matches somewhere in the haystack
        start = start.saturating_sub(32);
    }
}

/// Loads the cased needle into a __m256i vector, where the first 16 bytes are the uppercase
/// and the last 16 bytes are the lowercase version of the needle.
///
/// # Safety
/// Caller must ensure that AVX2 is available at runtime
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn load_window(haystack: &[u8], start: usize, len: usize) -> (__m256i, u32) {
    unsafe {
        if start + 32 <= len {
            (
                _mm256_loadu_si256(haystack.as_ptr().add(start) as *const __m256i),
                u32::MAX,
            )
        } else if can_overread_32(haystack.as_ptr().add(start)) {
            (
                _mm256_loadu_si256(haystack.as_ptr().add(start) as *const __m256i),
                (1u32 << (len - start)) - 1,
            )
        } else {
            let mut data = [0u8; 32];
            std::ptr::copy_nonoverlapping(
                haystack.as_ptr().add(start),
                data.as_mut_ptr(),
                len - start,
            );
            (
                _mm256_loadu_si256(data.as_ptr() as *const __m256i),
                u32::MAX,
            )
        }
    }
}

#[inline(always)]
fn can_overread_32(ptr: *const u8) -> bool {
    (ptr as usize & 0xFFF) <= (4096 - 32)
}
