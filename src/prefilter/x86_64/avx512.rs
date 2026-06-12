use std::arch::x86_64::*;

use crate::prefilter::{case_needle, scalar};

#[derive(Debug, Clone)]
pub struct PrefilterAVX512 {
    needle_scalar: Vec<(u8, u8)>,
    /// (case_a, case_b) broadcast to 512 bits per needle char
    needle_simd: Vec<(__m512i, __m512i)>,
}

impl PrefilterAVX512 {
    /// # Safety
    /// Caller must ensure that AVX-512F + AVX-512BW are available at runtime
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn new(needle: &[u8]) -> Self {
        let needle_scalar = case_needle(needle);
        let needle_simd = needle_scalar
            .iter()
            .map(|&(c1, c2)| (_mm512_set1_epi8(c1 as i8), _mm512_set1_epi8(c2 as i8)))
            .collect();
        Self {
            needle_scalar,
            needle_simd,
        }
    }

    pub fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_extended_feature_info()
            .is_some_and(|info| info.has_avx512f() && info.has_avx512bw())
    }

    /// Checks if the needle is wholly contained in the haystack, unordered within
    /// an aligned 16-byte sub-chunk but ordered across sub-chunks. Scans 64 bytes
    /// per outer iteration; ordering across sub-chunks within the 64-byte window
    /// is preserved by masking the cmpeq bitmask against a moving `min_bit`.
    ///
    /// # Safety
    /// The caller must ensure that AVX-512F + AVX-512BW are available.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize) {
        let len = haystack.len();
        match len {
            0 => return (true, 0),
            1..=7 => return (scalar::match_haystack(&self.needle_scalar, haystack), 0),
            _ => {}
        }

        let mut can_skip_chunks = true;
        let mut skipped_chars = 0usize;
        let mut needle_iter = self.needle_simd.iter();
        let mut needle_char = *needle_iter.next().unwrap();

        let mut start = 0usize;
        while start < len {
            let (window_start, haystack_chunk) = unsafe { load_window(haystack, start, len) };
            let mut min_bit: u32 = 0;

            loop {
                let mask_a = _mm512_cmpeq_epi8_mask(needle_char.0, haystack_chunk);
                let mask_b = _mm512_cmpeq_epi8_mask(needle_char.1, haystack_chunk);
                let mut mask = mask_a | mask_b;
                mask &= !((1u64 << min_bit).wrapping_sub(1));

                if mask == 0 {
                    break;
                }
                min_bit = mask.trailing_zeros();

                if can_skip_chunks {
                    skipped_chars = window_start + (min_bit as usize);
                    can_skip_chunks = false;
                }

                if let Some(next_needle_char) = needle_iter.next() {
                    needle_char = *next_needle_char;
                } else {
                    return (true, skipped_chars);
                }
            }

            start += 64;
        }

        (false, skipped_chars)
    }

    /// # Safety
    /// The caller must ensure that AVX-512F + AVX-512BW are available.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
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
        }

        if max_typos >= 3 {
            return (true, 0);
        }

        let mut needle_iter = self.needle_simd.iter();
        let mut needle_char = *needle_iter.next().unwrap();
        let mut typos = 0usize;

        loop {
            // Mirror the AVX2 / SSE typos loop: restart scan from the beginning
            // of the haystack each time we burn a typo.
            let mut start = 0usize;
            while start < len {
                let (_, haystack_chunk) = unsafe { load_window(haystack, start, len) };
                let mut min_bit: u32 = 0;

                loop {
                    let mask_a = _mm512_cmpeq_epi8_mask(needle_char.0, haystack_chunk);
                    let mask_b = _mm512_cmpeq_epi8_mask(needle_char.1, haystack_chunk);
                    let mut mask = mask_a | mask_b;
                    mask &= !((1u64 << min_bit).wrapping_sub(1));

                    if mask == 0 {
                        break;
                    }
                    min_bit = mask.trailing_zeros();

                    if let Some(next_needle_char) = needle_iter.next() {
                        needle_char = *next_needle_char;
                    } else {
                        return (true, 0);
                    }
                }

                start += 64;
            }

            typos += 1;
            if typos > max_typos as usize {
                return (false, 0);
            }

            if let Some(next_needle_char) = needle_iter.next() {
                needle_char = *next_needle_char;
            } else {
                return (true, 0);
            }
        }
    }
}

/// Loads a 64-byte window from the haystack, returning the byte offset of the
/// loaded data, a validity mask over the lanes (covers the full register when
/// the load is complete), and the loaded chunk.
///
/// - `start + 64 <= len`: aligned 64-byte load at `start`
/// - `len >= 64` but tail short: overlap-from-end (last 64 bytes)
/// - `len < 64`: masked load covering only the valid bytes (page-safe)
///
/// # Safety
/// Caller must ensure `len >= 8` (matching the prefilter's small-haystack
/// fallback) and AVX-512F + AVX-512BW availability.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load_window(haystack: &[u8], start: usize, len: usize) -> (usize, __m512i) {
    unsafe {
        if start + 64 <= len {
            let chunk = _mm512_loadu_si512(haystack.as_ptr().add(start) as *const __m512i);
            (start, chunk)
        } else if len >= 64 {
            let window_start = len - 64;
            let chunk = _mm512_loadu_si512(haystack.as_ptr().add(window_start) as *const __m512i);
            (window_start, chunk)
        } else {
            // len < 64: masked load, zero-fills inactive lanes
            let mask: u64 = (1u64 << len).wrapping_sub(1);
            let chunk = _mm512_maskz_loadu_epi8(mask, haystack.as_ptr() as *const i8);
            (0, chunk)
        }
    }
}
