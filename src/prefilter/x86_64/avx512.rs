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

    /// Checks if the needle is wholly contained in the haystack.
    ///
    /// Returns `(matched, start_pos, end_pos)` where `end_pos` is the
    /// exclusive byte offset just past the rightmost occurrence of the final
    /// needle char in `haystack[start_pos..end_pos]`.
    ///
    /// # Safety
    /// The caller must ensure that AVX-512F + AVX-512BW are available.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        match len {
            0 => return (true, 0, 0),
            1..=7 => {
                return (
                    scalar::match_haystack(&self.needle_scalar, haystack),
                    0,
                    len,
                );
            }
            _ => {}
        }

        let mut can_skip_chunks = true;
        let mut match_start_pos = 0usize;
        let mut needle_iter = self.needle_simd.iter();
        let mut needle_char = *needle_iter.next().unwrap();

        let mut start = 0usize;
        while start < len {
            let (haystack_chunk, mut haystack_chunk_mask) =
                unsafe { load_window(haystack, start, len) };

            loop {
                let mask_a = _mm512_cmpeq_epi8_mask(needle_char.0, haystack_chunk);
                let mask_b = _mm512_cmpeq_epi8_mask(needle_char.1, haystack_chunk);
                let mask = (mask_a | mask_b) & haystack_chunk_mask;

                if mask == 0 {
                    break;
                }
                // mask ^ (mask - 1) = all bits from 0..=lowest_set_bit
                haystack_chunk_mask &= !(mask ^ mask.wrapping_sub(1));

                if can_skip_chunks {
                    match_start_pos = mask.trailing_zeros() as usize;
                    can_skip_chunks = false;
                }

                if let Some(next_needle_char) = needle_iter.next() {
                    needle_char = *next_needle_char;
                }
                // on the last chunk and no more needle chars, reuse mask
                // for getting last needle char's last position in haystack
                else if start + 64 >= len {
                    return (
                        true,
                        match_start_pos,
                        start + 64 - (mask.leading_zeros() as usize),
                    );
                }
                // not on the last chunk, find the position of the last
                // needle char from the end of the haystack
                else {
                    let last_needle_char = *self.needle_simd.last().unwrap();
                    let end_pos =
                        unsafe { start + find_last_char_pos(last_needle_char, &haystack[start..]) };
                    return (true, match_start_pos, end_pos);
                }
            }

            start += 64;
        }

        (false, match_start_pos, len)
    }

    /// # Safety
    /// The caller must ensure that AVX-512F + AVX-512BW are available.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn match_haystack_typos(&self, haystack: &[u8], max_typos: u16) -> (bool, usize) {
        todo!();
        // let len = haystack.len();
        // match len {
        //     0 => return (true, 0),
        //     1..=7 => {
        //         return (
        //             scalar::match_haystack_typos(&self.needle_scalar, haystack, max_typos),
        //             0,
        //         );
        //     }
        //     _ => {}
        // }
        //
        // if max_typos >= 3 {
        //     return (true, 0);
        // }
        //
        // let mut needle_iter = self.needle_simd.iter();
        // let mut needle_char = *needle_iter.next().unwrap();
        // let mut typos = 0usize;
        //
        // loop {
        //     // Mirror the AVX2 / SSE typos loop: restart scan from the beginning
        //     // of the haystack each time we burn a typo.
        //     let mut start = 0usize;
        //     while start < len {
        //         let (haystack_chunk, mut haystack_chunk_mask) =
        //             unsafe { load_window(haystack, start, len) };
        //
        //         loop {
        //             let mask_a = _mm512_cmpeq_epi8_mask(needle_char.0, haystack_chunk);
        //             let mask_b = _mm512_cmpeq_epi8_mask(needle_char.1, haystack_chunk);
        //             let mask = (mask_a | mask_b) & haystack_chunk_mask;
        //
        //             if mask == 0 {
        //                 break;
        //             }
        //             let min_bit = mask.trailing_zeros();
        //             haystack_chunk_mask &= !((1u64 << min_bit).wrapping_sub(1));
        //
        //             if let Some(next_needle_char) = needle_iter.next() {
        //                 needle_char = *next_needle_char;
        //             } else {
        //                 return (true, 0);
        //             }
        //         }
        //
        //         start += 64;
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
/// The forward pass already proved the `needle_char` can be found somewhere within the provided
/// `haystack` slice.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512BW availability and `min_pos <= haystack.len()`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn find_last_char_pos(needle_char: (__m512i, __m512i), haystack: &[u8]) -> usize {
    let len = haystack.len();
    let mut start = len.saturating_sub(64);
    loop {
        let (haystack_chunk, haystack_chunk_mask) = unsafe { load_window(haystack, start, len) };
        let mask_a = _mm512_cmpeq_epi8_mask(needle_char.0, haystack_chunk);
        let mask_b = _mm512_cmpeq_epi8_mask(needle_char.1, haystack_chunk);
        let mask = (mask_a | mask_b) & haystack_chunk_mask;
        if mask != 0 {
            return start + 64 - (mask.leading_zeros() as usize);
        }
        // looping infinitely is safe, since the caller already ensured that
        // the needle char matches somewhere in the haystack
        start = start.saturating_sub(64);
    }
}

/// Loads a 64-byte window from the haystack, returning the byte offset of the
/// loaded data, a validity mask over the lanes (covers the full register when
/// the load is complete), and the loaded chunk.
///
/// - `start + 64 <= len`: aligned 64-byte load at `start`
/// - `len < 64`: masked load covering only the valid bytes (page-safe)
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512BW availability.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load_window(haystack: &[u8], start: usize, len: usize) -> (__m512i, u64) {
    unsafe {
        if start + 64 <= len {
            (
                _mm512_loadu_si512(haystack.as_ptr().add(start) as *const __m512i),
                u64::MAX,
            )
        } else if can_overread_64(haystack.as_ptr().add(start)) {
            (
                _mm512_loadu_si512(haystack.as_ptr().add(start) as *const __m512i),
                (1u64 << (len - start)) - 1,
            )
        } else {
            // len - start < 64: masked load, zero-fills inactive lanes
            let mask: u64 = (1u64 << (len - start)).wrapping_sub(1);
            (
                _mm512_maskz_loadu_epi8(mask, haystack.as_ptr().add(start) as *const i8),
                u64::MAX,
            )
        }
    }
}

#[inline(always)]
fn can_overread_64(ptr: *const u8) -> bool {
    (ptr as usize & 0xFFF) <= (4096 - 64)
}
