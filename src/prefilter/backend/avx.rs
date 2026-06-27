use std::arch::x86_64::*;

use crate::prefilter::{
    Kernel, Window,
    algo::{Prefilter, find_last_char_pos, load_window},
    case_needle,
};

use super::Backend;

#[derive(Debug, Clone)]
pub struct PrefilterAVX {
    inner: Prefilter<PrefilterAVXBackend>,
    needle_len: usize,
    /// padded with one extra pair so we can skip bounds checks and always
    /// assume we have a simd pair available
    needle_simd: Vec<(__m256i, __m256i)>,
}

impl Kernel for PrefilterAVX {
    #[inline(always)]
    fn new(needle: &str, case_sensitive: bool) -> Self {
        let needle_cases = case_needle(needle.as_bytes(), case_sensitive);
        let needle_len = needle_cases.len();
        let mut needle_simd = needle_cases
            .iter()
            .map(|&(c1, c2)| unsafe {
                (
                    PrefilterAVXBackend::splat(c1),
                    PrefilterAVXBackend::splat(c2),
                )
            })
            .collect::<Vec<_>>();

        needle_simd.push(unsafe { (PrefilterAVXBackend::splat(0), PrefilterAVXBackend::splat(0)) });
        if needle_simd.len() % 2 != 0 {
            needle_simd
                .push(unsafe { (PrefilterAVXBackend::splat(0), PrefilterAVXBackend::splat(0)) });
        }

        Self {
            inner: unsafe { Prefilter::new(needle, case_sensitive) },
            needle_len,
            needle_simd,
        }
    }

    #[inline(always)]
    fn is_available() -> bool {
        PrefilterAVXBackend::is_available()
    }

    #[inline(always)]
    fn match_haystack(&self, haystack: &[u8]) -> Window {
        let len = haystack.len();
        if len == 0 {
            return (false, 0, len);
        }

        let mut match_start_pos = 0;
        let mut can_skip_chunks = true;
        let mut needle_idx = 0;
        let needle_len = self.needle_len;
        let mut needle_char = self.needle_simd[needle_idx];
        let mut second_needle_char = self.needle_simd[needle_idx + 1];

        for start in (0..len).step_by(PrefilterAVXBackend::LANES) {
            let (haystack_chunk, mut haystack_chunk_mask) =
                unsafe { load_window::<PrefilterAVXBackend>(haystack, start, len) };

            // NOTE: for pure letter haystacks, we could instead OR the chunk
            // with 0x20 and compare against lowercase needles. it's about 2%
            // faster overall, so not worth the extra code
            loop {
                // CPUs can do 4 cmpeqs per cycle, so we compare two needles
                // at a time (each case-insensitive probe is two byte compares)
                // to maximize throughput
                let first_mask = unsafe {
                    _mm256_or_si256(
                        _mm256_cmpeq_epi8(needle_char.0, haystack_chunk),
                        _mm256_cmpeq_epi8(needle_char.1, haystack_chunk),
                    )
                };
                let second_mask = unsafe {
                    _mm256_or_si256(
                        _mm256_cmpeq_epi8(second_needle_char.0, haystack_chunk),
                        _mm256_cmpeq_epi8(second_needle_char.1, haystack_chunk),
                    )
                };

                let first_mask =
                    unsafe { _mm256_movemask_epi8(first_mask) } as u32 & haystack_chunk_mask;
                // no match, move to next chunk
                if first_mask == 0 {
                    break;
                }

                // mask out any match indices <= the current match
                let min_bit = first_mask.trailing_zeros() as usize;
                haystack_chunk_mask &= !1u32 << min_bit;

                if can_skip_chunks {
                    match_start_pos = start + min_bit;
                    can_skip_chunks = false;
                }

                // this was the final needle byte, return the match
                if needle_idx + 1 >= needle_len {
                    if start + PrefilterAVXBackend::LANES >= len {
                        return (
                            true,
                            match_start_pos,
                            start + PrefilterAVXBackend::LANES
                                - first_mask.leading_zeros() as usize,
                        );
                    }

                    let offset = start + min_bit;
                    let end_pos = offset
                        + unsafe {
                            find_last_char_pos::<PrefilterAVXBackend>(
                                needle_char,
                                &haystack[offset..],
                            )
                        };
                    return (true, match_start_pos, end_pos);
                }

                // second needle byte
                let second_mask =
                    unsafe { _mm256_movemask_epi8(second_mask) } as u32 & haystack_chunk_mask;
                if second_mask == 0 {
                    needle_idx += 1;
                    needle_char = second_needle_char;
                    second_needle_char = self.needle_simd[needle_idx + 1];
                    break;
                }
                haystack_chunk_mask &= !1u32 << second_mask.trailing_zeros();

                needle_idx += 2;
                if needle_idx < needle_len {
                    needle_char = self.needle_simd[needle_idx];
                    second_needle_char = self.needle_simd[needle_idx + 1];
                } else if start + PrefilterAVXBackend::LANES >= len {
                    return (
                        true,
                        match_start_pos,
                        start + PrefilterAVXBackend::LANES - second_mask.leading_zeros() as usize,
                    );
                } else {
                    let offset = start + min_bit;
                    let end_pos = offset
                        + unsafe {
                            find_last_char_pos::<PrefilterAVXBackend>(
                                second_needle_char,
                                &haystack[offset..],
                            )
                        };
                    return (true, match_start_pos, end_pos);
                }
            }
        }

        (false, match_start_pos, len)
    }

    #[inline(always)]
    fn match_haystack_unicode(&self, haystack: &[u8]) -> Window {
        unsafe { self.inner.match_haystack_unicode(haystack) }
    }

    #[inline(always)]
    fn match_haystack_1_typo(&self, haystack: &[u8]) -> Window {
        unsafe { self.inner.match_haystack_1_typo(haystack) }
    }

    #[inline(always)]
    fn match_haystack_unicode_1_typo(&self, haystack: &[u8]) -> Window {
        unsafe { self.inner.match_haystack_unicode_1_typo(haystack) }
    }

    #[inline(always)]
    fn match_haystack_2_typos(&self, haystack: &[u8]) -> Window {
        unsafe { self.inner.match_haystack_2_typos(haystack) }
    }

    #[inline(always)]
    fn match_haystack_unicode_2_typos(&self, haystack: &[u8]) -> Window {
        unsafe { self.inner.match_haystack_unicode_2_typos(haystack) }
    }

    #[inline(always)]
    fn match_haystack_many_typos(&mut self, haystack: &[u8], max_typos: u16) -> Window {
        unsafe { self.inner.match_haystack_many_typos(haystack, max_typos) }
    }

    #[inline(always)]
    fn match_haystack_unicode_many_typos(&mut self, haystack: &[u8], max_typos: u16) -> Window {
        unsafe {
            self.inner
                .match_haystack_unicode_many_typos(haystack, max_typos)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterAVXBackend;

impl Backend for PrefilterAVXBackend {
    const LANES: usize = 32;

    type Chunk = __m256i;
    type Mask = u32;

    fn is_available() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> Self::Chunk {
        unsafe { _mm256_set1_epi8(c as i8) }
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        unsafe { _mm256_cmpeq_epi8_mask(a, b) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        unsafe {
            let mask = _mm256_or_si256(
                _mm256_cmpeq_epi8(needle.0, chunk),
                _mm256_cmpeq_epi8(needle.1, chunk),
            );
            _mm256_movemask_epi8(mask) as u32
        }
    }
}
