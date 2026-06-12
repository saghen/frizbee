use std::arch::x86_64::*;

use crate::prefilter::{
    algo::{PrefilterImpl, find_last_char_pos, load_window},
    case_needle,
};

use super::Backend;

#[derive(Debug, Clone)]
pub struct PrefilterAVX {
    inner: PrefilterImpl<PrefilterAVXBackend>,
    needle_len: usize,
    /// Padded to an even length so the exact-match override can compare two
    /// needle bytes per loop without a tail branch.
    needle_simd: Vec<(__m256i, __m256i)>,
}

impl PrefilterAVX {
    /// # Safety
    /// Caller must ensure that AVX2 is available at runtime.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn new(needle: &[u8]) -> Self {
        assert!(!needle.is_empty(), "needle must not be empty");

        let needle_cases = case_needle(needle);
        let needle_len = needle_cases.len();
        let mut needle_simd = needle_cases
            .iter()
            .map(|&(c1, c2)| (_mm256_set1_epi8(c1 as i8), _mm256_set1_epi8(c2 as i8)))
            .collect::<Vec<_>>();

        // Keep an even number of elements so the exact-match override can
        // issue two needle probes per loop without a tail branch.
        if needle_simd.len() & 1 != 0 {
            needle_simd.push((_mm256_setzero_si256(), _mm256_setzero_si256()));
        }

        Self {
            inner: unsafe { PrefilterImpl::new(needle) },
            needle_len,
            needle_simd,
        }
    }

    pub fn is_available() -> bool {
        PrefilterImpl::<PrefilterAVXBackend>::is_available()
    }

    /// Checks if the needle is wholly contained in the haystack.
    ///
    /// # Safety
    /// Caller must ensure that AVX2 is available.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize, usize) {
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

            // TODO: for pure letter haystacks, we could instead OR the chunk
            // with 0x20 and compare against lowercase needles. That was about
            // 2% faster overall, but uses fewer compares instead of filling the
            // AVX2 compare throughput the paired-probe path targets here.
            loop {
                // Probe two needle bytes together: each case-insensitive probe
                // is two byte compares, so the pair fills AVX2 compare issue
                // bandwidth on CPUs with four such ports.
                let first_mask = _mm256_or_si256(
                    _mm256_cmpeq_epi8(needle_char.0, haystack_chunk),
                    _mm256_cmpeq_epi8(needle_char.1, haystack_chunk),
                );
                let second_mask = _mm256_or_si256(
                    _mm256_cmpeq_epi8(second_needle_char.0, haystack_chunk),
                    _mm256_cmpeq_epi8(second_needle_char.1, haystack_chunk),
                );

                let first_mask = _mm256_movemask_epi8(first_mask) as u32 & haystack_chunk_mask;
                if first_mask == 0 {
                    break;
                }

                let min_bit = first_mask.trailing_zeros() as usize;
                haystack_chunk_mask &= !1u32 << min_bit;

                if can_skip_chunks {
                    match_start_pos = start + min_bit;
                    can_skip_chunks = false;
                }

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

                let second_mask = _mm256_movemask_epi8(second_mask) as u32 & haystack_chunk_mask;
                if second_mask == 0 {
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

    /// # Safety
    /// Caller must ensure that AVX2 is available.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn match_haystack_1_typo(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.inner.match_haystack_1_typo(haystack) }
    }

    /// # Safety
    /// Caller must ensure that AVX2 is available.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn match_haystack_2_typos(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.inner.match_haystack_2_typos(haystack) }
    }

    /// # Safety
    /// Caller must ensure that AVX2 is available.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn match_haystack_typos(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> (bool, usize, usize) {
        match max_typos {
            0 => unsafe { self.match_haystack(haystack) },
            1 => unsafe { self.match_haystack_1_typo(haystack) },
            2 => unsafe { self.match_haystack_2_typos(haystack) },
            _ => unsafe { self.inner.match_haystack_typos(haystack, max_typos) },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterAVXBackend;

impl Backend for PrefilterAVXBackend {
    const LANES: usize = 32;

    type Chunk = __m256i;
    type Needle = (__m256i, __m256i);
    type Mask = u32;

    fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_extended_feature_info()
            .is_some_and(|info| info.has_avx2())
    }

    #[inline(always)]
    unsafe fn broadcast(c1: u8, c2: u8) -> Self::Needle {
        unsafe { (_mm256_set1_epi8(c1 as i8), _mm256_set1_epi8(c2 as i8)) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: Self::Needle) -> Self::Mask {
        unsafe {
            let mask = _mm256_or_si256(
                _mm256_cmpeq_epi8(needle.0, chunk),
                _mm256_cmpeq_epi8(needle.1, chunk),
            );
            _mm256_movemask_epi8(mask) as u32
        }
    }
}
