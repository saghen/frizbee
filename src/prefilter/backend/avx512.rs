use std::arch::x86_64::*;

use crate::prefilter::algo::PrefilterImpl;

use super::Backend;

#[derive(Debug, Clone)]
pub struct PrefilterAVX512(PrefilterImpl<PrefilterAVX512Backend>);

impl PrefilterAVX512 {
    #[inline(always)]
    pub fn new(needle: &[u8]) -> Self {
        Self(unsafe { PrefilterImpl::new(needle) })
    }

    pub fn is_available() -> bool {
        PrefilterImpl::<PrefilterAVX512Backend>::is_available()
    }

    #[inline(always)]
    pub fn match_haystack(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.0.match_haystack(haystack) }
    }

    #[inline(always)]
    pub fn match_haystack_1_typo(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.0.match_haystack_1_typo(haystack) }
    }

    #[inline(always)]
    pub fn match_haystack_2_typos(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.0.match_haystack_2_typos(haystack) }
    }

    #[inline(always)]
    pub fn match_haystack_typos(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> (bool, usize, usize) {
        match max_typos {
            0 => self.match_haystack(haystack),
            1 => self.match_haystack_1_typo(haystack),
            2 => self.match_haystack_2_typos(haystack),
            _ => unsafe { self.0.match_haystack_typos(haystack, max_typos) },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterAVX512Backend;

impl Backend for PrefilterAVX512Backend {
    const LANES: usize = 64;

    type Chunk = __m512i;
    type Needle = (__m512i, __m512i);
    type Mask = u64;

    fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_extended_feature_info()
            .is_some_and(|info| {
                info.has_avx512f() && info.has_avx512bw() && info.has_bmi1() && info.has_bmi2()
            })
    }

    #[inline(always)]
    unsafe fn broadcast(c1: u8, c2: u8) -> Self::Needle {
        unsafe { (_mm512_set1_epi8(c1 as i8), _mm512_set1_epi8(c2 as i8)) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { _mm512_loadu_si512(ptr as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const u8, _remaining: usize, mask: Self::Mask) -> Self::Chunk {
        // AVX-512 has native masked byte loads, so loads at page boundaries
        // don't need a copy based fallback
        unsafe { _mm512_maskz_loadu_epi8(mask, ptr as *const i8) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: Self::Needle) -> Self::Mask {
        unsafe { _mm512_cmpeq_epi8_mask(needle.0, chunk) | _mm512_cmpeq_epi8_mask(needle.1, chunk) }
    }

    #[inline(always)]
    unsafe fn first_hit_pos(hit: Self::Mask) -> usize {
        unsafe { _tzcnt_u64(hit) as usize }
    }

    #[inline(always)]
    unsafe fn clear_through_lowest(mask: Self::Mask, hit: Self::Mask) -> Self::Mask {
        unsafe { mask & !_blsmsk_u64(hit) }
    }
}
