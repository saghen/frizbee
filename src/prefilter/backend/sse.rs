use std::arch::x86_64::*;

use crate::prefilter::algo::PrefilterImpl;

use super::Backend;

#[derive(Debug, Clone)]
pub struct PrefilterSSE(PrefilterImpl<PrefilterSSEBackend>);

impl PrefilterSSE {
    /// # Safety
    /// Caller must ensure that SSE2 is available at runtime.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn new(needle: &[u8]) -> Self {
        Self(unsafe { PrefilterImpl::new(needle) })
    }

    pub fn is_available() -> bool {
        PrefilterImpl::<PrefilterSSEBackend>::is_available()
    }

    /// # Safety
    /// Caller must ensure that SSE2 is available.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.0.match_haystack(haystack) }
    }

    /// # Safety
    /// Caller must ensure that SSE2 is available.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn match_haystack_1_typo(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.0.match_haystack_1_typo(haystack) }
    }

    /// # Safety
    /// Caller must ensure that SSE2 is available.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn match_haystack_2_typos(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.0.match_haystack_2_typos(haystack) }
    }

    /// # Safety
    /// Caller must ensure that SSE2 is available.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn match_haystack_typos(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> (bool, usize, usize) {
        match max_typos {
            0 => unsafe { self.match_haystack(haystack) },
            1 => unsafe { self.match_haystack_1_typo(haystack) },
            2 => unsafe { self.match_haystack_2_typos(haystack) },
            _ => unsafe { self.0.match_haystack_typos(haystack, max_typos) },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterSSEBackend;

impl Backend for PrefilterSSEBackend {
    const LANES: usize = 16;

    type Chunk = __m128i;
    type Needle = (__m128i, __m128i);
    type Mask = u16;

    fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_feature_info()
            .is_some_and(|info| info.has_sse2())
    }

    #[inline(always)]
    unsafe fn broadcast(c1: u8, c2: u8) -> Self::Needle {
        unsafe { (_mm_set1_epi8(c1 as i8), _mm_set1_epi8(c2 as i8)) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { _mm_loadu_si128(ptr as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: Self::Needle) -> Self::Mask {
        unsafe {
            let mask = _mm_or_si128(
                _mm_cmpeq_epi8(needle.0, chunk),
                _mm_cmpeq_epi8(needle.1, chunk),
            );
            _mm_movemask_epi8(mask) as u16
        }
    }
}
