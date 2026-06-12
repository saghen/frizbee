use std::arch::x86_64::*;

use super::Backend;

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
