use std::arch::x86_64::*;

use super::Backend;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterSSEBackend;

impl Backend for PrefilterSSEBackend {
    const LANES: usize = 16;

    type Chunk = __m128i;
    type Mask = u16;

    fn is_available() -> bool {
        is_x86_feature_detected!("sse2")
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> __m128i {
        unsafe { _mm_set1_epi8(c as i8) }
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) as u16 }
    }

    #[inline(always)]
    unsafe fn broadcast(c: (u8, u8)) -> (Self::Chunk, Self::Chunk) {
        unsafe { (_mm_set1_epi8(c.0 as i8), _mm_set1_epi8(c.1 as i8)) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { _mm_loadu_si128(ptr as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        unsafe {
            let mask = _mm_or_si128(
                _mm_cmpeq_epi8(needle.0, chunk),
                _mm_cmpeq_epi8(needle.1, chunk),
            );
            _mm_movemask_epi8(mask) as u16
        }
    }
}
