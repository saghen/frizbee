use std::arch::x86_64::*;

use super::Backend;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterAVX512Backend;

impl Backend for PrefilterAVX512Backend {
    const LANES: usize = 64;

    type Chunk = __m512i;
    type Mask = u64;

    fn is_available() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("bmi1")
            && is_x86_feature_detected!("bmi2")
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> Self::Chunk {
        unsafe { _mm512_set1_epi8(c as i8) }
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        unsafe { _mm512_cmpeq_epi8_mask(a, b) }
    }

    #[inline(always)]
    unsafe fn broadcast(c: (u8, u8)) -> (Self::Chunk, Self::Chunk) {
        unsafe { (_mm512_set1_epi8(c.0 as i8), _mm512_set1_epi8(c.1 as i8)) }
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
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
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
