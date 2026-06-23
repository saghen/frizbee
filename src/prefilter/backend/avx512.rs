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
    unsafe fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
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

    #[inline(always)]
    unsafe fn shift_left<const N: usize>(a: __m512i, b: __m512i) -> __m512i {
        const { assert!(N <= 64, "shift amount must be <= 64") };
        let idx: __m512i = unsafe { core::mem::transmute(left_shift_idx::<N>()) };
        unsafe { _mm512_permutex2var_epi8(a, idx, b) }
    }
}

/// Build the VPERMI2B index table for a left shift by `N` bytes.
///
/// Sources: bytes 0..=63 are `a`, bytes 64..=127 are `b`.
/// result[i] = a[i - N]           for i >= N
/// result[i] = b[64 - N + i]      for i <  N   (top N bytes of b shifted in)
const fn left_shift_idx<const N: usize>() -> [i8; 64] {
    let mut idx = [0i8; 64];
    let mut i = 0;
    while i < 64 {
        idx[i] = if i >= N {
            (i - N) as i8 // pull from a
        } else {
            (128 - N + i) as i8 // pull from top of b
        };
        i += 1;
    }
    idx
}
