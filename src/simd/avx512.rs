use std::arch::x86_64::*;

use super::{Backend, BytesVec, MaskVec, ScoreVec};

/// 64-lane u8 scoring (512-bit __m512i), 64-lane u8 input (512-bit __m512i).
#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Backend;

#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Bytes(__m512i);

/// Native AVX-512 64-bit predicate mask. Bit `i` corresponds to lane `i`
/// (i.e. lane 0 is the LSB).
#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Mask(__mmask64);

#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Score(__m512i);

impl Backend for Avx512U8Backend {
    const LANES: usize = 64;
    const LANE_BYTES: usize = 1;
    type Bytes = Avx512U8Bytes;
    type Mask = Avx512U8Mask;
    type Score = Avx512U8Score;

    fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_extended_feature_info()
            .is_some_and(|info| info.has_avx512f() && info.has_avx512bw() && info.has_avx512vbmi())
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        // movm_epi8 spreads each mask bit to a full byte (0xFF / 0x00).
        unsafe { Avx512U8Score(_mm512_movm_epi8(m.0)) }
    }

    #[inline(always)]
    unsafe fn propagate_horizontal_gaps(
        row: Self::Score,
        adjacent_row: Self::Score,
        match_mask: Self::Score,
        adjacent_match_mask: Self::Score,
        gap_open_penalty: Self::Score,
        gap_extend_penalty: Self::Score,
    ) -> Self::Score {
        unsafe {
            super::propagate_64_lane::<Avx512U8Backend>(
                row,
                adjacent_row,
                match_mask,
                adjacent_match_mask,
                gap_open_penalty,
                gap_extend_penalty,
            )
        }
    }
}

impl BytesVec for Avx512U8Bytes {
    type Mask = Avx512U8Mask;

    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm512_setzero_si512()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        unsafe { Self(_mm512_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        unsafe { Avx512U8Mask(_mm512_cmpeq_epi8_mask(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        unsafe { Avx512U8Mask(_mm512_cmpgt_epu8_mask(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        unsafe { Avx512U8Mask(_mm512_cmplt_epu8_mask(self.0, other.0)) }
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            let remaining = len.saturating_sub(start);
            let ptr = data.add(start);
            Self(match remaining {
                0 => _mm512_setzero_si512(),
                1..64 => {
                    // lanes outside mask are zeroed and don't access memory,
                    // so a partial chunk at the haystack tail is page-safe
                    let mask: __mmask64 = (1u64 << remaining).wrapping_sub(1);
                    _mm512_maskz_loadu_epi8(mask, ptr as *const i8)
                }
                64.. => _mm512_loadu_si512(ptr as *const __m512i),
            })
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 64);
        Self(unsafe { _mm512_loadu_si512(values.as_ptr() as *const __m512i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 64];
        unsafe { _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, self.0) };
        buf.to_vec()
    }
}

impl MaskVec for Avx512U8Mask {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(0)
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self(!self.0)
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        // Lane i = bit i. shift_right_padded_1 places prev's highest lane
        // (bit 63) into lane 0 of the result, and shifts every other lane up
        // by one (lane i -> lane i+1). In bit terms:
        //   new = (self << 1) | (prev >> 63)
        Self((self.0 << 1) | (prev.0 >> 63))
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 64);
        let mut m: u64 = 0;
        for (i, &v) in values.iter().enumerate() {
            if v {
                m |= 1u64 << i;
            }
        }
        Self(m)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        (0..64).map(|i| (self.0 >> i) & 1 == 1).collect()
    }
}

impl ScoreVec for Avx512U8Score {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm512_setzero_si512()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(_mm512_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe {
            let lo = _mm_cvtsi32_si128((value & 0xFF) as i32);
            Self(_mm512_castsi128_si512(lo))
        }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe { Self(_mm512_max_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe {
            // Fold to a 256-bit half-max, then reuse the AVX2 u8 reduction
            // cascade. `vpreduce*` exists as an LLVM intrinsic but the
            // manual fold is easier to reason about and reliably codegens
            // to a tight chain.
            let lo = _mm512_castsi512_si256(self.0);
            let hi = _mm512_extracti64x4_epi64::<1>(self.0);
            let m = _mm256_max_epu8(lo, hi);
            let lo128 = _mm256_castsi256_si128(m);
            let hi128 = _mm256_extracti128_si256::<1>(m);
            let m = _mm_max_epu8(lo128, hi128);
            let m = _mm_max_epu8(m, _mm_srli_si128::<8>(m));
            let m = _mm_max_epu8(m, _mm_srli_si128::<4>(m));
            let m = _mm_max_epu8(m, _mm_srli_si128::<2>(m));
            let m = _mm_max_epu8(m, _mm_srli_si128::<1>(m));
            (_mm_extract_epi8::<0>(m) as u8) as u16
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm512_add_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(_mm512_subs_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm512_and_si512(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe { Self(shift_right_lanes::<L>(prev.0, self.0)) }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            let target = _mm512_set1_epi8(search as i8);
            let mask = _mm512_cmpeq_epi8_mask(self.0, target);
            if mask == 0 {
                64
            } else {
                mask.trailing_zeros() as usize
            }
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 64);
        let mut buf = [0u8; 64];
        for i in 0..64 {
            buf[i] = values[i] as u8;
        }
        Self(unsafe { _mm512_loadu_si512(buf.as_ptr() as *const __m512i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u8; 64];
        unsafe { _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, self.0) };
        buf.iter().map(|&v| v as u16).collect()
    }
}

/// Cross-lane byte shift used by both BytesVec and ScoreVec
/// Uses one VBMI vpermt2b with an idx vector unlike AVX2/SSE impls
#[inline(always)]
unsafe fn shift_right_lanes<const L: i32>(prev: __m512i, cur: __m512i) -> __m512i {
    const { assert!(L >= 0 && L <= 32) };
    if L == 0 {
        return cur;
    }
    // idx[i] = (64 - L) + i. For i < L this is in 0..64 (selects prev[64 - L + i]);
    // for i >= L this is in 64..128 (selects cur[i - L]).
    let idx_arr: [u8; 64] = const {
        let mut arr = [0u8; 64];
        let mut i = 0usize;
        while i < 64 {
            arr[i] = ((64 - L) + i as i32) as u8;
            i += 1;
        }
        arr
    };
    unsafe {
        let idx = _mm512_loadu_si512(idx_arr.as_ptr() as *const __m512i);
        _mm512_permutex2var_epi8(prev, idx, cur)
    }
}
