use std::arch::x86_64::*;

use crate::smith_waterman::algo::{ascii_gap, unicode_gap};

use super::{Backend, BytesVec, MaskVec, ScoreVec};

/// 32-lane u16 scoring (512-bit __m512i), 32-lane u8 input (low half of __m512i).
#[derive(Debug, Clone, Copy)]
pub struct BackendAVX512;

#[derive(Debug, Clone, Copy)]
pub struct Avx512Bytes(__m512i);

/// Native AVX-512 32-bit predicate mask. Bit `i` corresponds to lane `i`
/// (i.e. lane 0 is the LSB).
#[derive(Debug, Clone, Copy)]
pub struct Avx512Mask(__mmask32);

#[derive(Debug, Clone, Copy)]
pub struct Avx512Score(__m512i);

/// 64-lane u8 scoring (512-bit __m512i), 64-lane u8 input (512-bit __m512i).
#[derive(Debug, Clone, Copy)]
pub struct BackendAVX512U8;

#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Bytes(__m512i);

/// Native AVX-512 64-bit predicate mask. Bit `i` corresponds to lane `i`
/// (i.e. lane 0 is the LSB).
#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Mask(__mmask64);

#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Score(__m512i);

impl Backend for BackendAVX512 {
    const LANES: usize = 32;
    const LANE_BYTES: usize = 2;
    type Bytes = Avx512Bytes;
    type Mask = Avx512Mask;
    type Score = Avx512Score;

    fn is_available() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        // movm_epi16 spreads each mask bit to a full word (0xFFFF / 0x0000).
        unsafe { Avx512Score(_mm512_movm_epi16(m.0)) }
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
            ascii_gap::propagate_32_lane::<BackendAVX512>(
                row,
                adjacent_row,
                match_mask,
                adjacent_match_mask,
                gap_open_penalty,
                gap_extend_penalty,
            )
        }
    }

    #[inline(always)]
    unsafe fn propagate_horizontal_unicode_gaps(
        row: Self::Score,
        adjacent_row: Self::Score,
        pending_gap_open_mask: Self::Score,
        adjacent_pending_gap_open_mask: Self::Score,
        continuation_gap_extend_penalty: Self::Score,
        adjacent_continuation_gap_extend_penalty: Self::Score,
        scalar_end_mask: Self::Score,
        adjacent_scalar_end_mask: Self::Score,
        gap_open_penalty: Self::Score,
        gap_extend_penalty: Self::Score,
    ) -> (Self::Score, Self::Score) {
        unsafe {
            unicode_gap::propagate_unicode_32_lane::<BackendAVX512>(
                row,
                adjacent_row,
                pending_gap_open_mask,
                adjacent_pending_gap_open_mask,
                continuation_gap_extend_penalty,
                adjacent_continuation_gap_extend_penalty,
                scalar_end_mask,
                adjacent_scalar_end_mask,
                gap_open_penalty,
                gap_extend_penalty,
            )
        }
    }
}

impl Backend for BackendAVX512U8 {
    const LANES: usize = 64;
    const LANE_BYTES: usize = 1;
    type Bytes = Avx512U8Bytes;
    type Mask = Avx512U8Mask;
    type Score = Avx512U8Score;

    fn is_available() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
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
            ascii_gap::propagate_64_lane::<BackendAVX512U8>(
                row,
                adjacent_row,
                match_mask,
                adjacent_match_mask,
                gap_open_penalty,
                gap_extend_penalty,
            )
        }
    }

    #[inline(always)]
    unsafe fn propagate_horizontal_unicode_gaps(
        row: Self::Score,
        adjacent_row: Self::Score,
        pending_gap_open_mask: Self::Score,
        adjacent_pending_gap_open_mask: Self::Score,
        continuation_gap_extend_penalty: Self::Score,
        adjacent_continuation_gap_extend_penalty: Self::Score,
        scalar_end_mask: Self::Score,
        adjacent_scalar_end_mask: Self::Score,
        gap_open_penalty: Self::Score,
        gap_extend_penalty: Self::Score,
    ) -> (Self::Score, Self::Score) {
        unsafe {
            unicode_gap::propagate_unicode_64_lane::<BackendAVX512U8>(
                row,
                adjacent_row,
                pending_gap_open_mask,
                adjacent_pending_gap_open_mask,
                continuation_gap_extend_penalty,
                adjacent_continuation_gap_extend_penalty,
                scalar_end_mask,
                adjacent_scalar_end_mask,
                gap_open_penalty,
                gap_extend_penalty,
            )
        }
    }
}

impl BytesVec for Avx512Bytes {
    type Mask = Avx512Mask;

    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        unsafe { Self(_mm512_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        unsafe { Avx512Mask((_mm512_cmpeq_epi8_mask(self.0, other.0) as u32) as __mmask32) }
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        unsafe { Avx512Mask((_mm512_cmpgt_epu8_mask(self.0, other.0) as u32) as __mmask32) }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        unsafe { Avx512Mask((_mm512_cmplt_epu8_mask(self.0, other.0) as u32) as __mmask32) }
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            let remaining = len.saturating_sub(start).min(32);
            if remaining == 0 {
                return Self(_mm512_setzero_si512());
            }
            let ptr = data.add(start);
            let mask: __mmask64 = ((1u64 << remaining).wrapping_sub(1)) as __mmask64;
            Self(_mm512_maskz_loadu_epi8(mask, ptr as *const i8))
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 32);
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(values);
        Self(unsafe { _mm512_loadu_si512(buf.as_ptr() as *const __m512i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 64];
        unsafe { _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, self.0) };
        buf[..32].to_vec()
    }
}

impl BytesVec for Avx512U8Bytes {
    type Mask = Avx512U8Mask;

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
            if remaining == 0 {
                return Self(_mm512_setzero_si512());
            }
            let ptr = data.add(start);
            Self(match remaining {
                0 => unreachable!(),
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

impl MaskVec for Avx512Mask {
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
    unsafe fn is_zero(self) -> bool {
        self.0 == 0
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        // Lane i = bit i. shift_right_padded_1 places prev's highest lane
        // (bit 31) into lane 0 of the result, and shifts every other lane up
        // by one (lane i -> lane i+1).
        Self((self.0 << 1) | (prev.0 >> 31))
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 32);
        let mut m: u32 = 0;
        for (i, &v) in values.iter().enumerate() {
            if v {
                m |= 1u32 << i;
            }
        }
        Self(m)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        (0..32).map(|i| (self.0 >> i) & 1 == 1).collect()
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
    unsafe fn is_zero(self) -> bool {
        self.0 == 0
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

impl ScoreVec for Avx512Score {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm512_setzero_si512()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(_mm512_set1_epi16(value as i16)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe {
            let lo = _mm_cvtsi32_si128(value as i32);
            Self(_mm512_castsi128_si512(lo))
        }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe { Self(_mm512_max_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe {
            let lo = _mm512_castsi512_si256(self.0);
            let hi = _mm512_extracti64x4_epi64::<1>(self.0);
            let m = _mm256_max_epu16(lo, hi);
            let lo128 = _mm256_castsi256_si128(m);
            let hi128 = _mm256_extracti128_si256::<1>(m);
            let m = _mm_max_epu16(lo128, hi128);
            let m = _mm_max_epu16(m, _mm_srli_si128::<8>(m));
            let m = _mm_max_epu16(m, _mm_srli_si128::<4>(m));
            let m = _mm_max_epu16(m, _mm_srli_si128::<2>(m));
            _mm_extract_epi16::<0>(m) as u16
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm512_add_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(_mm512_subs_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm512_and_si512(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe { Self(shift_right_u16_lanes::<L>(prev.0, self.0)) }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            let target = _mm512_set1_epi16(search as i16);
            let mask = _mm512_cmpeq_epi16_mask(self.0, target);
            (mask.trailing_zeros() as usize).min(32)
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 32);
        Self(unsafe { _mm512_loadu_si512(values.as_ptr() as *const __m512i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u16; 32];
        unsafe { _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, self.0) };
        buf.to_vec()
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
            mask.trailing_zeros() as usize
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

#[inline(always)]
unsafe fn shift_right_u16_lanes<const L: i32>(prev: __m512i, cur: __m512i) -> __m512i {
    const { assert!(L >= 0 && L <= 32) };
    match L {
        0 => cur,
        32 => prev,

        _ if cfg!(miri) => unsafe {
            // Miri doesn't support _mm512_permutex2var_epi16, so use a fallback implementation
            // TODO: add support to Miri
            let mut cur_arr = [0u16; 32];
            _mm512_storeu_si512(cur_arr.as_mut_ptr() as *mut __m512i, cur);

            let mut prev_arr = [0u16; 32];
            _mm512_storeu_si512(prev_arr.as_mut_ptr() as *mut __m512i, prev);

            let mut shifted = [0u16; 32];
            std::ptr::copy_nonoverlapping(
                cur_arr.as_ptr(),
                shifted.as_mut_ptr().add(L as usize),
                32 - L as usize,
            );
            std::ptr::copy_nonoverlapping(
                prev_arr.as_ptr().add(32 - L as usize),
                shifted.as_mut_ptr(),
                L as usize,
            );
            _mm512_loadu_si512(shifted.as_ptr() as *const __m512i)
        },

        _ => {
            let idx_arr: [u16; 32] = const {
                let mut arr = [0u16; 32];
                let mut i = 0usize;
                while i < 32 {
                    arr[i] = ((32 - L) + i as i32) as u16;
                    i += 1;
                }
                arr
            };
            let idx = unsafe { _mm512_loadu_si512(idx_arr.as_ptr() as *const __m512i) };
            unsafe { _mm512_permutex2var_epi16(prev, idx, cur) }
        }
    }
}

/// Cross-lane byte shift used by both BytesVec and ScoreVec.
#[inline(always)]
unsafe fn shift_right_lanes<const L: i32>(prev: __m512i, cur: __m512i) -> __m512i {
    const { assert!(L >= 0 && L <= 32) };
    match L {
        0 => cur,

        // use native cross-lane byte shifts when available
        4 => unsafe { _mm512_alignr_epi32::<15>(cur, prev) },
        8 => unsafe { _mm512_alignr_epi64::<7>(cur, prev) },
        16 => unsafe { _mm512_alignr_epi64::<6>(cur, prev) },
        32 => unsafe { _mm512_alignr_epi64::<4>(cur, prev) },

        _ if cfg!(miri) => unsafe {
            // Miri doesn't support _mm512_permutex2var_epi16, so use a fallback implementation
            // TODO: add support to Miri
            let mut cur_arr = [0u8; 64];
            _mm512_store_si512(cur_arr.as_mut_ptr() as *mut __m512i, cur);

            let mut prev_arr = [0u8; 64];
            _mm512_store_si512(prev_arr.as_mut_ptr() as *mut __m512i, prev);

            let mut shifted = [0u8; 64];
            std::ptr::copy_nonoverlapping(
                cur_arr.as_ptr(),
                shifted.as_mut_ptr().add(L as usize),
                64 - L as usize,
            );
            std::ptr::copy_nonoverlapping(
                prev_arr.as_ptr().add(64 - L as usize),
                shifted.as_mut_ptr(),
                L as usize,
            );
            _mm512_loadu_si512(shifted.as_ptr() as *const __m512i)
        },

        // fallback to byte permute
        _ => {
            let idx_arr: [u8; 64] = const {
                let mut arr = [0u8; 64];
                let mut i = 0usize;
                while i < 64 {
                    arr[i] = ((64 - L) + i as i32) as u8;
                    i += 1;
                }
                arr
            };
            let idx = unsafe { _mm512_loadu_si512(idx_arr.as_ptr() as *const __m512i) };
            unsafe { _mm512_permutex2var_epi8(prev, idx, cur) }
        }
    }
}
