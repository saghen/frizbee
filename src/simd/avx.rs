use std::arch::x86_64::*;

use super::{Backend, BytesVec, MaskVec, ScoreVec};

/// 16-lane u16 scoring (256-bit __m256i), 16-lane u8 input (128-bit __m128i).
#[derive(Debug, Clone, Copy)]
pub struct AvxBackend;

#[derive(Debug, Clone, Copy)]
pub struct AvxBytes(__m128i);

#[derive(Debug, Clone, Copy)]
pub struct AvxScore(__m256i);

impl Backend for AvxBackend {
    const LANES: usize = 16;
    const LANE_BYTES: usize = 2;
    type Bytes = AvxBytes;
    type Mask = AvxBytes;
    type Score = AvxScore;

    fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_extended_feature_info()
            .is_some_and(|info| info.has_avx2())
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        unsafe { AvxScore(_mm256_cvtepi8_epi16(m.0)) }
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
            super::propagate_16_lane::<AvxBackend>(
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

/// Safe page-bounded read of 0..8 bytes into the low 64 bits of an __m128i.
/// The high 8 bytes are zero.
#[inline(always)]
unsafe fn load_partial_safe(ptr: *const u8, len: usize) -> __m128i {
    unsafe {
        debug_assert!(len < 8);
        let val: u64 = match len {
            0 => 0,
            1 => *ptr as u64,
            2 => (ptr as *const u16).read_unaligned() as u64,
            3 => {
                let lo = (ptr as *const u16).read_unaligned() as u64;
                let hi = *ptr.add(2) as u64;
                lo | (hi << 16)
            }
            4 => (ptr as *const u32).read_unaligned() as u64,
            5 => {
                let lo = (ptr as *const u32).read_unaligned() as u64;
                let hi = *ptr.add(4) as u64;
                lo | (hi << 32)
            }
            6 => {
                let lo = (ptr as *const u32).read_unaligned() as u64;
                let hi = (ptr.add(4) as *const u16).read_unaligned() as u64;
                lo | (hi << 32)
            }
            7 => {
                let lo = (ptr as *const u32).read_unaligned() as u64;
                let hi = (ptr.add(4) as *const u32).read_unaligned() as u64;
                lo | ((hi & 0xFFFFFF) << 32)
            }
            _ => std::hint::unreachable_unchecked(),
        };
        _mm_cvtsi64_si128(val as i64)
    }
}

#[inline(always)]
fn can_overread_8(ptr: *const u8) -> bool {
    (ptr as usize & 0xFFF) <= (4096 - 8)
}

/// Page-safe load of up to 16 bytes into an __m128i. `start` is the byte
/// offset within `data` (of total length `len`). Out-of-range bytes are zero.
#[inline(always)]
pub(crate) unsafe fn load_partial_m128i(data: *const u8, start: usize, len: usize) -> __m128i {
    unsafe {
        let remaining = len.saturating_sub(start);
        let ptr = data.add(start);
        match remaining {
            0 => _mm_setzero_si128(),
            8 => _mm_loadl_epi64(ptr as *const __m128i),
            16.. => _mm_loadu_si128(ptr as *const __m128i),
            1..=7 if can_overread_8(ptr) => {
                let lo = _mm_loadl_epi64(ptr as *const __m128i);
                let mask = _mm_set_epi64x(0, (1i64 << (remaining * 8)) - 1);
                _mm_and_si128(lo, mask)
            }
            1..=7 => load_partial_safe(ptr, remaining),
            9..=15 => {
                let lo = _mm_loadl_epi64(ptr as *const __m128i);
                let hi = _mm_loadl_epi64(ptr.add(remaining - 8) as *const __m128i);
                let hi = match 16 - remaining {
                    1 => _mm_srli_si128::<1>(hi),
                    2 => _mm_srli_si128::<2>(hi),
                    3 => _mm_srli_si128::<3>(hi),
                    4 => _mm_srli_si128::<4>(hi),
                    5 => _mm_srli_si128::<5>(hi),
                    6 => _mm_srli_si128::<6>(hi),
                    7 => _mm_srli_si128::<7>(hi),
                    _ => std::hint::unreachable_unchecked(),
                };
                _mm_unpacklo_epi64(lo, hi)
            }
        }
    }
}

impl BytesVec for AvxBytes {
    type Mask = AvxBytes;

    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm_setzero_si128()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        unsafe { Self(_mm_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        unsafe { Self(_mm_cmpeq_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        unsafe {
            let sign = _mm_set1_epi8(-128i8);
            let a = _mm_xor_si128(self.0, sign);
            let b = _mm_xor_si128(other.0, sign);
            Self(_mm_cmpgt_epi8(a, b))
        }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        unsafe {
            let sign = _mm_set1_epi8(-128i8);
            let a = _mm_xor_si128(self.0, sign);
            let b = _mm_xor_si128(other.0, sign);
            Self(_mm_cmplt_epi8(a, b))
        }
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe { Self(load_partial_m128i(data, start, len)) }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 16);
        Self(unsafe { _mm_loadu_si128(values.as_ptr() as *const __m128i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 16];
        unsafe { _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, self.0) };
        buf.to_vec()
    }
}

impl MaskVec for AvxBytes {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm_setzero_si128()) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        unsafe { Self(_mm_or_si128(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1))) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        unsafe { Self(_mm_alignr_epi8::<15>(self.0, prev.0)) }
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 16);
        let mut buf = [0u8; 16];
        for i in 0..16 {
            buf[i] = if values[i] { 0xFF } else { 0 };
        }
        Self(unsafe { _mm_loadu_si128(buf.as_ptr() as *const __m128i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        let mut buf = [0u8; 16];
        unsafe { _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, self.0) };
        buf.iter().map(|&v| v != 0).collect()
    }
}

impl ScoreVec for AvxScore {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm256_setzero_si256()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(_mm256_set1_epi16(value as i16)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe {
            // Lane 0 = value, lanes 1..16 = 0. Insert into low 128.
            let lo = _mm_cvtsi32_si128(value as i32);
            Self(_mm256_castsi128_si256(lo))
        }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe { Self(_mm256_max_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe {
            let high = _mm256_extracti128_si256::<1>(self.0);
            let low = _mm256_castsi256_si128(self.0);
            let m = _mm_max_epu16(low, high);
            let m = _mm_max_epu16(m, _mm_srli_si128::<8>(m));
            let m = _mm_max_epu16(m, _mm_srli_si128::<4>(m));
            let m = _mm_max_epu16(m, _mm_srli_si128::<2>(m));
            _mm_extract_epi16::<0>(m) as u16
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm256_add_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(_mm256_subs_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm256_and_si256(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe {
            const { assert!(L >= 0 && L <= 8) };
            // permute2x128(prev, self, 0x21) = [prev_high, self_low]
            // alignr_epi8(self, permuted, 16 - L*2) lane-pair-wise yields the
            // desired right-shift-with-fill.
            let permuted = _mm256_permute2x128_si256::<0x21>(prev.0, self.0);
            Self(match L {
                0 => self.0,
                1 => _mm256_alignr_epi8::<14>(self.0, permuted),
                2 => _mm256_alignr_epi8::<12>(self.0, permuted),
                3 => _mm256_alignr_epi8::<10>(self.0, permuted),
                4 => _mm256_alignr_epi8::<8>(self.0, permuted),
                5 => _mm256_alignr_epi8::<6>(self.0, permuted),
                6 => _mm256_alignr_epi8::<4>(self.0, permuted),
                7 => _mm256_alignr_epi8::<2>(self.0, permuted),
                8 => permuted,
                _ => std::hint::unreachable_unchecked(),
            })
        }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            let cmp = _mm256_cmpeq_epi16(self.0, _mm256_set1_epi16(search as i16));
            let mask = _mm256_movemask_epi8(cmp) as u32;
            (mask.trailing_zeros() as usize / 2).min(16)
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 16);
        Self(unsafe { _mm256_loadu_si256(values.as_ptr() as *const __m256i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u16; 16];
        unsafe { _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, self.0) };
        buf.to_vec()
    }
}

/// 32-lane u8 scoring (256-bit __m256i), 32-lane u8 input (256-bit __m256i).
#[derive(Debug, Clone, Copy)]
pub struct AvxU8Backend;

#[derive(Debug, Clone, Copy)]
pub struct AvxU8Bytes(__m256i);

#[derive(Debug, Clone, Copy)]
pub struct AvxU8Score(__m256i);

impl Backend for AvxU8Backend {
    const LANES: usize = 32;
    const LANE_BYTES: usize = 1;
    type Bytes = AvxU8Bytes;
    type Mask = AvxU8Bytes;
    type Score = AvxU8Score;

    fn is_available() -> bool {
        AvxBackend::is_available()
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        AvxU8Score(m.0)
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
            super::propagate_32_lane::<AvxU8Backend>(
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

impl BytesVec for AvxU8Bytes {
    type Mask = AvxU8Bytes;

    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm256_setzero_si256()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        unsafe { Self(_mm256_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        unsafe { Self(_mm256_cmpeq_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        unsafe {
            let sign = _mm256_set1_epi8(-128i8);
            Self(_mm256_cmpgt_epi8(
                _mm256_xor_si256(self.0, sign),
                _mm256_xor_si256(other.0, sign),
            ))
        }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        unsafe {
            let sign = _mm256_set1_epi8(-128i8);
            Self(_mm256_cmpgt_epi8(
                _mm256_xor_si256(other.0, sign),
                _mm256_xor_si256(self.0, sign),
            ))
        }
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            // Fast path: full 32-byte unaligned load when enough remains.
            if start + 32 <= len {
                return Self(_mm256_loadu_si256(data.add(start) as *const __m256i));
            }
            // Otherwise compose two 16-byte page-safe loads.
            let lo = load_partial_m128i(data, start, len);
            let hi = load_partial_m128i(data, start + 16, len);
            Self(_mm256_set_m128i(hi, lo))
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 32);
        Self(unsafe { _mm256_loadu_si256(values.as_ptr() as *const __m256i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 32];
        unsafe { _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, self.0) };
        buf.to_vec()
    }
}

impl MaskVec for AvxU8Bytes {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm256_setzero_si256()) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm256_and_si256(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        unsafe { Self(_mm256_or_si256(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self(_mm256_xor_si256(self.0, _mm256_set1_epi32(-1))) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        unsafe {
            // permute2x128(prev, self, 0x21) = (low = prev_high, high = self_low)
            // alignr_epi8 then runs per 128-bit lane, splicing
            //   low  = [prev[31], self[0..15]]
            //   high = [self[15], self[16..30]]
            // which together form the desired right-shift-by-1-byte result.
            let permuted = _mm256_permute2x128_si256::<0x21>(prev.0, self.0);
            Self(_mm256_alignr_epi8::<15>(self.0, permuted))
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 32);
        let mut buf = [0u8; 32];
        for i in 0..32 {
            buf[i] = if values[i] { 0xFF } else { 0 };
        }
        Self(unsafe { _mm256_loadu_si256(buf.as_ptr() as *const __m256i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        let mut buf = [0u8; 32];
        unsafe { _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, self.0) };
        buf.iter().map(|&v| v != 0).collect()
    }
}

impl ScoreVec for AvxU8Score {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm256_setzero_si256()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(_mm256_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe {
            let lo = _mm_cvtsi32_si128((value & 0xFF) as i32);
            Self(_mm256_castsi128_si256(lo))
        }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe { Self(_mm256_max_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe {
            let high = _mm256_extracti128_si256::<1>(self.0);
            let low = _mm256_castsi256_si128(self.0);
            let m = _mm_max_epu8(low, high);
            let m = _mm_max_epu8(m, _mm_srli_si128::<8>(m));
            let m = _mm_max_epu8(m, _mm_srli_si128::<4>(m));
            let m = _mm_max_epu8(m, _mm_srli_si128::<2>(m));
            let m = _mm_max_epu8(m, _mm_srli_si128::<1>(m));
            (_mm_extract_epi8::<0>(m) as u8) as u16
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm256_add_epi8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(_mm256_subs_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm256_and_si256(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe {
            const { assert!(L >= 0 && L <= 16) };
            // For L lane shift on a 32-byte vector composed of two 128-bit
            // halves, we permute (prev_high, self_low) into one register and
            // use alignr_epi8 byte count = 16 - L, which splices each half
            // correctly. See AVX2 u16 backend for the same trick at u16
            // granularity (byte count = 16 - 2L).
            let permuted = _mm256_permute2x128_si256::<0x21>(prev.0, self.0);
            Self(match L {
                0 => self.0,
                1 => _mm256_alignr_epi8::<15>(self.0, permuted),
                2 => _mm256_alignr_epi8::<14>(self.0, permuted),
                3 => _mm256_alignr_epi8::<13>(self.0, permuted),
                4 => _mm256_alignr_epi8::<12>(self.0, permuted),
                5 => _mm256_alignr_epi8::<11>(self.0, permuted),
                6 => _mm256_alignr_epi8::<10>(self.0, permuted),
                7 => _mm256_alignr_epi8::<9>(self.0, permuted),
                8 => _mm256_alignr_epi8::<8>(self.0, permuted),
                9 => _mm256_alignr_epi8::<7>(self.0, permuted),
                10 => _mm256_alignr_epi8::<6>(self.0, permuted),
                11 => _mm256_alignr_epi8::<5>(self.0, permuted),
                12 => _mm256_alignr_epi8::<4>(self.0, permuted),
                13 => _mm256_alignr_epi8::<3>(self.0, permuted),
                14 => _mm256_alignr_epi8::<2>(self.0, permuted),
                15 => _mm256_alignr_epi8::<1>(self.0, permuted),
                16 => permuted,
                _ => std::hint::unreachable_unchecked(),
            })
        }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            let target = _mm256_set1_epi8(search as i8);
            let cmp = _mm256_cmpeq_epi8(self.0, target);
            let mask = _mm256_movemask_epi8(cmp) as u32;
            (mask.trailing_zeros() as usize).min(32)
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 32);
        let mut buf = [0u8; 32];
        for i in 0..32 {
            buf[i] = values[i] as u8;
        }
        Self(unsafe { _mm256_loadu_si256(buf.as_ptr() as *const __m256i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u8; 32];
        unsafe { _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, self.0) };
        buf.iter().map(|&v| v as u16).collect()
    }
}
