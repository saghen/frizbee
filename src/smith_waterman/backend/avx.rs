use core::arch::x86_64::*;

use crate::prefilter::algo::can_overread;
use crate::prefilter::backend::copy_partial;

use super::{Backend, BytesVec, MaskVec, ScoreVec, keep_first};

/// 16-lane u16 scoring (256-bit __m256i), 16-lane u8 input (128-bit __m128i).
#[derive(Debug, Clone, Copy)]
pub struct BackendAVX;

#[derive(Debug, Clone, Copy)]
pub struct AvxBytes(__m128i);

#[derive(Debug, Clone, Copy)]
pub struct AvxScore(__m256i);

impl Backend for BackendAVX {
    const LANES: usize = 16;
    const LANE_BYTES: usize = 2;
    type Bytes = AvxBytes;
    type Mask = AvxBytes;
    type Score = AvxScore;

    fn is_available() -> bool {
        crate::cpuid::detect().avx2
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        unsafe { AvxScore(_mm256_cvtepi8_epi16(m.0)) }
    }

    gap_dispatch!(16);
}

/// Page-safe load of up to 16 bytes into an __m128i. `start` is the byte
/// offset within `data` (of total length `len`). Out-of-range bytes are zero.
#[inline(always)]
pub(crate) unsafe fn load_partial_m128i(data: *const u8, start: usize, len: usize) -> __m128i {
    unsafe {
        let remaining = len.saturating_sub(start);
        let ptr = data.wrapping_add(start);
        if remaining >= 16 {
            return _mm_loadu_si128(ptr as *const __m128i);
        }
        if remaining == 0 {
            return _mm_setzero_si128();
        }
        // A full load with the bytes past the end zeroed, unless it would
        // cross into the next page
        if can_overread(ptr, 16) {
            let keep = _mm_loadu_si128(keep_first(remaining) as *const __m128i);
            _mm_and_si128(_mm_loadu_si128(ptr as *const __m128i), keep)
        } else {
            _mm_loadu_si128(copy_partial::<16>(ptr, remaining).as_ptr() as *const __m128i)
        }
    }
}

impl BytesVec for AvxBytes {
    type Mask = AvxBytes;

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
            Self(_mm256_zextsi128_si256(lo))
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
            super::sse::horizontal_max_epu16(_mm_max_epu16(low, high))
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm256_adds_epu16(self.0, other.0)) }
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
            const { assert!(L >= 0 && L <= 16) };
            // permuted = [prev_high, self_low]
            // alignr_epi8(self, permuted, 16 - L*2) lane-pair-wise yields the
            // desired right-shift-with-fill. Past 8 lanes, each half splices
            // the next lower half instead: alignr_epi8(permuted, prev, 32 - L*2)
            let permuted = prev_high_self_low(prev.0, self.0);
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
                12 => _mm256_alignr_epi8::<8>(permuted, prev.0),
                16 => prev.0,
                _ => core::hint::unreachable_unchecked(),
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

/// `[prev_high, self_low]`, the window cross-lane shifts splice from. An
/// insert of `self`'s low half rather than `vperm2i128`, since `self` is
/// usually on the row's dependency chain and the insert is faster on AMD
#[inline(always)]
unsafe fn prev_high_self_low(prev: __m256i, this: __m256i) -> __m256i {
    unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm256_extracti128_si256::<1>(prev)),
            _mm256_castsi256_si128(this),
        )
    }
}

/// 32-lane u8 scoring (256-bit __m256i), 32-lane u8 input (256-bit __m256i).
#[derive(Debug, Clone, Copy)]
pub struct BackendAVXU8;

#[derive(Debug, Clone, Copy)]
pub struct AvxU8Bytes(__m256i);

#[derive(Debug, Clone, Copy)]
pub struct AvxU8Score(__m256i);

impl Backend for BackendAVXU8 {
    const LANES: usize = 32;
    const LANE_BYTES: usize = 1;
    type Bytes = AvxU8Bytes;
    type Mask = AvxU8Bytes;
    type Score = AvxU8Score;

    fn is_available() -> bool {
        BackendAVX::is_available()
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        AvxU8Score(m.0)
    }

    gap_dispatch!(32);
}

impl BytesVec for AvxU8Bytes {
    type Mask = AvxU8Bytes;

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
            let remaining = len.saturating_sub(start);
            // `start` may be past `len` (the shorter haystack of a pair): the
            // pointer is only formed with wrapping arithmetic
            let ptr = data.wrapping_add(start);
            if remaining >= 32 {
                return Self(_mm256_loadu_si256(ptr as *const __m256i));
            }
            if remaining == 0 {
                return Self(_mm256_setzero_si256());
            }
            // A full load with the bytes past the end zeroed, unless it would
            // cross into the next page
            Self(if can_overread(ptr, 32) {
                let keep = _mm256_loadu_si256(keep_first(remaining) as *const __m256i);
                _mm256_and_si256(_mm256_loadu_si256(ptr as *const __m256i), keep)
            } else {
                _mm256_loadu_si256(copy_partial::<32>(ptr, remaining).as_ptr() as *const __m256i)
            })
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
            // permuted = (low = prev_high, high = self_low)
            // alignr_epi8 then runs per 128-bit lane, splicing
            //   low  = [prev[31], self[0..15]]
            //   high = [self[15], self[16..30]]
            // which together form the desired right-shift-by-1-byte result.
            let permuted = prev_high_self_low(prev.0, self.0);
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
            Self(_mm256_zextsi128_si256(lo))
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
            super::sse::horizontal_max_epu8(_mm_max_epu8(low, high))
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm256_adds_epu8(self.0, other.0)) }
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
            // halves, we splice (prev_high, self_low) into one register and
            // use alignr_epi8 byte count = 16 - L, which splices each half
            // correctly. See AVX2 u16 backend for the same trick at u16
            // granularity (byte count = 16 - 2L).
            let permuted = prev_high_self_low(prev.0, self.0);
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
                _ => core::hint::unreachable_unchecked(),
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
