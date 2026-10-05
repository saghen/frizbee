use core::arch::x86_64::*;

use crate::prefilter::algo::can_overread;
use crate::smith_waterman::algo::{ascii_gap, unicode_gap};

use super::{Backend, BytesVec, MaskVec, ScoreVec};

/// `_mm_alignr_epi8`, built from SSE2 shifts on the tier without SSSE3
macro_rules! alignr {
    ($n:literal, $a:expr, $b:expr) => {
        if SSE41 {
            _mm_alignr_epi8::<$n>($a, $b)
        } else {
            _mm_or_si128(_mm_srli_si128::<$n>($b), _mm_slli_si128::<{ 16 - $n }>($a))
        }
    };
}

/// 8-lane u16 scoring (128-bit __mm128i)
/// 8-lane u8 input (low half of __m128i)
#[derive(Debug, Clone, Copy)]
pub struct BackendSSE<const SSE41: bool>;

/// Physically occupies a full 128-bit register, but only the low 8 bytes are
/// used since we'll eventually widen the 8-bit per lane to 16-bit per lane
/// for the Score vector
#[derive(Debug, Clone, Copy)]
pub struct SseBytes<const SSE41: bool>(__m128i);

#[derive(Debug, Clone, Copy)]
pub struct SseScore<const SSE41: bool>(__m128i);

impl<const SSE41: bool> Backend for BackendSSE<SSE41> {
    const LANES: usize = 8;
    type Bytes = SseBytes<SSE41>;
    type Mask = SseBytes<SSE41>;
    type Score = SseScore<SSE41>;

    fn is_available() -> bool {
        let features = crate::cpuid::detect();
        features.sse2 && (!SSE41 || (features.ssse3 && features.sse41))
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        // mask bytes are 0 or 0xff, so doubling them sign extends
        unsafe { SseScore(_mm_unpacklo_epi8(m.0, m.0)) }
    }

    gap_dispatch!(8);
}

impl<const SSE41: bool> SseBytes<SSE41> {
    /// Safe page-bounded read of 0..8 bytes into the low 64 bits of an __m128i
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
                    let mid = (ptr.add(4) as *const u16).read_unaligned() as u64;
                    let hi = *ptr.add(6) as u64;
                    lo | (mid << 32) | (hi << 48)
                }
                _ => core::hint::unreachable_unchecked(),
            };
            _mm_cvtsi64_si128(val as i64)
        }
    }
}

impl<const SSE41: bool> BytesVec for SseBytes<SSE41> {
    type Mask = SseBytes<SSE41>;

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
            Self(_mm_cmpgt_epi8(
                _mm_xor_si128(self.0, sign),
                _mm_xor_si128(other.0, sign),
            ))
        }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        unsafe {
            let sign = _mm_set1_epi8(-128i8);
            Self(_mm_cmplt_epi8(
                _mm_xor_si128(self.0, sign),
                _mm_xor_si128(other.0, sign),
            ))
        }
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            let remaining = len.saturating_sub(start);
            if remaining == 0 {
                return Self(_mm_setzero_si128());
            }
            let ptr = data.add(start);
            Self(match remaining {
                8.. => _mm_loadl_epi64(ptr as *const __m128i),
                1..=7 if can_overread(ptr, 8) => {
                    let lo = _mm_loadl_epi64(ptr as *const __m128i);
                    let mask = _mm_set_epi64x(0, (1i64 << (remaining * 8)) - 1);
                    _mm_and_si128(lo, mask)
                }
                _ => Self::load_partial_safe(ptr, remaining),
            })
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 8);
        // Place values in low 8 bytes; zero the high 8.
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(values);
        Self(unsafe { _mm_loadu_si128(buf.as_ptr() as *const __m128i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 16];
        unsafe { _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, self.0) };
        buf[..8].to_vec()
    }
}

impl<const SSE41: bool> MaskVec for SseBytes<SSE41> {
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
        unsafe {
            // Want low 8 bytes = [prev[7], self[0..7]].
            // Move prev's low 8 bytes to bytes 8..16 so prev[7] lands at byte
            // 15.
            let shifted = _mm_slli_si128::<8>(prev.0);
            // alignr<15>(self, shifted) = (self || shifted)[15..31]
            //   = [shifted[15], self[0..15]]
            //   = [prev[7], self[0..15]]
            // Low 8 bytes: [prev[7], self[0..7]]. Upper 8 are don't-care.
            Self(alignr!(15, self.0, shifted))
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 8);
        let mut buf = [0u8; 16];
        for i in 0..8 {
            buf[i] = if values[i] { 0xFF } else { 0 };
        }
        Self(unsafe { _mm_loadu_si128(buf.as_ptr() as *const __m128i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        let mut buf = [0u8; 16];
        unsafe { _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, self.0) };
        buf[..8].iter().map(|&v| v != 0).collect()
    }
}

impl<const SSE41: bool> ScoreVec for SseScore<SSE41> {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm_setzero_si128()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(_mm_set1_epi16(value as i16)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe { Self(_mm_cvtsi32_si128(value as i32)) }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe {
            Self(if SSE41 {
                _mm_max_epu16(self.0, other.0)
            } else {
                _mm_adds_epu16(_mm_subs_epu16(self.0, other.0), other.0)
            })
        }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe {
            if !SSE41 {
                let m = self.max(Self(_mm_srli_si128::<8>(self.0)));
                let m = m.max(Self(_mm_srli_si128::<4>(m.0)));
                let m = m.max(Self(_mm_srli_si128::<2>(m.0)));
                return _mm_extract_epi16::<0>(m.0) as u16;
            }
            // PHMINPOSUW finds the minimum; invert to find the max.
            let all_ones = _mm_set1_epi16(-1);
            let inverted = _mm_xor_si128(self.0, all_ones);
            let min_pos = _mm_minpos_epu16(inverted);
            let min_val = _mm_extract_epi16::<0>(min_pos) as u16;
            !min_val
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm_adds_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(_mm_subs_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe {
            const { assert!(L >= 0 && L <= 8) };
            Self(match L {
                0 => self.0,
                1 => alignr!(14, self.0, prev.0),
                2 => alignr!(12, self.0, prev.0),
                3 => alignr!(10, self.0, prev.0),
                4 => alignr!(8, self.0, prev.0),
                5 => alignr!(6, self.0, prev.0),
                6 => alignr!(4, self.0, prev.0),
                7 => alignr!(2, self.0, prev.0),
                8 => prev.0,
                _ => core::hint::unreachable_unchecked(),
            })
        }
    }
    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 8);
        Self(unsafe { _mm_loadu_si128(values.as_ptr() as *const __m128i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u16; 8];
        unsafe { _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, self.0) };
        buf.to_vec()
    }
}

/// 16-lane u8 scoring (128-bit __m128i), 16-lane u8 input (128-bit __m128i)
#[derive(Debug, Clone, Copy)]
pub struct BackendSSEU8<const SSE41: bool>;

#[derive(Debug, Clone, Copy)]
pub struct SseU8Bytes<const SSE41: bool>(__m128i);

#[derive(Debug, Clone, Copy)]
pub struct SseU8Score<const SSE41: bool>(__m128i);

impl<const SSE41: bool> Backend for BackendSSEU8<SSE41> {
    const LANES: usize = 16;
    type Bytes = SseU8Bytes<SSE41>;
    type Mask = SseU8Bytes<SSE41>;
    type Score = SseU8Score<SSE41>;

    fn is_available() -> bool {
        BackendSSE::<SSE41>::is_available()
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        SseU8Score(m.0)
    }

    gap_dispatch!(16);
}

impl<const SSE41: bool> BytesVec for SseU8Bytes<SSE41> {
    type Mask = SseU8Bytes<SSE41>;

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
            Self(_mm_cmpgt_epi8(
                _mm_xor_si128(self.0, sign),
                _mm_xor_si128(other.0, sign),
            ))
        }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        unsafe {
            let sign = _mm_set1_epi8(-128i8);
            Self(_mm_cmplt_epi8(
                _mm_xor_si128(self.0, sign),
                _mm_xor_si128(other.0, sign),
            ))
        }
    }
    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe { Self(super::avx::load_partial_m128i(data, start, len)) }
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

impl<const SSE41: bool> MaskVec for SseU8Bytes<SSE41> {
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
        // Full 16-byte register, full 16 meaningful bytes. alignr_epi8::<15>
        // gives [prev[15], self[0..15]] which is the desired result.
        unsafe { Self(alignr!(15, self.0, prev.0)) }
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

impl<const SSE41: bool> ScoreVec for SseU8Score<SSE41> {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm_setzero_si128()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(_mm_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe { Self(_mm_cvtsi32_si128((value & 0xFF) as i32)) }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe { Self(_mm_max_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe {
            // Cascade of pairwise maxes halving the lane count each step.
            let m = _mm_max_epu8(self.0, _mm_srli_si128::<8>(self.0));
            let m = _mm_max_epu8(m, _mm_srli_si128::<4>(m));
            let m = _mm_max_epu8(m, _mm_srli_si128::<2>(m));
            let m = _mm_max_epu8(m, _mm_srli_si128::<1>(m));
            (_mm_cvtsi128_si32(m) as u8) as u16
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm_adds_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(_mm_subs_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe {
            const { assert!(L >= 0 && L <= 16) };
            Self(match L {
                0 => self.0,
                1 => alignr!(15, self.0, prev.0),
                2 => alignr!(14, self.0, prev.0),
                3 => alignr!(13, self.0, prev.0),
                4 => alignr!(12, self.0, prev.0),
                5 => alignr!(11, self.0, prev.0),
                6 => alignr!(10, self.0, prev.0),
                7 => alignr!(9, self.0, prev.0),
                8 => alignr!(8, self.0, prev.0),
                9 => alignr!(7, self.0, prev.0),
                10 => alignr!(6, self.0, prev.0),
                11 => alignr!(5, self.0, prev.0),
                12 => alignr!(4, self.0, prev.0),
                13 => alignr!(3, self.0, prev.0),
                14 => alignr!(2, self.0, prev.0),
                15 => alignr!(1, self.0, prev.0),
                16 => prev.0,
                _ => core::hint::unreachable_unchecked(),
            })
        }
    }
    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 16);
        let mut buf = [0u8; 16];
        for i in 0..16 {
            buf[i] = values[i] as u8;
        }
        Self(unsafe { _mm_loadu_si128(buf.as_ptr() as *const __m128i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u8; 16];
        unsafe { _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, self.0) };
        buf.iter().map(|&v| v as u16).collect()
    }
}
