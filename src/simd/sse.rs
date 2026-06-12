use std::arch::x86_64::*;

use super::{Backend, BytesVec, MaskVec, ScoreVec};

/// 8-lane u16 scoring (128-bit __mm128i), 8-lane u8 input (low half of __m128i).
#[derive(Debug, Clone, Copy)]
pub struct SseBackend;

/// Physically occupies a full 128-bit register, but only the low 8 bytes are
/// used since we'll eventually widen the 8-bit per lane to 16-bit per lane
/// for the Score vector
#[derive(Debug, Clone, Copy)]
pub struct SseBytes(__m128i);

#[derive(Debug, Clone, Copy)]
pub struct SseScore(__m128i);

impl Backend for SseBackend {
    const LANES: usize = 8;
    const LANE_BYTES: usize = 2;
    type Bytes = SseBytes;
    type Mask = SseBytes;
    type Score = SseScore;

    fn is_available() -> bool {
        // SSE 4.1 covers _mm_minpos_epu16, _mm_cvtepi8_epi16, _mm_max_epu16,
        // _mm_alignr_epi8.
        raw_cpuid::CpuId::new()
            .get_feature_info()
            .is_some_and(|info| info.has_sse41())
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        unsafe { SseScore(_mm_cvtepi8_epi16(m.0)) }
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
            super::propagate_8_lane::<SseBackend>(
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

impl SseBytes {
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
}

impl BytesVec for SseBytes {
    type Mask = SseBytes;

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
            let ptr = data.add(start);
            Self(match remaining {
                0 => _mm_setzero_si128(),
                8.. => _mm_loadl_epi64(ptr as *const __m128i),
                1..=7 if Self::can_overread_8(ptr) => {
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

impl MaskVec for SseBytes {
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
            // Move prev's low 8 bytes to bytes 8..16 so prev[7] lands at byte 15.
            let shifted = _mm_slli_si128::<8>(prev.0);
            // alignr<15>(self, shifted) = (self || shifted)[15..31]
            //   = [shifted[15], self[0..15]]
            //   = [prev[7], self[0..15]]
            // Low 8 bytes: [prev[7], self[0..7]]. Upper 8 are don't-care.
            Self(_mm_alignr_epi8::<15>(self.0, shifted))
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

impl ScoreVec for SseScore {
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
        unsafe { Self(_mm_max_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe {
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
        unsafe { Self(_mm_add_epi16(self.0, other.0)) }
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
                1 => _mm_alignr_epi8::<14>(self.0, prev.0),
                2 => _mm_alignr_epi8::<12>(self.0, prev.0),
                3 => _mm_alignr_epi8::<10>(self.0, prev.0),
                4 => _mm_alignr_epi8::<8>(self.0, prev.0),
                5 => _mm_alignr_epi8::<6>(self.0, prev.0),
                6 => _mm_alignr_epi8::<4>(self.0, prev.0),
                7 => _mm_alignr_epi8::<2>(self.0, prev.0),
                8 => prev.0,
                _ => std::hint::unreachable_unchecked(),
            })
        }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            let cmp = _mm_cmpeq_epi16(self.0, _mm_set1_epi16(search as i16));
            let mask = _mm_movemask_epi8(cmp) as u32;
            (mask.trailing_zeros() as usize / 2).min(8)
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
pub struct SseU8Backend;

#[derive(Debug, Clone, Copy)]
pub struct SseU8Bytes(__m128i);

#[derive(Debug, Clone, Copy)]
pub struct SseU8Score(__m128i);

impl Backend for SseU8Backend {
    const LANES: usize = 16;
    const LANE_BYTES: usize = 1;
    type Bytes = SseU8Bytes;
    type Mask = SseU8Bytes;
    type Score = SseU8Score;

    fn is_available() -> bool {
        SseBackend::is_available()
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        SseU8Score(m.0)
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
            super::propagate_16_lane::<SseU8Backend>(
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

impl BytesVec for SseU8Bytes {
    type Mask = SseU8Bytes;

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

impl MaskVec for SseU8Bytes {
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

impl ScoreVec for SseU8Score {
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
            (_mm_extract_epi8::<0>(m) as u8) as u16
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm_add_epi8(self.0, other.0)) }
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
                1 => _mm_alignr_epi8::<15>(self.0, prev.0),
                2 => _mm_alignr_epi8::<14>(self.0, prev.0),
                3 => _mm_alignr_epi8::<13>(self.0, prev.0),
                4 => _mm_alignr_epi8::<12>(self.0, prev.0),
                5 => _mm_alignr_epi8::<11>(self.0, prev.0),
                6 => _mm_alignr_epi8::<10>(self.0, prev.0),
                7 => _mm_alignr_epi8::<9>(self.0, prev.0),
                8 => _mm_alignr_epi8::<8>(self.0, prev.0),
                9 => _mm_alignr_epi8::<7>(self.0, prev.0),
                10 => _mm_alignr_epi8::<6>(self.0, prev.0),
                11 => _mm_alignr_epi8::<5>(self.0, prev.0),
                12 => _mm_alignr_epi8::<4>(self.0, prev.0),
                13 => _mm_alignr_epi8::<3>(self.0, prev.0),
                14 => _mm_alignr_epi8::<2>(self.0, prev.0),
                15 => _mm_alignr_epi8::<1>(self.0, prev.0),
                16 => prev.0,
                _ => std::hint::unreachable_unchecked(),
            })
        }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            let cmp = _mm_cmpeq_epi8(self.0, _mm_set1_epi8(search as i8));
            let mask = _mm_movemask_epi8(cmp) as u32;
            (mask.trailing_zeros() as usize).min(16)
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
