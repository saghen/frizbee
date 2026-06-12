use std::arch::aarch64::*;

use super::{Backend, BytesVec, ScoreVec};

/// 8-lane scoring (uint16x8_t, 128-bit), 8-byte bytes (uint8x8_t, 64-bit).
#[derive(Debug, Clone, Copy)]
pub struct NeonBackend;

#[derive(Debug, Clone, Copy)]
pub struct NeonBytes(uint8x8_t);

#[derive(Debug, Clone, Copy)]
pub struct NeonScore(uint16x8_t);

impl Backend for NeonBackend {
    const LANES: usize = 8;
    const LANE_BYTES: usize = 2;
    type Bytes = NeonBytes;
    type Score = NeonScore;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn widen(b: Self::Bytes) -> Self::Score {
        unsafe { NeonScore(vreinterpretq_u16_s16(vmovl_s8(vreinterpret_s8_u8(b.0)))) }
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
            super::propagate_8_lane::<NeonBackend>(
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

impl NeonBytes {
    /// Safe page-bounded read of 0..8 bytes.
    #[inline(always)]
    unsafe fn load_partial_safe(ptr: *const u8, len: usize) -> uint8x8_t {
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
            vreinterpret_u8_u64(vdup_n_u64(val))
        }
    }

    #[inline(always)]
    fn can_overread_8(ptr: *const u8) -> bool {
        (ptr as usize & 0xFFF) <= (4096 - 8)
    }
}

impl BytesVec for NeonBytes {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(vdup_n_u8(0)) }
    }
    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        unsafe { Self(vdup_n_u8(value)) }
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self {
        unsafe { Self(vceq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self {
        unsafe { Self(vcgt_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self {
        unsafe { Self(vclt_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(vand_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        unsafe { Self(vorr_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self(vmvn_u8(self.0)) }
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            let remaining = len.saturating_sub(start);
            let ptr = data.add(start);
            Self(match remaining {
                0 => vdup_n_u8(0),
                8.. => vld1_u8(ptr),
                1..=7 if Self::can_overread_8(ptr) => {
                    let lo = vld1_u8(ptr);
                    // Build a per-byte mask: bytes 0..remaining = 0xFF, rest = 0x00.
                    let mask_bytes: [u8; 8] = [
                        if 0 < remaining { 0xFF } else { 0 },
                        if 1 < remaining { 0xFF } else { 0 },
                        if 2 < remaining { 0xFF } else { 0 },
                        if 3 < remaining { 0xFF } else { 0 },
                        if 4 < remaining { 0xFF } else { 0 },
                        if 5 < remaining { 0xFF } else { 0 },
                        if 6 < remaining { 0xFF } else { 0 },
                        if 7 < remaining { 0xFF } else { 0 },
                    ];
                    vand_u8(lo, vld1_u8(mask_bytes.as_ptr()))
                }
                _ => Self::load_partial_safe(ptr, remaining),
            })
        }
    }

    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        // vext_u8(prev, self, 7) takes bytes 7..15 of (prev || self)
        //   = [prev[7], self[0], self[1], ..., self[6]]
        // which is what shift-right-by-1-with-fill-from-prev produces.
        unsafe { Self(vext_u8::<7>(prev.0, self.0)) }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 8);
        Self(unsafe { vld1_u8(values.as_ptr()) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 8];
        unsafe { vst1_u8(buf.as_mut_ptr(), self.0) };
        buf.to_vec()
    }
}

impl ScoreVec for NeonScore {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(vdupq_n_u16(0)) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(vdupq_n_u16(value)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe { Self(vsetq_lane_u16::<0>(value, vdupq_n_u16(0))) }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe { Self(vmaxq_u16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe { vmaxvq_u16(self.0) }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(vaddq_u16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(vqsubq_u16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(vandq_u16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe {
            const { assert!(L >= 0 && L <= 8) };
            // vextq_u16(prev, self, N) takes 8 u16s starting at lane N of (prev || self).
            // For a right shift by L lanes filled from prev, we want lanes
            //   (8 - L) .. (8 - L + 8) of (prev || self), i.e. N = 8 - L.
            Self(match L {
                0 => self.0,
                1 => vextq_u16::<7>(prev.0, self.0),
                2 => vextq_u16::<6>(prev.0, self.0),
                3 => vextq_u16::<5>(prev.0, self.0),
                4 => vextq_u16::<4>(prev.0, self.0),
                5 => vextq_u16::<3>(prev.0, self.0),
                6 => vextq_u16::<2>(prev.0, self.0),
                7 => vextq_u16::<1>(prev.0, self.0),
                8 => prev.0,
                _ => std::hint::unreachable_unchecked(),
            })
        }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            // Compare lanes; each matching u16 becomes 0xFFFF, others 0.
            // Narrow to a byte-per-lane mask, then trailing_zeros locates the
            // first match.
            let cmp = vceqq_u16(self.0, vdupq_n_u16(search));
            let narrowed = vmovn_u16(cmp);
            let bits = vget_lane_u64::<0>(vreinterpret_u64_u8(narrowed));
            if bits == 0 {
                8
            } else {
                bits.trailing_zeros() as usize / 8
            }
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 8);
        Self(unsafe { vld1q_u16(values.as_ptr()) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u16; 8];
        unsafe { vst1q_u16(buf.as_mut_ptr(), self.0) };
        buf.to_vec()
    }
}

/// 16-lane scoring (uint8x16_t), 16-byte bytes (uint8x16_t).
#[derive(Debug, Clone, Copy)]
pub struct NeonU8Backend;

#[derive(Debug, Clone, Copy)]
pub struct NeonU8Bytes(uint8x16_t);

#[derive(Debug, Clone, Copy)]
pub struct NeonU8Score(uint8x16_t);

impl Backend for NeonU8Backend {
    const LANES: usize = 16;
    const LANE_BYTES: usize = 1;
    type Bytes = NeonU8Bytes;
    type Score = NeonU8Score;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn widen(b: Self::Bytes) -> Self::Score {
        NeonU8Score(b.0)
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
            super::propagate_16_lane::<NeonU8Backend>(
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

impl NeonU8Bytes {
    /// Safe page-bounded read of 0..16 bytes into a uint8x16_t.
    #[inline(always)]
    unsafe fn load_partial_safe(ptr: *const u8, len: usize) -> uint8x16_t {
        unsafe {
            debug_assert!(len < 16);
            // Build via two 8-byte halves; reuse NeonBytes::load_partial_safe
            // logic for chunks under 8.
            if len == 0 {
                return vdupq_n_u8(0);
            }
            if len <= 8 {
                let lo = NeonBytes::load_partial_safe(ptr, len);
                return vcombine_u8(lo, vdup_n_u8(0));
            }
            // 9..=15 — load 8 + remainder
            let lo = vld1_u8(ptr);
            let hi_len = len - 8;
            let hi = NeonBytes::load_partial_safe(ptr.add(8), hi_len);
            vcombine_u8(lo, hi)
        }
    }

    #[inline(always)]
    fn can_overread_16(ptr: *const u8) -> bool {
        (ptr as usize & 0xFFF) <= (4096 - 16)
    }
}

impl BytesVec for NeonU8Bytes {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(vdupq_n_u8(0)) }
    }
    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        unsafe { Self(vdupq_n_u8(value)) }
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self {
        unsafe { Self(vceqq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self {
        unsafe { Self(vcgtq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self {
        unsafe { Self(vcltq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(vandq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        unsafe { Self(vorrq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self(vmvnq_u8(self.0)) }
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            let remaining = len.saturating_sub(start);
            let ptr = data.add(start);
            Self(match remaining {
                0 => vdupq_n_u8(0),
                16.. => vld1q_u8(ptr),
                1..=15 if Self::can_overread_16(ptr) => {
                    let loaded = vld1q_u8(ptr);
                    // Mask off bytes >= remaining.
                    let mask_bytes: [u8; 16] = [
                        if 0 < remaining { 0xFF } else { 0 },
                        if 1 < remaining { 0xFF } else { 0 },
                        if 2 < remaining { 0xFF } else { 0 },
                        if 3 < remaining { 0xFF } else { 0 },
                        if 4 < remaining { 0xFF } else { 0 },
                        if 5 < remaining { 0xFF } else { 0 },
                        if 6 < remaining { 0xFF } else { 0 },
                        if 7 < remaining { 0xFF } else { 0 },
                        if 8 < remaining { 0xFF } else { 0 },
                        if 9 < remaining { 0xFF } else { 0 },
                        if 10 < remaining { 0xFF } else { 0 },
                        if 11 < remaining { 0xFF } else { 0 },
                        if 12 < remaining { 0xFF } else { 0 },
                        if 13 < remaining { 0xFF } else { 0 },
                        if 14 < remaining { 0xFF } else { 0 },
                        if 15 < remaining { 0xFF } else { 0 },
                    ];
                    vandq_u8(loaded, vld1q_u8(mask_bytes.as_ptr()))
                }
                _ => Self::load_partial_safe(ptr, remaining),
            })
        }
    }

    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        // vextq_u8(prev, self, 15) = (prev || self)[15..31]
        //   = [prev[15], self[0..15]]
        unsafe { Self(vextq_u8::<15>(prev.0, self.0)) }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 16);
        Self(unsafe { vld1q_u8(values.as_ptr()) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 16];
        unsafe { vst1q_u8(buf.as_mut_ptr(), self.0) };
        buf.to_vec()
    }
}

impl ScoreVec for NeonU8Score {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(vdupq_n_u8(0)) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(vdupq_n_u8(value as u8)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe { Self(vsetq_lane_u8::<0>(value as u8, vdupq_n_u8(0))) }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe { Self(vmaxq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe { vmaxvq_u8(self.0) as u16 }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(vaddq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(vqsubq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(vandq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe {
            const { assert!(L >= 0 && L <= 16) };
            // vextq_u8(prev, self, N) takes 16 bytes starting at byte N of
            // (prev || self). For a right-shift by L, we want N = 16 - L.
            Self(match L {
                0 => self.0,
                1 => vextq_u8::<15>(prev.0, self.0),
                2 => vextq_u8::<14>(prev.0, self.0),
                3 => vextq_u8::<13>(prev.0, self.0),
                4 => vextq_u8::<12>(prev.0, self.0),
                5 => vextq_u8::<11>(prev.0, self.0),
                6 => vextq_u8::<10>(prev.0, self.0),
                7 => vextq_u8::<9>(prev.0, self.0),
                8 => vextq_u8::<8>(prev.0, self.0),
                9 => vextq_u8::<7>(prev.0, self.0),
                10 => vextq_u8::<6>(prev.0, self.0),
                11 => vextq_u8::<5>(prev.0, self.0),
                12 => vextq_u8::<4>(prev.0, self.0),
                13 => vextq_u8::<3>(prev.0, self.0),
                14 => vextq_u8::<2>(prev.0, self.0),
                15 => vextq_u8::<1>(prev.0, self.0),
                16 => prev.0,
                _ => std::hint::unreachable_unchecked(),
            })
        }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            let cmp = vceqq_u8(self.0, vdupq_n_u8(search as u8));
            // Narrow each 8-bit lane to a 4-bit nibble so the full mask fits
            // in a u64, then trailing_zeros locates the first match.
            let narrowed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(cmp));
            let bits = vget_lane_u64::<0>(vreinterpret_u64_u8(narrowed));
            if bits == 0 {
                16
            } else {
                bits.trailing_zeros() as usize / 4
            }
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 16);
        let mut buf = [0u8; 16];
        for i in 0..16 {
            buf[i] = values[i] as u8;
        }
        Self(unsafe { vld1q_u8(buf.as_ptr()) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u8; 16];
        unsafe { vst1q_u8(buf.as_mut_ptr(), self.0) };
        buf.iter().map(|&v| v as u16).collect()
    }
}
