use core::arch::wasm32::*;

use crate::prefilter::algo::can_overread;
use crate::smith_waterman::algo::{ascii_gap, unicode_gap};

use super::{Backend, BytesVec, MaskVec, ScoreVec};

/// Lane indices 0..16, used to build variable byte-shift swizzles
/// (out-of-range swizzle indices read as zero).
const IOTA: v128 = u8x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

/// Safe page-bounded read of 0..8 bytes into a little-endian u64
#[inline(always)]
unsafe fn load_u64_partial_safe(ptr: *const u8, len: usize) -> u64 {
    unsafe {
        debug_assert!(len < 8);
        match len {
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
        }
    }
}

/// 8-lane u16 scoring (128-bit v128), 8-lane u8 input (low half of v128).
#[derive(Debug, Clone, Copy)]
pub struct BackendWasm;

/// Physically occupies a full 128-bit register, but only the low 8 bytes are
/// used since we'll eventually widen the 8-bit per lane to 16-bit per lane
/// for the Score vector
#[derive(Debug, Clone, Copy)]
pub struct WasmBytes(v128);

#[derive(Debug, Clone, Copy)]
pub struct WasmScore(v128);

impl Backend for BackendWasm {
    const LANES: usize = 8;
    const LANE_BYTES: usize = 2;
    type Bytes = WasmBytes;
    type Mask = WasmBytes;
    type Score = WasmScore;

    fn is_available() -> bool {
        // simd128 must be enabled at compile-time, it cannot be runtime-detected
        true
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        // Sign-extend so 0xFF mask bytes become 0xFFFF score lanes.
        WasmScore(i16x8_extend_low_i8x16(m.0))
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
            ascii_gap::propagate_8_lane::<BackendWasm>(
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
            unicode_gap::propagate_unicode_8_lane::<BackendWasm>(
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

impl BytesVec for WasmBytes {
    type Mask = WasmBytes;

    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        Self(u8x16_splat(value))
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        Self(u8x16_eq(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        Self(u8x16_gt(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        Self(u8x16_lt(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            let remaining = len.saturating_sub(start);
            if remaining == 0 {
                return Self(u64x2_splat(0));
            }
            let ptr = data.add(start);
            Self(match remaining {
                8.. => v128_load64_zero(ptr as *const u64),
                1..=7 if can_overread(ptr, 8) => {
                    let lo = v128_load64_zero(ptr as *const u64);
                    let mask = u64x2((1u64 << (remaining * 8)) - 1, 0);
                    v128_and(lo, mask)
                }
                _ => u64x2(load_u64_partial_safe(ptr, remaining), 0),
            })
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 8);
        // place values in low 8 bytes, leave high 8 zeroed
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(values);
        Self(unsafe { v128_load(buf.as_ptr() as *const v128) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 16];
        unsafe { v128_store(buf.as_mut_ptr() as *mut v128, self.0) };
        buf[..8].to_vec()
    }
}

impl MaskVec for WasmBytes {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(u64x2_splat(0))
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(v128_and(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        Self(v128_or(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self(v128_not(self.0))
    }
    #[inline(always)]
    unsafe fn is_zero(self) -> bool {
        (u8x16_bitmask(self.0) & 0x00ff) == 0
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        // low 8 bytes = [prev[7], self[0..7]], ignore upper 8 bytes
        Self(u8x16_shuffle::<
            7,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        >(prev.0, self.0))
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 8);
        let mut buf = [0u8; 16];
        for i in 0..8 {
            buf[i] = if values[i] { 0xFF } else { 0 };
        }
        Self(unsafe { v128_load(buf.as_ptr() as *const v128) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        let mut buf = [0u8; 16];
        unsafe { v128_store(buf.as_mut_ptr() as *mut v128, self.0) };
        buf[..8].iter().map(|&v| v != 0).collect()
    }
}

impl ScoreVec for WasmScore {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(u64x2_splat(0))
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        Self(u16x8_splat(value))
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        Self(u16x8_replace_lane::<0>(u64x2_splat(0), value))
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        Self(u16x8_max(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        // pairwise maxes, halving the lane count each step (8 -> 4 -> 2 -> 1)
        let m = u16x8_max(self.0, u64x2_shuffle::<1, 1>(self.0, self.0));
        let m = u16x8_max(m, u32x4_shuffle::<1, 1, 1, 1>(m, m));
        let m = u16x8_max(m, u16x8_shuffle::<1, 1, 1, 1, 1, 1, 1, 1>(m, m));
        u16x8_extract_lane::<0>(m)
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        Self(u16x8_add(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        Self(u16x8_sub_sat(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(v128_and(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe {
            const { assert!(L >= 0 && L <= 8) };
            // Lane j of the result is (prev || self)[8 - L + j].
            Self(match L {
                0 => self.0,
                1 => u16x8_shuffle::<7, 8, 9, 10, 11, 12, 13, 14>(prev.0, self.0),
                2 => u16x8_shuffle::<6, 7, 8, 9, 10, 11, 12, 13>(prev.0, self.0),
                3 => u16x8_shuffle::<5, 6, 7, 8, 9, 10, 11, 12>(prev.0, self.0),
                4 => u16x8_shuffle::<4, 5, 6, 7, 8, 9, 10, 11>(prev.0, self.0),
                5 => u16x8_shuffle::<3, 4, 5, 6, 7, 8, 9, 10>(prev.0, self.0),
                6 => u16x8_shuffle::<2, 3, 4, 5, 6, 7, 8, 9>(prev.0, self.0),
                7 => u16x8_shuffle::<1, 2, 3, 4, 5, 6, 7, 8>(prev.0, self.0),
                8 => prev.0,
                _ => core::hint::unreachable_unchecked(),
            })
        }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        let cmp = u16x8_eq(self.0, u16x8_splat(search));
        let mask = i16x8_bitmask(cmp) as u32;
        (mask.trailing_zeros() as usize).min(8)
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 8);
        Self(unsafe { v128_load(values.as_ptr() as *const v128) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u16; 8];
        unsafe { v128_store(buf.as_mut_ptr() as *mut v128, self.0) };
        buf.to_vec()
    }
}

/// 16-lane u8 scoring (128-bit v128), 16-lane u8 input (128-bit v128)
#[derive(Debug, Clone, Copy)]
pub struct BackendWasmU8;

#[derive(Debug, Clone, Copy)]
pub struct WasmU8Bytes(v128);

#[derive(Debug, Clone, Copy)]
pub struct WasmU8Score(v128);

impl Backend for BackendWasmU8 {
    const LANES: usize = 16;
    const LANE_BYTES: usize = 1;
    type Bytes = WasmU8Bytes;
    type Mask = WasmU8Bytes;
    type Score = WasmU8Score;

    fn is_available() -> bool {
        BackendWasm::is_available()
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        WasmU8Score(m.0)
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
            ascii_gap::propagate_16_lane::<BackendWasmU8>(
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
            unicode_gap::propagate_unicode_16_lane::<BackendWasmU8>(
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

impl BytesVec for WasmU8Bytes {
    type Mask = WasmU8Bytes;

    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        Self(u8x16_splat(value))
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        Self(u8x16_eq(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        Self(u8x16_gt(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        Self(u8x16_lt(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            let remaining = len.saturating_sub(start);
            if remaining == 0 {
                return Self(u64x2_splat(0));
            }
            let ptr = data.add(start);
            Self(match remaining {
                0 => unreachable!(),
                8 => v128_load64_zero(ptr as *const u64),
                16.. => v128_load(ptr as *const v128),
                1..=7 if can_overread(ptr, 8) => {
                    let lo = v128_load64_zero(ptr as *const u64);
                    let mask = u64x2((1u64 << (remaining * 8)) - 1, 0);
                    v128_and(lo, mask)
                }
                1..=7 => u64x2(load_u64_partial_safe(ptr, remaining), 0),
                9..=15 => {
                    // low half loads bytes 0..8
                    // high half re-loads the last 8 bytes and shifts them into place
                    let lo = v128_load64_zero(ptr as *const u64);
                    let hi = v128_load64_zero(ptr.add(remaining - 8) as *const u64);
                    let shift = u8x16_splat((16 - remaining) as u8);
                    let hi = u8x16_swizzle(hi, u8x16_add(IOTA, shift));
                    u64x2_shuffle::<0, 2>(lo, hi)
                }
            })
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 16);
        Self(unsafe { v128_load(values.as_ptr() as *const v128) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 16];
        unsafe { v128_store(buf.as_mut_ptr() as *mut v128, self.0) };
        buf.to_vec()
    }
}

impl MaskVec for WasmU8Bytes {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(u64x2_splat(0))
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(v128_and(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        Self(v128_or(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self(v128_not(self.0))
    }
    #[inline(always)]
    unsafe fn is_zero(self) -> bool {
        !v128_any_true(self.0)
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        // [prev[15], self[0..15]]
        Self(u8x16_shuffle::<
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        >(prev.0, self.0))
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 16);
        let mut buf = [0u8; 16];
        for i in 0..16 {
            buf[i] = if values[i] { 0xFF } else { 0 };
        }
        Self(unsafe { v128_load(buf.as_ptr() as *const v128) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        let mut buf = [0u8; 16];
        unsafe { v128_store(buf.as_mut_ptr() as *mut v128, self.0) };
        buf.iter().map(|&v| v != 0).collect()
    }
}

impl ScoreVec for WasmU8Score {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(u64x2_splat(0))
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        Self(u8x16_splat(value as u8))
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        Self(u8x16_replace_lane::<0>(
            u64x2_splat(0),
            (value & 0xFF) as u8,
        ))
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        Self(u8x16_max(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        // pairwise maxes, halving the lane count each step (16 -> 8 -> 4 -> 2 -> 1)
        let m = u8x16_max(self.0, u64x2_shuffle::<1, 1>(self.0, self.0));
        let m = u8x16_max(m, u32x4_shuffle::<1, 1, 1, 1>(m, m));
        let m = u8x16_max(m, u16x8_shuffle::<1, 1, 1, 1, 1, 1, 1, 1>(m, m));
        let m = u8x16_max(
            m,
            u8x16_shuffle::<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(m, m),
        );
        u8x16_extract_lane::<0>(m) as u16
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        Self(u8x16_add(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        Self(u8x16_sub_sat(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(v128_and(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe {
            const { assert!(L >= 0 && L <= 16) };
            // Lane j of the result is (prev || self)[16 - L + j].
            Self(match L {
                0 => self.0,
                1 => {
                    u8x16_shuffle::<15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>(
                        prev.0, self.0,
                    )
                }
                2 => {
                    u8x16_shuffle::<14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29>(
                        prev.0, self.0,
                    )
                }
                3 => {
                    u8x16_shuffle::<13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28>(
                        prev.0, self.0,
                    )
                }
                4 => {
                    u8x16_shuffle::<12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27>(
                        prev.0, self.0,
                    )
                }
                5 => {
                    u8x16_shuffle::<11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26>(
                        prev.0, self.0,
                    )
                }
                6 => {
                    u8x16_shuffle::<10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25>(
                        prev.0, self.0,
                    )
                }
                7 => {
                    u8x16_shuffle::<9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24>(
                        prev.0, self.0,
                    )
                }
                8 => u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23>(
                    prev.0, self.0,
                ),
                9 => u8x16_shuffle::<7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22>(
                    prev.0, self.0,
                ),
                10 => u8x16_shuffle::<6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21>(
                    prev.0, self.0,
                ),
                11 => u8x16_shuffle::<5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20>(
                    prev.0, self.0,
                ),
                12 => u8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19>(
                    prev.0, self.0,
                ),
                13 => u8x16_shuffle::<3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18>(
                    prev.0, self.0,
                ),
                14 => u8x16_shuffle::<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17>(
                    prev.0, self.0,
                ),
                15 => u8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(
                    prev.0, self.0,
                ),
                16 => prev.0,
                _ => core::hint::unreachable_unchecked(),
            })
        }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        let cmp = u8x16_eq(self.0, u8x16_splat(search as u8));
        let mask = u8x16_bitmask(cmp) as u32;
        (mask.trailing_zeros() as usize).min(16)
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 16);
        let mut buf = [0u8; 16];
        for i in 0..16 {
            buf[i] = values[i] as u8;
        }
        Self(unsafe { v128_load(buf.as_ptr() as *const v128) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u8; 16];
        unsafe { v128_store(buf.as_mut_ptr() as *mut v128, self.0) };
        buf.iter().map(|&v| v as u16).collect()
    }
}
