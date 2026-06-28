//! Scalar backend used when no SIMD instruction set is available, and for
//! cross-backend property tests.

use crate::smith_waterman::algo::{ascii_gap, unicode_gap};

use super::{Backend, BytesVec, MaskVec, ScoreVec};

// ---------------------------------------------------------------------------
// 8-lane scalar backend
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Scalar8Bytes([u8; 8]);

#[derive(Debug, Clone, Copy)]
pub struct Scalar8Score([u16; 8]);

#[derive(Debug, Clone, Copy)]
pub struct BackendScalar8;

impl Backend for BackendScalar8 {
    const LANES: usize = 8;
    const LANE_BYTES: usize = 2;
    type Bytes = Scalar8Bytes;
    type Mask = Scalar8Bytes;
    type Score = Scalar8Score;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        let mut out = [0u16; 8];
        for i in 0..8 {
            out[i] = (m.0[i] as i8 as i16) as u16;
        }
        Scalar8Score(out)
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
            ascii_gap::propagate_8_lane::<BackendScalar8>(
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
            unicode_gap::propagate_unicode_8_lane::<BackendScalar8>(
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

impl BytesVec for Scalar8Bytes {
    type Mask = Scalar8Bytes;

    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        Self([value; 8])
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        let mut o = [0u8; 8];
        for i in 0..8 {
            o[i] = if self.0[i] == other.0[i] { 0xFF } else { 0 };
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        let mut o = [0u8; 8];
        for i in 0..8 {
            o[i] = if self.0[i] > other.0[i] { 0xFF } else { 0 };
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        let mut o = [0u8; 8];
        for i in 0..8 {
            o[i] = if self.0[i] < other.0[i] { 0xFF } else { 0 };
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        let mut o = [0u8; 8];
        let available = len.saturating_sub(start);
        let take = available.min(8);
        for i in 0..take {
            o[i] = unsafe { *data.add(start + i) };
        }
        Self(o)
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 8);
        let mut o = [0u8; 8];
        o.copy_from_slice(values);
        Self(o)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        self.0.to_vec()
    }
}

impl MaskVec for Scalar8Bytes {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0; 8])
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        let mut o = [0u8; 8];
        for i in 0..8 {
            o[i] = self.0[i] & other.0[i];
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        let mut o = [0u8; 8];
        for i in 0..8 {
            o[i] = self.0[i] | other.0[i];
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        let mut o = [0u8; 8];
        for i in 0..8 {
            o[i] = !self.0[i];
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn is_zero(self) -> bool {
        self.0.iter().all(|&v| v == 0)
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        let mut o = [0u8; 8];
        o[0] = prev.0[7];
        o[1..8].copy_from_slice(&self.0[0..7]);
        Self(o)
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 8);
        let mut o = [0u8; 8];
        for i in 0..8 {
            o[i] = if values[i] { 0xFF } else { 0 };
        }
        Self(o)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        self.0.iter().map(|&v| v != 0).collect()
    }
}

impl ScoreVec for Scalar8Score {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0; 8])
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        Self([value; 8])
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        let mut o = [0u16; 8];
        o[0] = value;
        Self(o)
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        let mut o = [0u16; 8];
        for i in 0..8 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        *self.0.iter().max().unwrap()
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        let mut o = [0u16; 8];
        for i in 0..8 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        let mut o = [0u16; 8];
        for i in 0..8 {
            o[i] = self.0[i].saturating_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        let mut o = [0u16; 8];
        for i in 0..8 {
            o[i] = self.0[i] & other.0[i];
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        const { assert!(L >= 0 && L <= 8) };
        let n = L as usize;
        let mut o = [0u16; 8];
        for i in 0..n {
            o[i] = prev.0[8 - n + i];
        }
        for i in n..8 {
            o[i] = self.0[i - n];
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        for i in 0..8 {
            if self.0[i] == search {
                return i;
            }
        }
        8
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 8);
        let mut o = [0u16; 8];
        o.copy_from_slice(values);
        Self(o)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        self.0.to_vec()
    }
}

// ---------------------------------------------------------------------------
// 16-lane u8-scoring scalar backend
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Scalar16U8Bytes([u8; 16]);

#[derive(Debug, Clone, Copy)]
pub struct Scalar16U8Score([u8; 16]);

#[derive(Debug, Clone, Copy)]
pub struct BackendScalar16U8;

impl Backend for BackendScalar16U8 {
    const LANES: usize = 16;
    const LANE_BYTES: usize = 1;
    type Bytes = Scalar16U8Bytes;
    type Mask = Scalar16U8Bytes;
    type Score = Scalar16U8Score;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        Scalar16U8Score(m.0)
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
            ascii_gap::propagate_16_lane::<BackendScalar16U8>(
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
            unicode_gap::propagate_unicode_16_lane::<BackendScalar16U8>(
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

impl BytesVec for Scalar16U8Bytes {
    type Mask = Scalar16U8Bytes;

    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        Self([value; 16])
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = if self.0[i] == other.0[i] { 0xFF } else { 0 };
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = if self.0[i] > other.0[i] { 0xFF } else { 0 };
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = if self.0[i] < other.0[i] { 0xFF } else { 0 };
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        let mut o = [0u8; 16];
        let available = len.saturating_sub(start);
        let take = available.min(16);
        for i in 0..take {
            o[i] = unsafe { *data.add(start + i) };
        }
        Self(o)
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 16);
        let mut o = [0u8; 16];
        o.copy_from_slice(values);
        Self(o)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        self.0.to_vec()
    }
}

impl MaskVec for Scalar16U8Bytes {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0; 16])
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = self.0[i] & other.0[i];
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = self.0[i] | other.0[i];
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = !self.0[i];
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn is_zero(self) -> bool {
        self.0.iter().all(|&v| v == 0)
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        let mut o = [0u8; 16];
        o[0] = prev.0[15];
        for i in 1..16 {
            o[i] = self.0[i - 1];
        }
        Self(o)
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 16);
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = if values[i] { 0xFF } else { 0 };
        }
        Self(o)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        self.0.iter().map(|&v| v != 0).collect()
    }
}

impl ScoreVec for Scalar16U8Score {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0; 16])
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        Self([value as u8; 16])
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        let mut o = [0u8; 16];
        o[0] = value as u8;
        Self(o)
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = self.0[i].max(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        *self.0.iter().max().unwrap() as u16
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = self.0[i].wrapping_add(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = self.0[i].saturating_sub(other.0[i]);
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = self.0[i] & other.0[i];
        }
        Self(o)
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        const { assert!(L >= 0 && L <= 16) };
        let n = L as usize;
        let mut o = [0u8; 16];
        for i in 0..n {
            o[i] = prev.0[16 - n + i];
        }
        o[n..16].copy_from_slice(&self.0[..(16 - n)]);
        Self(o)
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        let target = search as u8;
        for i in 0..16 {
            if self.0[i] == target {
                return i;
            }
        }
        16
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 16);
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = values[i] as u8;
        }
        Self(o)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        self.0.iter().map(|&v| v as u16).collect()
    }
}
