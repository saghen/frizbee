//! Scalar backend used when no SIMD instruction set is available (non-x86, non-ARM)

use crate::smith_waterman::algo::{ascii_gap, unicode_gap};

use super::{Backend, BytesVec, MaskVec, ScoreVec};

/// Doubles as both the byte vector and the comparison mask (a mask lane is 0xFF
/// for true, 0x00 for false)
#[derive(Debug, Clone, Copy)]
pub struct ScalarBytes<const LANES: usize>([u8; LANES]);

#[derive(Debug, Clone, Copy)]
pub struct ScalarScoreU16<const LANES: usize>([u16; LANES]);

#[derive(Debug, Clone, Copy)]
pub struct ScalarScoreU8<const LANES: usize>([u8; LANES]);

impl<const LANES: usize> ScalarScoreU16<LANES> {
    /// Widen a comparison mask (0xFF/0x00 per byte) to full-width score lanes.
    #[inline(always)]
    fn widen(mask: ScalarBytes<LANES>) -> Self {
        let mut out = [0u16; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = if mask.0[idx] != 0 { u16::MAX } else { 0 };
        }
        Self(out)
    }
}

impl<const LANES: usize> ScalarScoreU8<LANES> {
    /// Widen a comparison mask (0xFF/0x00 per byte) to full-width score lanes.
    #[inline(always)]
    fn widen(mask: ScalarBytes<LANES>) -> Self {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = if mask.0[idx] != 0 { u8::MAX } else { 0 };
        }
        Self(out)
    }
}

impl<const LANES: usize> BytesVec for ScalarBytes<LANES> {
    type Mask = ScalarBytes<LANES>;

    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        Self([value; LANES])
    }

    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = if self.0[idx] == other.0[idx] { 0xFF } else { 0 };
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = if self.0[idx] > other.0[idx] { 0xFF } else { 0 };
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = if self.0[idx] < other.0[idx] { 0xFF } else { 0 };
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        let mut out = [0u8; LANES];
        let take = len.saturating_sub(start).min(LANES);
        for (idx, lane) in out.iter_mut().take(take).enumerate() {
            *lane = unsafe { *data.add(start + idx) };
        }
        Self(out)
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), LANES);
        let mut out = [0u8; LANES];
        out.copy_from_slice(values);
        Self(out)
    }

    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        self.0.to_vec()
    }
}

impl<const LANES: usize> MaskVec for ScalarBytes<LANES> {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0; LANES])
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx] & other.0[idx];
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx] | other.0[idx];
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn not(self) -> Self {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = !self.0[idx];
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn is_zero(self) -> bool {
        self.0.iter().all(|&v| v == 0)
    }

    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        let mut out = [0u8; LANES];
        out[0] = prev.0[LANES - 1];
        out[1..LANES].copy_from_slice(&self.0[..(LANES - 1)]);
        Self(out)
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), LANES);
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = if values[idx] { 0xFF } else { 0 };
        }
        Self(out)
    }

    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        self.0.iter().map(|&v| v != 0).collect()
    }
}

impl<const LANES: usize> ScoreVec for ScalarScoreU16<LANES> {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0; LANES])
    }

    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        Self([value; LANES])
    }

    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        let mut out = [0u16; LANES];
        out[0] = value;
        Self(out)
    }

    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        let mut out = [0u16; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx].max(other.0[idx]);
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        *self.0.iter().max().unwrap()
    }

    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        let mut out = [0u16; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx].wrapping_add(other.0[idx]);
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        let mut out = [0u16; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx].saturating_sub(other.0[idx]);
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        let mut out = [0u16; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx] & other.0[idx];
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        const { assert!(L >= 0 && (L as usize) <= LANES) };
        let n = L as usize;
        let mut out = [0u16; LANES];
        for idx in 0..n {
            out[idx] = prev.0[LANES - n + idx];
        }
        out[n..LANES].copy_from_slice(&self.0[..(LANES - n)]);
        Self(out)
    }

    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        for (idx, &lane) in self.0.iter().enumerate() {
            if lane == search {
                return idx;
            }
        }
        LANES
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), LANES);
        let mut out = [0u16; LANES];
        out.copy_from_slice(values);
        Self(out)
    }

    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        self.0.to_vec()
    }
}

impl<const LANES: usize> ScoreVec for ScalarScoreU8<LANES> {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0; LANES])
    }

    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        Self([value as u8; LANES])
    }

    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        let mut out = [0u8; LANES];
        out[0] = value as u8;
        Self(out)
    }

    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx].max(other.0[idx]);
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        *self.0.iter().max().unwrap() as u16
    }

    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx].wrapping_add(other.0[idx]);
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx].saturating_sub(other.0[idx]);
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = self.0[idx] & other.0[idx];
        }
        Self(out)
    }

    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        const { assert!(L >= 0 && (L as usize) <= LANES) };
        let n = L as usize;
        let mut out = [0u8; LANES];
        for idx in 0..n {
            out[idx] = prev.0[LANES - n + idx];
        }
        out[n..LANES].copy_from_slice(&self.0[..(LANES - n)]);
        Self(out)
    }

    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        let search = search as u8;
        for (idx, &lane) in self.0.iter().enumerate() {
            if lane == search {
                return idx;
            }
        }
        LANES
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), LANES);
        let mut out = [0u8; LANES];
        for (idx, lane) in out.iter_mut().enumerate() {
            *lane = values[idx] as u8;
        }
        Self(out)
    }

    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        self.0.iter().map(|&v| v as u16).collect()
    }
}

// Each scalar backend differs only in lane count, score-element width, and
// which lane-count-specialized gap-propagation helper it needs. We use a
// macro to avoid writing these all out by hand.
macro_rules! scalar_backend {
    (
        $backend:ident,
        $lanes:literal,
        $lane_bytes:literal,
        $score:ty,
        $propagate:ident,
        $propagate_unicode:ident
    ) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $backend;

        impl Backend for $backend {
            const LANES: usize = $lanes;
            const LANE_BYTES: usize = $lane_bytes;
            type Bytes = ScalarBytes<$lanes>;
            type Mask = ScalarBytes<$lanes>;
            type Score = $score;

            fn is_available() -> bool {
                true
            }

            #[inline(always)]
            unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
                <$score>::widen(m)
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
                    ascii_gap::$propagate::<Self>(
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
                    unicode_gap::$propagate_unicode::<Self>(
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
    };
}

scalar_backend!(
    BackendScalar8,
    8,
    2,
    ScalarScoreU16<8>,
    propagate_8_lane,
    propagate_unicode_8_lane
);
scalar_backend!(
    BackendScalar16U8,
    16,
    1,
    ScalarScoreU8<16>,
    propagate_16_lane,
    propagate_unicode_16_lane
);

// Test only backends used to assert correctness in the SIMD backends
#[cfg(target_arch = "x86_64")]
#[cfg(test)]
scalar_backend!(
    TestScalar16,
    16,
    2,
    ScalarScoreU16<16>,
    propagate_16_lane,
    propagate_unicode_16_lane
);
#[cfg(target_arch = "x86_64")]
#[cfg(test)]
scalar_backend!(
    TestScalar32,
    32,
    2,
    ScalarScoreU16<32>,
    propagate_32_lane,
    propagate_unicode_32_lane
);
#[cfg(target_arch = "x86_64")]
#[cfg(test)]
scalar_backend!(
    TestScalar32U8,
    32,
    1,
    ScalarScoreU8<32>,
    propagate_32_lane,
    propagate_unicode_32_lane
);
#[cfg(target_arch = "x86_64")]
#[cfg(test)]
scalar_backend!(
    TestScalar64U8,
    64,
    1,
    ScalarScoreU8<64>,
    propagate_64_lane,
    propagate_unicode_64_lane
);
