//! SIMD backend abstraction for the Smith-Waterman kernel.
//!
//! Each backend exposes its native vector width via [`Backend::LANES`] and
//! three associated vector types: [`BytesVec`] for haystack bytes (LANES bytes
//! wide), [`MaskVec`] for boolean comparison results, and [`ScoreVec`] for the
//! score matrix (LANES × u16 or LANES × u8). Keeping masks as a distinct type
//! lets AVX-512 carry them as native `__mmask*` bitmasks rather than
//! 64-byte vectors, while other backends can alias [`MaskVec`] to their
//! [`BytesVec`] type with no loss.
//!
//! The kernel itself is generic over [`Backend`] and the lane count threads
//! through chunk arithmetic, the matrix stride, the alignment-path iterator,
//! and the horizontal-gap propagation unroll.
//!
//! Availabled backends:
//!   - AVX-512: LANES = 32/64 (scoring u16 x 32 = 512-bit or u8 x 64 = 512-bit)
//!   - AVX2:    LANES = 16/32 (scoring u16 x 16 = 256-bit or u8 x 32 = 256-bit)
//!   - SSE:     LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - NEON:    LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - Scalar:  LANES = 16/32 (fallback for non-SIMD systems)

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod sse;

#[cfg(target_arch = "x86_64")]
pub use avx::{AvxBackend, AvxU8Backend};
#[cfg(target_arch = "x86_64")]
pub use avx512::{Avx512Backend, Avx512U8Backend};
#[cfg(target_arch = "aarch64")]
pub use neon::{NeonBackend, NeonU8Backend};
pub use scalar::{Scalar8Backend, Scalar16U8Backend};
#[cfg(target_arch = "x86_64")]
pub use sse::{SseBackend, SseU8Backend};

/// A SIMD backend for Smith-Waterman matching that supports variable-width
/// lanes (8, 16, 32, 64) and score primitives (u8, u16).
///
/// `LANES` is the number of cells per chunk in the score matrix. The matching
/// [`BytesVec`] holds `LANES` meaningful bytes (one per lane); comparisons
/// produce a [`MaskVec`] which can be widened element-wise to the [`ScoreVec`]
/// via [`Backend::widen_mask`].
pub trait Backend: Sized + core::fmt::Debug + Clone + 'static {
    const LANES: usize;
    /// Size of a single score lane in bytes. 1 for u8-element backends,
    /// 2 for u16-element backends. Used by the alignment-path iterator to
    /// read individual cells from the matrix's raw byte view.
    const LANE_BYTES: usize;
    type Bytes: BytesVec<Mask = Self::Mask>;
    type Mask: MaskVec;
    type Score: ScoreVec;

    /// Whether this backend's required CPU features are available at runtime
    fn is_available() -> bool;

    /// Widen a comparison [`MaskVec`] into a [`ScoreVec`] with each lane set
    /// to `0xFF` (u8 score) or `0xFFFF` (u16 score) where the mask is true and
    /// zero elsewhere.
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score;

    /// Propagate horizontal (left-direction) gaps across a score row.
    ///
    /// The number of unrolled stages is fixed per backend: 8-lane backends do
    /// 3 stages (shifts of 1, 2, 4 lanes), 16-lane backends do 4 stages (1, 2,
    /// 4, 8), 32-lane backends do 5 stages, and 64-lane backends do 6 stages.
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    unsafe fn propagate_horizontal_gaps(
        row: Self::Score,
        adjacent_row: Self::Score,
        match_mask: Self::Score,
        adjacent_match_mask: Self::Score,
        gap_open_penalty: Self::Score,
        gap_extend_penalty: Self::Score,
    ) -> Self::Score;
}

/// A byte-wide vector with `LANES` meaningful bytes
pub trait BytesVec: Copy + core::fmt::Debug {
    /// The backend's mask type, produced by comparison operations.
    type Mask: MaskVec;

    /// Zero in every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn zero() -> Self;

    /// Broadcast `value` into every byte lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn splat(value: u8) -> Self;

    /// Per-lane equality, producing a true bit in each lane where equal.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn eq(self, other: Self) -> Self::Mask;

    /// Per-lane unsigned greater-than, producing a true bit where
    /// `self > other`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn gt(self, other: Self) -> Self::Mask;

    /// Per-lane unsigned less-than, producing a true bit where `self < other`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn lt(self, other: Self) -> Self::Mask;

    /// Load up to `LANES` bytes from `data + start`. `len` is the total length
    /// of the buffer pointed to by `data`.
    ///
    /// # Safety
    /// Caller must guarantee `data` points to at least `len` bytes and the
    /// backend's target features are enabled.
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self;

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8>;
}

/// A boolean mask over `LANES` lanes, produced by [`BytesVec`] comparisons.
///
/// Each lane holds a logical true/false. Non-AVX-512 backends typically alias
/// this to their [`BytesVec`] type and represent the mask as 0xFF/0x00 per
/// byte. AVX-512 uses a native `__mmask64` bitmask.
pub trait MaskVec: Copy + core::fmt::Debug {
    /// False in every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn zero() -> Self;

    /// Per-lane logical AND.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn and(self, other: Self) -> Self;

    /// Per-lane logical OR.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn or(self, other: Self) -> Self;

    /// Per-lane logical NOT.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn not(self) -> Self;

    /// Shift right by 1 lane, filling lane 0 with the highest meaningful lane
    /// of `prev`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self;

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool>;
}

/// A score-wide vector holding `LANES` u16 elements.
pub trait ScoreVec: Copy + core::fmt::Debug {
    /// Zero in every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn zero() -> Self;

    /// Broadcast `value` into every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn splat(value: u16) -> Self;

    /// `value` in lane 0, zero in all other lanes
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn first_lane(value: u16) -> Self;

    /// Per-lane unsigned max
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn max(self, other: Self) -> Self;

    /// Horizontal unsigned max across all lanes
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn horizontal_max(self) -> u16;

    /// Per-lane wrapping add
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn add(self, other: Self) -> Self;

    /// Per-lane saturating subtract (saturating at zero)
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn subs(self, other: Self) -> Self;

    /// Bitwise AND
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn and(self, other: Self) -> Self;

    /// Shift right by `L` lanes, filling the low `L` lanes with the high `L`
    /// lanes of `prev`. `L` must be in `0..=LANES`. `L == LANES` returns
    /// `prev` unmodified.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self;

    /// Index of the lowest lane equal to `search`, or `LANES` if absent
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn find_lane(self, search: u16) -> usize;

    // ----- Test-only helpers ---------------------------------------------

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16>;
}

// ---------------------------------------------------------------------------
// Gap propagation helpers shared by all backends.
//
// Each backend's Backend::propagate_horizontal_gaps calls one of these
// generic-over-B functions, selected by lane count. The macro hides the
// fully-qualified `<<B as Backend>::Score as ScoreVec>::*` boilerplate.
// ---------------------------------------------------------------------------

macro_rules! gap_step {
    ($B:ty, $shift:literal, $row:ident, $adj:ident, $mm:ident, $amm:ident, $gop:ident, $gex:ident) => {
        let shifted_row =
            <<$B as Backend>::Score as ScoreVec>::shift_right_padded::<$shift>($row, $adj);
        let shifted_match_mask =
            <<$B as Backend>::Score as ScoreVec>::shift_right_padded::<$shift>($mm, $amm);
        let gap_penalty = <<$B as Backend>::Score as ScoreVec>::add(
            $gex,
            <<$B as Backend>::Score as ScoreVec>::and($gop, shifted_match_mask),
        );
        let decayed = <<$B as Backend>::Score as ScoreVec>::subs(shifted_row, gap_penalty);
        let $row = <<$B as Backend>::Score as ScoreVec>::max($row, decayed);
        let $gex = <<$B as Backend>::Score as ScoreVec>::add($gex, $gex);
    };
}

#[inline(always)]
pub(crate) unsafe fn propagate_8_lane<B: Backend>(
    row: B::Score,
    adj: B::Score,
    mm: B::Score,
    amm: B::Score,
    gop: B::Score,
    gex: B::Score,
) -> B::Score {
    unsafe {
        gap_step!(B, 1, row, adj, mm, amm, gop, gex);
        gap_step!(B, 2, row, adj, mm, amm, gop, gex);
        gap_step!(B, 4, row, adj, mm, amm, gop, gex);
        let _ = gex;
        row
    }
}

#[inline(always)]
pub(crate) unsafe fn propagate_16_lane<B: Backend>(
    row: B::Score,
    adj: B::Score,
    mm: B::Score,
    amm: B::Score,
    gop: B::Score,
    gex: B::Score,
) -> B::Score {
    unsafe {
        gap_step!(B, 1, row, adj, mm, amm, gop, gex);
        gap_step!(B, 2, row, adj, mm, amm, gop, gex);
        gap_step!(B, 4, row, adj, mm, amm, gop, gex);
        gap_step!(B, 8, row, adj, mm, amm, gop, gex);
        let _ = gex;
        row
    }
}

#[inline(always)]
pub(crate) unsafe fn propagate_32_lane<B: Backend>(
    row: B::Score,
    adj: B::Score,
    mm: B::Score,
    amm: B::Score,
    gop: B::Score,
    gex: B::Score,
) -> B::Score {
    unsafe {
        gap_step!(B, 1, row, adj, mm, amm, gop, gex);
        gap_step!(B, 2, row, adj, mm, amm, gop, gex);
        gap_step!(B, 4, row, adj, mm, amm, gop, gex);
        gap_step!(B, 8, row, adj, mm, amm, gop, gex);
        gap_step!(B, 16, row, adj, mm, amm, gop, gex);
        let _ = gex;
        row
    }
}

#[inline(always)]
pub(crate) unsafe fn propagate_64_lane<B: Backend>(
    row: B::Score,
    adj: B::Score,
    mm: B::Score,
    amm: B::Score,
    gop: B::Score,
    gex: B::Score,
) -> B::Score {
    unsafe {
        gap_step!(B, 1, row, adj, mm, amm, gop, gex);
        gap_step!(B, 2, row, adj, mm, amm, gop, gex);
        gap_step!(B, 4, row, adj, mm, amm, gop, gex);
        gap_step!(B, 8, row, adj, mm, amm, gop, gex);
        gap_step!(B, 16, row, adj, mm, amm, gop, gex);
        gap_step!(B, 32, row, adj, mm, amm, gop, gex);
        let _ = gex;
        row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- BytesVec property tests ---------------------------------------

    fn check_bytes_zero<B: Backend>() {
        unsafe {
            let z = B::Bytes::zero();
            assert_eq!(BytesVec::to_lanes(z), vec![0u8; B::LANES]);
        }
    }

    fn check_bytes_splat<B: Backend>() {
        unsafe {
            let v = B::Bytes::splat(0x42);
            assert_eq!(BytesVec::to_lanes(v), vec![0x42; B::LANES]);
        }
    }

    fn check_bytes_eq<B: Backend>() {
        unsafe {
            let a_in = (0u8..(B::LANES as u8)).collect::<Vec<_>>();
            let b_in = a_in
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { v.wrapping_add(1) })
                .collect::<Vec<_>>();
            let a = B::Bytes::from_lanes(&a_in);
            let b = B::Bytes::from_lanes(&b_in);
            let result = MaskVec::to_lanes(a.eq(b));
            for (i, r) in result.iter().enumerate() {
                assert_eq!(*r, i % 2 == 0, "lane {i}");
            }
        }
    }

    fn check_bytes_gt_lt<B: Backend>() {
        unsafe {
            let a_in = vec![5u8; B::LANES];
            let mut b_in = vec![5u8; B::LANES];
            if B::LANES >= 2 {
                b_in[0] = 4;
                b_in[1] = 6;
            }
            let a = B::Bytes::from_lanes(&a_in);
            let b = B::Bytes::from_lanes(&b_in);
            let gt = MaskVec::to_lanes(a.gt(b));
            let lt = MaskVec::to_lanes(a.lt(b));
            if B::LANES >= 2 {
                assert!(gt[0]);
                assert!(!gt[1]);
                assert!(!lt[0]);
                assert!(lt[1]);
            }
        }
    }

    fn check_mask_and_or_not<B: Backend>() {
        unsafe {
            let pattern_a: Vec<bool> = (0..B::LANES).map(|i| i % 2 == 0).collect();
            let pattern_b: Vec<bool> = (0..B::LANES).map(|i| i % 3 == 0).collect();
            let a = B::Mask::from_lanes(&pattern_a);
            let b = B::Mask::from_lanes(&pattern_b);
            let and = a.and(b).to_lanes();
            let or = a.or(b).to_lanes();
            let not_a = a.not().to_lanes();
            for i in 0..B::LANES {
                assert_eq!(and[i], pattern_a[i] && pattern_b[i], "and lane {i}");
                assert_eq!(or[i], pattern_a[i] || pattern_b[i], "or lane {i}");
                assert_eq!(not_a[i], !pattern_a[i], "not lane {i}");
            }
        }
    }

    fn check_mask_zero<B: Backend>() {
        unsafe {
            let z = B::Mask::zero();
            assert_eq!(MaskVec::to_lanes(z), vec![false; B::LANES]);
        }
    }

    fn check_bytes_load_partial<B: Backend>() {
        unsafe {
            let data: Vec<u8> = (1..=64).collect();
            let lanes = B::LANES;
            for start in (0..32).step_by(lanes) {
                for len in (start + 1)..=32 {
                    let v = B::Bytes::load_partial(data.as_ptr(), start, len);
                    let got = BytesVec::to_lanes(v);
                    let mut expected = vec![0u8; lanes];
                    let take = (len - start).min(lanes);
                    expected[..take].copy_from_slice(&data[start..start + take]);
                    assert_eq!(got, expected, "start={start} len={len}");
                }
            }
        }
    }

    fn check_mask_shift_right_padded_1<B: Backend>() {
        unsafe {
            let a_in: Vec<bool> = (0..B::LANES).map(|i| i % 2 == 0).collect();
            let p_in: Vec<bool> = (0..B::LANES).map(|i| i % 3 == 0).collect();
            let a = B::Mask::from_lanes(&a_in);
            let p = B::Mask::from_lanes(&p_in);
            let got = MaskVec::to_lanes(a.shift_right_padded_1(p));

            let mut expected = vec![false; B::LANES];
            expected[0] = p_in[B::LANES - 1];
            expected[1..B::LANES].copy_from_slice(&a_in[0..(B::LANES - 1)]);
            assert_eq!(got, expected);
        }
    }

    // ----- ScoreVec property tests ---------------------------------------

    fn check_score_zero<B: Backend>() {
        unsafe {
            assert_eq!(B::Score::zero().to_lanes(), vec![0u16; B::LANES]);
        }
    }

    fn check_score_splat<B: Backend>() {
        // Use a value that fits in u8 so the same test applies to both
        // u8- and u16-element backends.
        unsafe {
            assert_eq!(B::Score::splat(0xAB).to_lanes(), vec![0xABu16; B::LANES]);
        }
    }

    fn check_score_first_lane<B: Backend>() {
        unsafe {
            let v = B::Score::first_lane(0xCD);
            let lanes = v.to_lanes();
            assert_eq!(lanes[0], 0xCD);
            for &l in &lanes[1..] {
                assert_eq!(l, 0);
            }
        }
    }

    fn check_score_max_add_subs<B: Backend>() {
        // Values stay in u8 range so the same inputs work for both widths.
        unsafe {
            let a_in: Vec<u16> = (0..B::LANES).map(|i| (i % 100) as u16).collect();
            let b_in: Vec<u16> = (0..B::LANES).map(|i| (B::LANES - i) as u16 % 50).collect();
            let a = B::Score::from_lanes(&a_in);
            let b = B::Score::from_lanes(&b_in);
            let max = a.max(b).to_lanes();
            for i in 0..B::LANES {
                assert_eq!(max[i], a_in[i].max(b_in[i]));
            }
            let added = a.add(b).to_lanes();
            for i in 0..B::LANES {
                assert_eq!(added[i], a_in[i].wrapping_add(b_in[i]));
            }
            let subbed = a.subs(b).to_lanes();
            for i in 0..B::LANES {
                assert_eq!(subbed[i], a_in[i].saturating_sub(b_in[i]));
            }
        }
    }

    fn check_score_horizontal_max<B: Backend>() {
        unsafe {
            let mut a_in = vec![10u16; B::LANES];
            a_in[B::LANES / 2] = 222;
            let a = B::Score::from_lanes(&a_in);
            assert_eq!(a.horizontal_max(), 222);

            let zero = B::Score::zero();
            assert_eq!(zero.horizontal_max(), 0);
        }
    }

    fn check_score_find_lane<B: Backend>() {
        unsafe {
            // For 32 lanes, step of 7 stays under 224 — fits in u8.
            let a_in: Vec<u16> = (0..B::LANES).map(|i| (i as u16) * 7).collect();
            let a = B::Score::from_lanes(&a_in);
            for (i, &v) in a_in.iter().enumerate() {
                assert_eq!(a.find_lane(v), i);
            }
            // Search for a value that can't appear in the data; works for
            // both u8 and u16 backends.
            assert_eq!(a.find_lane(251), B::LANES);
        }
    }

    fn check_score_shift_right_padded_each<B: Backend, const L: i32>() {
        unsafe {
            // Distinct values for `a` and `p` that all fit in u8.
            let a_in: Vec<u16> = (0..B::LANES).map(|i| (i as u16) + 1).collect();
            let p_in: Vec<u16> = (0..B::LANES).map(|i| (i as u16) + 100).collect();
            let a = B::Score::from_lanes(&a_in);
            let p = B::Score::from_lanes(&p_in);
            let got = a.shift_right_padded::<L>(p).to_lanes();

            let n = L as usize;
            let mut expected = vec![0u16; B::LANES];
            for i in 0..n {
                expected[i] = p_in[B::LANES - n + i];
            }
            expected[n..B::LANES].copy_from_slice(&a_in[0..(B::LANES - n)]);
            assert_eq!(got, expected, "L = {}", L);
        }
    }

    fn check_score_shift_right_padded_8<B: Backend>() {
        check_score_shift_right_padded_each::<B, 0>();
        check_score_shift_right_padded_each::<B, 1>();
        check_score_shift_right_padded_each::<B, 2>();
        check_score_shift_right_padded_each::<B, 3>();
        check_score_shift_right_padded_each::<B, 4>();
        if B::LANES >= 8 {
            check_score_shift_right_padded_each::<B, 5>();
            check_score_shift_right_padded_each::<B, 6>();
            check_score_shift_right_padded_each::<B, 7>();
            check_score_shift_right_padded_each::<B, 8>();
        }
        // L = 16 is used by the 32-lane u8 backends' propagate_horizontal_gaps.
        // Can't gate on B::LANES inside a const-generic instantiation because
        // dead branches still monomorphize. Use a separate top-level test
        // function for 32-lane coverage.
    }

    fn check_score_shift_right_padded_16<B: Backend>() {
        check_score_shift_right_padded_each::<B, 16>();
    }

    fn check_score_shift_right_padded_32<B: Backend>() {
        check_score_shift_right_padded_each::<B, 32>();
    }

    // ----- Dispatch -------------------------------------------------------

    macro_rules! backend_tests {
        ($mod_name:ident, $backend:ty) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn bytes_zero() {
                    check_bytes_zero::<$backend>();
                }
                #[test]
                fn bytes_splat() {
                    check_bytes_splat::<$backend>();
                }
                #[test]
                fn bytes_eq() {
                    check_bytes_eq::<$backend>();
                }
                #[test]
                fn bytes_gt_lt() {
                    check_bytes_gt_lt::<$backend>();
                }
                #[test]
                fn mask_and_or_not() {
                    check_mask_and_or_not::<$backend>();
                }
                #[test]
                fn mask_zero() {
                    check_mask_zero::<$backend>();
                }
                #[test]
                fn bytes_load_partial() {
                    check_bytes_load_partial::<$backend>();
                }
                #[test]
                fn mask_shift_right_padded_1() {
                    check_mask_shift_right_padded_1::<$backend>();
                }
                #[test]
                fn score_zero() {
                    check_score_zero::<$backend>();
                }
                #[test]
                fn score_splat() {
                    check_score_splat::<$backend>();
                }
                #[test]
                fn score_first_lane() {
                    check_score_first_lane::<$backend>();
                }
                #[test]
                fn score_max_add_subs() {
                    check_score_max_add_subs::<$backend>();
                }
                #[test]
                fn score_horizontal_max() {
                    check_score_horizontal_max::<$backend>();
                }
                #[test]
                fn score_find_lane() {
                    check_score_find_lane::<$backend>();
                }
                #[test]
                fn score_shift_right_padded() {
                    check_score_shift_right_padded_8::<$backend>();
                }
            }
        };
    }

    /// Extra coverage for 32-lane backends only (uses L = 16 const).
    macro_rules! backend_tests_32_lane {
        ($mod_name:ident, $backend:ty) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn score_shift_right_padded_16() {
                    check_score_shift_right_padded_16::<$backend>();
                }
            }
        };
    }

    /// Backend tests gated on `Backend::is_available()` at runtime. Used for
    /// backends whose required CPU features may not be present in CI
    /// (notably AVX-512). Tests no-op on machines without the features
    macro_rules! backend_tests_runtime_gated {
        ($mod_name:ident, $backend:ty) => {
            mod $mod_name {
                use super::*;

                #[inline]
                fn skip_if_unavailable() -> bool {
                    !<$backend>::is_available()
                }

                #[test]
                fn bytes_zero() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_bytes_zero::<$backend>();
                }
                #[test]
                fn bytes_splat() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_bytes_splat::<$backend>();
                }
                #[test]
                fn bytes_eq() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_bytes_eq::<$backend>();
                }
                #[test]
                fn bytes_gt_lt() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_bytes_gt_lt::<$backend>();
                }
                #[test]
                fn mask_and_or_not() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_mask_and_or_not::<$backend>();
                }
                #[test]
                fn mask_zero() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_mask_zero::<$backend>();
                }
                #[test]
                fn bytes_load_partial() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_bytes_load_partial::<$backend>();
                }
                #[test]
                fn mask_shift_right_padded_1() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_mask_shift_right_padded_1::<$backend>();
                }
                #[test]
                fn score_zero() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_zero::<$backend>();
                }
                #[test]
                fn score_splat() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_splat::<$backend>();
                }
                #[test]
                fn score_first_lane() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_first_lane::<$backend>();
                }
                #[test]
                fn score_max_add_subs() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_max_add_subs::<$backend>();
                }
                #[test]
                fn score_horizontal_max() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_horizontal_max::<$backend>();
                }
                #[test]
                fn score_find_lane() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_find_lane::<$backend>();
                }
                #[test]
                fn score_shift_right_padded() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_shift_right_padded_8::<$backend>();
                }
                #[test]
                fn score_shift_right_padded_16() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_shift_right_padded_16::<$backend>();
                }
                #[test]
                fn score_shift_right_padded_32() {
                    if skip_if_unavailable() {
                        return;
                    }
                    check_score_shift_right_padded_32::<$backend>();
                }
            }
        };
    }

    backend_tests!(scalar8, super::Scalar8Backend);
    backend_tests!(scalar16_u8, super::Scalar16U8Backend);
    #[cfg(target_arch = "x86_64")]
    backend_tests!(sse, super::SseBackend);
    #[cfg(target_arch = "x86_64")]
    backend_tests!(sse_u8, super::SseU8Backend);
    #[cfg(target_arch = "x86_64")]
    backend_tests!(avx, super::AvxBackend);
    #[cfg(target_arch = "x86_64")]
    backend_tests!(avx_u8, super::AvxU8Backend);
    #[cfg(target_arch = "x86_64")]
    backend_tests_32_lane!(avx_u8_extra, super::AvxU8Backend);
    #[cfg(target_arch = "x86_64")]
    backend_tests_runtime_gated!(avx512, super::Avx512Backend);
    #[cfg(target_arch = "x86_64")]
    backend_tests_runtime_gated!(avx512_u8, super::Avx512U8Backend);
    #[cfg(target_arch = "aarch64")]
    backend_tests!(neon, super::NeonBackend);
    #[cfg(target_arch = "aarch64")]
    backend_tests!(neon_u8, super::NeonU8Backend);
}
