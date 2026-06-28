//! With the default gap propagation, `hw` matched against `h😀w` will receive a penalty
//! to the score of `gap_open_penality + gap_extend_penalty * 4` due to the emoji taking
//! up 4 bytes. If the haystack was instead `hew`, the penalty would be
//! `gap_open_penality + gap_extend_penalty`. As a result, we need to be aware of the unicode
//! boundaries and only apply the gap penalty when reaching the end of a multi-byte
//! codepoint.
//!
//! Each row becomes a single needle char (multi-byte utf-8 codepoint)
//! Haystack stays as bytes
//!
//! C = lead or continuation byte
//! E = end of utf-8 codepoint
//!
//! ```text
//!       f  C  C  C  E   C   C   E
//!    0  0  0  0  0  0   0   0   0
//! \u 0  0  0  0  0  16  11  10  9
//! \u 0  0  0  0  0  11  10  9   25
//! ```
//!
//! We can solve the issue by keeping track of the continuation bytes as we apply the
//! gap propagation.
//!
//! ```text
//! input:
//! E   C   C   E   E   C   E   <- codepoint types
//! 16  0   0   0   0   0   0   <- score = match scores
//! Y   N   N   N   N   N   N   <- pending gap-open mask = match mask
//! N   Y   Y   N   N   Y   N   <- continuation byte mask (C)
//! 0   1   1   0   0   1   0   <- continuation (C) count
//! Y   N   N   Y   Y   N   Y   <- end of utf-8 codepoint (E) mask
//!
//! ---
//! shift 1
//!
//! 0   16  0   0   0   0   0   <- score shifted by 1 lane
//! 0   1   1   0   0   1   0   <- total C count = C count
//!
//! N   Y   N   N   N   N   N   <- pending gap-open mask shifted by 1 lane
//! Y   N   N   Y   Y   N   Y   <- E crossed = E mask
//! N   N   N   N   N   N   N   <- pending gap-open crossed E = shifted pending gap-open mask & E mask
//!
//! 1   0   0   1   1   0   1   <- gap extend penalty = 1 - total C count
//! 0   0   0   0   0   0   0   <- gap open extra
//! 0   16  0   0   0   0   0   <- shifted score - penalties
//!
//! state:
//! Y   N   N   Y   Y   N   Y   <- E crossed
//! 0   1   1   0   0   1   0   <- total C count
//! Y   Y   N   N   N   N   N   <- pending gap-open mask |= shifted pending gap-open mask & !E mask
//! 16  16  0   0   0   0   0   <- score = max(score, shifted score - penalties)
//!
//! ---
//! shift 2
//!
//! 0   0   16  16  0   0   0   <- score shifted by 2 lanes
//!
//! 0   0   1   1   0   0   1   <- C count shifted by 1 lane
//! 0   1   2   1   0   1   1   <- total C count += shifted C count
//!
//! N   N   Y   Y   N   N   N   <- pending gap-open mask shifted by 2 lanes
//! N   Y   N   N   Y   Y   N   <- E mask shifted by 1 lane
//! Y   Y   N   Y   Y   Y   Y   <- E crossed |= shifted E mask
//! N   N   N   Y   N   N   N   <- pending gap-open crossed E = pending gap-open mask & E crossed
//!
//! 0   0   0   4   0   0   0   <- gap open penalty
//! 2   1   0   1   2   1   1   <- gap extend penalty = 2 - total C count
//! 0   0   16  11  0   0   0   <- shifted score - penalties
//!
//! state updates:
//! Y   Y   N   Y   Y   Y   Y   <- E crossed
//! 0   1   2   1   0   1   1   <- total C count
//! Y   Y   Y   N   N   N   N   <- pending gap-open mask |= shifted pending gap-open mask & !E crossed
//! 16  16  16  11  0   0   0   <- score = max(score, shifted score - penalties)
//!
//! ---
//! shift 4
//!
//! 0   0   0   0   16  16  16  <- score shifted by 4 lanes
//!
//! 0   0   0   1   2   1   0   <- total C count shifted by 2 lanes
//! 0   1   2   2   2   2   1   <- total C count += shifted C count
//!
//! N   N   N   N   Y   Y   Y   <- pending gap-open mask shifted by 4 lanes
//! N   N   Y   N   N   Y   Y   <- E mask shifted by 2 lane
//! Y   Y   Y   Y   Y   Y   Y   <- E crossed |= shifted E mask
//! N   N   N   N   Y   Y   Y   <- pending gap-open crossed E = shifted pending gap-open mask & E crossed
//!
//! 0   0   0   0   4   4   4   <- gap open penalty
//! 4   3   2   2   2   2   3   <- gap extend penalty = 4 - total C count
//! 0   0   0   0   10  10  9   <- shifted score - penalties
//!
//! state updates:
//! Y   Y   Y   Y   Y   Y   Y   <- E crossed
//! 0   1   2   2   2   2   1   <- total C count
//! Y   Y   Y   N   N   N   N   <- pending gap-open mask |= shifted pending gap-open mask & !E crossed
//! 16  16  16  11  10  10  9   <- score = max(score, shifted score - penalties)
//!
//! ---
//! final
//!
//! E   C   C   E   E   C   E
//! 16  16  16  11  10  10  9
//! ```

use crate::smith_waterman::backend::{Backend, ScoreVec};

#[inline(always)]
unsafe fn unicode_gap_step<B: Backend, const SHIFT: i32>(
    row: &mut B::Score,
    pending_gap_open_mask: &mut B::Score,
    adjacent_row: B::Score,
    adjacent_pending_gap_open_mask: B::Score,
    continuation_gap_extend_penalty: B::Score,
    scalar_end_mask: B::Score,
    total_gap_extend_penalty: B::Score,
    gap_open_penalty: B::Score,
) {
    unsafe {
        let shifted_row = row.shift_right_padded::<SHIFT>(adjacent_row);
        let shifted_pending_gap_open_mask =
            pending_gap_open_mask.shift_right_padded::<SHIFT>(adjacent_pending_gap_open_mask);

        let scalar_gap_extend_penalty =
            total_gap_extend_penalty.subs(continuation_gap_extend_penalty);
        let pending_gap_open_crossed_scalar_end =
            shifted_pending_gap_open_mask.and(scalar_end_mask);
        let gap_penalty = scalar_gap_extend_penalty
            .add(gap_open_penalty.and(pending_gap_open_crossed_scalar_end));
        let candidate_row = shifted_row.subs(gap_penalty);

        *row = row.max(candidate_row);

        let candidate_pending_gap_open_mask = shifted_pending_gap_open_mask.subs(scalar_end_mask);
        *pending_gap_open_mask = pending_gap_open_mask.max(candidate_pending_gap_open_mask);
    }
}

#[inline(always)]
unsafe fn prepare_next_unicode_gap_step<B: Backend, const SHIFT: i32>(
    continuation_gap_extend_penalty: &mut B::Score,
    adjacent_continuation_gap_extend_penalty: &mut B::Score,
    scalar_end_mask: &mut B::Score,
    adjacent_scalar_end_mask: &mut B::Score,
    total_gap_extend_penalty: &mut B::Score,
) {
    unsafe {
        let zero = B::Score::zero();

        let shifted_continuation_gap_extend_penalty = continuation_gap_extend_penalty
            .shift_right_padded::<SHIFT>(*adjacent_continuation_gap_extend_penalty);
        *continuation_gap_extend_penalty =
            continuation_gap_extend_penalty.add(shifted_continuation_gap_extend_penalty);
        *adjacent_continuation_gap_extend_penalty = adjacent_continuation_gap_extend_penalty
            .add(adjacent_continuation_gap_extend_penalty.shift_right_padded::<SHIFT>(zero));

        let shifted_scalar_end_mask =
            scalar_end_mask.shift_right_padded::<SHIFT>(*adjacent_scalar_end_mask);
        *scalar_end_mask = scalar_end_mask.max(shifted_scalar_end_mask);
        *adjacent_scalar_end_mask = adjacent_scalar_end_mask
            .max(adjacent_scalar_end_mask.shift_right_padded::<SHIFT>(zero));

        *total_gap_extend_penalty = total_gap_extend_penalty.add(*total_gap_extend_penalty);
    }
}

macro_rules! unicode_propagator {
    ($name:ident, [$($prepare_shift:literal),*], $final_shift:literal) => {
        #[inline(always)]
        pub(crate) unsafe fn $name<B: Backend>(
            row: B::Score,
            adjacent_row: B::Score,
            pending_gap_open_mask: B::Score,
            adjacent_pending_gap_open_mask: B::Score,
            continuation_gap_extend_penalty: B::Score,
            adjacent_continuation_gap_extend_penalty: B::Score,
            scalar_end_mask: B::Score,
            adjacent_scalar_end_mask: B::Score,
            gap_open_penalty: B::Score,
            gap_extend_penalty: B::Score,
        ) -> (B::Score, B::Score) {
            unsafe {
                let mut row = row;
                let mut pending_gap_open_mask = pending_gap_open_mask;
                let mut continuation_gap_extend_penalty = continuation_gap_extend_penalty;
                let mut adjacent_continuation_gap_extend_penalty =
                    adjacent_continuation_gap_extend_penalty;
                let mut scalar_end_mask = scalar_end_mask;
                let mut adjacent_scalar_end_mask = adjacent_scalar_end_mask;
                let mut total_gap_extend_penalty = gap_extend_penalty;

                $(
                    unicode_gap_step::<B, $prepare_shift>(
                        &mut row,
                        &mut pending_gap_open_mask,
                        adjacent_row,
                        adjacent_pending_gap_open_mask,
                        continuation_gap_extend_penalty,
                        scalar_end_mask,
                        total_gap_extend_penalty,
                        gap_open_penalty,
                    );
                    prepare_next_unicode_gap_step::<B, $prepare_shift>(
                        &mut continuation_gap_extend_penalty,
                        &mut adjacent_continuation_gap_extend_penalty,
                        &mut scalar_end_mask,
                        &mut adjacent_scalar_end_mask,
                        &mut total_gap_extend_penalty,
                    );
                )*

                unicode_gap_step::<B, $final_shift>(
                    &mut row,
                    &mut pending_gap_open_mask,
                    adjacent_row,
                    adjacent_pending_gap_open_mask,
                    continuation_gap_extend_penalty,
                    scalar_end_mask,
                    total_gap_extend_penalty,
                    gap_open_penalty,
                );

                (row, pending_gap_open_mask)
            }
        }
    };
}

unicode_propagator!(propagate_unicode_8_lane, [1, 2], 4);
unicode_propagator!(propagate_unicode_16_lane, [1, 2, 4], 8);
unicode_propagator!(propagate_unicode_32_lane, [1, 2, 4, 8], 16);
unicode_propagator!(propagate_unicode_64_lane, [1, 2, 4, 8, 16], 32);
