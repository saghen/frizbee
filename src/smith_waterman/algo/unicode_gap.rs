//! With the default gap propagation, `hw` matched against `h😀w` will receive a
//! penalty to the score of `gap_open_penalty + gap_extend_penalty * 4` due to
//! the emoji taking up 4 bytes. If the haystack was instead `hew`, the penalty
//! would be `gap_open_penalty + gap_extend_penalty`. As a result, we need to be
//! aware of the unicode boundaries and only apply the gap penalty when reaching
//! the end of a multi-byte codepoint.
//!
//! Each row becomes a single needle char (multi-byte utf-8 codepoint)
//! Haystack stays as bytes
//!
//! - UTF-8 continuation bytes are transport lanes: a scalar's score rides
//!   across them for free (two shift stages cover the at most three
//!   continuation bytes), so the next row's diagonal sees it at the following
//!   scalar's lead byte.
//! - Gaps use the ASCII kernel's decayed prefix max (see [`ascii_gap`]), but
//!   each step charges the gap extend penalty once per non-continuation lane
//!   crossed, computed once per chunk.
//!
//! [`ascii_gap`]: super::ascii_gap

use super::ascii_gap::GAP_STEPS;
use crate::smith_waterman::backend::{Backend, ScoreVec};

/// `steps[a] + shift_A(steps[b])`: the step over `A + shift(b)` lanes, filled
/// from the previous chunk's steps
#[inline(always)]
unsafe fn join<B: Backend, const A: i32>(
    steps: &[B::Score; GAP_STEPS],
    prev_steps: &[B::Score; GAP_STEPS],
    a: usize,
    b: usize,
) -> B::Score {
    unsafe { steps[a].add(steps[b].shift_right_padded::<A>(prev_steps[b])) }
}

/// Builds the per-chunk steps in the slots of [`GAP_SHIFTS`] (then the full
/// chunk at [`FULL_STEP`]) that the lane count's scan reads, from `step_1`:
/// the gap extend penalty in non-continuation lanes. Each entry is
/// `(slot, A, a, b)` with `A` the shift of slot `a`, see [`join`].
///
/// [`GAP_SHIFTS`]: super::ascii_gap::GAP_SHIFTS
/// [`FULL_STEP`]: super::ascii_gap::FULL_STEP
macro_rules! unicode_steps {
    ($name:ident, $(($slot:literal, $shift:literal, $a:literal, $b:literal)),*) => {
        #[inline(always)]
        pub(crate) unsafe fn $name<B: Backend>(
            step_1: B::Score,
            prev_steps: &[B::Score; GAP_STEPS],
        ) -> [B::Score; GAP_STEPS] {
            unsafe {
                let mut steps = [step_1; GAP_STEPS];
                $(steps[$slot] = join::<B, $shift>(&steps, prev_steps, $a, $b);)*
                steps
            }
        }
    };
}

// Wider steps read the 1, 4 or 16 lane step, so 64 lanes take 3 dependent
// shifts instead of 6. The radix-2 16-lane scan needs no 3-lane decay.
unicode_steps!(
    unicode_steps_8_lane,
    (1, 1, 0, 0),
    (2, 2, 1, 0),
    (3, 3, 2, 0),
    (9, 4, 3, 3)
);
unicode_steps!(
    unicode_steps_16_lane,
    (1, 1, 0, 0),
    (3, 2, 1, 1),
    (4, 4, 3, 3),
    (9, 8, 4, 4)
);
#[cfg(target_arch = "x86_64")]
unicode_steps!(
    unicode_steps_32_lane,
    (1, 1, 0, 0),
    (2, 2, 1, 0),
    (3, 3, 2, 0),
    (4, 4, 3, 3),
    (5, 8, 4, 3),
    (6, 12, 5, 3),
    (9, 16, 6, 6)
);
#[cfg(target_arch = "x86_64")]
unicode_steps!(
    unicode_steps_64_lane,
    (1, 1, 0, 0),
    (2, 2, 1, 0),
    (3, 3, 2, 0),
    (4, 4, 3, 3),
    (5, 8, 4, 3),
    (6, 12, 5, 3),
    (7, 16, 6, 6),
    (8, 32, 7, 6),
    (9, 48, 8, 6)
);
