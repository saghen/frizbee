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
//! - Gaps use the ASCII kernel's decayed prefix-max scan (see [`ascii_gap`]),
//!   with the gap open penalty folded in at the source lane, but each shift's
//!   decay charges the gap extend penalty once per *non-continuation* lane
//!   crossed. The decays only depend on the chunk, so they're computed here
//!   once per chunk and shared by every needle row.
//!
//! [`ascii_gap`]: super::ascii_gap

use crate::smith_waterman::backend::{Backend, ScoreVec};

/// `d[a] + shift_A(d[b])`: the decay over `A + shift(b)` lanes, filled from the
/// previous chunk's decays
#[inline(always)]
unsafe fn join<B: Backend, const A: i32>(
    d: &[B::Score; 10],
    prev: &[B::Score; 10],
    a: usize,
    b: usize,
) -> B::Score {
    unsafe { d[a].add(d[b].shift_right_padded::<A>(prev[b])) }
}

/// Builds the per-chunk decays in the slots of [`ascii_gap::GAP_SHIFTS`] (then
/// the full chunk at [`ascii_gap::FULL_STEP`]) that the lane count's scan reads,
/// from `d1`: the gap extend penalty in non-continuation lanes. Each entry is
/// `(slot, A, a, b)`, see [`join`].
///
/// [`ascii_gap::GAP_SHIFTS`]: super::ascii_gap::GAP_SHIFTS
/// [`ascii_gap::FULL_STEP`]: super::ascii_gap::FULL_STEP
macro_rules! unicode_steps {
    ($name:ident, $(($slot:literal, $shift:literal, $a:literal, $b:literal)),*) => {
        #[inline(always)]
        pub(crate) unsafe fn $name<B: Backend>(
            d1: B::Score,
            prev: &[B::Score; 10],
        ) -> [B::Score; 10] {
            unsafe {
                let mut d = [d1; 10];
                $(d[$slot] = join::<B, $shift>(&d, prev, $a, $b);)*
                d
            }
        }
    };
}

// Wider decays read `d1`, `d4` or `d16`, so 64 lanes take 3 dependent shifts
// instead of 6. The radix-2 16-lane scan needs no 3-lane decay.
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
