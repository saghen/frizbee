//! ASCII gap propagation shared by all backends. Horizontal gaps are a
//! decayed prefix max: lane `j` takes the max over `k < j` of
//! `values[k] - gap_extend * (j - k)`, with the gap open penalty folded into
//! `values` at the source. The scan is radix-4 (shifts by `s`, `2s` and `3s`
//! per stage), and `prev`, the previous chunk's scan, fills the shifts.

use crate::smith_waterman::backend::{Backend, ScoreVec};

/// Lane shifts of the scan stages. [`ScoringVecs::gap_steps`] holds
/// `gap_extend * GAP_SHIFTS[i]`, then `gap_extend * LANES` at [`FULL_STEP`].
///
/// [`ScoringVecs::gap_steps`]: crate::smith_waterman::ScoringVecs
pub(crate) const GAP_SHIFTS: [u16; 9] = [1, 2, 3, 4, 8, 12, 16, 32, 48];
pub(crate) const FULL_STEP: usize = 9;

/// Shifts `v` right by `L` lanes, filling from `prev`, and decays it
#[inline(always)]
unsafe fn decay<B: Backend, const L: i32>(v: B::Score, prev: B::Score, step: B::Score) -> B::Score {
    unsafe { v.shift_right_padded::<L>(prev).subs(step) }
}

/// One radix-4 stage: shifts of `A`, `B` and `C` lanes, with steps from slot
/// `i` on. `fold` joins the max tree next to `v`, where it adds no latency.
#[inline(always)]
unsafe fn stage<Bk: Backend, const A: i32, const B: i32, const C: i32>(
    v: Bk::Score,
    prev: Bk::Score,
    steps: &[Bk::Score; 10],
    i: usize,
    fold: Bk::Score,
) -> Bk::Score {
    unsafe {
        let a = decay::<Bk, A>(v, prev, steps[i]);
        let b = decay::<Bk, B>(v, prev, steps[i + 1]);
        let c = decay::<Bk, C>(v, prev, steps[i + 2]);
        v.max(fold).max(a).max(b.max(c))
    }
}

/// Final radix-4 stage, returning `(scan, row)` where `row` also folds in the
/// row's `scores`
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn last_stage<Bk: Backend, const A: i32, const B: i32, const C: i32>(
    v: Bk::Score,
    prev: Bk::Score,
    steps: &[Bk::Score; 10],
    i: usize,
    scores: Bk::Score,
) -> (Bk::Score, Bk::Score) {
    unsafe {
        let a = decay::<Bk, A>(v, prev, steps[i]);
        let bc = decay::<Bk, B>(v, prev, steps[i + 1]).max(decay::<Bk, C>(v, prev, steps[i + 2]));
        (v.max(a).max(bc), v.max(scores).max(a).max(bc))
    }
}

/// Final single-shift stage, like [`last_stage`]
#[inline(always)]
unsafe fn last_shift<B: Backend, const A: i32>(
    v: B::Score,
    prev: B::Score,
    steps: &[B::Score; 10],
    i: usize,
    scores: B::Score,
) -> (B::Score, B::Score) {
    unsafe {
        let a = decay::<B, A>(v, prev, steps[i]);
        (v.max(a), v.max(scores).max(a))
    }
}

/// Scans `v`, returning `(scan, row)` where `row` is the scan folded with the
/// row's `scores`
#[inline(always)]
pub(crate) unsafe fn scan_8_lane<B: Backend>(
    v: B::Score,
    prev: B::Score,
    steps: &[B::Score; 10],
    scores: B::Score,
) -> (B::Score, B::Score) {
    unsafe {
        let full = prev.subs(steps[FULL_STEP]);
        let v = stage::<B, 1, 2, 3>(v, prev, steps, 0, full);
        last_shift::<B, 4>(v, prev, steps, 3, scores)
    }
}

/// Radix-2: on 16 lanes consecutive chunks' rows overlap, so radix-4's extra
/// shifts cost more than its shorter chain saves
#[inline(always)]
pub(crate) unsafe fn scan_16_lane<B: Backend>(
    v: B::Score,
    prev: B::Score,
    steps: &[B::Score; 10],
    scores: B::Score,
) -> (B::Score, B::Score) {
    unsafe {
        // `prev` is itself a decayed prefix max, so `full` carried further
        // never beats `full` in its own lane: it joins once, after the scan
        let full = prev.subs(steps[FULL_STEP]);
        let v = v.max(decay::<B, 1>(v, prev, steps[0]));
        let v = v.max(decay::<B, 2>(v, prev, steps[1]));
        let v = v.max(decay::<B, 4>(v, prev, steps[3]));
        let scan = v.max(decay::<B, 8>(v, prev, steps[4])).max(full);
        (scan, scan.max(scores))
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn scan_32_lane<B: Backend>(
    v: B::Score,
    prev: B::Score,
    steps: &[B::Score; 10],
    scores: B::Score,
) -> (B::Score, B::Score) {
    unsafe {
        let full = prev.subs(steps[FULL_STEP]);
        let v = stage::<B, 1, 2, 3>(v, prev, steps, 0, full);
        let v = stage::<B, 4, 8, 12>(v, prev, steps, 3, B::Score::zero());
        last_shift::<B, 16>(v, prev, steps, 6, scores)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn scan_64_lane<B: Backend>(
    v: B::Score,
    prev: B::Score,
    steps: &[B::Score; 10],
    scores: B::Score,
) -> (B::Score, B::Score) {
    unsafe {
        let full = prev.subs(steps[FULL_STEP]);
        let v = stage::<B, 1, 2, 3>(v, prev, steps, 0, full);
        let v = stage::<B, 4, 8, 12>(v, prev, steps, 3, B::Score::zero());
        last_stage::<B, 16, 32, 48>(v, prev, steps, 6, scores)
    }
}
