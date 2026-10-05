//! ASCII gap propagation shared by all backends. Horizontal gaps are a
//! decayed prefix max: lane `j` takes the max over `k < j` of
//! `sources[k] - gap_extend * (j - k)`, with the gap open penalty folded into
//! `sources`. The scan is radix-4 (shifts by `s`, `2s` and `3s` per stage),
//! and `prev_scan`, the previous chunk's scan, fills the shifts.

use crate::smith_waterman::backend::{Backend, ScoreVec};

/// Lane shifts of the scan stages. [`ScoringVecs::gap_steps`] holds
/// `gap_extend * GAP_SHIFTS[i]`, then `gap_extend * LANES` at [`FULL_STEP`].
///
/// [`ScoringVecs::gap_steps`]: crate::smith_waterman::ScoringVecs
pub(crate) const GAP_SHIFTS: [u16; 9] = [1, 2, 3, 4, 8, 12, 16, 32, 48];
pub(crate) const FULL_STEP: usize = GAP_SHIFTS.len();
pub(crate) const GAP_STEPS: usize = FULL_STEP + 1;

/// The steps [`Backend::propagate_horizontal_gaps`] takes in the ASCII
/// kernel, for a gap extend penalty
#[inline(always)]
pub(crate) unsafe fn gap_steps<B: Backend>(gap_extend: u16) -> [B::Score; GAP_STEPS] {
    unsafe {
        let mut steps = [B::Score::zero(); GAP_STEPS];
        for (i, step) in steps.iter_mut().enumerate() {
            let shift = GAP_SHIFTS.get(i).copied().unwrap_or(B::LANES as u16);
            *step = B::Score::splat(gap_extend.saturating_mul(shift));
        }
        steps
    }
}

/// Shifts `v` right by `L` lanes, filling from `prev_scan`, and decays it
#[inline(always)]
unsafe fn decay<B: Backend, const L: i32>(
    v: B::Score,
    prev_scan: B::Score,
    step: B::Score,
) -> B::Score {
    unsafe { v.shift_right_padded::<L>(prev_scan).subs(step) }
}

/// One radix-4 stage: shifts of `A`, `B` and `C` lanes, with steps from slot
/// `i` on. `fold` joins the max tree next to `v`, where it adds no latency.
#[inline(always)]
unsafe fn stage<Bk: Backend, const A: i32, const B: i32, const C: i32>(
    v: Bk::Score,
    prev_scan: Bk::Score,
    steps: &[Bk::Score; GAP_STEPS],
    i: usize,
    fold: Bk::Score,
) -> Bk::Score {
    unsafe {
        let a = decay::<Bk, A>(v, prev_scan, steps[i]);
        let b = decay::<Bk, B>(v, prev_scan, steps[i + 1]);
        let c = decay::<Bk, C>(v, prev_scan, steps[i + 2]);
        v.max(fold).max(a).max(b.max(c))
    }
}

/// Scans `sources`, returning `(scan, row)` where `row` is the scan folded
/// with the row's `scores`
#[inline(always)]
pub(crate) unsafe fn scan_8_lane<B: Backend>(
    sources: B::Score,
    prev_scan: B::Score,
    steps: &[B::Score; GAP_STEPS],
    scores: B::Score,
) -> (B::Score, B::Score) {
    unsafe {
        let full_chunk = prev_scan.subs(steps[FULL_STEP]);
        let v = stage::<B, 1, 2, 3>(sources, prev_scan, steps, 0, full_chunk);
        let a = decay::<B, 4>(v, prev_scan, steps[3]);
        (v.max(a), v.max(scores).max(a))
    }
}

/// Radix-2: on 16 lanes consecutive chunks' rows overlap, so radix-4's extra
/// shifts cost more than its shorter chain saves
#[inline(always)]
pub(crate) unsafe fn scan_16_lane<B: Backend>(
    sources: B::Score,
    prev_scan: B::Score,
    steps: &[B::Score; GAP_STEPS],
    scores: B::Score,
) -> (B::Score, B::Score) {
    unsafe {
        // `prev_scan` is itself a decayed prefix max, so `full_chunk` carried
        // further never beats `full_chunk` in its own lane: it joins once,
        // after the scan
        let full_chunk = prev_scan.subs(steps[FULL_STEP]);
        let v = sources.max(decay::<B, 1>(sources, prev_scan, steps[0]));
        let v = v.max(decay::<B, 2>(v, prev_scan, steps[1]));
        let v = v.max(decay::<B, 4>(v, prev_scan, steps[3]));
        let scan = v.max(decay::<B, 8>(v, prev_scan, steps[4])).max(full_chunk);
        (scan, scan.max(scores))
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn scan_32_lane<B: Backend>(
    sources: B::Score,
    prev_scan: B::Score,
    steps: &[B::Score; GAP_STEPS],
    scores: B::Score,
) -> (B::Score, B::Score) {
    unsafe {
        let full_chunk = prev_scan.subs(steps[FULL_STEP]);
        let v = stage::<B, 1, 2, 3>(sources, prev_scan, steps, 0, full_chunk);
        let v = stage::<B, 4, 8, 12>(v, prev_scan, steps, 3, B::Score::zero());
        let a = decay::<B, 16>(v, prev_scan, steps[6]);
        (v.max(a), v.max(scores).max(a))
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn scan_64_lane<B: Backend>(
    sources: B::Score,
    prev_scan: B::Score,
    steps: &[B::Score; GAP_STEPS],
    scores: B::Score,
) -> (B::Score, B::Score) {
    unsafe {
        let full_chunk = prev_scan.subs(steps[FULL_STEP]);
        let v = stage::<B, 1, 2, 3>(sources, prev_scan, steps, 0, full_chunk);
        let v = stage::<B, 4, 8, 12>(v, prev_scan, steps, 3, B::Score::zero());
        let a = decay::<B, 16>(v, prev_scan, steps[6]);
        let bc = decay::<B, 32>(v, prev_scan, steps[7]).max(decay::<B, 48>(v, prev_scan, steps[8]));
        (v.max(a).max(bc), v.max(scores).max(a).max(bc))
    }
}
