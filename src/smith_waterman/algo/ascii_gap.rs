use crate::smith_waterman::backend::{Backend, ScoreVec};

// ---------------------------------------------------------------------------
// ASCII gap propagation helpers shared by all backends.
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

#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "x86_64")]
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
