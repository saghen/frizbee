#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

use crate::Scoring;

mod alignment;
use alignment::AlignmentChunk;

pub unsafe fn smith_waterman(
    haystack: &str,
    needle: &str,
    scoring: &Scoring,
) -> (__m256i, [AlignmentChunk; 4]) {
    let mut max_scores = _mm256_setzero_si256();

    let gap_extend = _mm256_set1_epi16(scoring.gap_extend_penalty as i16);
    let gap_open = _mm256_set1_epi16(scoring.gap_open_penalty as i16);
    let match_score = _mm256_set1_epi16((scoring.match_score + scoring.matching_case_bonus) as i16);
    let mismatch_penalty = _mm256_set1_epi16(scoring.mismatch_penalty as i16);

    let haystack = _mm_loadu_si128(haystack.as_ptr() as *const __m128i);
    let haystack = _mm256_cvtepu8_epi16(haystack); // 16xu8 -> 16xu16

    let mut up_gap_mask = _mm256_setzero_si256();
    let mut prev_row_scores = _mm256_setzero_si256();

    // TODO: should be the length of the needle
    let mut alignment: [AlignmentChunk; 4] = std::array::repeat(AlignmentChunk::default());

    for (i, needle) in needle.as_bytes().iter().enumerate() {
        let needle = _mm256_set1_epi16(*needle as i16);
        let match_mask = _mm256_cmpeq_epi16(needle, haystack);

        // Up - skipping char in needle
        let up_scores = _mm256_subs_epu16(
            prev_row_scores,
            _mm256_blendv_epi8(up_gap_mask, gap_extend, gap_open),
        );

        // Diagonal - typical match/mismatch, moving along one haystack and needle char
        let diag_scores = {
            let diag = _mm256_slli_si256::<2>(prev_row_scores);
            let diag_matched = _mm256_add_epi16(diag, match_score);
            let diag_mismatched = _mm256_subs_epu16(diag, mismatch_penalty);
            _mm256_blendv_epi8(diag_matched, diag_mismatched, match_mask)
        };

        // Max of diagonal, up and left (after gap extension)
        let row_scores = propagate_horizontal_gaps(
            _mm256_max_epu16(diag_scores, up_scores),
            match_mask,
            scoring.gap_open_penalty,
            scoring.gap_extend_penalty,
        );

        alignment[i] = AlignmentChunk::new(
            _mm256_cmpeq_epi16(diag_scores, row_scores),
            _mm256_setzero_si256(),
            _mm256_cmpeq_epi16(up_scores, row_scores),
        );
        prev_row_scores = row_scores;
        up_gap_mask = match_mask;
        max_scores = _mm256_max_epu16(max_scores, row_scores);
    }

    (max_scores, alignment)
}

#[inline(always)]
unsafe fn propagate_horizontal_gaps(
    row: __m256i,
    match_mask: __m256i,
    gap_open_penalty: u16,
    gap_extend_penalty: u16,
) -> __m256i {
    // TODO: explain this trick, due to shift not crossing 128-bit lanes
    let crossed = _mm256_permute2x128_si256::<0x81>(row, row);

    // apply gap open penalty if the previous element was a match
    let match_mask_crossed = _mm256_permute2x128_si256::<0x81>(match_mask, match_mask);
    let match_mask_shifted = _mm256_alignr_epi8::<1>(match_mask_crossed, match_mask);
    let gap_penalty = _mm256_blendv_epi8(
        _mm256_set1_epi16(gap_open_penalty as i16),
        _mm256_set1_epi16(gap_extend_penalty as i16),
        match_mask_shifted,
    );

    // shift by 1 element (2 bytes), decay by 1
    // row: [0, 4, 0, 0, 0, ...]
    let shifted = _mm256_alignr_epi8::<2>(crossed, row); // [0, 0, 4, 0, 0, ...]
    let decayed = _mm256_subs_epu16(shifted, gap_penalty); // [0, 0, 3, 0, ...]
    let row = _mm256_max_epu16(row, decayed); // [0, 4, 3, 0, 0, ...]

    // shift by 2 elements (4 bytes), decay by 2
    let shifted = _mm256_alignr_epi8::<4>(crossed, row);
    let decayed = _mm256_subs_epu16(shifted, _mm256_set1_epi16((gap_extend_penalty * 2) as i16));
    let row = _mm256_max_epu16(row, decayed);

    // recompute crossed since prev_row changed
    let crossed = _mm256_permute2x128_si256::<0x81>(row, row);

    // shift by 4 elements (8 bytes), decay by 4
    let shifted = _mm256_alignr_epi8::<8>(crossed, row);
    let decayed = _mm256_subs_epu16(shifted, _mm256_set1_epi16((gap_extend_penalty * 4) as i16));
    _mm256_max_epu16(row, decayed)
}

unsafe fn _mm256_cmpneq_epi16(a: __m256i, b: __m256i) -> __m256i {
    let eq = _mm256_cmpeq_epi16(a, b);
    _mm256_xor_si256(eq, _mm256_set1_epi16(-1)) // not
}
