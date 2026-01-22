use std::arch::x86_64::*;

use super::ops::*;

#[inline(always)]
pub unsafe fn propagate_horizontal_gaps(
    adjacent_row: __m256i,
    row: __m256i,
    match_mask: __m256i,
    gap_open_penalty: u16,
    gap_extend_penalty: u16,
) -> __m256i {
    // TODO: need adjacent match mask too

    // shift by 1 element (2 bytes), decay by 1
    // row: [0, 4, 0, 0, 0, ...]
    let shifted = _mm256_shift_right_padded_epi16(row, adjacent_row);
    let gap_penalty = _mm256_blendv_epi8(
        _mm256_set1_epi16(gap_extend_penalty as i16),
        _mm256_set1_epi16(gap_open_penalty as i16),
        _mm256_shift_right_epi16(match_mask),
    );
    let decayed = _mm256_subs_epu16(shifted, gap_penalty); // [0, 0, 3, 0, ...]
    let row = _mm256_max_epu16(row, decayed); // [0, 4, 3, 0, 0, ...]

    // shift by 2 elements (4 bytes), decay by 2
    let shifted = _mm256_shift_right_two_padded_epi16(row, adjacent_row);
    let gap_penalty = _mm256_blendv_epi8(
        _mm256_set1_epi16((gap_extend_penalty * 2) as i16),
        _mm256_set1_epi16((gap_open_penalty + gap_extend_penalty) as i16),
        _mm256_shift_right_two_epi16(match_mask),
    );
    let decayed = _mm256_subs_epu16(shifted, gap_penalty);
    let row = _mm256_max_epu16(row, decayed);

    // shift by 4 elements (8 bytes), decay by 4
    let shifted = _mm256_shift_right_four_padded_epi16(row, adjacent_row);
    let gap_penalty = _mm256_blendv_epi8(
        _mm256_set1_epi16((gap_extend_penalty * 4) as i16),
        _mm256_set1_epi16((gap_open_penalty + gap_extend_penalty * 3) as i16),
        _mm256_shift_right_four_epi16(match_mask),
    );
    let decayed = _mm256_subs_epu16(shifted, gap_penalty);
    _mm256_max_epu16(row, decayed)
}
