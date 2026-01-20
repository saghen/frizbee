#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

use crate::Scoring;

mod gaps;
mod ops;
mod typos;

use gaps::propagate_horizontal_gaps;
use ops::*;
pub use typos::typos_from_score_matrix;

pub unsafe fn smith_waterman(
    needle: &str,
    haystack: &str,
    score_matrix: &mut [Vec<__m256i>],
    scoring: &Scoring,
) -> __m256i {
    let needle = needle.as_bytes();
    let haystack = haystack.as_bytes();

    let mut max_scores = _mm256_setzero_si256();

    let gap_extend = _mm256_set1_epi16(scoring.gap_extend_penalty as i16);
    let gap_open = _mm256_set1_epi16(scoring.gap_open_penalty as i16);
    let match_score = _mm256_set1_epi16((scoring.match_score + scoring.matching_case_bonus) as i16);
    let mismatch_penalty = _mm256_set1_epi16(scoring.mismatch_penalty as i16);

    for (i, haystack) in haystack.chunks(16).enumerate().map(|(i, c)| (i + 1, c)) {
        let haystack = _mm_loadu_si128(haystack.as_ptr() as *const __m128i);
        let haystack = _mm256_cvtepu8_epi16(haystack); // 16xu8 -> 16xu16

        let mut up_gap_mask = _mm256_setzero_si256();
        let mut prev_row_scores = _mm256_setzero_si256();

        for (j, needle) in needle.iter().enumerate().map(|(i, c)| (i + 1, c)) {
            let needle = _mm256_set1_epi16(*needle as i16);
            let match_mask = _mm256_cmpeq_epi16(needle, haystack);

            // Up - skipping char in needle
            let up_scores = _mm256_subs_epu16(
                prev_row_scores,
                _mm256_blendv_epi8(gap_extend, gap_open, up_gap_mask),
            );

            // Diagonal - typical match/mismatch, moving along one haystack and needle char
            let diag_scores = {
                let diag =
                    _mm256_shift_right_padded_epi16(prev_row_scores, score_matrix[i - 1][j - 1]);
                let diag_matched = _mm256_add_epi16(diag, match_score);
                let diag_mismatched = _mm256_subs_epu16(diag, mismatch_penalty);
                _mm256_blendv_epi8(diag_mismatched, diag_matched, match_mask)
            };

            // Max of diagonal, up and left (after gap extension)
            let row_scores = propagate_horizontal_gaps(
                score_matrix[i - 1][j],
                _mm256_max_epu16(diag_scores, up_scores),
                match_mask,
                scoring.gap_open_penalty,
                scoring.gap_extend_penalty,
            );

            score_matrix[i][j] = row_scores;
            prev_row_scores = row_scores;
            up_gap_mask = match_mask;
            max_scores = _mm256_max_epu16(max_scores, row_scores);
        }
    }

    max_scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::x86_64::__m256i;

    fn make_score_matrix(needle: &str, haystack: &str) -> Vec<Vec<__m256i>> {
        (0..(haystack.len() / 16 + 1))
            .map(|_| {
                (0..=needle.len())
                    .map(|_| unsafe { _mm256_setzero_si256() })
                    .collect()
            })
            .collect::<Vec<_>>()
    }

    fn print_score_matrix(needle: &str, haystack: &str, score_matrix: &[Vec<__m256i>]) {
        let score_matrix =
            unsafe { std::mem::transmute::<_, &[Vec<std::simd::Simd<u16, 16>>]>(score_matrix) };

        print!("     ");
        for char in haystack.chars() {
            print!("{:<4} ", char);
        }
        println!();

        for (i, col_chunk) in score_matrix.iter().enumerate().skip(1) {
            for (j, row) in col_chunk.iter().skip(1).enumerate() {
                print!("{:>2}   ", needle.chars().nth(j).unwrap());
                for col in row.to_array().iter() {
                    print!("{:<4} ", col);
                }
                println!();
            }
        }
    }

    #[test]
    fn thisthisthis() {
        let needle = "test";
        let haystack = "~~~~~~t~est~~~~~";

        let scoring = Scoring::default();
        let mut score_matrix = make_score_matrix(needle, haystack);

        unsafe {
            smith_waterman(needle, haystack, &mut score_matrix, &scoring);
        }
        print_score_matrix(needle, haystack, &score_matrix);
    }
}
