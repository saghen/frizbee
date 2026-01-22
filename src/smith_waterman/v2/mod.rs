#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

use crate::Scoring;

mod gaps;
mod ops;
mod typos;

use gaps::propagate_horizontal_gaps;
use ops::*;
pub use typos::typos_from_score_matrix;

pub struct Needle(Vec<(u8, u8)>);

impl Needle {
    pub fn new(needle: &str) -> Self {
        let mut data = Vec::with_capacity(needle.len());
        for c in needle.chars() {
            if c.is_lowercase() {
                data.push((c.to_ascii_uppercase() as u8, c as u8));
            } else if c.is_uppercase() {
                data.push((c.to_ascii_lowercase() as u8, c as u8));
            } else {
                data.push((c as u8, c as u8));
            }
        }

        Self(data)
    }

    pub fn iter(&self) -> impl Iterator<Item = &(u8, u8)> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub fn generate_score_matrix(needle_len: usize, haystack_len: usize) -> Vec<Vec<__m256i>> {
    (0..=(haystack_len.div_ceil(16) + 1))
        .map(|_| {
            (0..=needle_len)
                .map(|_| unsafe { _mm256_setzero_si256() })
                .collect()
        })
        .collect::<Vec<_>>()
}

pub fn smith_waterman(
    needle: &Needle,
    haystack: &[u8],
    scoring: &Scoring,
    score_matrix: &mut [Vec<__m256i>],
) -> u16 {
    unsafe {
        // TODO: capitalization, prefix, offset_prefix bonuses
        let mut max_scores = _mm256_setzero_si256();

        // constants
        let gap_extend = _mm256_set1_epi16(scoring.gap_extend_penalty as i16);
        let gap_open =
            _mm256_set1_epi16((scoring.gap_open_penalty - scoring.gap_extend_penalty) as i16);
        let match_score =
            _mm256_set1_epi16((scoring.match_score + scoring.mismatch_penalty) as i16);
        let mismatch_penalty = _mm256_set1_epi16(scoring.mismatch_penalty as i16);
        let matching_case_bonus = _mm256_set1_epi16(scoring.matching_case_bonus as i16);

        let delimiter_bonus = _mm256_set1_epi16(scoring.delimiter_bonus as i16);
        let delimiters = [b' ', b'/', b'.', b',', b'_', b'-', b':'];

        let mut prev_delimiter_mask = _mm256_setzero_si256();

        for mut col_idx in 0..(haystack.len().div_ceil(16)) {
            let haystack = _mm_loadu(haystack, col_idx * 16, haystack.len());
            col_idx += 1;

            let mut up_gap_mask = _mm256_setzero_si256();
            let mut prev_row_scores = _mm256_setzero_si256();

            for (row_idx, (needle_char, flipped_case_needle_char)) in
                needle.iter().enumerate().map(|(i, c)| (i + 1, c))
            {
                let exact_case_match_mask =
                    _mm_cmpeq_epi8(_mm_set1_epi8(*needle_char as i8), haystack);
                let flipped_case_match_mask =
                    _mm_cmpeq_epi8(_mm_set1_epi8(*flipped_case_needle_char as i8), haystack);
                let match_mask = _mm_or_si128(exact_case_match_mask, flipped_case_match_mask);
                let exact_case_match_mask = _mm256_cvtepi8_epi16(exact_case_match_mask); // 16xu8 -> 16xu16
                let match_mask = _mm256_cvtepi8_epi16(match_mask); // 16xu8 -> 16xu16

                // Bonus for matching after a delimiter character
                let is_delimiter_mask =
                    delimiters.iter().fold(_mm_setzero_si128(), |acc, delim| {
                        let mask = _mm_cmpeq_epi8(haystack, _mm_set1_epi8(*delim as i8));
                        _mm_or_si128(acc, mask)
                    });
                let is_delimiter_mask = _mm256_cvtepi8_epi16(is_delimiter_mask); // 16xu8 -> 16xu16
                let prev_is_delimiter_mask =
                    _mm256_shift_right_padded_epi16(is_delimiter_mask, prev_delimiter_mask);
                let delimiter_mask =
                    _mm256_and_si256(prev_is_delimiter_mask, _mm256_not_epi16(is_delimiter_mask));
                prev_delimiter_mask = is_delimiter_mask;

                // Up - skipping char in needle
                let up_scores = {
                    // Always apply gap extend penalty
                    let score_after_gap_extend = _mm256_subs_epu16(prev_row_scores, gap_extend);
                    // Apply gap open penalty - gap extend penalty for opened gaps, avoiding blendv
                    _mm256_subs_epu16(
                        score_after_gap_extend,
                        _mm256_and_si256(up_gap_mask, gap_open),
                    )
                };

                // Diagonal - typical match/mismatch, moving along one haystack and needle char
                let diag_scores = {
                    let diag = _mm256_shift_right_padded_epi16(
                        prev_row_scores,
                        score_matrix[col_idx - 1][row_idx - 1],
                    );

                    // Always add mismatch penalty
                    let diag = _mm256_subs_epu16(diag, mismatch_penalty);
                    // Add match score (+ mismatch penalty) for matches, avoiding blendv
                    let diag = _mm256_add_epi16(diag, _mm256_and_si256(match_mask, match_score));
                    // Add delimiter bonus
                    let diag =
                        _mm256_add_epi16(diag, _mm256_and_si256(delimiter_mask, delimiter_bonus));
                    // Add matching case bonus
                    _mm256_add_epi16(
                        diag,
                        _mm256_and_si256(exact_case_match_mask, matching_case_bonus),
                    )
                };

                // Max of diagonal, up and left (after gap extension)
                let row_scores = propagate_horizontal_gaps(
                    score_matrix[col_idx - 1][row_idx],
                    _mm256_max_epu16(diag_scores, up_scores),
                    match_mask,
                    scoring.gap_open_penalty,
                    scoring.gap_extend_penalty,
                );

                score_matrix[col_idx][row_idx] = row_scores;
                prev_row_scores = row_scores;
                up_gap_mask = match_mask;

                if row_idx == needle.len() {
                    max_scores = _mm256_max_epu16(max_scores, row_scores);
                }
            }
        }

        _mm256_smax_epu16(max_scores)
    }
}

#[cfg(test)]
pub fn print_score_matrix(needle: &str, haystack: &str, score_matrix: &[Vec<__m256i>]) {
    let score_matrix =
        unsafe { std::mem::transmute::<&[Vec<__m256i>], &[Vec<[u16; 16]>]>(score_matrix) };

    print!("     ");
    for char in haystack.chars() {
        print!("{:<4} ", char);
    }
    println!();

    for (i, col_chunk) in score_matrix[0..=(haystack.len() / 16 + 1)]
        .iter()
        .skip(1)
        .enumerate()
    {
        for (j, row) in col_chunk.iter().skip(1).enumerate() {
            print!("{:>2}   ", needle.chars().nth(j).unwrap());
            for col in row[0..(haystack.len().saturating_sub(i * 16))].iter() {
                print!("{:<4} ", col);
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thisthisthis() {
        let needle = "deadbe";
        let haystack = "deadbeef";

        let scoring = Scoring::default();
        let mut score_matrix = generate_score_matrix(needle.len(), haystack.len());

        let score = smith_waterman(
            &Needle::new(needle),
            haystack.as_bytes(),
            &scoring,
            &mut score_matrix,
        );
        print_score_matrix(needle, haystack, &score_matrix);
        println!("max_score: {}", score);

        let haystack = "deadbf";
        let score = smith_waterman(
            &Needle::new(needle),
            haystack.as_bytes(),
            &scoring,
            &mut score_matrix,
        );

        print_score_matrix(needle, haystack, &score_matrix);
        println!("max_score: {}", score);
    }
}
