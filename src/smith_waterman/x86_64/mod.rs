#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

use crate::Scoring;

mod gaps;
mod ops;
mod typos;

use gaps::propagate_horizontal_gaps;
use ops::*;
pub use typos::typos_from_score_matrix;

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
    needle: &[(u8, u8)],
    haystack: &[u8],
    scoring: &Scoring,
    score_matrix: &mut [Vec<__m256i>],
) -> u16 {
    unsafe {
        // TODO: capitalization bonus
        // TODO: have prefix bonus scale based on distance

        // Constants
        let gap_extend = _mm256_set1_epi16(scoring.gap_extend_penalty as i16);
        let gap_open =
            _mm256_set1_epi16((scoring.gap_open_penalty - scoring.gap_extend_penalty) as i16);
        let match_score =
            _mm256_set1_epi16((scoring.match_score + scoring.mismatch_penalty) as i16);
        let mismatch_penalty = _mm256_set1_epi16(scoring.mismatch_penalty as i16);
        let matching_case_bonus = _mm256_set1_epi16(scoring.matching_case_bonus as i16);
        let prefix_bonus = _mm256_set1_epi16(scoring.prefix_bonus as i16);

        let delimiter_bonus = _mm256_set1_epi16(scoring.delimiter_bonus as i16);
        let delimiters = [b' ', b'/', b'.', b',', b'_', b'-', b':'];

        // State
        let mut prev_delimiter_mask = _mm256_setzero_si256();
        let mut prefix_mask = get_prefix_mask();
        let mut max_scores = _mm256_setzero_si256();

        for mut col_idx in 0..(haystack.len().div_ceil(16)) {
            let haystack = _mm_loadu(haystack, col_idx * 16, haystack.len());
            col_idx += 1;

            let mut up_gap_mask = _mm256_setzero_si256();
            let mut prev_row_scores = _mm256_setzero_si256();

            for (row_idx, (needle_char, flipped_case_needle_char)) in
                needle.iter().enumerate().map(|(i, c)| (i + 1, c))
            {
                // Match needle chars against the haystack (case insensitive)
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

                    // Add match score (+ mismatch penalty) for matches, avoiding blendv
                    let diag = _mm256_add_epi16(diag, _mm256_and_si256(match_mask, match_score));
                    // Always add mismatch penalty
                    let diag = _mm256_subs_epu16(diag, mismatch_penalty);
                    // Add prefix bonus
                    let diag = _mm256_add_epi16(
                        diag,
                        _mm256_and_si256(_mm256_and_si256(prefix_mask, match_mask), prefix_bonus),
                    );
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

                // Store results
                score_matrix[col_idx][row_idx] = row_scores;
                prev_row_scores = row_scores;
                up_gap_mask = match_mask;

                if row_idx == needle.len() {
                    max_scores = _mm256_max_epu16(max_scores, row_scores);
                }
            }

            prefix_mask = _mm256_setzero_si256();
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
    use crate::r#const::*;
    use crate::prefilter::Prefilter;

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn get_score(needle: &str, haystack: &str) -> u16 {
        let mut score_matrix = generate_score_matrix(needle.len(), haystack.len());
        let score = smith_waterman(
            &Prefilter::case_needle(needle),
            haystack.as_bytes(),
            &Scoring::default(),
            &mut score_matrix,
        );
        print_score_matrix(needle, haystack, &score_matrix);
        score
    }

    #[test]
    fn test_score_basic() {
        assert_eq!(get_score("b", "abc"), CHAR_SCORE);
        assert_eq!(get_score("c", "abc"), CHAR_SCORE);
    }

    #[test]
    fn test_score_prefix() {
        assert_eq!(get_score("a", "abc"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("a", "aabc"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("a", "babc"), CHAR_SCORE);
    }

    #[test]
    fn test_score_exact_match() {
        assert_eq!(get_score("a", "a"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("abc", "abc"), 3 * CHAR_SCORE + PREFIX_BONUS);
    }

    #[test]
    fn test_score_delimiter() {
        assert_eq!(get_score("-", "a--bc"), CHAR_SCORE);
        assert_eq!(get_score("b", "a-b"), CHAR_SCORE + DELIMITER_BONUS);
        assert_eq!(get_score("a", "a-b-c"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("b", "a--b"), CHAR_SCORE + DELIMITER_BONUS);
        assert_eq!(get_score("c", "a--bc"), CHAR_SCORE);
        assert_eq!(get_score("a", "-a--bc"), CHAR_SCORE + DELIMITER_BONUS);
    }

    #[test]
    fn test_score_no_delimiter_for_delimiter_chars() {
        assert_eq!(get_score("-", "a-bc"), CHAR_SCORE);
        assert_eq!(get_score("-", "a--bc"), CHAR_SCORE);
        assert!(get_score("a_b", "a_bb") > get_score("a_b", "a__b"));
    }

    #[test]
    fn test_score_affine_gap() {
        assert_eq!(
            get_score("test", "Uterst"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY
        );
        assert_eq!(
            get_score("test", "Uterrst"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY - GAP_EXTEND_PENALTY
        );
    }

    #[test]
    fn test_score_capital_bonus() {
        assert_eq!(get_score("a", "A"), MATCH_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("A", "Aa"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("D", "forDist"), CHAR_SCORE + CAPITALIZATION_BONUS);
        assert_eq!(get_score("D", "foRDist"), CHAR_SCORE);
        assert_eq!(get_score("D", "FOR_DIST"), CHAR_SCORE + DELIMITER_BONUS);
    }

    #[test]
    fn test_score_prefix_beats_delimiter() {
        assert!(get_score("swap", "swap(test)") > get_score("swap", "iter_swap(test)"));
        assert!(get_score("_", "_private_member") > get_score("_", "public_member"));
    }

    #[test]
    fn test_score_prefix_beats_capitalization() {
        assert!(get_score("H", "HELLO") > get_score("H", "fooHello"));
    }

    #[test]
    fn test_score_continuous_beats_delimiter() {
        assert!(get_score("foo", "fooo") > get_score("foo", "f_o_o_o"));
    }

    #[test]
    fn test_score_continuous_beats_capitalization() {
        assert!(get_score("fo", "foo") > get_score("fo", "faOo"));
    }
}
