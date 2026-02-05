use std::arch::x86_64::*;

use crate::{Scoring, prefilter::case_needle};

mod gaps;
mod ops;
mod typos;

use gaps::propagate_horizontal_gaps;
use ops::*;
pub use typos::typos_from_score_matrix;

const MAX_HAYSTACK_LEN: usize = 512;

#[derive(Debug, Clone)]
pub struct SmithWatermanMatcher<const ALIGNED: bool> {
    #[cfg(test)]
    pub needle: String,
    pub needle_simd: Vec<(__m128i, __m128i)>,
    pub scoring: Scoring,
}

impl<const ALIGNED: bool> SmithWatermanMatcher<ALIGNED> {
    pub fn new(needle: &[u8], scoring: &Scoring) -> Self {
        Self {
            #[cfg(test)]
            needle: String::from_utf8_lossy(needle).to_string(),
            needle_simd: Self::broadcast_needle(needle),
            scoring: scoring.clone(),
        }
    }

    fn broadcast_needle(needle: &[u8]) -> Vec<(__m128i, __m128i)> {
        let needle_cased = case_needle(needle);
        needle_cased
            .iter()
            .map(|(c1, c2)| unsafe { (_mm_set1_epi8(*c1 as i8), _mm_set1_epi8(*c2 as i8)) })
            .collect()
    }

    pub fn generate_score_matrix(needle_len: usize, haystack_len: usize) -> Vec<__m256i> {
        (0..((needle_len + 1) * (haystack_len.div_ceil(16) + 1)))
            .map(|_| unsafe { _mm256_setzero_si256() })
            .collect::<Vec<_>>()
    }

    pub fn generate_generic_score_matrix(needle_len: usize) -> Vec<__m256i> {
        Self::generate_score_matrix(needle_len, MAX_HAYSTACK_LEN)
    }

    pub fn match_haystack(
        &mut self,
        haystack: &[u8],
        max_typos: Option<u16>,
        score_matrix: &mut [__m256i],
    ) -> Option<u16> {
        let score = self.score_haystack(haystack, score_matrix);
        if let Some(max_typos) = max_typos {
            let typos = typos_from_score_matrix(score_matrix, score, max_typos, haystack.len());
            if typos > max_typos {
                return None;
            }
        }
        Some(score)
    }

    pub fn score_haystack(&self, haystack: &[u8], score_matrix: &mut [__m256i]) -> u16 {
        let haystack_chunks = haystack.len().div_ceil(16);
        let scoring = &self.scoring;
        unsafe {
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
            let capitalization_bonus = _mm256_set1_epi16(scoring.capitalization_bonus as i16);
            let delimiter_bonus = _mm256_set1_epi16(scoring.delimiter_bonus as i16);

            // State
            let mut prev_chunk_char_is_delimiter_mask = _mm_setzero_si128();
            let mut prev_chunk_is_lower_mask = _mm_setzero_si128();
            let mut prefix_mask = PREFIX_MASK.as_m256i();
            let mut max_scores = _mm256_setzero_si256();

            for mut col_idx in 0..haystack_chunks {
                let haystack = if ALIGNED {
                    _mm_load_si128(haystack.as_ptr().add(col_idx * 16) as *const __m128i)
                } else {
                    _mm_loadu(haystack, col_idx * 16, haystack.len())
                };
                col_idx += 1;

                // Bonus for matching a capital letter after a lowercase letter
                let is_upper_mask = _mm_and_si128(
                    _mm_cmplt_epi8(haystack, _mm_set1_epi8((b'Z' + 1) as i8)),
                    _mm_cmpgt_epi8(haystack, _mm_set1_epi8((b'A' - 1) as i8)),
                );
                let is_lower_mask = _mm_and_si128(
                    _mm_cmplt_epi8(haystack, _mm_set1_epi8((b'z' + 1) as i8)),
                    _mm_cmpgt_epi8(haystack, _mm_set1_epi8((b'a' - 1) as i8)),
                );
                let is_letter_mask = _mm_or_si128(is_upper_mask, is_lower_mask);
                let capitalization_mask = _mm_and_si128(
                    is_upper_mask,
                    _mm_alignr_epi8::<15>(is_lower_mask, prev_chunk_is_lower_mask),
                );
                let capitalization_bonus_masked = _mm256_and_si256(
                    _mm256_cvtepi8_epi16(capitalization_mask),
                    capitalization_bonus,
                );
                prev_chunk_is_lower_mask = is_lower_mask;

                // Bonus for matching after a delimiter character
                // We consider anything that isn't a digit or a letter, and within ASCII range, to
                // be a delimiter
                let is_digit_mask = _mm_and_si128(
                    _mm_cmpgt_epi8(haystack, _mm_set1_epi8((b'0' - 1) as i8)),
                    _mm_cmplt_epi8(haystack, _mm_set1_epi8((b'9' + 1) as i8)),
                );
                let char_is_delimiter_mask = _mm_not_si128(_mm_or_si128(
                    is_letter_mask,
                    _mm_or_si128(is_digit_mask, _mm_cmpgt_epi8(haystack, _mm_set1_epi8(127))),
                ));
                let prev_char_is_delimiter_mask = _mm_alignr_epi8::<15>(
                    char_is_delimiter_mask,
                    prev_chunk_char_is_delimiter_mask,
                );
                let delimiter_mask = _mm_and_si128(
                    prev_char_is_delimiter_mask,
                    _mm_not_si128(char_is_delimiter_mask),
                );
                let delimiter_bonus_masked =
                    _mm256_and_si256(_mm256_cvtepi8_epi16(delimiter_mask), delimiter_bonus);
                prev_chunk_char_is_delimiter_mask = char_is_delimiter_mask;

                let mut up_gap_mask = _mm256_setzero_si256();
                let mut prev_row_scores = score_matrix[col_idx];
                let mut row_scores = _mm256_setzero_si256();

                for (row_idx, (needle_char, flipped_case_needle_char)) in
                    self.needle_simd.iter().enumerate().map(|(i, c)| (i + 1, c))
                {
                    // Match needle chars against the haystack (case insensitive)
                    let exact_case_match_mask = _mm_cmpeq_epi8(*needle_char, haystack);
                    let flipped_case_match_mask =
                        _mm_cmpeq_epi8(*flipped_case_needle_char, haystack);
                    let match_mask = _mm_or_si128(exact_case_match_mask, flipped_case_match_mask);
                    let exact_case_match_mask = _mm256_cvtepi8_epi16(exact_case_match_mask); // 16xu8 -> 16xu16
                    let match_mask = _mm256_cvtepi8_epi16(match_mask); // 16xu8 -> 16xu16

                    // Diagonal - typical match/mismatch, moving along one haystack and needle char
                    let diag_scores = {
                        let diag = _mm256_shift_right_padded_epi16(
                            prev_row_scores,
                            score_matrix[(row_idx - 1) * haystack_chunks + col_idx - 1],
                        );

                        // Add match score (+ mismatch penalty) for matches, avoiding blendv
                        let diag =
                            _mm256_add_epi16(diag, _mm256_and_si256(match_mask, match_score));
                        // Always add mismatch penalty
                        let diag = _mm256_subs_epu16(diag, mismatch_penalty);
                        // Add prefix bonus
                        let diag = _mm256_add_epi16(
                            diag,
                            _mm256_and_si256(
                                _mm256_and_si256(prefix_mask, match_mask),
                                prefix_bonus,
                            ),
                        );
                        // Add delimiter bonus
                        let diag = _mm256_add_epi16(
                            diag,
                            _mm256_and_si256(match_mask, delimiter_bonus_masked),
                        );
                        // Add capitalization bonus
                        let diag = _mm256_add_epi16(
                            diag,
                            _mm256_and_si256(match_mask, capitalization_bonus_masked),
                        );
                        // Add matching case bonus
                        _mm256_add_epi16(
                            diag,
                            _mm256_and_si256(exact_case_match_mask, matching_case_bonus),
                        )
                    };

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

                    // Max of diagonal, up and left (after gap extension)
                    row_scores = propagate_horizontal_gaps(
                        score_matrix[row_idx * haystack_chunks + col_idx - 1], // Left
                        _mm256_max_epu16(diag_scores, up_scores),              // Current
                        match_mask,
                        scoring.gap_open_penalty,
                        scoring.gap_extend_penalty,
                    );

                    // Store results
                    score_matrix[row_idx * haystack_chunks + col_idx] = row_scores;
                    prev_row_scores = row_scores;
                    up_gap_mask = match_mask;
                }

                // because we do this after the loop, we're guaranteed to be on the last row
                max_scores = _mm256_max_epu16(max_scores, row_scores);
                prefix_mask = _mm256_setzero_si256();
            }

            _mm256_smax_epu16(max_scores)
        }
    }

    #[cfg(test)]
    pub fn print_score_matrix(&self, haystack: &str, score_matrix: &[__m256i]) {
        let haystack_chunks = haystack.len().div_ceil(16) + 1;
        let score_matrix = unsafe { std::mem::transmute::<&[__m256i], &[[u16; 16]]>(score_matrix) };

        print!("     ");
        for char in haystack.chars() {
            print!("{:<4} ", char);
        }
        println!();

        for (i, row) in score_matrix
            .chunks_exact(haystack_chunks)
            .enumerate()
            .skip(1)
        {
            print!("{:<4} ", self.needle.chars().nth(i - 1).unwrap_or(' '));
            for col in row.iter().skip(1).flatten() {
                print!("{:<4} ", col);
            }
            println!();
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#const::*;

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn get_score(needle: &str, haystack: &str) -> u16 {
        let mut score_matrix =
            SmithWatermanMatcher::<false>::generate_score_matrix(needle.len(), haystack.len());
        let matcher = SmithWatermanMatcher::<false>::new(needle.as_bytes(), &Scoring::default());
        let score = matcher.score_haystack(haystack.as_bytes(), &mut score_matrix);
        matcher.print_score_matrix(haystack, &score_matrix);
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
