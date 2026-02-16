use crate::Scoring;

pub fn smith_waterman(
    needle: &str,
    haystack: &str,
    scoring: &Scoring,
) -> (u16, Vec<Vec<u16>>, bool) {
    let needle = needle.as_bytes();
    let haystack = haystack.as_bytes();

    let delimiter_bytes: Vec<u8> = scoring.delimiters.bytes().collect();

    // State
    let mut score_matrix = vec![vec![0; haystack.len()]; needle.len()];
    let mut all_time_max_score = 0;

    for i in 0..needle.len() {
        let (prev_col_scores, curr_col_scores) = if i > 0 {
            let (prev_col_scores_slice, curr_col_scores_slice) = score_matrix.split_at_mut(i);
            (&prev_col_scores_slice[i - 1], &mut curr_col_scores_slice[0])
        } else {
            (&vec![0; haystack.len()], &mut score_matrix[i])
        };

        let mut up_score_simd: u16 = 0;
        let mut up_gap_penalty_mask = true;

        let needle_char = needle[i];
        let needle_is_uppercase = needle_char.is_ascii_uppercase();
        let needle_char = needle_char.to_ascii_lowercase();

        let mut left_gap_penalty_mask = true;
        let mut delimiter_bonus_enabled = false;
        let mut prev_haystack_is_delimiter = false;
        let mut prev_haystack_is_lowercase = false;

        for j in 0..haystack.len() {
            let is_prefix = j == 0;
            let is_offset_prefix =
                j == 1 && prev_col_scores[0] == 0 && !haystack[0].is_ascii_alphabetic();

            // Load chunk and remove casing
            let haystack_char = haystack[j];
            let haystack_is_uppercase = haystack_char.is_ascii_uppercase();
            let haystack_is_lowercase = haystack_char.is_ascii_lowercase();
            let haystack_char = haystack_char.to_ascii_lowercase();

            let haystack_is_delimiter = delimiter_bytes.contains(&haystack_char);
            let matched_casing_mask = needle_is_uppercase == haystack_is_uppercase;

            // Give a bonus for prefix matches
            let match_score = if is_prefix {
                scoring.match_score + scoring.prefix_bonus
            } else if is_offset_prefix {
                scoring.match_score + scoring.offset_prefix_bonus
            } else {
                scoring.match_score
            };

            // Calculate diagonal (match/mismatch) scores
            let diag = if is_prefix { 0 } else { prev_col_scores[j - 1] };
            let is_match = needle_char == haystack_char;
            let diag_score = if is_match {
                diag + match_score
                    + if prev_haystack_is_delimiter && delimiter_bonus_enabled && !haystack_is_delimiter { scoring.delimiter_bonus } else { 0 }
                    // ignore capitalization on the prefix
                    + if !is_prefix && haystack_is_uppercase && prev_haystack_is_lowercase { scoring.capitalization_bonus } else { 0 }
                    + if matched_casing_mask { scoring.matching_case_bonus } else { 0 }
            } else {
                diag.saturating_sub(scoring.mismatch_penalty)
            };

            // Load and calculate up scores (skipping char in haystack)
            let up_gap_penalty = if up_gap_penalty_mask {
                scoring.gap_open_penalty
            } else {
                scoring.gap_extend_penalty
            };
            let up_score = up_score_simd.saturating_sub(up_gap_penalty);

            // Load and calculate left scores (skipping char in needle)
            let left = prev_col_scores[j];
            let left_gap_penalty = if left_gap_penalty_mask {
                scoring.gap_open_penalty
            } else {
                scoring.gap_extend_penalty
            };
            let left_score = left.saturating_sub(left_gap_penalty);

            // Calculate maximum scores
            let max_score = diag_score.max(up_score).max(left_score);

            // Update gap penalty mask
            let diag_mask = max_score == diag_score;
            up_gap_penalty_mask = max_score != up_score || diag_mask;
            left_gap_penalty_mask = max_score != left_score || diag_mask;

            // Update haystack char masks
            prev_haystack_is_lowercase = haystack_is_lowercase;
            prev_haystack_is_delimiter = haystack_is_delimiter;
            // Only enable delimiter bonus if we've seen a non-delimiter char
            delimiter_bonus_enabled |= !prev_haystack_is_delimiter;

            // Store the scores for the next iterations
            up_score_simd = max_score;
            curr_col_scores[j] = max_score;

            // Store the maximum score across all runs
            all_time_max_score = all_time_max_score.max(max_score);
        }
    }

    let mut max_score = all_time_max_score;
    let exact = haystack == needle;
    if exact {
        max_score += scoring.exact_match_bonus;
    }

    (max_score, score_matrix, exact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Scoring, r#const::*, smith_waterman::simd::smith_waterman as smith_waterman_simd};

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn get_score(needle: &str, haystack: &str) -> u16 {
        let scoring = Scoring::default();
        let ref_score = smith_waterman(needle, haystack, &scoring).0;
        let simd_score = smith_waterman_simd::<16, 1>(needle, &[haystack], None, &scoring).0[0];

        assert_eq!(
            ref_score, simd_score,
            "Reference and SIMD scores don't match"
        );

        ref_score
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
    fn test_score_offset_prefix() {
        // Give prefix bonus on second char if the first char isn't a letter
        assert_eq!(get_score("a", "-a"), CHAR_SCORE + OFFSET_PREFIX_BONUS);
        assert_eq!(get_score("-a", "-ab"), 2 * CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("a", "'a"), CHAR_SCORE + OFFSET_PREFIX_BONUS);
        assert_eq!(get_score("a", "Ba"), CHAR_SCORE);
    }

    #[test]
    fn test_score_exact_match() {
        assert_eq!(
            get_score("a", "a"),
            CHAR_SCORE + EXACT_MATCH_BONUS + PREFIX_BONUS
        );
        assert_eq!(
            get_score("abc", "abc"),
            3 * CHAR_SCORE + EXACT_MATCH_BONUS + PREFIX_BONUS
        );
        assert_eq!(get_score("ab", "abc"), 2 * CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("abc", "ab"), 2 * CHAR_SCORE + PREFIX_BONUS);
    }

    #[test]
    fn test_score_delimiter() {
        assert_eq!(get_score("-", "a--bc"), CHAR_SCORE);
        assert_eq!(get_score("b", "a-b"), CHAR_SCORE + DELIMITER_BONUS);
        assert_eq!(get_score("a", "a-b-c"), CHAR_SCORE + PREFIX_BONUS);
        assert_eq!(get_score("b", "a--b"), CHAR_SCORE + DELIMITER_BONUS);
        assert_eq!(get_score("c", "a--bc"), CHAR_SCORE);
        assert_eq!(get_score("a", "-a--bc"), CHAR_SCORE + OFFSET_PREFIX_BONUS);
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

    /// Verifies that scattered character matches across long haystacks produce
    /// lower scores than contiguous ones with default gap penalties.
    #[test]
    fn test_scattered_match_should_score_low() {
        let scoring = Scoring::default();
        // "SortedMap" vs a good match (contiguous subsequence)
        let good_score = smith_waterman("SortedMap", "SortedArrayMap", &scoring).0;
        // "SortedMap" vs a bad match (scattered characters across unrelated words)
        let bad_score = smith_waterman("SortedMap", "LightSourceTeamApiKeys", &scoring).0;

        // The good match should score higher than the scattered one
        assert!(
            good_score > bad_score,
            "SortedArrayMap (score={}) should score higher than \
             LightSourceTeamApiKeys (score={}) for needle 'SortedMap'",
            good_score,
            bad_score
        );
    }

    /// Verifies that high gap penalties in custom Scoring cause scattered matches
    /// to be heavily penalized in the reference path (used by match_indices).
    /// The good (nearly contiguous) match should score significantly better.
    #[test]
    fn test_high_gap_penalties_reject_scattered_matches() {
        let strict_scoring = Scoring {
            gap_open_penalty: 100,
            gap_extend_penalty: 80,
            ..Scoring::default()
        };

        // Good match — nearly contiguous (only one gap for "Array" insertion)
        let good_score = smith_waterman("SortedMap", "SortedArrayMap", &strict_scoring).0;
        // Bad match — scattered across unrelated words
        let bad_score = smith_waterman("SortedMap", "LightSourceTeamApiKeys", &strict_scoring).0;

        assert!(
            good_score > 0,
            "Good match should still have a positive score, got {}",
            good_score
        );
        // With high gap penalties, good match should beat the scattered one
        assert!(
            good_score > bad_score,
            "Good match (score={}) should score higher than \
             scattered match (score={}) with high gap penalties",
            good_score,
            bad_score
        );

        // Verify the gap penalties actually change behavior compared to defaults
        let default_scoring = Scoring::default();
        let default_bad_score =
            smith_waterman("SortedMap", "LightSourceTeamApiKeys", &default_scoring).0;
        assert!(
            default_bad_score > bad_score,
            "Default scoring bad_score ({}) should be higher than \
             strict scoring bad_score ({}) — confirms gap penalties take effect",
            default_bad_score,
            bad_score
        );
    }

    /// Verifies that with very high gap penalties, the algorithm is forced to pick
    /// shorter contiguous matches rather than longer scattered ones.
    /// This confirms the Scoring parameter is actually being used.
    #[test]
    fn test_gap_penalties_affect_reference_scoring() {
        let default_scoring = Scoring::default();
        let strict_scoring = Scoring {
            gap_open_penalty: 100,
            gap_extend_penalty: 80,
            ..Scoring::default()
        };

        // "test" vs "Uterrrrrst" — has a gap of 5 extra 'r's
        let default_score = smith_waterman("test", "Uterrrrrst", &default_scoring).0;
        let strict_score = smith_waterman("test", "Uterrrrrst", &strict_scoring).0;

        // With strict gap penalties, the score should drop significantly
        assert!(
            default_score > strict_score,
            "Default score ({}) should be higher than strict score ({}) for gapped match",
            default_score,
            strict_score,
        );
    }
}
