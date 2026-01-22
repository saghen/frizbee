use crate::r#const::*;

pub fn smith_waterman(needle: &str, haystack: &str) -> (u16, Vec<Vec<u16>>) {
    let needle = needle.as_bytes();
    let haystack = haystack.as_bytes();

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
        let mut prev_haystack_is_delimiter = false;
        let mut prev_haystack_is_lowercase = false;

        for j in 0..haystack.len() {
            let is_prefix = j == 0;

            // Load chunk and remove casing
            let haystack_char = haystack[j];
            let haystack_is_uppercase = haystack_char.is_ascii_uppercase();
            let haystack_is_lowercase = haystack_char.is_ascii_lowercase();
            let haystack_char = haystack_char.to_ascii_lowercase();

            let haystack_is_delimiter =
                [b' ', b'/', b'.', b',', b'_', b'-', b':'].contains(&haystack_char);
            let matched_casing_mask = needle_is_uppercase == haystack_is_uppercase;

            // Give a bonus for prefix matches
            let match_score = if is_prefix {
                MATCH_SCORE + PREFIX_BONUS
            } else {
                MATCH_SCORE
            };

            // Calculate diagonal (match/mismatch) scores
            let diag = if is_prefix { 0 } else { prev_col_scores[j - 1] };
            let is_match = needle_char == haystack_char;
            let diag_score = if is_match {
                diag + match_score
                    + if prev_haystack_is_delimiter && !haystack_is_delimiter { DELIMITER_BONUS } else { 0 }
                    // ignore capitalization on the prefix
                    + if !is_prefix && haystack_is_uppercase && prev_haystack_is_lowercase { CAPITALIZATION_BONUS } else { 0 }
                    + if matched_casing_mask { MATCHING_CASE_BONUS } else { 0 }
            } else {
                diag.saturating_sub(MISMATCH_PENALTY)
            };

            // Load and calculate up scores (skipping char in haystack)
            let up_gap_penalty = if up_gap_penalty_mask {
                GAP_OPEN_PENALTY
            } else {
                GAP_EXTEND_PENALTY
            };
            let up_score = up_score_simd.saturating_sub(up_gap_penalty);

            // Load and calculate left scores (skipping char in needle)
            let left = prev_col_scores[j];
            let left_gap_penalty = if left_gap_penalty_mask {
                GAP_OPEN_PENALTY
            } else {
                GAP_EXTEND_PENALTY
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

            // Store the scores for the next iterations
            up_score_simd = max_score;
            curr_col_scores[j] = max_score;

            // Store the maximum score across all runs
            all_time_max_score = all_time_max_score.max(max_score);
        }
    }

    (all_time_max_score, score_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn get_score(needle: &str, haystack: &str) -> u16 {
        smith_waterman(needle, haystack).0
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
}
