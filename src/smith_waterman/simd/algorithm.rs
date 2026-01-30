use multiversion::multiversion;
use std::ops::Not;
use std::simd::cmp::*;
use std::simd::num::SimdUint;
use std::simd::{Mask, Select, Simd};

use super::{HaystackChar, NeedleChar, interleave};
use crate::Scoring;

#[inline(always)]
pub(crate) fn smith_waterman_inner<const L: usize>(
    start: usize,
    end: usize,
    needle_char: NeedleChar<L>,
    haystack: &[HaystackChar<L>],
    prev_score_col: Option<&[Simd<u16, L>]>,
    curr_score_col: &mut [Simd<u16, L>],
    scoring: &Scoring,
) {
    let mut up_score_simd = Simd::splat(0);
    let mut up_gap_penalty_mask = Mask::splat(true);
    let mut left_gap_penalty_mask = Mask::splat(true);
    let mut delimiter_bonus_enabled_mask = Mask::splat(false);

    for haystack_idx in start..end {
        let haystack_char = haystack[haystack_idx];

        let (diag, left) = if haystack_idx == 0 {
            (Simd::splat(0), Simd::splat(0))
        } else {
            prev_score_col
                .map(|c| (c[haystack_idx - 1], c[haystack_idx]))
                .unwrap_or((Simd::splat(0), Simd::splat(0)))
        };

        // Calculate diagonal (match/mismatch) scores
        let match_mask: Mask<i16, L> = needle_char.lowercase.simd_eq(haystack_char.lowercase);
        let matched_casing_mask: Mask<i16, L> = needle_char
            .is_capital_mask
            .simd_eq(haystack_char.is_capital_mask);

        let match_score = if haystack_idx > 0 {
            let match_score = {
                let prev_haystack_char = haystack[haystack_idx - 1];

                // ignore capitalization on the prefix
                let capitalization_bonus_mask: Mask<i16, L> =
                    haystack_char.is_capital_mask & prev_haystack_char.is_lower_mask;
                let capitalization_bonus = capitalization_bonus_mask
                    .select(Simd::splat(scoring.capitalization_bonus), Simd::splat(0));

                let delimiter_bonus_mask: Mask<i16, L> = prev_haystack_char.is_delimiter_mask
                    & delimiter_bonus_enabled_mask
                    & !haystack_char.is_delimiter_mask;
                let delimiter_bonus = delimiter_bonus_mask
                    .select(Simd::splat(scoring.delimiter_bonus), Simd::splat(0));

                capitalization_bonus + delimiter_bonus + Simd::splat(scoring.match_score)
            };

            if haystack_idx == 1 {
                // If the first char is not a letter, apply the offset prefix bonus on the second char
                // if we didn't match on the first char
                // I.e. `a` matching on `-a` would get the offset prefix bonus
                // but  `b` matching on `ab` would not get the offset prefix bonus
                let offset_prefix_mask = !(haystack[0].is_lower_mask | haystack[0].is_capital_mask)
                    & diag.simd_eq(Simd::splat(0));

                offset_prefix_mask.select(
                    Simd::splat(scoring.offset_prefix_bonus + scoring.match_score),
                    match_score,
                )
            } else {
                match_score
            }
        } else {
            // Give a bonus for prefix matches
            Simd::splat(scoring.prefix_bonus + scoring.match_score)
        };

        let diag_score = match_mask.select(
            diag + matched_casing_mask
                .select(Simd::splat(scoring.matching_case_bonus), Simd::splat(0))
                + match_score,
            diag.saturating_sub(Simd::splat(scoring.mismatch_penalty)),
        );

        // Load and calculate up scores (skipping char in haystack)
        let up_gap_penalty = up_gap_penalty_mask.select(
            Simd::splat(scoring.gap_open_penalty),
            Simd::splat(scoring.gap_extend_penalty),
        );
        let up_score = up_score_simd.saturating_sub(up_gap_penalty);

        // Load and calculate left scores (skipping char in needle)
        let left_gap_penalty = left_gap_penalty_mask.select(
            Simd::splat(scoring.gap_open_penalty),
            Simd::splat(scoring.gap_extend_penalty),
        );
        let left_score = left.saturating_sub(left_gap_penalty);

        // Calculate maximum scores
        let max_score = diag_score.simd_max(up_score).simd_max(left_score);

        // Update gap penalty mask
        let diag_mask: Mask<i16, L> = max_score.simd_eq(diag_score);
        up_gap_penalty_mask = max_score.simd_ne(up_score) | diag_mask;
        left_gap_penalty_mask = max_score.simd_ne(left_score) | diag_mask;

        // Only enable delimiter bonus if we've seen a non-delimiter char
        delimiter_bonus_enabled_mask |= haystack_char.is_delimiter_mask.not();

        // Store the scores for the next iterations
        up_score_simd = max_score;
        curr_score_col[haystack_idx] = max_score;
    }
}

#[multiversion(targets(
    // x86-64-v4 without lahfsahf
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    // x86-64-v3 without lahfsahf
    "x86_64+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    // x86-64-v2 without lahfsahf
    "x86_64+cmpxchg16b+fxsr+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3",
))]
pub fn smith_waterman<const W: usize, const L: usize>(
    needle_str: &str,
    haystack_strs: &[&str; L],
    max_typos: Option<u16>,
    scoring: &Scoring,
) -> ([u16; L], Vec<[Simd<u16, L>; W]>, [bool; L]) {
    let needle = needle_str.as_bytes();
    let haystacks = interleave::<W, L>(*haystack_strs).map(HaystackChar::new);

    // State
    let mut score_matrix = vec![[Simd::splat(0); W]; needle.len()];

    for (needle_idx, haystack_start, haystack_end) in (0..needle.len()).map(|needle_idx| {
        // When matching "asd" against "qwerty" with max_typos = 0, we can avoid matching "s"
        // against the "q" since it's impossible for this to be a valid match
        // And likewise, we avoid matching "d" against "q" and "w"
        let haystack_start = max_typos
            .map(|max_typos| needle_idx.saturating_sub(max_typos as usize))
            .unwrap_or(0);
        // When matching "foo" against "foobar" with max_typos = 0, we can avoid matching "f"
        // againt "a" and "r" since it's impossible for this to be a valid match
        let haystack_end = max_typos
            .map(|max_typos| {
                (W + needle_idx + (max_typos as usize))
                    .saturating_sub(needle.len())
                    .min(W)
            })
            .unwrap_or(W);
        (needle_idx, haystack_start, haystack_end)
    }) {
        let needle_char = NeedleChar::new(needle[needle_idx] as u16);

        let (prev_score_col, curr_score_col) = if needle_idx == 0 {
            (None, &mut score_matrix[needle_idx])
        } else {
            let (a, b) = score_matrix.split_at_mut(needle_idx);
            (Some(a[needle_idx - 1].as_slice()), &mut b[0])
        };

        smith_waterman_inner(
            haystack_start,
            haystack_end,
            needle_char,
            &haystacks,
            prev_score_col,
            curr_score_col,
            scoring,
        );
    }

    let mut all_time_max_score = Simd::splat(0);
    for score_col in score_matrix.iter() {
        for score in score_col {
            all_time_max_score = score.simd_max(all_time_max_score);
        }
    }

    let exact_matches: [bool; L] = std::array::from_fn(|i| haystack_strs[i] == needle_str);

    let max_scores = std::array::from_fn(|i| {
        let mut score = all_time_max_score[i];
        if exact_matches[i] {
            score += scoring.exact_match_bonus;
        }
        score
    });

    (max_scores, score_matrix, exact_matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#const::*;

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn get_score(needle: &str, haystack: &str) -> u16 {
        smith_waterman::<16, 1>(needle, &[haystack], None, &Scoring::default()).0[0]
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
