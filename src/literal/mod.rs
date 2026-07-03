//! Literal matching: exact / prefix / suffix / substring
//!
//! Unlike the fuzzy [`crate::matcher`], literal matching requires the needle to appear as a
//! *contiguous* run of characters. The only step that benefits from SIMD is finding *where*
//! the needle occurs (substring search), which reuses [`crate::prefilter::backend::Backend`].
//!
//! All of the implementations ignore the `max_typos` parameter for now.

mod algo;
mod backend;
mod rank;

pub(crate) use backend::*;

#[cfg(test)]
mod tests {
    use crate::r#const::*;
    use crate::{CaseMatching, Config, Match, Matching, match_list, match_list_indices};

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn config(matching: Matching) -> Config {
        Config {
            matching,
            sort: false,
            ..Config::default()
        }
    }

    fn scores(matching: Matching, needle: &str, haystacks: &[&str]) -> Vec<(u32, u16, bool)> {
        match_list(needle, haystacks, &config(matching))
            .iter()
            .map(|m: &Match| (m.index, m.score, m.exact))
            .collect()
    }

    /// Score of the best-scoring substring occurrence of `needle` in `haystack`. Panics when the
    /// needle is absent, since the scoring tests always use a present needle.
    fn get_score(needle: &str, haystack: &str) -> u16 {
        get_score_case(needle, haystack, CaseMatching::Ignore).expect("needle should be present")
    }

    /// Best-scoring substring occurrence under the given casing, or `None` when the needle is absent.
    fn get_score_case(needle: &str, haystack: &str, casing: CaseMatching) -> Option<u16> {
        let config = Config {
            matching: Matching::Substring,
            casing,
            sort: false,
            ..Config::default()
        };
        match_list(needle, &[haystack], &config)
            .first()
            .map(|m| m.score)
    }

    #[test]
    fn exact_matches_whole_haystack_only() {
        let haystacks = ["foo", "foobar", "xfoo", "FOO"];
        let got = scores(Matching::Exact, "foo", &haystacks);
        // "foo" (exact, case match) and "FOO" (exact, case-insensitive) match; nothing else.
        assert_eq!(got.iter().map(|m| m.0).collect::<Vec<_>>(), vec![0, 3]);
        assert!(got.iter().all(|m| m.2), "all exact");
    }

    #[test]
    fn prefix_and_suffix() {
        let haystacks = ["foobar", "barfoo", "foo", "xfoobar"];
        assert_eq!(
            scores(Matching::Prefix, "foo", &haystacks)
                .iter()
                .map(|m| m.0)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(
            scores(Matching::Suffix, "foo", &haystacks)
                .iter()
                .map(|m| m.0)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn substring_finds_interior_matches() {
        let haystacks = ["xxbarxx", "bar", "nope", "foo_bar"];
        assert_eq!(
            scores(Matching::Substring, "bar", &haystacks)
                .iter()
                .map(|m| m.0)
                .collect::<Vec<_>>(),
            vec![0, 1, 3]
        );
    }

    #[test]
    fn exact_and_prefix_scores_match_fuzzy() {
        // For matches anchored at position 0, the literal score equals the fuzzy score.
        for (needle, haystack) in [
            ("foo", "foo"),
            ("foo", "foobar"),
            ("fooBar", "fooBarBaz"),
            ("a", "abc"),
        ] {
            let fuzzy = match_list(needle, &[haystack], &Config::default())[0].score;
            let prefix = match_list(needle, &[haystack], &config(Matching::Prefix))[0].score;
            assert_eq!(
                prefix, fuzzy,
                "prefix vs fuzzy mismatch for {needle:?} on {haystack:?}"
            );
        }

        let fuzzy = match_list("foo", &["foo"], &Config::default())[0].score;
        let exact = match_list("foo", &["foo"], &config(Matching::Exact))[0].score;
        assert_eq!(exact, fuzzy);
    }

    #[test]
    fn test_score_multibyte_needle() {
        // Interior "bar" in "foobar": three matched bytes, no prefix or delimiter.
        assert_eq!(get_score("bar", "foobar"), 3 * CHAR_SCORE);
        // "bar" after a delimiter earns the delimiter bonus on its first byte.
        assert_eq!(
            get_score("bar", "foo_bar"),
            3 * CHAR_SCORE + DELIMITER_BONUS
        );
    }

    #[test]
    fn substring_picks_best_scoring_occurrence() {
        // In "ab_ab" the position-0 occurrence (prefix bonus) must beat the one after '_'
        // (delimiter bonus).
        assert_eq!(get_score("ab", "ab_ab"), 2 * CHAR_SCORE + PREFIX_BONUS);
    }

    #[test]
    fn casing_respect_and_smart() {
        let haystacks = ["foo", "FOO", "fOo"];
        // Respect: only exact case.
        let respect = Config {
            matching: Matching::Prefix,
            casing: CaseMatching::Respect,
            sort: false,
            ..Config::default()
        };
        assert_eq!(
            match_list("foo", &haystacks, &respect)
                .iter()
                .map(|m| m.index)
                .collect::<Vec<_>>(),
            vec![0]
        );
        // Smart: lowercase needle => case-insensitive.
        assert_eq!(
            scores(Matching::Prefix, "foo", &haystacks)
                .iter()
                .map(|m| m.0)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn unicode_substring_and_exact() {
        let haystacks = ["é다😀", "xxé다😀yy", "é다", "plain"];
        assert_eq!(
            scores(Matching::Substring, "é다😀", &haystacks)
                .iter()
                .map(|m| m.0)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let exact = scores(Matching::Exact, "é다😀", &haystacks);
        assert_eq!(exact.iter().map(|m| m.0).collect::<Vec<_>>(), vec![0]);
        assert!(exact[0].2);
    }

    #[test]
    fn indices_are_contiguous_reversed() {
        let matches = match_list_indices("abc", &["xxabcxx"], &config(Matching::Substring));
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].indices, vec![4, 3, 2]);
    }

    #[test]
    fn no_match_when_needle_longer_than_haystack() {
        assert!(scores(Matching::Substring, "abcd", &["abc"]).is_empty());
        assert!(scores(Matching::Prefix, "abcd", &["abc"]).is_empty());
        assert!(scores(Matching::Suffix, "abcd", &["abc"]).is_empty());
        assert!(scores(Matching::Exact, "abcd", &["abc"]).is_empty());
    }

    #[test]
    fn substring_scan_handles_chunk_boundaries() {
        // Exercise occurrences straddling the SIMD lane boundaries of every backend.
        for prefix_len in [0usize, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65] {
            let haystack = format!("{}bar", "x".repeat(prefix_len));
            let got = scores(Matching::Substring, "bar", &[&haystack]);
            assert_eq!(got.len(), 1, "prefix_len={prefix_len}");
            assert_eq!(got[0].0, 0);
        }
    }

    // The tests below mirror the Smith-Waterman scoring suite (`src/smith_waterman/mod.rs`),
    // adapted to literal matching: matches are contiguous (so the gap/affine/typo cases do not
    // apply) and the exact-match bonus is included when the run spans the whole haystack.

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
        // Unlike the raw Smith-Waterman scorer, the literal scorer adds the exact-match bonus when
        // the run covers the whole haystack.
        assert_eq!(
            get_score("a", "a"),
            CHAR_SCORE + PREFIX_BONUS + EXACT_MATCH_BONUS
        );
        assert_eq!(
            get_score("abc", "abc"),
            3 * CHAR_SCORE + PREFIX_BONUS + EXACT_MATCH_BONUS
        );
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
    }

    #[test]
    fn test_score_capital_bonus() {
        assert_eq!(get_score("a", "Ab"), MATCH_SCORE + PREFIX_BONUS);
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
    fn bonus_precedence_manual_cases() {
        assert!(get_score("b", "b") > get_score("b", "a-b"));
        assert!(get_score("b", "a-b") > get_score("b", "ab"));
        assert!(get_score("B", "aB") > get_score("b", "aB"));
    }

    #[test]
    fn case_sensitive_scoring_rejects_folded_bytes() {
        // A digit prefix keeps the seed byte free of capitalization/delimiter bonuses.
        assert_eq!(
            get_score_case("A", "0A", CaseMatching::Respect),
            Some(CHAR_SCORE)
        );
        assert_eq!(get_score_case("A", "0a", CaseMatching::Respect), None);
        assert_eq!(
            get_score_case("A", "0a", CaseMatching::Ignore),
            Some(MATCH_SCORE)
        );
    }

    // Unicode matching: a non-ASCII needle takes the per-codepoint path. Multibyte characters are
    // scored once (not once per UTF-8 byte) and case folding compares whole codepoints.

    #[test]
    fn test_score_unicode_per_codepoint() {
        // "é" is two UTF-8 bytes but scores as a single matched character.
        assert_eq!(
            get_score("é", "é"),
            CHAR_SCORE + PREFIX_BONUS + EXACT_MATCH_BONUS
        );
        // Two codepoints (é is 2 bytes, x is 1) score as two characters, not three bytes.
        assert_eq!(
            get_score("éx", "éx"),
            2 * CHAR_SCORE + PREFIX_BONUS + EXACT_MATCH_BONUS
        );
        // An interior occurrence: only "é" is scored, the leading "x" is not.
        assert_eq!(get_score("é", "xé"), CHAR_SCORE);
    }

    #[test]
    fn unicode_case_insensitive_fold() {
        // Whole-codepoint case folding across several scripts (é/É, Cyrillic и/И, Greek α/Α).
        for (needle, upper) in [("é", "É"), ("и", "И"), ("α", "Α")] {
            assert!(
                get_score_case(needle, upper, CaseMatching::Ignore).is_some(),
                "{needle:?} should fold to {upper:?}"
            );
            assert_eq!(
                get_score_case(needle, upper, CaseMatching::Respect),
                None,
                "{needle:?} must not match {upper:?} when respecting case"
            );
        }
    }

    #[test]
    fn unicode_rejects_hybrid_case_bytes() {
        // Cherokee case pairs are the same length but differ in every byte:
        //   'Ꭰ' U+13A0 = E1 8E A0, 'ꭰ' U+AB70 = EA AD B0.
        // A per-byte verifier would accept the hybrid E1 AD B0 (= U+1B70 '᭰'); the codepoint
        // verifier must reject it while still matching the true lowercase form.
        assert_eq!(
            get_score_case("Ꭰ", "\u{1b70}", CaseMatching::Ignore),
            None,
            "hybrid byte sequence must not match"
        );
        assert!(
            get_score_case("Ꭰ", "ꭰ", CaseMatching::Ignore).is_some(),
            "true lowercase form must match"
        );
    }

    #[test]
    fn unicode_length_changing_fold_is_case_sensitive() {
        // 'ß' folds to "SS" (a length change), so it is treated case-sensitively: it matches only
        // itself, never "SS"/"ss".
        assert!(get_score_case("ß", "ß", CaseMatching::Ignore).is_some());
        assert_eq!(get_score_case("ß", "SS", CaseMatching::Ignore), None);
        assert_eq!(get_score_case("ß", "ss", CaseMatching::Ignore), None);
    }

    #[test]
    fn unicode_indices_span_whole_utf8_run() {
        // Matched indices are byte offsets covering the full UTF-8 run, reversed.
        let matches = match_list_indices("é다", &["xxé다yy"], &config(Matching::Substring));
        assert_eq!(matches.len(), 1);
        // "xx" is two bytes; "é" occupies bytes 2..4, "다" bytes 4..7.
        assert_eq!(matches[0].indices, vec![6, 5, 4, 3, 2]);
    }
}
