use crate::Scoring;
use backend::Backend;
use matrix::Matrix;

mod algo;
mod alignment;
mod alignment_iter;
pub(crate) mod backend;
mod matrix;

pub use alignment_iter::{Alignment, AlignmentPathIter};

/// Returns true if every possible Smith-Waterman matrix cell value for this
/// needle length and scoring config fits in a u8. The u8 backends are
/// otherwise identical to the u16 backends but with double the lane count
/// (64 cells/chunk on AVX-512, 32 on AVX2, 16 on SSE/NEON).
#[inline]
pub(crate) fn score_fits_in_u8(needle_len: usize, scoring: &Scoring) -> bool {
    let max_per_char = scoring.match_score as usize
        + scoring.matching_case_bonus as usize
        + scoring
            .delimiter_bonus
            .max(scoring.capitalization_bonus)
            .saturating_sub(scoring.gap_open_penalty) as usize;
    let max_matrix_score = max_per_char * needle_len + scoring.prefix_bonus as usize;
    max_matrix_score <= u8::MAX as usize
}

#[derive(Debug, Clone)]
pub(crate) struct SmithWaterman<B: Backend> {
    needle: String,
    needle_simd: Vec<(B::Bytes, B::Bytes)>,
    scoring: Scoring,
    score_matrix: Matrix<B>,
    match_masks: Matrix<B>,
    /// Number of LANES-wide chunks (incl. the leading zero column) actually
    /// consumed by the most recent `score_haystack` call. The matrix stride is
    /// always sized for `MAX_HAYSTACK_LEN` for zero-free reuse.
    haystack_chunks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#const::*;
    #[cfg(target_arch = "x86_64")]
    use crate::smith_waterman::simd::backend::{
        Avx512Backend, Avx512U8Backend, AvxBackend, AvxU8Backend, SseBackend, SseU8Backend,
    };
    use crate::smith_waterman::simd::backend::{Backend, Scalar8Backend, Scalar16U8Backend};

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn get_score(needle: &str, haystack: &str) -> u16 {
        let mut matcher =
            SmithWaterman::<Scalar8Backend>::new(needle.as_bytes(), &Scoring::default());
        let score = matcher.match_haystack(haystack.as_bytes(), Some(0));
        score.unwrap()
    }

    fn get_score_typos(needle: &str, haystack: &str, max_typos: u16) -> Option<u16> {
        let mut matcher =
            SmithWaterman::<Scalar8Backend>::new(needle.as_bytes(), &Scoring::default());
        let score = matcher.match_haystack(haystack.as_bytes(), Some(max_typos));
        score
    }

    fn get_indices(needle: &str, haystack: &str) -> Option<Vec<usize>> {
        let mut matcher =
            SmithWaterman::<Scalar8Backend>::new(needle.as_bytes(), &Scoring::default());
        let indices = matcher
            .match_haystack_indices(haystack.as_bytes(), 0, None)
            .map(|(_, indices)| indices);
        indices
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
            get_score("test", "Uteost"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY
        );
        assert_eq!(
            get_score("test", "Uteoost"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY - GAP_EXTEND_PENALTY
        );
        assert_eq!(
            get_score("test", "Utooooeoooosoooot"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY * 3 - GAP_EXTEND_PENALTY * 9
        );
        assert_eq!(
            get_score("test", "Utooooooeoooooosoooooot"),
            CHAR_SCORE * 4 - GAP_OPEN_PENALTY * 3 - GAP_EXTEND_PENALTY * 15
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

    #[cfg(feature = "match_end_col")]
    fn get_end_col(needle: &str, haystack: &str) -> u16 {
        let mut matcher =
            SmithWaterman::<Scalar8Backend>::new(needle.as_bytes(), &Scoring::default());
        matcher.match_haystack(haystack.as_bytes(), None);
        matcher.match_end_col(haystack.as_bytes())
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn test_end_col_basic() {
        assert_eq!(get_end_col("abc", "abcdef"), 2);
        assert_eq!(get_end_col("a", "abc"), 0);
        assert_eq!(get_end_col("c", "abc"), 2);
        assert_eq!(get_end_col("def", "abcdef"), 5);
        assert_eq!(get_end_col("def", "________________abcdef"), 21);
    }

    #[test]
    fn test_score_typos() {
        assert_eq!(get_score_typos("foo", "Ufooo", 0), Some(CHAR_SCORE * 3));
        assert_eq!(get_score_typos("foo", "Ufo", 0), None);
        assert_eq!(
            get_score_typos("foo", "Ufo", 1),
            Some(CHAR_SCORE * 2 - GAP_OPEN_PENALTY)
        );
        assert_eq!(
            get_score_typos("foo", "Ufo", 2),
            Some(CHAR_SCORE * 2 - GAP_OPEN_PENALTY)
        );
        assert_eq!(get_score_typos("foo", "Uf", 1), None);
        assert_eq!(
            get_score_typos("foo", "Uf", 2),
            Some(CHAR_SCORE - GAP_OPEN_PENALTY - GAP_EXTEND_PENALTY)
        );
        assert_eq!(get_score_typos("foo", "U", 2), None);
        assert_eq!(get_score_typos("foo", "U", 3), Some(0));
        assert_eq!(get_score_typos("foo", "U", 4), Some(0));
    }

    #[test]
    fn test_indices_basic() {
        assert_eq!(get_indices("_", "abc"), Some(vec![]));
        assert_eq!(get_indices("a", "abc"), Some(vec![0]));
        assert_eq!(get_indices("b", "abc"), Some(vec![1]));
        assert_eq!(get_indices("c", "abc"), Some(vec![2]));
        assert_eq!(get_indices("ac", "________________abc"), Some(vec![18, 16]));
        assert_eq!(get_indices("foo", "Uf"), Some(vec![1]));
    }

    // ---------------------------------------------------------------
    // Cross-backend parity: every available backend should produce the same
    // scores and the same alignment-path indices as the runtime-selected
    // backend. With Phase 2 this covers u8 and u16 paths on each
    // architecture.
    // ---------------------------------------------------------------

    fn cases() -> Vec<(&'static str, &'static str)> {
        vec![
            // short
            ("a", "abc"),
            ("abc", "abc"),
            ("foo", "fooBar"),
            // crossing 8-byte chunk boundary (SSE u16 LANES = 8)
            ("foo", "012345foo"),
            ("foo", "01234567foo"),
            ("foo", "0123456789foo"),
            // crossing 16-byte boundary (AVX u16, SSE u8 LANES = 16)
            ("foo", "0123456789012345foo"),
            // crossing 32-byte boundary (AVX u8 LANES = 32)
            ("foo", "0123456789012345678901234567foo"),
            // ranges that cross multiple chunks for all widths
            ("test", "Utooooeoooosoooot"),
            ("test", "Utooooooeoooooosoooooot"),
            // typos
            ("foo", "Ufooo"),
            ("foo", "Ufo"),
            // delimiter / capitalization
            ("hw", "hello_world"),
            ("fBr", "fooBar"),
            ("D", "FOR_DIST"),
            // long needles (some short enough for u8, some not)
            ("needle", "____________needle____________"),
            ("abcdefghij", "abcdefghij"),
            ("abcdefghijklmnopqrst", "abcdefghijklmnopqrst"),
        ]
    }

    fn score_with<B: Backend>(needle: &str, haystack: &str) -> u16 {
        let mut matcher = SmithWaterman::<B>::new(needle.as_bytes(), &Scoring::default());
        matcher.match_haystack(haystack.as_bytes(), None).unwrap()
    }

    fn indices_with<B: Backend>(needle: &str, haystack: &str) -> Option<Vec<usize>> {
        let mut matcher = SmithWaterman::<B>::new(needle.as_bytes(), &Scoring::default());
        matcher
            .match_haystack_indices(haystack.as_bytes(), 0, None)
            .map(|(_, indices)| indices)
    }

    fn assert_score_backend<B: Backend>(label: &str, needle: &str, haystack: &str, want: u16) {
        if B::is_available() {
            let got = score_with::<B>(needle, haystack);
            assert_eq!(
                got, want,
                "{label} score mismatch for needle={needle:?} haystack={haystack:?}"
            );
        }
    }

    fn assert_indices_backend<B: Backend>(
        label: &str,
        needle: &str,
        haystack: &str,
        want: Option<Vec<usize>>,
    ) {
        if B::is_available() {
            let got = indices_with::<B>(needle, haystack);
            assert_eq!(
                got, want,
                "{label} indices mismatch for needle={needle:?} haystack={haystack:?}"
            );
        }
    }

    #[test]
    fn cross_backend_parity_score() {
        for (needle, haystack) in cases() {
            let want = score_with::<Scalar8Backend>(needle, haystack);

            #[cfg(target_arch = "x86_64")]
            {
                assert_score_backend::<SseBackend>("SSE-u16", needle, haystack, want);
                assert_score_backend::<Avx512Backend>("AVX-512-u16", needle, haystack, want);
                assert_score_backend::<AvxBackend>("AVX-u16", needle, haystack, want);

                if score_fits_in_u8(needle.len(), &Scoring::default()) {
                    assert_score_backend::<SseU8Backend>("SSE-u8", needle, haystack, want);
                    assert_score_backend::<AvxU8Backend>("AVX-u8", needle, haystack, want);
                    assert_score_backend::<Avx512U8Backend>("AVX-512-u8", needle, haystack, want);
                }
            }

            assert_score_backend::<Scalar8Backend>("Scalar-u16", needle, haystack, want);

            if score_fits_in_u8(needle.len(), &Scoring::default()) {
                assert_score_backend::<Scalar16U8Backend>("Scalar-u8", needle, haystack, want);
            }
        }
    }

    #[test]
    fn cross_backend_parity_indices() {
        for (needle, haystack) in cases() {
            let want = indices_with::<Scalar8Backend>(needle, haystack);

            #[cfg(target_arch = "x86_64")]
            {
                assert_indices_backend::<SseBackend>("SSE-u16", needle, haystack, want.clone());
                assert_indices_backend::<Avx512Backend>(
                    "AVX-512-u16",
                    needle,
                    haystack,
                    want.clone(),
                );
                assert_indices_backend::<AvxBackend>("AVX-u16", needle, haystack, want.clone());

                if score_fits_in_u8(needle.len(), &Scoring::default()) {
                    assert_indices_backend::<SseU8Backend>(
                        "SSE-u8",
                        needle,
                        haystack,
                        want.clone(),
                    );
                    assert_indices_backend::<AvxU8Backend>(
                        "AVX-u8",
                        needle,
                        haystack,
                        want.clone(),
                    );
                    assert_indices_backend::<Avx512U8Backend>(
                        "AVX-512-u8",
                        needle,
                        haystack,
                        want.clone(),
                    );
                }
            }

            assert_indices_backend::<Scalar8Backend>("Scalar-u16", needle, haystack, want.clone());

            if score_fits_in_u8(needle.len(), &Scoring::default()) {
                assert_indices_backend::<Scalar16U8Backend>("Scalar-u8", needle, haystack, want);
            }
        }
    }
}
