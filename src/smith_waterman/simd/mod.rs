#[cfg(target_arch = "x86_64")]
use crate::simd::{AVXVector, SSE256Vector, SSEVector};
#[cfg(target_arch = "aarch64")]
use crate::simd::{NEON256Vector, NEONVector};
use crate::simd::{Scalar128Vector, Scalar256Vector};
use crate::{Scoring, simd::Vector};

mod algo;
mod alignment;
mod alignment_iter;
mod gaps;
mod matrix;

use algo::SmithWatermanMatcherInternal;
pub use alignment_iter::{Alignment, AlignmentPathIter};

/// SIMD Smith Waterman matcher with affine gaps and sequential layout parallelism.
/// Chooses the fastest algorithm via runtime feature detection.
#[derive(Debug, Clone)]
pub enum SmithWatermanMatcher {
    #[cfg(target_arch = "x86_64")]
    AVX2(SmithWatermanMatcherAVX2),
    #[cfg(target_arch = "x86_64")]
    SSE(SmithWatermanMatcherSSE),
    #[cfg(target_arch = "aarch64")]
    NEON(SmithWatermanMatcherNEON),
    Scalar(SmithWatermanMatcherScalar),
}

impl SmithWatermanMatcher {
    pub fn new(needle: &[u8], scoring: &Scoring) -> Self {
        #[cfg(target_arch = "x86_64")]
        if SmithWatermanMatcherAVX2::is_available() {
            return Self::AVX2(unsafe { SmithWatermanMatcherAVX2::new(needle, scoring) });
        }
        #[cfg(target_arch = "x86_64")]
        if SmithWatermanMatcherSSE::is_available() {
            return Self::SSE(unsafe { SmithWatermanMatcherSSE::new(needle, scoring) });
        }

        #[cfg(target_arch = "aarch64")]
        return Self::NEON(unsafe { SmithWatermanMatcherNEON::new(needle, scoring) });

        #[cfg(not(target_arch = "aarch64"))]
        Self::Scalar(SmithWatermanMatcherScalar::new(needle, scoring))
    }

    pub fn match_haystack(&mut self, haystack: &[u8], max_typos: Option<u16>) -> Option<u16> {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX2(matcher) => unsafe { matcher.match_haystack(haystack, max_typos) },
            #[cfg(target_arch = "x86_64")]
            Self::SSE(matcher) => unsafe { matcher.match_haystack(haystack, max_typos) },
            #[cfg(target_arch = "aarch64")]
            Self::NEON(matcher) => unsafe { matcher.match_haystack(haystack, max_typos) },
            Self::Scalar(matcher) => matcher.match_haystack(haystack, max_typos),
        }
    }

    #[cfg(feature = "match_end_col")]
    pub fn match_haystack_with_end_col(
        &mut self,
        haystack: &[u8],
        max_typos: Option<u16>,
    ) -> Option<(u16, u16)> {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX2(matcher) => unsafe {
                matcher.match_haystack_with_end_col(haystack, max_typos)
            },
            #[cfg(target_arch = "x86_64")]
            Self::SSE(matcher) => unsafe {
                matcher.match_haystack_with_end_col(haystack, max_typos)
            },
            #[cfg(target_arch = "aarch64")]
            Self::NEON(matcher) => unsafe {
                matcher.match_haystack_with_end_col(haystack, max_typos)
            },
            Self::Scalar(matcher) => matcher.match_haystack_with_end_col(haystack, max_typos),
        }
    }

    pub fn match_haystack_indices(
        &mut self,
        haystack: &[u8],
        skipped_chunks: usize,
        max_typos: Option<u16>,
    ) -> Option<(u16, Vec<usize>)> {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX2(matcher) => unsafe {
                matcher.match_haystack_indices(haystack, skipped_chunks, max_typos)
            },
            #[cfg(target_arch = "x86_64")]
            Self::SSE(matcher) => unsafe {
                matcher.match_haystack_indices(haystack, skipped_chunks, max_typos)
            },
            #[cfg(target_arch = "aarch64")]
            Self::NEON(matcher) => unsafe {
                matcher.match_haystack_indices(haystack, skipped_chunks, max_typos)
            },
            Self::Scalar(matcher) => {
                matcher.match_haystack_indices(haystack, skipped_chunks, max_typos)
            }
        }
    }

    pub fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX2(matcher) => unsafe { matcher.score_haystack(haystack) },
            #[cfg(target_arch = "x86_64")]
            Self::SSE(matcher) => unsafe { matcher.score_haystack(haystack) },
            #[cfg(target_arch = "aarch64")]
            Self::NEON(matcher) => unsafe { matcher.score_haystack(haystack) },
            Self::Scalar(matcher) => matcher.score_haystack(haystack),
        }
    }

    #[cfg(feature = "match_end_col")]
    pub fn match_end_col(&self, haystack: &[u8]) -> u16 {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX2(matcher) => unsafe { matcher.match_end_col(haystack) },
            #[cfg(target_arch = "x86_64")]
            Self::SSE(matcher) => unsafe { matcher.match_end_col(haystack) },
            #[cfg(target_arch = "aarch64")]
            Self::NEON(matcher) => unsafe { matcher.match_end_col(haystack) },
            Self::Scalar(matcher) => matcher.match_end_col(haystack),
        }
    }

    /// Iterate over the alignment path positions with support for max typos.
    ///
    /// Yields `Some((needle_idx, haystack_idx))` for each matched position,
    /// or `None` if max_typos was exceeded.
    pub fn iter_alignment_path(
        &self,
        skipped_chunks: usize,
        score: u16,
        max_typos: Option<u16>,
    ) -> AlignmentPathIter<'_> {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX2(m) => AlignmentPathIter::new(
                &m.0.score_matrix,
                &m.0.match_masks,
                m.0.needle.len(),
                m.0.haystack_chunks,
                skipped_chunks,
                score,
                max_typos,
            ),
            #[cfg(target_arch = "x86_64")]
            Self::SSE(m) => AlignmentPathIter::new(
                &m.0.score_matrix,
                &m.0.match_masks,
                m.0.needle.len(),
                m.0.haystack_chunks,
                skipped_chunks,
                score,
                max_typos,
            ),
            #[cfg(target_arch = "aarch64")]
            Self::NEON(m) => AlignmentPathIter::new(
                &m.0.score_matrix,
                &m.0.match_masks,
                m.0.needle.len(),
                m.0.haystack_chunks,
                skipped_chunks,
                score,
                max_typos,
            ),
            Self::Scalar(m) => AlignmentPathIter::new(
                &m.0.score_matrix,
                &m.0.match_masks,
                m.0.needle.len(),
                m.0.haystack_chunks,
                skipped_chunks,
                score,
                max_typos,
            ),
        }
    }

    #[cfg(test)]
    pub fn print_score_matrix(&self, haystack: &str) {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX2(matcher) => unsafe { matcher.print_score_matrix(haystack) },
            #[cfg(target_arch = "x86_64")]
            Self::SSE(matcher) => unsafe { matcher.print_score_matrix(haystack) },
            #[cfg(target_arch = "aarch64")]
            Self::NEON(matcher) => unsafe { matcher.print_score_matrix(haystack) },
            Self::Scalar(matcher) => matcher.print_score_matrix(haystack),
        }
    }
}

macro_rules! define_matcher {
    (
        $name:ident,
        small = $small:ty,
        large = $large:ty,
        target_feature = $feature:literal,
        available = $available:expr
    ) => {
        #[derive(Debug, Clone)]
        pub struct $name(SmithWatermanMatcherInternal<$small, $large>);

        impl $name {
            #[doc = concat!("# Safety\n\nCaller must ensure that the target feature `", $feature, "` is available")]
            #[target_feature(enable = $feature)]
            pub unsafe fn new(needle: &[u8], scoring: &Scoring) -> Self {
                Self(SmithWatermanMatcherInternal::new(needle, scoring))
            }

            pub fn is_available() -> bool {
                $available
            }

            #[doc = concat!(
                "Match the haystack against the needle, with an optional maximum number of typos\n\n",
                "# Safety\n\n",
                "Caller must ensure that the target feature `", $feature, "` is available"
            )]
            #[target_feature(enable = $feature)]
            pub unsafe fn match_haystack(
                &mut self,
                haystack: &[u8],
                max_typos: Option<u16>,
            ) -> Option<u16> {
                self.0.match_haystack(haystack, max_typos)
            }

            #[cfg(feature = "match_end_col")]
            #[doc = concat!("# Safety\n\nCaller must ensure that the target feature `", $feature, "` is available")]
            #[target_feature(enable = $feature)]
            pub unsafe fn match_haystack_with_end_col(
                &mut self,
                haystack: &[u8],
                max_typos: Option<u16>,
            ) -> Option<(u16, u16)> {
                self.0.match_haystack_with_end_col(haystack, max_typos)
            }

            #[doc = concat!("# Safety\n\nCaller must ensure that the target feature `", $feature, "` is available")]
            #[target_feature(enable = $feature)]
            pub unsafe fn match_haystack_indices(
                &mut self,
                haystack: &[u8],
                skipped_chunks: usize,
                max_typos: Option<u16>,
            ) -> Option<(u16, Vec<usize>)> {
                self.0.match_haystack_indices(haystack, skipped_chunks, max_typos)
            }

            #[doc = concat!(
                "Match the haystack against the needle, returning the score on the final row of the matrix\n\n",
                "# Safety\n\n",
                "Caller must ensure that the target feature `", $feature, "` is available"
            )]
            #[target_feature(enable = $feature)]
            pub unsafe fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
                self.0.score_haystack(haystack)
            }

            #[doc = concat!(
                "Get the index of the final needle char in the haystack\n\n",
                "# Safety\n\n",
                "Caller must ensure that the target feature `", $feature, "` is available"
            )]
            #[target_feature(enable = $feature)]
            pub unsafe fn match_end_col(&self, haystack: &[u8]) -> u16 {
                self.0.match_end_col(haystack)
            }

            #[cfg(test)]
            #[doc = concat!("# Safety\n\nCaller must ensure that the target feature `", $feature, "` is available")]
            #[target_feature(enable = $feature)]
            pub fn print_score_matrix(&self, haystack: &str) {
                self.0.print_score_matrix(haystack)
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
define_matcher!(
    SmithWatermanMatcherAVX2,
    small = SSEVector,
    large = AVXVector,
    target_feature = "avx2",
    available = AVXVector::is_available() && SSEVector::is_available()
);

#[cfg(target_arch = "x86_64")]
define_matcher!(
    SmithWatermanMatcherSSE,
    small = SSEVector,
    large = SSE256Vector,
    target_feature = "ssse3,sse4.1",
    available = SSEVector::is_available() && SSE256Vector::is_available()
);

#[cfg(target_arch = "aarch64")]
define_matcher!(
    SmithWatermanMatcherNEON,
    small = NEONVector,
    large = NEON256Vector,
    target_feature = "neon",
    available = NEONVector::is_available() && NEON256Vector::is_available()
);

#[derive(Debug, Clone)]
pub struct SmithWatermanMatcherScalar(
    SmithWatermanMatcherInternal<Scalar128Vector, Scalar256Vector>,
);

impl SmithWatermanMatcherScalar {
    pub fn new(needle: &[u8], scoring: &Scoring) -> Self {
        Self(SmithWatermanMatcherInternal::new(needle, scoring))
    }

    pub fn is_available() -> bool {
        true
    }

    pub fn match_haystack(&mut self, haystack: &[u8], max_typos: Option<u16>) -> Option<u16> {
        self.0.match_haystack(haystack, max_typos)
    }

    pub fn match_haystack_indices(
        &mut self,
        haystack: &[u8],
        skipped_chunks: usize,
        max_typos: Option<u16>,
    ) -> Option<(u16, Vec<usize>)> {
        self.0
            .match_haystack_indices(haystack, skipped_chunks, max_typos)
    }

    pub fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
        self.0.score_haystack(haystack)
    }

    #[cfg(feature = "match_end_col")]
    pub fn match_haystack_with_end_col(
        &mut self,
        haystack: &[u8],
        max_typos: Option<u16>,
    ) -> Option<(u16, u16)> {
        self.0.match_haystack_with_end_col(haystack, max_typos)
    }

    #[cfg(feature = "match_end_col")]
    pub fn match_end_col(&self, haystack: &[u8]) -> u16 {
        self.0.match_end_col(haystack)
    }

    #[cfg(test)]
    pub fn print_score_matrix(&self, haystack: &str) {
        self.0.print_score_matrix(haystack)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#const::*;

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn get_score(needle: &str, haystack: &str) -> u16 {
        let mut matcher = SmithWatermanMatcher::new(needle.as_bytes(), &Scoring::default());
        let score = matcher.match_haystack(haystack.as_bytes(), Some(0));
        matcher.print_score_matrix(haystack);
        score.unwrap()
    }

    fn get_score_typos(needle: &str, haystack: &str, max_typos: u16) -> Option<u16> {
        let mut matcher = SmithWatermanMatcher::new(needle.as_bytes(), &Scoring::default());
        let score = matcher.match_haystack(haystack.as_bytes(), Some(max_typos));
        matcher.print_score_matrix(haystack);
        score
    }

    fn get_indices(needle: &str, haystack: &str) -> Option<Vec<usize>> {
        let mut matcher = SmithWatermanMatcher::new(needle.as_bytes(), &Scoring::default());
        let indices = matcher
            .match_haystack_indices(haystack.as_bytes(), 0, None)
            .map(|(_, indices)| indices);
        matcher.print_score_matrix(haystack);
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
        let mut matcher = SmithWatermanMatcher::new(needle.as_bytes(), &Scoring::default());
        matcher.match_haystack(haystack.as_bytes(), None);
        matcher.match_end_col(haystack.as_bytes())
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn test_end_col_basic() {
        // "abc" in "abcdef" should end at column 2 (0-indexed byte position of 'c')
        assert_eq!(get_end_col("abc", "abcdef"), 2);
        // "a" in "abc" should end at column 0
        assert_eq!(get_end_col("a", "abc"), 0);
        // "c" in "abc" should end at column 2
        assert_eq!(get_end_col("c", "abc"), 2);
        // "def" in "abcdef" should end at column 5
        assert_eq!(get_end_col("def", "abcdef"), 5);
        // "def" in "abcdef" should end at column 21
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
}
