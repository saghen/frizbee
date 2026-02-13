#[cfg(target_arch = "x86_64")]
use crate::simd::{AVXVector, SSE256Vector, SSEVector};
#[cfg(target_arch = "aarch64")]
use crate::simd::{NEON256Vector, NEONVector};
use crate::{Scoring, simd::Vector};

mod algo;
mod gaps;
mod typos;

use algo::SmithWatermanMatcherInternal;

#[derive(Debug, Clone)]
pub enum SmithWatermanMatcher {
    #[cfg(target_arch = "x86_64")]
    AVX2(SmithWatermanMatcherAVX2),
    #[cfg(target_arch = "x86_64")]
    SSE(SmithWatermanMatcherSSE),
    #[cfg(target_arch = "aarch64")]
    NEON(SmithWatermanMatcherNEON),
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
        #[cfg(target_arch = "x86_64")]
        panic!("no smith waterman implementation available due to missing SSE4.1 support");

        #[cfg(target_arch = "aarch64")]
        return Self::NEON(unsafe { SmithWatermanMatcherNEON::new(needle, scoring) });
    }

    pub fn match_haystack(&mut self, haystack: &[u8], max_typos: Option<u16>) -> Option<u16> {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX2(matcher) => unsafe { matcher.match_haystack(haystack, max_typos) },
            #[cfg(target_arch = "x86_64")]
            Self::SSE(matcher) => unsafe { matcher.match_haystack(haystack, max_typos) },
            #[cfg(target_arch = "aarch64")]
            Self::NEON(matcher) => unsafe { matcher.match_haystack(haystack, max_typos) },
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

            #[doc = concat!(
                "Match the haystack against the needle, returning the score on the final row of the matrix\n\n",
                "# Safety\n\n",
                "Caller must ensure that the target feature `", $feature, "` is available"
            )]
            #[target_feature(enable = $feature)]
            pub unsafe fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
                self.0.score_haystack(haystack)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#const::*;

    const CHAR_SCORE: u16 = MATCH_SCORE + MATCHING_CASE_BONUS;

    fn get_score(needle: &str, haystack: &str) -> u16 {
        let mut matcher = SmithWatermanMatcher::new(needle.as_bytes(), &Scoring::default());
        let score = matcher.score_haystack(haystack.as_bytes());
        // matcher.print_score_matrix(haystack);
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
