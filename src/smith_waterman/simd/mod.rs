use crate::Scoring;
use backend::Backend as _;
#[cfg(target_arch = "x86_64")]
use backend::{Avx512Backend, Avx512U8Backend, AvxBackend, AvxU8Backend, SseBackend, SseU8Backend};
#[cfg(target_arch = "aarch64")]
use backend::{NeonBackend, NeonU8Backend};
use backend::{Scalar8Backend, Scalar16U8Backend};

mod algo;
mod alignment;
mod alignment_iter;
pub(crate) mod backend;
mod matrix;

use algo::SmithWatermanMatcherInternal;
pub use alignment_iter::{Alignment, AlignmentPathIter};

/// Returns true if every possible Smith-Waterman matrix cell value for this
/// needle length and scoring config fits in a u8. The u8 backends are
/// otherwise identical to the u16 backends but with double the lane count
/// (64 cells/chunk on AVX-512, 32 on AVX2, 16 on SSE/NEON).
#[inline]
fn score_fits_in_u8(needle_len: usize, scoring: &Scoring) -> bool {
    let max_per_char = scoring.match_score as usize
        + scoring.matching_case_bonus as usize
        + scoring
            .delimiter_bonus
            .max(scoring.capitalization_bonus)
            .saturating_sub(scoring.gap_open_penalty) as usize;
    let max_matrix_score = max_per_char * needle_len + scoring.prefix_bonus as usize;
    max_matrix_score <= u8::MAX as usize
}

/// SIMD Smith Waterman matcher with affine gaps and sequential layout
/// parallelism. Chooses the fastest backend via runtime feature detection.
///
/// Each architecture has both a u16-scoring and a u8-scoring variant.
/// `new()` picks the u8 variant when the maximum possible matrix score fits
/// in a u8, doubling effective throughput
#[derive(Debug, Clone)]
pub enum SmithWatermanMatcher {
    #[cfg(target_arch = "x86_64")]
    AVX512(SmithWatermanMatcherAVX512),
    #[cfg(target_arch = "x86_64")]
    AVX512U8(SmithWatermanMatcherAVX512U8),
    #[cfg(target_arch = "x86_64")]
    AVX2(SmithWatermanMatcherAVX2),
    #[cfg(target_arch = "x86_64")]
    AVX2U8(SmithWatermanMatcherAVX2U8),
    #[cfg(target_arch = "x86_64")]
    SSE(SmithWatermanMatcherSSE),
    #[cfg(target_arch = "x86_64")]
    SSEU8(SmithWatermanMatcherSSEU8),
    #[cfg(target_arch = "aarch64")]
    NEON(SmithWatermanMatcherNEON),
    #[cfg(target_arch = "aarch64")]
    NEONU8(SmithWatermanMatcherNEONU8),
    Scalar(SmithWatermanMatcherScalar),
    ScalarU8(SmithWatermanMatcherScalarU8),
}

/// Dispatch to the active backend's matcher. All variants' inner methods are
/// `unsafe`: the SIMD variants because of `#[target_feature]`, the scalar
/// ones for uniformity (no actual safety obligation). The wrapping methods
/// on [`SmithWatermanMatcher`] are safe due to runtime feature detection.
macro_rules! dispatch {
    ($self:expr, $m:ident => $body:expr) => {
        match $self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX512($m) => unsafe { $body },
            #[cfg(target_arch = "x86_64")]
            Self::AVX512U8($m) => unsafe { $body },
            #[cfg(target_arch = "x86_64")]
            Self::AVX2($m) => unsafe { $body },
            #[cfg(target_arch = "x86_64")]
            Self::AVX2U8($m) => unsafe { $body },
            #[cfg(target_arch = "x86_64")]
            Self::SSE($m) => unsafe { $body },
            #[cfg(target_arch = "x86_64")]
            Self::SSEU8($m) => unsafe { $body },
            #[cfg(target_arch = "aarch64")]
            Self::NEON($m) => unsafe { $body },
            #[cfg(target_arch = "aarch64")]
            Self::NEONU8($m) => unsafe { $body },
            Self::Scalar($m) => unsafe { $body },
            Self::ScalarU8($m) => unsafe { $body },
        }
    };
}

impl SmithWatermanMatcher {
    pub fn new(needle: &[u8], scoring: &Scoring) -> Self {
        let use_u8 = score_fits_in_u8(needle.len(), scoring);

        #[cfg(target_arch = "x86_64")]
        if use_u8 && SmithWatermanMatcherAVX512U8::is_available() {
            return Self::AVX512U8(unsafe { SmithWatermanMatcherAVX512U8::new(needle, scoring) });
        }
        #[cfg(target_arch = "x86_64")]
        if !use_u8 && SmithWatermanMatcherAVX512::is_available() {
            return Self::AVX512(unsafe { SmithWatermanMatcherAVX512::new(needle, scoring) });
        }
        #[cfg(target_arch = "x86_64")]
        if SmithWatermanMatcherAVX2::is_available() {
            return if use_u8 {
                Self::AVX2U8(unsafe { SmithWatermanMatcherAVX2U8::new(needle, scoring) })
            } else {
                Self::AVX2(unsafe { SmithWatermanMatcherAVX2::new(needle, scoring) })
            };
        }
        #[cfg(target_arch = "x86_64")]
        if SmithWatermanMatcherSSE::is_available() {
            return if use_u8 {
                Self::SSEU8(unsafe { SmithWatermanMatcherSSEU8::new(needle, scoring) })
            } else {
                Self::SSE(unsafe { SmithWatermanMatcherSSE::new(needle, scoring) })
            };
        }

        #[cfg(target_arch = "aarch64")]
        return if use_u8 {
            Self::NEONU8(unsafe { SmithWatermanMatcherNEONU8::new(needle, scoring) })
        } else {
            Self::NEON(unsafe { SmithWatermanMatcherNEON::new(needle, scoring) })
        };

        #[cfg(not(target_arch = "aarch64"))]
        if use_u8 {
            Self::ScalarU8(unsafe { SmithWatermanMatcherScalarU8::new(needle, scoring) })
        } else {
            Self::Scalar(unsafe { SmithWatermanMatcherScalar::new(needle, scoring) })
        }
    }

    pub fn match_haystack(&mut self, haystack: &[u8], max_typos: Option<u16>) -> Option<u16> {
        dispatch!(self, m => m.match_haystack(haystack, max_typos))
    }

    pub fn match_haystack_indices(
        &mut self,
        haystack: &[u8],
        skipped_chars: usize,
        max_typos: Option<u16>,
    ) -> Option<(u16, Vec<usize>)> {
        dispatch!(self, m => m.match_haystack_indices(haystack, skipped_chars, max_typos))
    }

    pub fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
        dispatch!(self, m => m.score_haystack(haystack))
    }

    #[cfg(feature = "match_end_col")]
    pub fn match_end_col(&self, haystack: &[u8]) -> u16 {
        dispatch!(self, m => m.match_end_col(haystack))
    }

    #[allow(unused_unsafe)] // body is safe; `dispatch!` wraps uniformly.
    pub fn iter_alignment_path(
        &self,
        skipped_chars: usize,
        score: u16,
        max_typos: Option<u16>,
    ) -> AlignmentPathIter<'_> {
        dispatch!(self, m => m.0.iter_alignment_path(skipped_chars, score, max_typos))
    }

    #[cfg(test)]
    #[allow(unused_unsafe)] // SIMD `print_score_matrix` is `unsafe`; scalar is not.
    pub fn print_score_matrix(&self, haystack: &str) {
        dispatch!(self, m => m.print_score_matrix(haystack))
    }
}

macro_rules! define_matcher {
    (
        $name:ident,
        backend = $backend:ty,
        target_feature = $feature:literal,
    ) => {
        #[derive(Debug, Clone)]
        pub struct $name(SmithWatermanMatcherInternal<$backend>);

        impl $name {
            #[doc = concat!("# Safety\n\nCaller must ensure that the target feature `", $feature, "` is available")]
            #[target_feature(enable = $feature)]
            pub unsafe fn new(needle: &[u8], scoring: &Scoring) -> Self {
                Self(SmithWatermanMatcherInternal::new(needle, scoring))
            }

            pub fn is_available() -> bool {
                <$backend>::is_available()
            }

            #[doc = concat!(
                "Match the haystack against the needle, with an optional maximum number of typos.\n\n",
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

            #[doc = concat!("# Safety\n\nCaller must ensure that the target feature `", $feature, "` is available")]
            #[target_feature(enable = $feature)]
            pub unsafe fn match_haystack_indices(
                &mut self,
                haystack: &[u8],
                skipped_chars: usize,
                max_typos: Option<u16>,
            ) -> Option<(u16, Vec<usize>)> {
                self.0
                    .match_haystack_indices(haystack, skipped_chars, max_typos)
            }

            #[doc = concat!(
                "Match the haystack and return the score on the final row of the matrix.\n\n",
                "# Safety\n\n",
                "Caller must ensure that the target feature `", $feature, "` is available"
            )]
            #[target_feature(enable = $feature)]
            pub unsafe fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
                self.0.score_haystack(haystack)
            }

            #[doc = concat!(
                "Get the column of the final needle char in the haystack.\n\n",
                "# Safety\n\n",
                "Caller must ensure that the target feature `", $feature, "` is available"
            )]
            #[cfg(feature = "match_end_col")]
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
    SmithWatermanMatcherAVX512,
    backend = Avx512Backend,
    target_feature = "avx512f,avx512bw",
);

#[cfg(target_arch = "x86_64")]
define_matcher!(
    SmithWatermanMatcherAVX512U8,
    backend = Avx512U8Backend,
    target_feature = "avx512f,avx512bw,avx512vbmi",
);

#[cfg(target_arch = "x86_64")]
define_matcher!(
    SmithWatermanMatcherAVX2,
    backend = AvxBackend,
    target_feature = "avx2",
);

#[cfg(target_arch = "x86_64")]
define_matcher!(
    SmithWatermanMatcherAVX2U8,
    backend = AvxU8Backend,
    target_feature = "avx2",
);

#[cfg(target_arch = "x86_64")]
define_matcher!(
    SmithWatermanMatcherSSE,
    backend = SseBackend,
    target_feature = "ssse3,sse4.1",
);

#[cfg(target_arch = "x86_64")]
define_matcher!(
    SmithWatermanMatcherSSEU8,
    backend = SseU8Backend,
    target_feature = "ssse3,sse4.1",
);

#[cfg(target_arch = "aarch64")]
define_matcher!(
    SmithWatermanMatcherNEON,
    backend = NeonBackend,
    target_feature = "neon",
);

#[cfg(target_arch = "aarch64")]
define_matcher!(
    SmithWatermanMatcherNEONU8,
    backend = NeonU8Backend,
    target_feature = "neon",
);

/// Scalar fallback. Always available, no target_feature needed. Methods are
/// marked `unsafe` so the `dispatch!` macro can wrap them uniformly with
/// the SIMD variants.
macro_rules! define_scalar_matcher {
    ($name:ident, backend = $backend:ty) => {
        #[derive(Debug, Clone)]
        pub struct $name(SmithWatermanMatcherInternal<$backend>);

        impl $name {
            /// # Safety
            /// Trivially safe — kept `unsafe` for uniformity with the SIMD matchers.
            pub unsafe fn new(needle: &[u8], scoring: &Scoring) -> Self {
                Self(SmithWatermanMatcherInternal::new(needle, scoring))
            }

            pub fn is_available() -> bool {
                true
            }

            /// # Safety
            /// Trivially safe — kept `unsafe` for uniformity.
            pub unsafe fn match_haystack(
                &mut self,
                haystack: &[u8],
                max_typos: Option<u16>,
            ) -> Option<u16> {
                self.0.match_haystack(haystack, max_typos)
            }

            /// # Safety
            /// Trivially safe — kept `unsafe` for uniformity.
            pub unsafe fn match_haystack_indices(
                &mut self,
                haystack: &[u8],
                skipped_chars: usize,
                max_typos: Option<u16>,
            ) -> Option<(u16, Vec<usize>)> {
                self.0
                    .match_haystack_indices(haystack, skipped_chars, max_typos)
            }

            /// # Safety
            /// Trivially safe — kept `unsafe` for uniformity.
            pub unsafe fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
                self.0.score_haystack(haystack)
            }

            /// # Safety
            /// Trivially safe — kept `unsafe` for uniformity.
            #[cfg(feature = "match_end_col")]
            pub unsafe fn match_end_col(&self, haystack: &[u8]) -> u16 {
                self.0.match_end_col(haystack)
            }

            #[cfg(test)]
            pub fn print_score_matrix(&self, haystack: &str) {
                self.0.print_score_matrix(haystack)
            }
        }
    };
}

define_scalar_matcher!(SmithWatermanMatcherScalar, backend = Scalar8Backend);
define_scalar_matcher!(SmithWatermanMatcherScalarU8, backend = Scalar16U8Backend);

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

    #[test]
    fn cross_backend_parity_score() {
        let reference = |needle: &str, haystack: &str| {
            let mut m = SmithWatermanMatcher::new(needle.as_bytes(), &Scoring::default());
            m.match_haystack(haystack.as_bytes(), None).unwrap()
        };

        for (needle, haystack) in cases() {
            let want = reference(needle, haystack);

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherSSE::is_available() {
                let mut sse =
                    unsafe { SmithWatermanMatcherSSE::new(needle.as_bytes(), &Scoring::default()) };
                let got = unsafe { sse.match_haystack(haystack.as_bytes(), None) }.unwrap();
                assert_eq!(
                    got, want,
                    "SSE-u16 score mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherAVX512::is_available() {
                let mut avx = unsafe {
                    SmithWatermanMatcherAVX512::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { avx.match_haystack(haystack.as_bytes(), None) }.unwrap();
                assert_eq!(
                    got, want,
                    "AVX-512-u16 score mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherSSEU8::is_available()
                && score_fits_in_u8(needle.len(), &Scoring::default())
            {
                let mut sse = unsafe {
                    SmithWatermanMatcherSSEU8::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { sse.match_haystack(haystack.as_bytes(), None) }.unwrap();
                assert_eq!(
                    got, want,
                    "SSE-u8 score mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherAVX2U8::is_available()
                && score_fits_in_u8(needle.len(), &Scoring::default())
            {
                let mut avx = unsafe {
                    SmithWatermanMatcherAVX2U8::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { avx.match_haystack(haystack.as_bytes(), None) }.unwrap();
                assert_eq!(
                    got, want,
                    "AVX-u8 score mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherAVX512U8::is_available()
                && score_fits_in_u8(needle.len(), &Scoring::default())
            {
                let mut avx = unsafe {
                    SmithWatermanMatcherAVX512U8::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { avx.match_haystack(haystack.as_bytes(), None) }.unwrap();
                assert_eq!(
                    got, want,
                    "AVX-512-u8 score mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            let mut scalar =
                unsafe { SmithWatermanMatcherScalar::new(needle.as_bytes(), &Scoring::default()) };
            let got = unsafe { scalar.match_haystack(haystack.as_bytes(), None) }.unwrap();
            assert_eq!(
                got, want,
                "Scalar-u16 score mismatch for needle={needle:?} haystack={haystack:?}"
            );

            if score_fits_in_u8(needle.len(), &Scoring::default()) {
                let mut scalar_u8 = unsafe {
                    SmithWatermanMatcherScalarU8::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { scalar_u8.match_haystack(haystack.as_bytes(), None) }.unwrap();
                assert_eq!(
                    got, want,
                    "Scalar-u8 score mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }
        }
    }

    #[test]
    fn cross_backend_parity_indices() {
        let reference = |needle: &str, haystack: &str| {
            let mut m = SmithWatermanMatcher::new(needle.as_bytes(), &Scoring::default());
            m.match_haystack_indices(haystack.as_bytes(), 0, None)
                .map(|(_, indices)| indices)
        };

        for (needle, haystack) in cases() {
            let want = reference(needle, haystack);

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherSSE::is_available() {
                let mut sse =
                    unsafe { SmithWatermanMatcherSSE::new(needle.as_bytes(), &Scoring::default()) };
                let got = unsafe { sse.match_haystack_indices(haystack.as_bytes(), 0, None) }
                    .map(|(_, indices)| indices);
                assert_eq!(
                    got, want,
                    "SSE-u16 indices mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherAVX512::is_available() {
                let mut avx = unsafe {
                    SmithWatermanMatcherAVX512::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { avx.match_haystack_indices(haystack.as_bytes(), 0, None) }
                    .map(|(_, indices)| indices);
                assert_eq!(
                    got, want,
                    "AVX-512-u16 indices mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherSSEU8::is_available()
                && score_fits_in_u8(needle.len(), &Scoring::default())
            {
                let mut sse = unsafe {
                    SmithWatermanMatcherSSEU8::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { sse.match_haystack_indices(haystack.as_bytes(), 0, None) }
                    .map(|(_, indices)| indices);
                assert_eq!(
                    got, want,
                    "SSE-u8 indices mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherAVX2U8::is_available()
                && score_fits_in_u8(needle.len(), &Scoring::default())
            {
                let mut avx = unsafe {
                    SmithWatermanMatcherAVX2U8::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { avx.match_haystack_indices(haystack.as_bytes(), 0, None) }
                    .map(|(_, indices)| indices);
                assert_eq!(
                    got, want,
                    "AVX-u8 indices mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            #[cfg(target_arch = "x86_64")]
            if SmithWatermanMatcherAVX512U8::is_available()
                && score_fits_in_u8(needle.len(), &Scoring::default())
            {
                let mut avx = unsafe {
                    SmithWatermanMatcherAVX512U8::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { avx.match_haystack_indices(haystack.as_bytes(), 0, None) }
                    .map(|(_, indices)| indices);
                assert_eq!(
                    got, want,
                    "AVX-512-u8 indices mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }

            let mut scalar =
                unsafe { SmithWatermanMatcherScalar::new(needle.as_bytes(), &Scoring::default()) };
            let got = unsafe { scalar.match_haystack_indices(haystack.as_bytes(), 0, None) }
                .map(|(_, indices)| indices);
            assert_eq!(
                got, want,
                "Scalar-u16 indices mismatch for needle={needle:?} haystack={haystack:?}"
            );

            if score_fits_in_u8(needle.len(), &Scoring::default()) {
                let mut scalar_u8 = unsafe {
                    SmithWatermanMatcherScalarU8::new(needle.as_bytes(), &Scoring::default())
                };
                let got = unsafe { scalar_u8.match_haystack_indices(haystack.as_bytes(), 0, None) }
                    .map(|(_, indices)| indices);
                assert_eq!(
                    got, want,
                    "Scalar-u8 indices mismatch for needle={needle:?} haystack={haystack:?}"
                );
            }
        }
    }

    #[test]
    fn u8_path_selected_for_short_needle() {
        // Default scoring: u8 fits for needles up to ~12 chars.
        let m = SmithWatermanMatcher::new(b"abc", &Scoring::default());
        let is_u8 = match m {
            #[cfg(target_arch = "x86_64")]
            SmithWatermanMatcher::AVX512U8(_)
            | SmithWatermanMatcher::AVX2U8(_)
            | SmithWatermanMatcher::SSEU8(_) => true,
            #[cfg(target_arch = "aarch64")]
            SmithWatermanMatcher::NEONU8(_) => true,
            SmithWatermanMatcher::ScalarU8(_) => true,
            _ => false,
        };
        assert!(is_u8);
    }

    #[test]
    fn u16_path_selected_for_long_needle() {
        // 20 chars × ~20 per char + 12 prefix = 412 → overflows u8.
        let needle = b"abcdefghijklmnopqrst";
        let m = SmithWatermanMatcher::new(needle, &Scoring::default());
        let is_u16 = match m {
            #[cfg(target_arch = "x86_64")]
            SmithWatermanMatcher::AVX512(_)
            | SmithWatermanMatcher::AVX2(_)
            | SmithWatermanMatcher::SSE(_) => true,
            #[cfg(target_arch = "aarch64")]
            SmithWatermanMatcher::NEON(_) => true,
            SmithWatermanMatcher::Scalar(_) => true,
            _ => false,
        };
        assert!(is_u16);
    }
}
