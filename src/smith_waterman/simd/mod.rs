use crate::{
    Scoring,
    simd::{AVXVector, SSE256Vector, SSEVector, Vector},
};

mod algo;
mod gaps;
mod typos;

use algo::SmithWatermanMatcherInternal;

#[derive(Debug, Clone)]
pub enum SmithWatermanMatcher {
    AVX2(SmithWatermanMatcherAVX2),
    SSE(SmithWatermanMatcherSSE),
}

impl SmithWatermanMatcher {
    pub fn new(needle: &[u8], scoring: &Scoring) -> Self {
        if SmithWatermanMatcherAVX2::is_available() {
            Self::AVX2(unsafe { SmithWatermanMatcherAVX2::new(needle, scoring) })
        } else if SmithWatermanMatcherSSE::is_available() {
            Self::SSE(unsafe { SmithWatermanMatcherSSE::new(needle, scoring) })
        } else {
            panic!("frizbee requires SSE4.1 at minimum which your CPU does not support");
        }
    }

    pub fn match_haystack(&mut self, haystack: &[u8], max_typos: Option<u16>) -> Option<u16> {
        match self {
            Self::AVX2(matcher) => unsafe { matcher.match_haystack(haystack, max_typos) },
            Self::SSE(matcher) => unsafe { matcher.match_haystack(haystack, max_typos) },
        }
    }

    pub fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
        match self {
            Self::AVX2(matcher) => unsafe { matcher.score_haystack(haystack) },
            Self::SSE(matcher) => unsafe { matcher.score_haystack(haystack) },
        }
    }

    #[cfg(test)]
    pub fn print_score_matrix(&self, haystack: &str) {
        match self {
            Self::AVX2(matcher) => unsafe { matcher.print_score_matrix(haystack) },
            Self::SSE(matcher) => unsafe { matcher.print_score_matrix(haystack) },
        }
    }
}

macro_rules! define_matcher {
    (
        $name:ident,
        small = $small:ty,
        large = $large:ty,
        target_feature = $feature:literal,
        available = |$cpu:ident| $available:expr
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
                let $cpu = raw_cpuid::CpuId::new();
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
            #[target_feature(enable = $feature)]
            pub fn print_score_matrix(&self, haystack: &str) {
                self.0.print_score_matrix(haystack)
            }
        }
    };
}

define_matcher!(
    SmithWatermanMatcherAVX2,
    small = SSEVector,
    large = AVXVector,
    target_feature = "avx2",
    available = |cpu| AVXVector::is_available(&cpu) && SSEVector::is_available(&cpu)
);

define_matcher!(
    SmithWatermanMatcherSSE,
    small = SSEVector,
    large = SSE256Vector,
    target_feature = "ssse3,sse4.1",
    available = |cpu| SSEVector::is_available(&cpu) && SSE256Vector::is_available(&cpu)
);
