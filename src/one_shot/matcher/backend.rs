//! Definitions for all target feature-specific implementations of the matcher

use super::algo::MatcherImpl;
use crate::{Config, Match, MatchIndices};

#[cfg(target_arch = "aarch64")]
use crate::prefilter::backend::PrefilterNEON;
use crate::prefilter::backend::PrefilterScalar;
#[cfg(target_arch = "x86_64")]
use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};
#[cfg(target_arch = "x86_64")]
use crate::smith_waterman::simd::{
    SmithWatermanAVX, SmithWatermanAVX512, SmithWatermanAVX512U8, SmithWatermanAVXU8,
    SmithWatermanSSE, SmithWatermanSSEU8,
};
#[cfg(target_arch = "aarch64")]
use crate::smith_waterman::simd::{SmithWatermanNEON, SmithWatermanNEONU8};
use crate::smith_waterman::simd::{SmithWatermanScalar, SmithWatermanScalarU8};

#[cfg(target_arch = "x86_64")]
pub type MatcherAVX512U8 = MatcherImpl<PrefilterAVX512, SmithWatermanAVX512U8>;
#[cfg(target_arch = "x86_64")]
pub type MatcherAVX512 = MatcherImpl<PrefilterAVX512, SmithWatermanAVX512>;
#[cfg(target_arch = "x86_64")]
pub type MatcherAVXU8 = MatcherImpl<PrefilterAVX, SmithWatermanAVXU8>;
#[cfg(target_arch = "x86_64")]
pub type MatcherAVX = MatcherImpl<PrefilterAVX, SmithWatermanAVX>;
#[cfg(target_arch = "x86_64")]
pub type MatcherSSEU8 = MatcherImpl<PrefilterSSE, SmithWatermanSSEU8>;
#[cfg(target_arch = "x86_64")]
pub type MatcherSSE = MatcherImpl<PrefilterSSE, SmithWatermanSSE>;
#[cfg(target_arch = "aarch64")]
pub type MatcherNEONU8 = MatcherImpl<PrefilterNEON, SmithWatermanNEONU8>;
#[cfg(target_arch = "aarch64")]
pub type MatcherNEON = MatcherImpl<PrefilterNEON, SmithWatermanNEON>;
pub type MatcherScalar = MatcherImpl<PrefilterScalar, SmithWatermanScalar>;
pub type MatcherScalarU8 = MatcherImpl<PrefilterScalar, SmithWatermanScalarU8>;

#[derive(Debug, Clone)]
pub enum MatcherBackend {
    #[cfg(target_arch = "x86_64")]
    AVX512U8(MatcherAVX512U8),
    #[cfg(target_arch = "x86_64")]
    AVX512(MatcherAVX512),
    #[cfg(target_arch = "x86_64")]
    AVXU8(MatcherAVXU8),
    #[cfg(target_arch = "x86_64")]
    AVX(MatcherAVX),
    #[cfg(target_arch = "x86_64")]
    SSEU8(MatcherSSEU8),
    #[cfg(target_arch = "x86_64")]
    SSE(MatcherSSE),
    #[cfg(target_arch = "aarch64")]
    NEONU8(MatcherNEONU8),
    #[cfg(target_arch = "aarch64")]
    NEON(MatcherNEON),
    ScalarU8(MatcherScalarU8),
    Scalar(MatcherScalar),
}

macro_rules! impl_matcher_entrypoints {
    ($prefilter:ty, $smith_waterman:ty $(, target_feature = $feature:literal)?) => {
        impl MatcherImpl<$prefilter, $smith_waterman> {
            #[inline]
            $(#[target_feature(enable = $feature)])?
            pub unsafe fn new(needle: &str, config: &Config) -> Self {
                Self::new_impl(needle, config)
            }

            #[inline]
            $(#[target_feature(enable = $feature)])?
            pub unsafe fn match_list<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<Match> {
                self.match_list_impl(haystacks)
            }

            #[inline]
            $(#[target_feature(enable = $feature)])?
            pub unsafe fn match_list_into<S: AsRef<str>>(
                &mut self,
                haystacks: &[S],
                haystack_index_offset: u32,
                matches: &mut Vec<Match>,
            ) {
                self.match_list_into_impl(haystacks, haystack_index_offset, matches)
            }

            #[inline]
            $(#[target_feature(enable = $feature)])?
            pub unsafe fn match_list_indices<S: AsRef<str>>(
                &mut self,
                haystacks: &[S],
            ) -> Vec<MatchIndices> {
                self.match_list_indices_impl(haystacks)
            }

            #[inline]
            $(#[target_feature(enable = $feature)])?
            pub unsafe fn match_one(&mut self, haystack: &[u8], index: u32) -> Option<Match> {
                self.match_one_impl(haystack, index)
            }

            #[inline]
            $(#[target_feature(enable = $feature)])?
            pub unsafe fn match_indices_one(
                &mut self,
                haystack: &[u8],
                index: u32,
            ) -> Option<MatchIndices> {
                self.match_indices_one_impl(haystack, index)
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
impl_matcher_entrypoints!(
    PrefilterAVX512,
    SmithWatermanAVX512U8,
    target_feature = "avx512f,avx512bw,avx512vbmi,bmi1,bmi2"
);
#[cfg(target_arch = "x86_64")]
impl_matcher_entrypoints!(
    PrefilterAVX512,
    SmithWatermanAVX512,
    target_feature = "avx512f,avx512bw,bmi1,bmi2"
);
#[cfg(target_arch = "x86_64")]
impl_matcher_entrypoints!(PrefilterAVX, SmithWatermanAVXU8, target_feature = "avx2");
#[cfg(target_arch = "x86_64")]
impl_matcher_entrypoints!(PrefilterAVX, SmithWatermanAVX, target_feature = "avx2");
#[cfg(target_arch = "x86_64")]
impl_matcher_entrypoints!(
    PrefilterSSE,
    SmithWatermanSSEU8,
    target_feature = "sse2,ssse3,sse4.1"
);
#[cfg(target_arch = "x86_64")]
impl_matcher_entrypoints!(
    PrefilterSSE,
    SmithWatermanSSE,
    target_feature = "sse2,ssse3,sse4.1"
);
#[cfg(target_arch = "aarch64")]
impl_matcher_entrypoints!(PrefilterNEON, SmithWatermanNEONU8, target_feature = "neon");
#[cfg(target_arch = "aarch64")]
impl_matcher_entrypoints!(PrefilterNEON, SmithWatermanNEON, target_feature = "neon");
impl_matcher_entrypoints!(PrefilterScalar, SmithWatermanScalarU8);
impl_matcher_entrypoints!(PrefilterScalar, SmithWatermanScalar);
