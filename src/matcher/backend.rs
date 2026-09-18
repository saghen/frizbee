//! Definitions for all target feature-specific implementations of the matcher

use super::algo::{MatcherImpl, Scorer, Specialized};
use crate::smith_waterman::ScoreWidth;
use crate::{Config, Match, MatchIndices};
use alloc::vec::Vec;

#[cfg(target_arch = "aarch64")]
use crate::literal::LiteralNEON;
#[cfg(scalar_fallback)]
use crate::literal::LiteralScalar;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::literal::LiteralWasm;
#[cfg(target_arch = "x86_64")]
use crate::literal::{LiteralAVX, LiteralAVX512, LiteralSSE};

#[cfg(target_arch = "aarch64")]
use crate::prefilter::backend::PrefilterNEON;
#[cfg(scalar_fallback)]
use crate::prefilter::backend::PrefilterScalar;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::prefilter::backend::PrefilterWasm;
#[cfg(target_arch = "x86_64")]
use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};
#[cfg(target_arch = "x86_64")]
use crate::smith_waterman::{
    SmithWatermanAVX, SmithWatermanAVX512, SmithWatermanAVX512U8, SmithWatermanAVXU8,
    SmithWatermanSSE, SmithWatermanSSEU8,
};
#[cfg(target_arch = "aarch64")]
use crate::smith_waterman::{SmithWatermanNEON, SmithWatermanNEONU8};
#[cfg(scalar_fallback)]
use crate::smith_waterman::{SmithWatermanScalar, SmithWatermanScalarU8};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::smith_waterman::{SmithWatermanWasm, SmithWatermanWasmU8};

#[cfg(target_arch = "x86_64")]
pub type MatcherAVX512U8 = MatcherImpl<PrefilterAVX512, SmithWatermanAVX512U8>;
#[cfg(target_arch = "x86_64")]
pub type MatcherAVX512 = MatcherImpl<PrefilterAVX512, SmithWatermanAVX512>;
// The u8 and u16 AVX-512 kernels need different features (VBMI), so they
// stay separate backends; every other tier picks the width at construction
#[cfg(target_arch = "x86_64")]
pub type MatcherAVX = MatcherImpl<PrefilterAVX, ScoreWidth<SmithWatermanAVXU8, SmithWatermanAVX>>;
#[cfg(target_arch = "x86_64")]
pub type MatcherSSE = MatcherImpl<PrefilterSSE, ScoreWidth<SmithWatermanSSEU8, SmithWatermanSSE>>;
#[cfg(target_arch = "aarch64")]
pub type MatcherNEON =
    MatcherImpl<PrefilterNEON, ScoreWidth<SmithWatermanNEONU8, SmithWatermanNEON>>;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub type MatcherWasm =
    MatcherImpl<PrefilterWasm, ScoreWidth<SmithWatermanWasmU8, SmithWatermanWasm>>;
#[cfg(scalar_fallback)]
pub type MatcherScalar =
    MatcherImpl<PrefilterScalar, ScoreWidth<SmithWatermanScalarU8, SmithWatermanScalar>>;

#[derive(Debug, Clone)]
pub enum MatcherBackend {
    #[cfg(target_arch = "x86_64")]
    AVX512U8(MatcherAVX512U8),
    #[cfg(target_arch = "x86_64")]
    AVX512(MatcherAVX512),
    #[allow(clippy::upper_case_acronyms)]
    #[cfg(target_arch = "x86_64")]
    AVX(MatcherAVX),
    #[allow(clippy::upper_case_acronyms)]
    #[cfg(target_arch = "x86_64")]
    SSE(MatcherSSE),
    #[cfg(target_arch = "aarch64")]
    NEON(MatcherNEON),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Wasm(MatcherWasm),
    // Compiled out when a SIMD tier is statically available (see build.rs)
    #[cfg(scalar_fallback)]
    Scalar(MatcherScalar),

    // Literal matching backends (exact / prefix / suffix / substring). Selected when
    // `Config::matching` is not `Matching::Fuzzy`.
    #[cfg(target_arch = "x86_64")]
    LiteralAVX512(LiteralAVX512),
    #[cfg(target_arch = "x86_64")]
    LiteralAVX(LiteralAVX),
    #[cfg(target_arch = "x86_64")]
    LiteralSSE(LiteralSSE),
    #[cfg(target_arch = "aarch64")]
    LiteralNEON(LiteralNEON),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    LiteralWasm(LiteralWasm),
    #[cfg(scalar_fallback)]
    LiteralScalar(LiteralScalar),
}

/// Implements [`Specialized`] for one concrete backend, attaching the
/// backend's `#[target_feature]` to each method (see the [`Specialized`] docs
/// for how this establishes the feature boundary). The bodies forward to the
/// `#[inline(always)]` loop helpers on [`MatcherImpl`]. Scalar backends omit
/// the feature argument.
macro_rules! impl_specialized {
    ($prefilter:ty, $smith_waterman:ty $(, target_feature = $feature:literal)?) => {
        impl Specialized for MatcherImpl<$prefilter, $smith_waterman> {
            #[inline]
            $(#[target_feature(enable = $feature)])?
            unsafe fn build(needle: &str, config: &Config) -> Self {
                Self::new(needle, config)
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_list<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystacks: &[H],
                haystack_index_offset: u32,
                matches: &mut Vec<Match>,
            ) {
                self.match_list_into_impl::<TYPOS, UNICODE, H>(
                    haystacks,
                    haystack_index_offset,
                    matches,
                )
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_one_indices<const UNICODE: bool>(
                &mut self,
                haystack: &str,
                index: u32,
            ) -> Option<MatchIndices> {
                self.match_one_indices_impl::<UNICODE>(haystack, index)
            }
        }

        impl Scorer for MatcherImpl<$prefilter, $smith_waterman> {
            #[inline(never)]
            $(#[target_feature(enable = $feature)])?
            unsafe fn score<const UNICODE: bool>(
                &mut self,
                haystack: &[u8],
                haystack_start_pos: usize,
            ) -> u16 {
                self.score_inline::<UNICODE>(haystack, haystack_start_pos)
            }

            #[inline(never)]
            $(#[target_feature(enable = $feature)])?
            unsafe fn score_indices<const UNICODE: bool>(
                &mut self,
                haystack: &[u8],
                haystack_start_pos: usize,
                max_typos: Option<u16>,
            ) -> (u16, Vec<u32>) {
                self.score_indices_inline::<UNICODE>(haystack, haystack_start_pos, max_typos)
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
impl_specialized!(
    PrefilterAVX512,
    SmithWatermanAVX512U8,
    target_feature = "avx512f,avx512bw,avx512vbmi,bmi1,bmi2"
);
#[cfg(target_arch = "x86_64")]
impl_specialized!(
    PrefilterAVX512,
    SmithWatermanAVX512,
    target_feature = "avx512f,avx512bw,bmi1,bmi2"
);
#[cfg(target_arch = "x86_64")]
impl_specialized!(
    PrefilterAVX,
    ScoreWidth<SmithWatermanAVXU8, SmithWatermanAVX>,
    target_feature = "avx2"
);
#[cfg(target_arch = "x86_64")]
impl_specialized!(
    PrefilterSSE,
    ScoreWidth<SmithWatermanSSEU8, SmithWatermanSSE>,
    target_feature = "sse2,ssse3,sse4.1"
);
#[cfg(target_arch = "aarch64")]
impl_specialized!(
    PrefilterNEON,
    ScoreWidth<SmithWatermanNEONU8, SmithWatermanNEON>,
    target_feature = "neon"
);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl_specialized!(
    PrefilterWasm,
    ScoreWidth<SmithWatermanWasmU8, SmithWatermanWasm>,
    target_feature = "simd128"
);
#[cfg(scalar_fallback)]
impl_specialized!(
    PrefilterScalar,
    ScoreWidth<SmithWatermanScalarU8, SmithWatermanScalar>
);
