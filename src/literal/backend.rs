//! Target-feature-specific instantiations of the literal matcher, mirroring
//! `src/matcher/backend.rs`. Each backend attaches its `#[target_feature]` to
//! the [`Specialized`] methods, which forward to the `#[inline(always)]`
//! helpers on [`LiteralImpl`].

use super::algo::LiteralImpl;
use crate::matcher::algo::Specialized;
use crate::{Config, Match, MatchIndices};
use alloc::vec::Vec;

#[cfg(target_arch = "aarch64")]
use crate::prefilter::backend::PrefilterNEONBackend;
#[cfg(any(test, scalar_fallback))]
use crate::prefilter::backend::PrefilterScalarBackend;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::prefilter::backend::PrefilterWasmBackend;
#[cfg(target_arch = "x86_64")]
use crate::prefilter::backend::{PrefilterAVX512Backend, PrefilterAVXBackend, PrefilterSSEBackend};

#[cfg(target_arch = "x86_64")]
pub(crate) type LiteralAVX512 = LiteralImpl<PrefilterAVX512Backend>;
#[cfg(target_arch = "x86_64")]
pub(crate) type LiteralAVX = LiteralImpl<PrefilterAVXBackend>;
#[cfg(target_arch = "x86_64")]
pub(crate) type LiteralSSE = LiteralImpl<PrefilterSSEBackend>;
#[cfg(target_arch = "aarch64")]
pub(crate) type LiteralNEON = LiteralImpl<PrefilterNEONBackend>;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) type LiteralWasm = LiteralImpl<PrefilterWasmBackend>;
#[cfg(any(test, scalar_fallback))]
pub(crate) type LiteralScalar = LiteralImpl<PrefilterScalarBackend>;

/// Implements [`Specialized`] for one literal backend. `TYPOS` is ignored
/// (literal matching has no typo tolerance), and so is `UNICODE`: the needle
/// holds its own scalar boundaries.
macro_rules! impl_specialized_literal {
    ($backend:ty $(, target_feature = $feature:literal)?) => {
        impl Specialized for LiteralImpl<$backend> {
            #[inline]
            $(#[target_feature(enable = $feature)])?
            unsafe fn build(needle: &str, config: &Config) -> Self {
                unsafe { Self::new(needle, config) }
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_list<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
                &mut self,
                haystacks: &[H],
                haystack_index_offset: u32,
                matches: &mut Vec<Match>,
            ) {
                unsafe {
                    self.match_list_impl(haystacks, haystack_index_offset, matches)
                }
            }

            $(#[target_feature(enable = $feature)])?
            unsafe fn match_one_indices<const UNICODE: bool>(
                &mut self,
                haystack: &str,
                index: u32,
            ) -> Option<MatchIndices> {
                unsafe { self.match_one_indices_impl(haystack, index) }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
impl_specialized_literal!(
    PrefilterAVX512Backend,
    target_feature = "avx512f,avx512bw,bmi1,bmi2,popcnt"
);
#[cfg(target_arch = "x86_64")]
impl_specialized_literal!(PrefilterAVXBackend, target_feature = "avx2,popcnt");
#[cfg(target_arch = "x86_64")]
impl_specialized_literal!(PrefilterSSEBackend, target_feature = "sse2");
#[cfg(target_arch = "aarch64")]
impl_specialized_literal!(PrefilterNEONBackend, target_feature = "neon");
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl_specialized_literal!(PrefilterWasmBackend, target_feature = "simd128");
#[cfg(any(test, scalar_fallback))]
impl_specialized_literal!(PrefilterScalarBackend);

#[cfg(test)]
mod backend_parity {
    use crate::matcher::algo::Specialized;
    use crate::{Config, Matching, SortStrategy};

    /// Runs one needle/haystack through a specialized literal backend,
    /// returning the observable result of both `match_list` and
    /// `match_one_indices`. `UNICODE` is chosen the same way the real
    /// dispatch chooses it (`respects_unicode_for`), so non-ASCII needles
    /// exercise the codepoint path on every backend.
    #[allow(clippy::type_complexity)]
    unsafe fn probe<T: Specialized>(
        needle: &str,
        haystack: &str,
        config: &Config,
    ) -> (Option<(u16, bool)>, Option<Vec<u32>>) {
        let mut matcher = unsafe { T::build(needle, config) };
        let mut m = Vec::new();
        let i = if config.unicode.respects_unicode_for(needle) {
            unsafe { matcher.match_list::<0, true, &str>(&[haystack], 0, &mut m) };
            unsafe { matcher.match_one_indices::<true>(haystack, 0) }
        } else {
            unsafe { matcher.match_list::<0, false, &str>(&[haystack], 0, &mut m) };
            unsafe { matcher.match_one_indices::<false>(haystack, 0) }
        };
        (m.pop().map(|m| (m.score, m.exact)), i.map(|i| i.indices))
    }

    #[test]
    fn backends_agree_with_reference_on_edge_cases() {
        let backends = backends();
        let cases: &[(&str, &str)] = &[
            ("bar", "foobar"),
            ("bar", "foo_bar"),
            ("bar", "barbarbar"),
            ("ab", "ab_ab"),
            ("foo", "foo"),
            ("x", &"y".repeat(200)),
            ("needle", "xxneedlexxneedle_needle"),
            ("é다😀", "xxé다😀yyé다😀"),
            ("다", "가나다라마"),
            ("z", "abcdefghijklmnopqrstuvwxyz"),
            ("FoO", "prefix_FoO_FOO_foo"),
            // Two-byte prefilter stress: common first byte, len 1/2, first ==
            // last, match at the end, and overlapping occurrences.
            ("a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            ("ab", "abababababababababababab"),
            ("aa", "baaab"),
            ("aaaa", "xaaaaaaaax"),
            ("aaa", "aaaaa"),
            ("bar", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxbar"),
            ("ba", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaba"),
            ("foobar", "foobatefoobarfoobar"),
            // Unicode codepoint path with case folding: mixed-case occurrences,
            // script mixes, and the Cherokee hybrid (E1 8E A0 / EA AD B0) that
            // per-byte matching would wrongly accept.
            ("é", "xÉyéZÉ"),
            ("café", "un CAFÉ, deux cafés"),
            ("Ꭰ", "\u{1b70}Ꭰꭰ\u{1b70}"),
            ("иха", "МУХА_ИХА_иха"),
            ("αβ", "ΑΒβα_αβ"),
        ];

        for &(needle, haystack) in cases {
            for matching in [
                Matching::Exact,
                Matching::Prefix,
                Matching::Suffix,
                Matching::Substring,
            ] {
                let config = Config::default()
                    .matching(matching)
                    .sort(SortStrategy::IndexAsc);
                let expected = reference(needle, haystack, &config);
                for (name, probe) in &backends {
                    assert_eq!(
                        probe(needle, haystack, &config),
                        expected,
                        "{name} mismatch: needle={needle:?} haystack={haystack:?} {matching:?}"
                    );
                }
            }
        }
    }

    type Probe = fn(&str, &str, &Config) -> (Option<(u16, bool)>, Option<Vec<u32>>);

    fn backends() -> Vec<(&'static str, Probe)> {
        use crate::literal::LiteralScalar;
        let mut backends: Vec<(&'static str, Probe)> = vec![("scalar", |n, h, c| unsafe {
            probe::<LiteralScalar>(n, h, c)
        })];
        #[cfg(target_arch = "x86_64")]
        {
            use crate::literal::{LiteralAVX, LiteralAVX512, LiteralSSE};
            if LiteralAVX512::is_available() {
                backends.push(("avx512", |n, h, c| unsafe {
                    probe::<LiteralAVX512>(n, h, c)
                }));
            }
            if LiteralAVX::is_available() {
                backends.push(("avx2", |n, h, c| unsafe { probe::<LiteralAVX>(n, h, c) }));
            }
            if LiteralSSE::is_available() {
                backends.push(("sse", |n, h, c| unsafe { probe::<LiteralSSE>(n, h, c) }));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            use crate::literal::LiteralNEON;
            if LiteralNEON::is_available() {
                backends.push(("neon", |n, h, c| unsafe { probe::<LiteralNEON>(n, h, c) }));
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            use crate::literal::LiteralWasm;
            if LiteralWasm::is_available() {
                backends.push(("wasm", |n, h, c| unsafe { probe::<LiteralWasm>(n, h, c) }));
            }
        }
        backends
    }

    /// The literal semantics spelled out a scalar at a time: the earliest of
    /// the best-scoring positions the mode allows, in the shape of [`probe`]
    fn reference(
        needle: &str,
        haystack: &str,
        config: &Config,
    ) -> (Option<(u16, bool)>, Option<Vec<u32>>) {
        use crate::prefilter::{case_byte, case_needle_unicode};

        let case_sensitive = config.casing.respects_case_for(needle);
        let scalars: Vec<(Vec<u8>, Vec<u8>)> = if config.unicode.respects_unicode_for(needle) {
            case_needle_unicode(needle, case_sensitive)
                .iter()
                .map(|c| (c.chars[..c.len].to_vec(), c.flipped_chars[..c.len].to_vec()))
                .collect()
        } else {
            needle
                .bytes()
                .map(|byte| (vec![byte], vec![case_byte(byte, case_sensitive).1]))
                .collect()
        };

        let (haystack, scoring) = (haystack.as_bytes(), &config.scoring);
        if haystack.len() < needle.len() {
            return (None, None);
        }
        let last = haystack.len() - needle.len();
        let is_delimiter = |byte: u8| byte.is_ascii() && !byte.is_ascii_alphanumeric();
        let score_at = |pos: usize| {
            let (mut score, mut start) = (0u16, pos);
            let mut add = |bonus: u16| score = score.saturating_add(bonus);
            for (orig, flipped) in &scalars {
                let bytes = &haystack[start..start + orig.len()];
                if bytes != orig && bytes != flipped {
                    return None;
                }
                add(scoring.match_score);
                if bytes == orig {
                    add(scoring.matching_case_bonus);
                }
                if start == 0 {
                    add(scoring.prefix_bonus);
                } else {
                    let (prev, byte) = (haystack[start - 1], haystack[start]);
                    if prev.is_ascii_lowercase() && byte.is_ascii_uppercase() {
                        add(scoring.capitalization_bonus);
                    }
                    if is_delimiter(prev) && !is_delimiter(byte) {
                        add(scoring.delimiter_bonus);
                    }
                }
                start += orig.len();
            }
            if pos == 0 && last == 0 {
                add(scoring.exact_match_bonus);
            }
            Some(score)
        };

        let positions = match config.matching {
            Matching::Exact if last != 0 => return (None, None),
            Matching::Exact | Matching::Prefix => 0..=0,
            Matching::Suffix => last..=last,
            Matching::Substring => 0..=last,
            Matching::Fuzzy => unreachable!(),
        };
        // `max_by_key` keeps the last maximum, so walk the positions backwards
        let best = positions
            .rev()
            .filter_map(|pos| Some((score_at(pos)?, pos)))
            .max_by_key(|&(score, _)| score);
        best.map(|(score, pos)| {
            let indices = (pos..pos + needle.len()).rev().map(|i| i as u32).collect();
            ((score, pos == 0 && last == 0), indices)
        })
        .unzip()
    }

    #[test]
    fn backends_agree_with_reference_on_random_cases() {
        use crate::smith_waterman::reference::Rng;
        use crate::{CaseMatching, Scoring, UnicodeMatching};

        let alphabet: Vec<char> = "abcABC_/.1éÉиИ다Ꭰꭰß".chars().collect();
        let scorings = [
            Scoring::default(),
            // a capitalization worth more than a matching case
            Scoring {
                capitalization_bonus: 9,
                matching_case_bonus: 2,
                delimiter_bonus: 5,
                prefix_bonus: 7,
                exact_match_bonus: 3,
                ..Scoring::default()
            },
            // saturates on long needles
            Scoring {
                match_score: 9000,
                ..Scoring::default()
            },
        ];
        let backends = backends();
        let mut rng = Rng(0x0117_e2a1);
        for round in 0..if cfg!(miri) { 40 } else { 40_000 } {
            // mostly short needles, some longer than every backend's chunk
            let needle_len = 1 + rng.below(if round % 16 == 0 { 80 } else { 5 });
            let needle = rng.string(&alphabet, needle_len);
            // random text around needles planted in a random case
            let gap = if round % 8 == 0 { 90 } else { 10 };
            let mut haystack = String::new();
            for _ in 0..rng.below(4) {
                let filler = rng.below(gap);
                haystack += &rng.string(&alphabet, filler);
                for c in needle.chars() {
                    let flipped = if c.is_lowercase() {
                        c.to_uppercase().next()
                    } else {
                        c.to_lowercase().next()
                    };
                    haystack.push(if rng.below(4) == 0 {
                        flipped.unwrap()
                    } else {
                        c
                    });
                }
            }
            let filler = rng.below(gap);
            haystack += &rng.string(&alphabet, filler);

            let config = Config::default()
                .matching(rng.pick(&[
                    Matching::Exact,
                    Matching::Prefix,
                    Matching::Suffix,
                    Matching::Substring,
                    Matching::Substring,
                ]))
                .casing(rng.pick(&[
                    CaseMatching::Smart,
                    CaseMatching::Ignore,
                    CaseMatching::Respect,
                ]))
                .unicode(rng.pick(&[
                    UnicodeMatching::Smart,
                    UnicodeMatching::Ignore,
                    UnicodeMatching::Always,
                ]))
                .scoring(scorings[rng.below(scorings.len())].clone())
                .sort(SortStrategy::IndexAsc);
            let expected = reference(&needle, &haystack, &config);
            for (name, probe) in &backends {
                assert_eq!(
                    probe(&needle, &haystack, &config),
                    expected,
                    "{name} mismatch: needle={needle:?} haystack={haystack:?} {config:?}"
                );
            }
        }
    }
}
