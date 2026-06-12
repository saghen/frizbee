//! Fast prefiltering algorithms, which run before Smith Waterman since in the typical case,
//! a small percentage of the haystack will match the needle. Automatically used by the Matcher
//! and match_list APIs.
//!
//! The prefilter proves that an ordered alignment exists after deleting at
//! most `max_typos` needle bytes. Substitution is relaxed to deletion here:
//! any alignment with a mismatched byte is also accepted by deleting that
//! needle byte. This can still produce score-level false positives, but it
//! cannot reject a haystack that Smith-Waterman could accept.
//!
//! The `Prefilter` struct chooses the fastest algorithm via runtime feature detection.
//! All algorithms assume that needle.len() > 0

pub(crate) mod algo;
pub(crate) mod backend;

#[cfg(target_arch = "aarch64")]
use backend::PrefilterNEON;
use backend::PrefilterScalar;
#[cfg(target_arch = "x86_64")]
use backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};

pub(crate) fn case_needle(needle: &[u8]) -> Vec<(u8, u8)> {
    needle
        .iter()
        .map(|&c| {
            (
                c,
                if c.is_ascii_lowercase() {
                    c.to_ascii_uppercase()
                } else {
                    c.to_ascii_lowercase()
                },
            )
        })
        .collect()
}

/// Ordered prefiltering algorithm which allows score-level false positives.
/// Chooses the fastest implementation via runtime feature detection.
#[derive(Debug, Clone)]
pub enum Prefilter {
    #[cfg(target_arch = "x86_64")]
    AVX512(PrefilterAVX512),
    #[cfg(target_arch = "x86_64")]
    AVX(PrefilterAVX),
    #[cfg(target_arch = "x86_64")]
    SSE(PrefilterSSE),
    #[cfg(target_arch = "aarch64")]
    NEON(PrefilterNEON),
    Scalar(PrefilterScalar),
}

impl Prefilter {
    pub fn new(needle: &[u8]) -> Self {
        #[cfg(target_arch = "x86_64")]
        if PrefilterAVX512::is_available() {
            return Prefilter::AVX512(unsafe { PrefilterAVX512::new(needle) });
        }
        #[cfg(target_arch = "x86_64")]
        if PrefilterAVX::is_available() {
            return Prefilter::AVX(unsafe { PrefilterAVX::new(needle) });
        }
        #[cfg(target_arch = "x86_64")]
        if PrefilterSSE::is_available() {
            return Prefilter::SSE(unsafe { PrefilterSSE::new(needle) });
        }

        #[cfg(target_arch = "aarch64")]
        return Prefilter::NEON(PrefilterNEON::new(needle));

        #[cfg(not(target_arch = "aarch64"))]
        Prefilter::Scalar(PrefilterScalar::new(needle))
    }

    /// Checks whether the needle can be aligned to the haystack after deleting
    /// at most `max_typos` needle bytes.
    ///
    /// Returns `(matched, skipped_chars, end_pos)`:
    /// - `skipped_chars`: conservative byte offset of the first possible matched
    ///   needle byte.
    /// - `end_pos`: conservative exclusive byte offset after the final possible
    ///   matched needle byte.
    ///
    /// The caller can slice `haystack[skipped_chars..end_pos]` to drop unmatched
    /// prefix and suffix bytes before scoring. If all needle bytes may be
    /// deleted, the full haystack is returned so scoring and exact matching can
    /// still use the original bytes.
    ///
    /// The caller must ensure needle.len() > 0
    #[inline]
    pub fn match_haystack(&mut self, haystack: &[u8], max_typos: u16) -> (bool, usize, usize) {
        match (self, max_typos) {
            #[cfg(target_arch = "x86_64")]
            (Prefilter::AVX512(p), 0) => unsafe { p.match_haystack(haystack) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::AVX512(p), 1) => unsafe { p.match_haystack_1_typo(haystack) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::AVX512(p), 2) => unsafe { p.match_haystack_2_typos(haystack) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::AVX512(p), _) => unsafe { p.match_haystack_typos(haystack, max_typos) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::AVX(p), 0) => unsafe { p.match_haystack(haystack) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::AVX(p), 1) => unsafe { p.match_haystack_1_typo(haystack) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::AVX(p), 2) => unsafe { p.match_haystack_2_typos(haystack) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::AVX(p), _) => unsafe { p.match_haystack_typos(haystack, max_typos) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::SSE(p), 0) => unsafe { p.match_haystack(haystack) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::SSE(p), 1) => unsafe { p.match_haystack_1_typo(haystack) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::SSE(p), 2) => unsafe { p.match_haystack_2_typos(haystack) },
            #[cfg(target_arch = "x86_64")]
            (Prefilter::SSE(p), _) => unsafe { p.match_haystack_typos(haystack, max_typos) },
            #[cfg(target_arch = "aarch64")]
            (Prefilter::NEON(p), 0) => unsafe { p.match_haystack(haystack) },
            #[cfg(target_arch = "aarch64")]
            (Prefilter::NEON(p), 1) => unsafe { p.match_haystack_1_typo(haystack) },
            #[cfg(target_arch = "aarch64")]
            (Prefilter::NEON(p), 2) => unsafe { p.match_haystack_2_typos(haystack) },
            #[cfg(target_arch = "aarch64")]
            (Prefilter::NEON(p), _) => unsafe { p.match_haystack_typos(haystack, max_typos) },
            (Prefilter::Scalar(p), 0) => p.match_haystack(haystack),
            (Prefilter::Scalar(p), _) => p.match_haystack_typos(haystack, max_typos),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Prefilter, backend::PrefilterScalar};

    fn result(needle: &str, haystack: &str, max_typos: u16) -> (bool, usize, usize) {
        result_generic(needle, haystack, max_typos)
    }

    fn matched(needle: &str, haystack: &str, max_typos: u16) -> bool {
        result(needle, haystack, max_typos).0
    }

    #[test]
    fn ordered_matching_cases() {
        for (needle, haystack, max_typos, want) in [
            ("foo", "foo", 0, true),
            ("foo", "f_o_o", 0, true),
            ("foo", "FOO", 0, true),
            ("abc", "xaxbxcx", 0, true),
            ("fo", "_______________fo", 0, true),
            ("foo", "f_______________o_______________o", 0, true),
            ("foo", "oof", 0, false),
            ("abc", "cba", 0, false),
            ("foo", "fo", 0, false),
            ("foo", "f_________________________o______", 0, false),
            ("a", "", 0, false),
            ("\0", "abc", 0, false),
            ("aa", "a", 0, false),
        ] {
            assert_eq!(
                matched(needle, haystack, max_typos),
                want,
                "needle={needle:?} haystack={haystack:?} max_typos={max_typos}"
            );
        }
    }

    #[test]
    fn typo_matching_cases() {
        for (needle, haystack, max_typos, want) in [
            ("bar", "ba", 1, true),
            ("bar", "ar", 1, true),
            ("hello", "hll", 2, true),
            ("abcdef", "abdf", 2, true),
            ("TeSt", "ES", 2, true),
            ("abc", "c", 2, true),
            ("a\0b", "ab", 1, true),
            ("abc", "", 3, true),
            ("foo", "fo", 5, true),
            ("abc", "a_______________b", 1, true),
            ("test", "t_______________s_______________t", 1, true),
            ("bar", "rb", 1, false),
            ("abcdef", "fcda", 2, false),
            ("TeSt", "ES", 1, false),
            ("abc", "", 2, false),
        ] {
            assert_eq!(
                matched(needle, haystack, max_typos),
                want,
                "needle={needle:?} haystack={haystack:?} max_typos={max_typos}"
            );
        }
    }

    #[test]
    fn returned_windows_are_conservative() {
        assert_eq!(result("foo", "xxfooxfoo", 0), (true, 2, 9));
        assert_eq!(result("abc", "xxaybzczz", 0), (true, 2, 7));
        assert_eq!(result("abcd", "xxaydz", 2), (true, 2, 5));
        assert_eq!(result("abc", "xyz", 3), (true, 0, 3));
    }

    fn result_generic(needle: &str, haystack: &str, max_typos: u16) -> (bool, usize, usize) {
        let haystack = haystack.as_bytes();
        let scalar_result = PrefilterScalar::new(needle.as_bytes()).match_haystack(haystack);
        let scalar_result = if max_typos == 0 {
            scalar_result
        } else {
            PrefilterScalar::new(needle.as_bytes()).match_haystack_typos(haystack, max_typos)
        };

        let mut selected_prefilter = Prefilter::new(needle.as_bytes());
        let selected = selected_prefilter.match_haystack(haystack, max_typos);
        assert_same_result(
            selected,
            scalar_result,
            &format!(
                "selected backend result mismatch for needle={needle:?} haystack={:?} max_typos={max_typos}",
                String::from_utf8_lossy(haystack)
            ),
        );

        #[cfg(target_arch = "x86_64")]
        {
            use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};

            if PrefilterAVX::is_available() {
                let avx_result = unsafe {
                    let mut prefilter = PrefilterAVX::new(needle.as_bytes());
                    if max_typos == 0 {
                        prefilter.match_haystack(haystack)
                    } else {
                        prefilter.match_haystack_typos(haystack, max_typos)
                    }
                };
                assert_same_result(avx_result, scalar_result, "AVX2 mismatch");
            }

            if PrefilterSSE::is_available() {
                let sse_result = unsafe {
                    let mut prefilter = PrefilterSSE::new(needle.as_bytes());
                    if max_typos == 0 {
                        prefilter.match_haystack(haystack)
                    } else {
                        prefilter.match_haystack_typos(haystack, max_typos)
                    }
                };
                assert_same_result(sse_result, scalar_result, "SSE mismatch");
            }

            if PrefilterAVX512::is_available() {
                let mut prefilter =
                    Prefilter::AVX512(unsafe { PrefilterAVX512::new(needle.as_bytes()) });
                let avx512_result = prefilter.match_haystack(haystack, max_typos);
                assert_same_result(avx512_result, scalar_result, "AVX-512 mismatch");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon_result = unsafe {
                use crate::prefilter::backend::PrefilterNEON;
                let mut prefilter = PrefilterNEON::new(needle.as_bytes());
                if max_typos == 0 {
                    prefilter.match_haystack(haystack)
                } else {
                    prefilter.match_haystack_typos(haystack, max_typos)
                }
            };
            assert_same_result(neon_result, scalar_result, "NEON mismatch");
        }

        scalar_result
    }

    fn assert_same_result(got: (bool, usize, usize), want: (bool, usize, usize), context: &str) {
        if want.0 {
            assert_eq!(got, want, "{context}");
        } else {
            assert_eq!(got.0, want.0, "{context}");
        }
    }
}
