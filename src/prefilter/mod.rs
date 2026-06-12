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
//! Matcher chooses the concrete prefilter backend via runtime feature detection.
//! All algorithms assume that needle.len() > 0

pub(crate) mod algo;
pub(crate) mod backend;

use algo::Prefilter;
use backend::Backend;

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

pub(crate) type Window = (bool, usize, usize);

/// Ordered prefiltering kernel which allows score-level false positives.
pub(crate) trait Kernel: Clone + std::fmt::Debug + 'static {
    fn new(needle: &[u8]) -> Self;

    fn match_haystack(&self, haystack: &[u8]) -> Window;
    fn match_haystack_1_typo(&self, haystack: &[u8]) -> Window;
    fn match_haystack_2_typos(&self, haystack: &[u8]) -> Window;
    fn match_haystack_typos(&mut self, haystack: &[u8], max_typos: u16) -> Window;
}

impl<B: Backend> Kernel for Prefilter<B> {
    #[inline(always)]
    fn new(needle: &[u8]) -> Self {
        unsafe { Self::new(needle) }
    }

    #[inline(always)]
    fn match_haystack(&self, haystack: &[u8]) -> Window {
        unsafe { self.match_haystack(haystack) }
    }

    #[inline(always)]
    fn match_haystack_1_typo(&self, haystack: &[u8]) -> Window {
        unsafe { self.match_haystack_1_typo(haystack) }
    }

    #[inline(always)]
    fn match_haystack_2_typos(&self, haystack: &[u8]) -> Window {
        unsafe { self.match_haystack_2_typos(haystack) }
    }

    #[inline(always)]
    fn match_haystack_typos(&mut self, haystack: &[u8], max_typos: u16) -> Window {
        match max_typos {
            0 => Kernel::match_haystack(self, haystack),
            1 => Kernel::match_haystack_1_typo(self, haystack),
            2 => Kernel::match_haystack_2_typos(self, haystack),
            _ => unsafe { self.match_haystack_many_typos(haystack, max_typos) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Kernel, Window, backend::PrefilterScalar};

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

    #[test]
    fn backend_parity_suite() {
        for (needle, haystack, max_typos) in [
            ("foo", "foo", 0),
            ("foo", "oof", 0),
            ("foo", "f_o_o", 0),
            ("foo", "f_______________o_______________o", 0),
            ("\0", "abc", 0),
            ("a", "", 0),
            ("bar", "ba", 1),
            ("abc", "c", 2),
            ("bar", "rb", 1),
            ("a\0b", "ab", 1),
            ("abcdef", "abdf", 2),
            ("abcdef", "fcda", 2),
            ("abc", "", 3),
            ("abcdefghij", "abxxcxxdxxe", 5),
            ("abcdefghij", "jihgfedcba", 5),
            ("abcdefghij", "abc", 8),
        ] {
            result_generic(needle, haystack, max_typos);
        }
    }

    fn result_generic(needle: &str, haystack: &str, max_typos: u16) -> (bool, usize, usize) {
        let haystack = haystack.as_bytes();
        let scalar_result =
            kernel_result::<PrefilterScalar>(needle.as_bytes(), haystack, max_typos);

        #[cfg(target_arch = "x86_64")]
        {
            use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};

            if PrefilterAVX::is_available() {
                let avx_result =
                    kernel_result::<PrefilterAVX>(needle.as_bytes(), haystack, max_typos);
                assert_same_result(avx_result, scalar_result, "AVX2 mismatch");
            }

            if PrefilterSSE::is_available() {
                let sse_result =
                    kernel_result::<PrefilterSSE>(needle.as_bytes(), haystack, max_typos);
                assert_same_result(sse_result, scalar_result, "SSE mismatch");
            }

            if PrefilterAVX512::is_available() {
                let avx512_result =
                    kernel_result::<PrefilterAVX512>(needle.as_bytes(), haystack, max_typos);
                assert_same_result(avx512_result, scalar_result, "AVX-512 mismatch");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            use crate::prefilter::backend::PrefilterNEON;

            let neon_result =
                kernel_result::<PrefilterNEON>(needle.as_bytes(), haystack, max_typos);
            assert_same_result(neon_result, scalar_result, "NEON mismatch");
        }

        scalar_result
    }

    fn kernel_result<P: Kernel>(needle: &[u8], haystack: &[u8], max_typos: u16) -> Window {
        let mut prefilter = P::new(needle);
        prefilter.match_haystack_typos(haystack, max_typos)
    }

    fn assert_same_result(got: (bool, usize, usize), want: (bool, usize, usize), context: &str) {
        if want.0 {
            assert_eq!(got, want, "{context}");
        } else {
            assert_eq!(got.0, want.0, "{context}");
        }
    }
}
