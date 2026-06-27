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
//! Matching assumes that needle.len() > 0, but backends may be constructed for
//! empty needles so `Matcher` can still select a concrete backend up front.

pub(crate) mod algo;
pub(crate) mod backend;

use algo::Prefilter;
use backend::Backend;

#[derive(Debug, Clone, Copy)]
pub struct UnicodeChar {
    pub chars: [u8; 4],
    pub len: usize,
}

impl UnicodeChar {
    pub fn new(c: char) -> Self {
        let mut chars = [0; 4];
        c.encode_utf8(&mut chars);
        Self {
            chars,
            len: c.len_utf8(),
        }
    }
}

pub(crate) fn case_needle(needle: &[u8], case_sensitive: bool) -> Vec<(u8, u8)> {
    needle
        .iter()
        .map(|&c| {
            (
                c,
                if case_sensitive {
                    c
                } else if c.is_ascii_lowercase() {
                    c.to_ascii_uppercase()
                } else {
                    c.to_ascii_lowercase()
                },
            )
        })
        .collect()
}

// pub(crate) fn case_needle_unicode(needle: &str, case_sensitive: bool) -> Vec<UnicodeChar<u8>> {
//     needle
//         .chars()
//         .map(|c| {
//             let len = c.len_utf8();
//             // TODO: for now, this should ignore cases where more than one byte is changed
//             let opposite_case = (if !case_sensitive && c.is_uppercase() {
//                 let mut lower = c.to_lowercase();
//                 let lower_char = lower.next().unwrap_or(c);
//
//                 // ignore cases where there's multiple variations or the length doesnt match
//                 (lower.next().is_none() && lower_char.len_utf8() == len).then_some(lower_char)
//             } else if !case_sensitive && c.is_lowercase() {
//                 let mut upper = c.to_uppercase();
//                 let upper_char = upper.next().unwrap_or(c);
//
//                 (upper.next().is_none() && upper_char.len_utf8() == len).then_some(upper_char)
//             } else {
//                 None
//             })
//             .unwrap_or(c);
//
//             // TODO: nicer way to write this?
//             let mut normal_case_char = [0u8; 4];
//             let mut opposite_case_char = [0u8; 4];
//             c.encode_utf8(&mut normal_case_char);
//             opposite_case.encode_utf8(&mut opposite_case_char);
//
//             let mut chars = [(0u8, 0u8); 4];
//             for i in 0..c.len_utf8() {
//                 chars[i] = (normal_case_char[i], opposite_case_char[i]);
//             }
//
//             UnicodeChar {
//                 chars,
//                 len: c.len_utf8(),
//             }
//         })
//         .collect()
// }

pub(crate) type Window = (bool, usize, usize);

/// Ordered prefiltering kernel which allows score-level false positives.
pub(crate) trait Kernel: Clone + std::fmt::Debug + 'static {
    fn new(needle: &str, case_sensitive: bool) -> Self;
    fn is_available() -> bool;

    fn match_haystack(&self, haystack: &[u8]) -> Window;
    fn match_haystack_unicode(&self, haystack: &[u8]) -> Window;
    fn match_haystack_1_typo(&self, haystack: &[u8]) -> Window;
    fn match_haystack_2_typos(&self, haystack: &[u8]) -> Window;
    fn match_haystack_many_typos(&mut self, haystack: &[u8], max_typos: u16) -> Window;
}

impl<B: Backend> Kernel for Prefilter<B> {
    #[inline(always)]
    fn new(needle: &str, case_sensitive: bool) -> Self {
        unsafe { Self::new(needle, case_sensitive) }
    }

    #[inline(always)]
    fn is_available() -> bool {
        B::is_available()
    }

    #[inline(always)]
    fn match_haystack(&self, haystack: &[u8]) -> Window {
        unsafe { self.match_haystack(haystack) }
    }

    #[inline(always)]
    fn match_haystack_unicode(&self, haystack: &[u8]) -> Window {
        unsafe { self.match_haystack_unicode(haystack) }
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
    fn match_haystack_many_typos(&mut self, haystack: &[u8], max_typos: u16) -> Window {
        unsafe { self.match_haystack_many_typos(haystack, max_typos) }
    }
}

#[cfg(test)]
mod tests {
    use super::{Kernel, Window, backend::PrefilterScalar};
    use bolero::check;

    fn result(needle: &str, haystack: &str, max_typos: u16) -> (bool, usize, usize) {
        result_generic(needle, haystack, max_typos)
    }

    fn matched(needle: &str, haystack: &str, max_typos: u16) -> bool {
        result(needle, haystack, max_typos).0
    }

    fn matched_sensitive(needle: &str, haystack: &str, max_typos: u16) -> bool {
        kernel_result::<PrefilterScalar>(needle, haystack.as_bytes(), max_typos, true).0
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
            ("abc", "", 2, false),
            ("abc", "", 3, true),
            ("abc", "bc", 1, true),
            ("abc", "ac", 1, true),
            ("abc", "ab", 1, true),
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
            ("d63NacaDJaaaa", "63aeeaaaeeaaaaaaaNacaDJaaAa", 1, true),
            ("bar", "rb", 1, false),
            ("abcdef", "fcda", 2, false),
            ("TeSt", "ES", 1, false),
            ("abc", "cba", 1, false),
            ("abc", "cba", 2, true),
            ("aaa", "aa", 0, false),
            ("aaa", "aa", 1, true),
            ("aba", "aa", 1, true),
            ("aaba", "aba", 1, true),
        ] {
            assert_eq!(
                matched(needle, haystack, max_typos),
                want,
                "needle={needle:?} haystack={haystack:?} max_typos={max_typos}"
            );
        }
    }

    #[test]
    fn case_sensitive_matching_cases() {
        for (needle, haystack, max_typos, want) in [
            ("foo", "foo", 0, true),
            ("foo", "FOO", 0, false),
            ("FoO", "xxFoOxx", 0, true),
            ("abc", "xaxbxcx", 0, true),
            ("abc", "xAxBxCx", 0, false),
            ("TeSt", "eS", 2, true),
            ("TeSt", "ES", 2, false),
            ("Ab", "b", 1, true),
            ("Ab", "ab", 0, false),
            ("Ab", "ab", 1, true),
        ] {
            assert_eq!(
                matched_sensitive(needle, haystack, max_typos),
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
    fn unicode_prefilter_matches_full_utf8_chars() {
        for (needle, haystack, want) in [
            ("إن", "xxإنyy", (true, 2, 6)),
            ("니다", "xx니__다yy", (true, 2, 10)),
            ("😀", "xx😀yy", (true, 2, 6)),
        ] {
            assert_eq!(
                unicode_result_generic(needle, haystack, false),
                want,
                "needle={needle:?} haystack={haystack:?}"
            );
        }
    }

    #[test]
    fn unicode_prefilter_rejects_same_final_bytes_with_wrong_prefixes() {
        let wrong_first = "\u{06e5}";
        let wrong_second = "\u{0606}";
        assert_eq!("إ".as_bytes()[1], wrong_first.as_bytes()[1]);
        assert_ne!("إ".as_bytes()[0], wrong_first.as_bytes()[0]);
        assert_eq!("ن".as_bytes()[1], wrong_second.as_bytes()[1]);
        assert_ne!("ن".as_bytes()[0], wrong_second.as_bytes()[0]);

        let false_positive_bytes = format!("{wrong_first}{wrong_second}");
        assert_eq!(
            unicode_result_generic("إن", &false_positive_bytes, false).0,
            false,
            "last-byte-only match should not pass UTF-8 sequence verification"
        );

        let haystack = format!("{false_positive_bytes}__إن");
        assert_eq!(
            unicode_result_generic("إن", &haystack, false),
            (true, false_positive_bytes.len() + 2, haystack.len())
        );
    }

    #[test]
    fn unicode_prefilter_matches_across_chunk_boundaries() {
        for prefix_len in [0usize, 1, 7, 14, 15, 16, 31, 32, 63, 64] {
            let haystack = format!("{}إن", "x".repeat(prefix_len));
            assert_eq!(
                unicode_result_generic("إن", &haystack, false),
                (true, prefix_len, haystack.len()),
                "prefix_len={prefix_len}"
            );
        }
    }

    #[test]
    fn unicode_prefilter_back_scans_final_char() {
        let haystack = format!("xxإن{}نzz", "x".repeat(32));
        let expected_end = haystack.rfind('ن').unwrap() + 'ن'.len_utf8();

        assert_eq!(
            unicode_result_generic("إن", &haystack, false),
            (true, 2, expected_end)
        );
    }

    #[test]
    fn unicode_prefilter_respects_case_setting() {
        assert_eq!(unicode_result_generic("É", "é", false), (true, 0, 2));
        assert_eq!(unicode_result_generic("É", "é", true).0, false);
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
            ("abcdefghijklmnop", "abcxxxdefxxxghixxxjklxxxmnop", 4),
            ("abcdefghijklmnop", "ponmlkjihgfedcba", 10),
        ] {
            result_generic(needle, haystack, max_typos);
        }
    }

    #[test]
    fn reference_oracle_manual_cases() {
        for (needle, haystack, max_typos, case_sensitive, want) in [
            ("abc", b"".as_slice(), 2, false, false),
            ("abc", b"".as_slice(), 3, false, true),
            ("abc", b"bc".as_slice(), 1, false, true),
            ("abc", b"ac".as_slice(), 1, false, true),
            ("abc", b"ab".as_slice(), 1, false, true),
            ("abc", b"cba".as_slice(), 2, false, true),
            ("aaa", b"aa".as_slice(), 1, false, true),
            ("aba", b"aa".as_slice(), 1, false, true),
            ("Ab", b"ab".as_slice(), 0, false, true),
            ("Ab", b"ab".as_slice(), 0, true, false),
            ("A\0b", b"a\0b".as_slice(), 0, false, true),
            ("A\0b", b"a\0b".as_slice(), 0, true, false),
            ("éa", "é_a".as_bytes(), 0, false, true),
            ("ÿA", "ÿa".as_bytes(), 0, false, true),
            ("ÿA", "ÿa".as_bytes(), 0, true, false),
        ] {
            let oracle = reference_matches_by_deleting_needle_bytes(
                needle.as_bytes(),
                haystack,
                max_typos,
                case_sensitive,
            );
            assert_eq!(
                oracle, want,
                "oracle mismatch needle={needle:?} haystack={haystack:?} max_typos={max_typos} case_sensitive={case_sensitive}"
            );

            let case = PrefilterCase {
                needle: needle.to_owned(),
                haystack: haystack.to_vec(),
                max_typos,
                case_sensitive,
            };
            assert_case_matches_oracle(&case);
        }
    }

    #[test]
    fn reference_oracle_chunk_boundaries() {
        for prefix_len in [0usize, 1, 7, 8, 15, 16, 31, 32, 63, 64] {
            let mut haystack = vec![b'x'; prefix_len];
            haystack.extend_from_slice(b"abc");

            for (needle, max_typos, want) in [
                ("abc", 0, true),
                ("ac", 0, true),
                ("abcd", 0, false),
                ("abcd", 1, true),
            ] {
                let case = PrefilterCase {
                    needle: needle.to_owned(),
                    haystack: haystack.clone(),
                    max_typos,
                    case_sensitive: false,
                };
                assert_eq!(
                    reference_matches_by_deleting_needle_bytes(
                        case.needle.as_bytes(),
                        &case.haystack,
                        case.max_typos,
                        case.case_sensitive,
                    ),
                    want,
                    "prefix_len={prefix_len} needle={needle:?} max_typos={max_typos}"
                );
                assert_case_matches_oracle(&case);
            }
        }
    }

    fn result_generic(needle: &str, haystack: &str, max_typos: u16) -> (bool, usize, usize) {
        let haystack = haystack.as_bytes();
        let scalar_result = kernel_result::<PrefilterScalar>(needle, haystack, max_typos, false);

        #[cfg(target_arch = "x86_64")]
        {
            use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};

            if PrefilterAVX::is_available() {
                let avx_result = kernel_result::<PrefilterAVX>(needle, haystack, max_typos, false);
                assert_same_result(avx_result, scalar_result, "AVX2 mismatch");
            }

            if PrefilterSSE::is_available() {
                let sse_result = kernel_result::<PrefilterSSE>(needle, haystack, max_typos, false);
                assert_same_result(sse_result, scalar_result, "SSE mismatch");
            }

            if PrefilterAVX512::is_available() {
                let avx512_result =
                    kernel_result::<PrefilterAVX512>(needle, haystack, max_typos, false);
                assert_same_result(avx512_result, scalar_result, "AVX-512 mismatch");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            use crate::prefilter::backend::PrefilterNEON;

            let neon_result = kernel_result::<PrefilterNEON>(needle, haystack, max_typos, false);
            assert_same_result(neon_result, scalar_result, "NEON mismatch");
        }

        scalar_result
    }

    fn unicode_result_generic(
        needle: &str,
        haystack: &str,
        case_sensitive: bool,
    ) -> (bool, usize, usize) {
        let haystack = haystack.as_bytes();
        let scalar_result =
            kernel_result_unicode::<PrefilterScalar>(needle, haystack, case_sensitive);

        #[cfg(target_arch = "x86_64")]
        {
            use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};

            if PrefilterAVX::is_available() {
                let avx_result =
                    kernel_result_unicode::<PrefilterAVX>(needle, haystack, case_sensitive);
                assert_same_result(avx_result, scalar_result, "AVX2 unicode mismatch");
            }

            if PrefilterSSE::is_available() {
                let sse_result =
                    kernel_result_unicode::<PrefilterSSE>(needle, haystack, case_sensitive);
                assert_same_result(sse_result, scalar_result, "SSE unicode mismatch");
            }

            if PrefilterAVX512::is_available() {
                let avx512_result =
                    kernel_result_unicode::<PrefilterAVX512>(needle, haystack, case_sensitive);
                assert_same_result(avx512_result, scalar_result, "AVX-512 unicode mismatch");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            use crate::prefilter::backend::PrefilterNEON;

            if PrefilterNEON::is_available() {
                let neon_result =
                    kernel_result_unicode::<PrefilterNEON>(needle, haystack, case_sensitive);
                assert_same_result(neon_result, scalar_result, "NEON unicode mismatch");
            }
        }

        scalar_result
    }

    fn kernel_result<P: Kernel>(
        needle: &str,
        haystack: &[u8],
        max_typos: u16,
        case_sensitive: bool,
    ) -> Window {
        let mut prefilter = P::new(needle, case_sensitive);
        match max_typos {
            0 => prefilter.match_haystack(haystack),
            1 => prefilter.match_haystack_1_typo(haystack),
            2 => prefilter.match_haystack_2_typos(haystack),
            _ => prefilter.match_haystack_many_typos(haystack, max_typos),
        }
    }

    fn kernel_result_unicode<P: Kernel>(
        needle: &str,
        haystack: &[u8],
        case_sensitive: bool,
    ) -> Window {
        P::new(needle, case_sensitive).match_haystack_unicode(haystack)
    }

    #[derive(Debug, Clone)]
    struct PrefilterCase {
        needle: String,
        haystack: Vec<u8>,
        max_typos: u16,
        case_sensitive: bool,
    }

    impl PrefilterCase {
        fn from_bytes(input: &[u8]) -> Self {
            let mut cursor = ByteCursor::new(input);
            let needle_len = cursor
                .len(test_bound(96, 32), &[1, 7, 8, 15, 16, 31, 32, 63, 64])
                .max(1);
            let haystack_len = cursor.len(
                test_bound(768, 128),
                &[0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 511, 512, 513],
            );
            let max_typos = (cursor.next() as u16) % 17;
            let case_sensitive = cursor.bool();

            Self {
                needle: cursor.string(needle_len),
                haystack: cursor.bytes(haystack_len),
                max_typos,
                case_sensitive,
            }
        }
    }

    struct ByteCursor<'a> {
        input: &'a [u8],
        pos: usize,
    }

    impl<'a> ByteCursor<'a> {
        fn new(input: &'a [u8]) -> Self {
            Self { input, pos: 0 }
        }

        fn next(&mut self) -> u8 {
            let byte = if self.input.is_empty() {
                (self.pos as u8).wrapping_mul(37).wrapping_add(11)
            } else {
                self.input[self.pos % self.input.len()]
                    .wrapping_add(((self.pos / self.input.len()) as u8).wrapping_mul(17))
            };
            self.pos += 1;
            byte
        }

        fn bool(&mut self) -> bool {
            self.next() & 1 == 1
        }

        fn usize(&mut self) -> usize {
            let mut value = 0usize;
            for shift in (0..usize::BITS as usize).step_by(8) {
                value |= (self.next() as usize) << shift;
            }
            value
        }

        fn len(&mut self, max: usize, boundaries: &[usize]) -> usize {
            if self.next() % 4 == 0 {
                boundaries[(self.next() as usize) % boundaries.len()].min(max)
            } else {
                self.usize() % (max + 1)
            }
        }

        fn bytes(&mut self, len: usize) -> Vec<u8> {
            (0..len).map(|_| self.byte()).collect()
        }

        fn string(&mut self, len: usize) -> String {
            (0..len).map(|_| self.char()).collect()
        }

        fn char(&mut self) -> char {
            let byte = self.next();
            match byte % 16 {
                0 => '\0',
                1 => ' ',
                2 => '/',
                3 => '.',
                4 => ',',
                5 => '_',
                6 => '-',
                7 => ':',
                8..=10 => (b'a' + (byte % 26)) as char,
                11..=13 => (b'A' + (byte % 26)) as char,
                _ => (b'0' + (byte % 10)) as char,
            }
        }

        fn byte(&mut self) -> u8 {
            let byte = self.next();
            match byte % 18 {
                0 => 0,
                1 => b' ',
                2 => b'/',
                3 => b'.',
                4 => b',',
                5 => b'_',
                6 => b'-',
                7 => b':',
                8..=10 => b'a' + (byte % 26),
                11..=13 => b'A' + (byte % 26),
                14..=15 => b'0' + (byte % 10),
                16 => 0x80 | (byte & 0x3f),
                _ => byte,
            }
        }
    }

    fn test_bound(max: usize, miri_max: usize) -> usize {
        if cfg!(miri) { max.min(miri_max) } else { max }
    }

    fn test_iterations(default: usize) -> usize {
        if cfg!(miri) { default.min(4) } else { default }
    }

    #[test]
    fn randomized_backend_parity_and_oracle() {
        if cfg!(miri) {
            for input in miri_inputs() {
                let case = PrefilterCase::from_bytes(input);
                assert_case_matches_oracle(&case);
            }
            return;
        }

        check!()
            .with_iterations(test_iterations(256))
            .with_max_len(test_bound(2048, 384))
            .for_each(|input: &[u8]| {
                let case = PrefilterCase::from_bytes(input);
                assert_case_matches_oracle(&case);
            });
    }

    fn miri_inputs() -> &'static [&'static [u8]] {
        &[b"", b"\0", b"abcABC012 /.,_-:", b"\xff\x80\0needlehaystack"]
    }

    fn assert_case_matches_oracle(case: &PrefilterCase) {
        let scalar_result = kernel_result::<PrefilterScalar>(
            &case.needle,
            &case.haystack,
            case.max_typos,
            case.case_sensitive,
        );
        let oracle = reference_matches_by_deleting_needle_bytes(
            case.needle.as_bytes(),
            &case.haystack,
            case.max_typos,
            case.case_sensitive,
        );
        assert_eq!(
            scalar_result.0, oracle,
            "scalar/oracle mismatch for {case:?}"
        );
        assert_valid_window(scalar_result, &case.haystack, "Scalar", case);

        #[cfg(target_arch = "x86_64")]
        {
            use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};

            if PrefilterSSE::is_available() {
                let result = kernel_result::<PrefilterSSE>(
                    &case.needle,
                    &case.haystack,
                    case.max_typos,
                    case.case_sensitive,
                );
                assert_same_case_result(result, scalar_result, "SSE", case);
                assert_valid_window(result, &case.haystack, "SSE", case);
            }

            if PrefilterAVX::is_available() {
                let result = kernel_result::<PrefilterAVX>(
                    &case.needle,
                    &case.haystack,
                    case.max_typos,
                    case.case_sensitive,
                );
                assert_same_case_result(result, scalar_result, "AVX2", case);
                assert_valid_window(result, &case.haystack, "AVX2", case);
            }

            if PrefilterAVX512::is_available() {
                let result = kernel_result::<PrefilterAVX512>(
                    &case.needle,
                    &case.haystack,
                    case.max_typos,
                    case.case_sensitive,
                );
                assert_same_case_result(result, scalar_result, "AVX-512", case);
                assert_valid_window(result, &case.haystack, "AVX-512", case);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            use crate::prefilter::backend::PrefilterNEON;

            if PrefilterNEON::is_available() {
                let result = kernel_result::<PrefilterNEON>(
                    &case.needle,
                    &case.haystack,
                    case.max_typos,
                    case.case_sensitive,
                );
                assert_same_case_result(result, scalar_result, "NEON", case);
                assert_valid_window(result, &case.haystack, "NEON", case);
            }
        }
    }

    fn reference_matches_by_deleting_needle_bytes(
        needle: &[u8],
        haystack: &[u8],
        max_typos: u16,
        case_sensitive: bool,
    ) -> bool {
        if max_typos as usize >= needle.len() {
            return true;
        }

        longest_common_subsequence_len(needle, haystack, case_sensitive) + max_typos as usize
            >= needle.len()
    }

    fn longest_common_subsequence_len(
        needle: &[u8],
        haystack: &[u8],
        case_sensitive: bool,
    ) -> usize {
        let mut previous = vec![0usize; haystack.len() + 1];
        let mut current = vec![0usize; haystack.len() + 1];
        for &needle_byte in needle {
            current[0] = 0;
            for (idx, &haystack_byte) in haystack.iter().enumerate() {
                current[idx + 1] = if bytes_match(needle_byte, haystack_byte, case_sensitive) {
                    previous[idx] + 1
                } else {
                    previous[idx + 1].max(current[idx])
                };
            }
            std::mem::swap(&mut previous, &mut current);
        }

        previous[haystack.len()]
    }

    fn bytes_match(needle: u8, haystack: u8, case_sensitive: bool) -> bool {
        needle == haystack || (!case_sensitive && needle.eq_ignore_ascii_case(&haystack))
    }

    fn assert_valid_window(
        result: (bool, usize, usize),
        haystack: &[u8],
        context: &str,
        case: &PrefilterCase,
    ) {
        if !result.0 {
            return;
        }

        assert!(
            result.1 <= result.2 && result.2 <= haystack.len(),
            "{} returned invalid window {:?} for haystack_len={} case={:?}",
            context,
            result,
            haystack.len(),
            case
        );
    }

    fn assert_same_result(got: (bool, usize, usize), want: (bool, usize, usize), context: &str) {
        if want.0 {
            assert_eq!(got, want, "{context}");
        } else {
            assert_eq!(got.0, want.0, "{context}");
        }
    }

    fn assert_same_case_result(
        got: (bool, usize, usize),
        want: (bool, usize, usize),
        context: &str,
        case: &PrefilterCase,
    ) {
        if want.0 {
            assert_eq!(got, want, "{context} mismatch for {case:?}");
        } else {
            assert_eq!(got.0, want.0, "{context} mismatch for {case:?}");
        }
    }
}
