//! Fast prefiltering algorithms, which run before Smith Waterman since in the typical case,
//! a small percentage of the haystack will match the needle. Automatically used by the Matcher
//! and match_list APIs.
//!
//! Unordered algorithms are much faster than ordered algorithms, but don't guarantee that the
//! needle is contained in the haystack, unlike ordered algorithms. As a result, a backwards
//! pass must be performed after Smith Waterman to verify the number of typos. But the faster
//! prefilter generally seems to outweigh this extra cost.
//!
//! The `Prefilter` struct chooses the fastest algorithm via runtime feature detection.
//!
//! All algorithms, except scalar, assume that needle.len() > 0 && haystack.len() >= 8

use std::arch::x86_64::__m256i;

use crate::prefilter::x86_64::needle_to_insensitive_avx2;

pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[derive(Clone, Debug)]
pub struct Prefilter {
    pub needle: String,
    pub needle_cased: Vec<(u8, u8)>,
    pub needle_cased_avx2: Vec<__m256i>,
    pub max_typos: u16,
}

impl Prefilter {
    pub fn new(needle: &str, max_typos: u16) -> Self {
        let needle_cased = Self::case_needle(needle);
        let needle_cased_avx2 = unsafe { needle_to_insensitive_avx2(&needle_cased) };
        Prefilter {
            needle: needle.to_string(),
            needle_cased,
            needle_cased_avx2,
            max_typos,
        }
    }

    pub fn case_needle(needle: &str) -> Vec<(u8, u8)> {
        needle
            .as_bytes()
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

    #[inline(always)]
    pub fn match_haystack_sensitive(&self, haystack: &[u8]) -> (bool, usize) {
        self.match_haystack::<true>(haystack)
    }

    #[inline(always)]
    pub fn match_haystack_insensitive(&self, haystack: &[u8]) -> (bool, usize) {
        self.match_haystack::<false>(haystack)
    }

    #[inline(always)]
    fn match_haystack<const CASE_SENSITIVE: bool>(&self, haystack: &[u8]) -> (bool, usize) {
        match haystack.len() {
            0 => (true, 0),
            1..8 => (self.match_haystack_scalar::<CASE_SENSITIVE>(haystack), 0),
            _ => unsafe { self.match_haystack_x86_64::<CASE_SENSITIVE>(haystack) },
        }
    }

    #[inline(always)]
    fn match_haystack_scalar<const CASE_SENSITIVE: bool>(&self, haystack: &[u8]) -> bool {
        match (self.max_typos, CASE_SENSITIVE) {
            (0, true) => scalar::match_haystack(self.needle.as_bytes(), haystack),
            (0, false) => scalar::match_haystack_insensitive(&self.needle_cased, haystack),
            (_, true) => {
                scalar::match_haystack_typos(self.needle.as_bytes(), haystack, self.max_typos)
            }
            (_, false) => scalar::match_haystack_typos_insensitive(
                &self.needle_cased,
                haystack,
                self.max_typos,
            ),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn match_haystack_x86_64<const CASE_SENSITIVE: bool>(
        &self,
        haystack: &[u8],
    ) -> (bool, usize) {
        unsafe {
            match (self.max_typos, CASE_SENSITIVE) {
                (0, false) => {
                    x86_64::match_haystack_unordered_insensitive(&self.needle_cased_avx2, haystack)
                }
                (0, true) => x86_64::match_haystack_unordered_insensitive_typos(
                    &self.needle_cased_avx2,
                    haystack,
                    self.max_typos,
                ),
                (_, false) => x86_64::match_haystack_unordered(self.needle.as_bytes(), haystack),
                (_, true) => x86_64::match_haystack_unordered_typos(
                    self.needle.as_bytes(),
                    haystack,
                    self.max_typos,
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Prefilter;

    /// Ensures both the ordered and unordered implementations return the same result
    fn match_haystack(needle: &str, haystack: &str) -> bool {
        match_haystack_generic::<true>(needle, haystack, 0)
    }

    fn match_haystack_insensitive(needle: &str, haystack: &str) -> bool {
        match_haystack_generic::<false>(needle, haystack, 0)
    }

    fn match_haystack_typos(needle: &str, haystack: &str, max_typos: u16) -> bool {
        match_haystack_generic::<true>(needle, haystack, max_typos)
    }

    fn match_haystack_typos_insensitive(needle: &str, haystack: &str, max_typos: u16) -> bool {
        match_haystack_generic::<false>(needle, haystack, max_typos)
    }

    #[test]
    fn test_exact_match() {
        assert!(match_haystack("foo", "foo"));
        assert!(match_haystack("a", "a"));
        assert!(match_haystack("hello", "hello"));
    }

    #[test]
    fn test_fuzzy_match_with_gaps() {
        assert!(match_haystack("foo", "f_o_o"));
        assert!(match_haystack("foo", "f__o__o"));
        assert!(match_haystack("abc", "a_b_c"));
        assert!(match_haystack("test", "t_e_s_t"));
    }

    #[test]
    fn test_unordered_within_chunk() {
        assert!(match_haystack("foo", "oof"));
        assert!(match_haystack("abc", "cba"));
        assert!(match_haystack("test", "tset"));
        assert!(match_haystack("hello", "olleh"));
    }

    #[test]
    fn test_case_sensitivity() {
        assert!(!match_haystack("foo", "FOO"));
        assert!(match_haystack_insensitive("foo", "FOO"));

        assert!(!match_haystack("Foo", "foo"));
        assert!(match_haystack_insensitive("Foo", "foo"));

        assert!(!match_haystack("ABC", "abc"));
        assert!(match_haystack_insensitive("ABC", "abc"));
    }

    #[test]
    fn test_chunk_boundary() {
        // Characters must be within same 16-byte chunk
        let haystack = "oo_______________f"; // 'f' is at position 17 (18th byte)
        assert!(!match_haystack("foo", haystack));

        // But if all within one chunk, should work
        let haystack = "oof_____________"; // All within first 16 bytes
        assert!(match_haystack("foo", haystack));
    }

    #[test]
    fn test_overlapping_load() {
        // Because we load the last 16 bytes of the haystack in the final iteration,
        // when the haystack.len() % 16 != 0, we end up matching on the 'o' twice
        assert!(match_haystack("foo", "f_________________________o______"));
    }

    #[test]
    fn test_multiple_chunks() {
        assert!(match_haystack("foo", "f_______________o_______________o"));
        assert!(match_haystack(
            "abc",
            "a_______________b_______________c_______________"
        ));
    }

    #[test]
    fn test_partial_matches() {
        assert!(!match_haystack("fob", "fo"));
        assert!(!match_haystack("test", "tet"));
        assert!(!match_haystack("abc", "a"));
    }

    #[test]
    fn test_duplicate_characters_in_needle() {
        assert!(match_haystack("foo", "foo"));
        assert!(match_haystack("foo", "ofo"));
        assert!(match_haystack("foo", "fo")); // Missing one 'o'

        assert!(match_haystack("aaa", "aaa"));
        assert!(match_haystack("aaa", "aa"));
    }

    #[test]
    fn test_haystack_with_extra_characters() {
        assert!(match_haystack("foo", "foobar"));
        assert!(match_haystack("foo", "prefoobar"));
        assert!(match_haystack("abc", "xaxbxcx"));
    }

    #[test]
    fn test_edge_cases_at_16_byte_boundary() {
        let haystack = "123456789012345f"; // 'f' at position 15 (last position in chunk)
        assert!(match_haystack("f", haystack));

        let haystack = "o_______________of"; // Two 'o's in first chunk, 'f' in second
        // Due to overlapping loads, we end up loading the 'of' in the final chunk
        assert!(match_haystack("foo", haystack));
    }

    #[test]
    fn test_overlapping_chunks() {
        // The function uses overlapping loads, so test edge cases
        // where characters might be found in overlapping regions
        let haystack = "_______________fo"; // 'f' at position 15, 'o' at position 16
        assert!(match_haystack("fo", haystack));
    }

    #[test]
    fn test_single_character_needle() {
        // Single character needles
        assert!(match_haystack("a", "a"));
        assert!(match_haystack("a", "ba"));
        assert!(match_haystack("a", "_______________a"));
        assert!(!match_haystack("a", ""));
    }

    #[test]
    fn test_repeated_character_haystack() {
        // Haystack with repeated characters
        assert!(match_haystack("abc", "aaabbbccc"));
        assert!(match_haystack("foo", "fofofoooo"));
    }

    #[test]
    fn test_typos_single_missing_character() {
        // One character missing from haystack
        assert!(match_haystack_typos("bar", "ba", 1));
        assert!(match_haystack_typos("bar", "ar", 1));
        assert!(match_haystack_typos("hello", "hllo", 1));
        assert!(match_haystack_typos("test", "tst", 1));

        // Should fail with 0 typos allowed
        assert!(!match_haystack_typos("bar", "ba", 0));
        assert!(!match_haystack_typos("hello", "hllo", 0));
    }

    #[test]
    fn test_typos_multiple_missing_characters() {
        assert!(match_haystack_typos("hello", "hll", 2));
        assert!(match_haystack_typos("testing", "tstng", 2));
        assert!(match_haystack_typos("abcdef", "abdf", 2));

        assert!(!match_haystack_typos("hello", "hll", 1));
        assert!(!match_haystack_typos("testing", "tstng", 1));
    }

    #[test]
    fn test_typos_with_gaps() {
        assert!(match_haystack_typos("bar", "b_r", 1));
        assert!(match_haystack_typos("test", "t__s_t", 1));
        assert!(match_haystack_typos("helo", "h_l_", 2));
    }

    #[test]
    fn test_typos_unordered_permutations() {
        assert!(match_haystack_typos("bar", "rb", 1));
        assert!(match_haystack_typos("abcdef", "fcda", 2));
    }

    #[test]
    fn test_typos_case_insensitive() {
        // Case insensitive with typos
        assert!(match_haystack_typos_insensitive("BAR", "ba", 1));
        assert!(match_haystack_typos_insensitive("Hello", "HLL", 2));
        assert!(match_haystack_typos_insensitive("TeSt", "ES", 2));
        assert!(!match_haystack_typos_insensitive("TeSt", "ES", 1));
    }

    #[test]
    fn test_typos_edge_cases() {
        // All characters missing (typos == needle length)
        assert!(match_haystack_typos("abc", "", 3));

        // More typos allowed than necessary
        assert!(match_haystack_typos("foo", "fo", 5));
    }

    #[test]
    fn test_typos_across_chunks() {
        assert!(match_haystack_typos("abc", "a_______________b", 1));

        assert!(match_haystack_typos(
            "test",
            "t_______________s_______________t",
            1
        ));
    }

    #[test]
    fn test_typos_single_character_needle() {
        assert!(match_haystack_typos("a", "a", 0));
        assert!(match_haystack_typos("a", "", 1));
        assert!(!match_haystack_typos("a", "", 0));
    }

    fn normalize_haystack(haystack: &str) -> String {
        if haystack.len() < 8 {
            "_".repeat(8 - haystack.len()) + haystack
        } else {
            haystack.to_string()
        }
    }

    fn match_haystack_generic<const CASE_SENSITIVE: bool>(
        needle: &str,
        haystack: &str,
        max_typos: u16,
    ) -> bool {
        let prefilter = Prefilter::new(needle, max_typos);
        let haystack = normalize_haystack(haystack);
        let haystack = haystack.as_bytes();
        prefilter.match_haystack::<CASE_SENSITIVE>(haystack).0
    }
}
