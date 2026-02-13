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

pub mod scalar;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

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

#[derive(Debug, Clone)]
pub enum Prefilter {
    AVX2(x86_64::PrefilterAVX2),
    SSE(x86_64::PrefilterSSE),
}

impl Prefilter {
    pub fn new(needle: &[u8]) -> Self {
        #[cfg(target_arch = "x86_64")]
        if x86_64::PrefilterAVX2::is_available() {
            Prefilter::AVX2(unsafe { x86_64::PrefilterAVX2::new(needle) })
        } else if x86_64::PrefilterSSE::is_available() {
            Prefilter::SSE(unsafe { x86_64::PrefilterSSE::new(needle) })
        } else {
            panic!("no prefilter algorithm available due to missing SSE2 support");
        }
    }

    #[inline]
    pub fn match_haystack(&self, haystack: &[u8], max_typos: u16) -> (bool, usize) {
        match (self, max_typos) {
            (Prefilter::AVX2(p), 0) => unsafe { p.match_haystack(haystack) },
            (Prefilter::AVX2(p), _) => unsafe { p.match_haystack_typos(haystack, max_typos) },
            (Prefilter::SSE(p), 0) => unsafe { p.match_haystack(haystack) },
            (Prefilter::SSE(p), _) => unsafe { p.match_haystack_typos(haystack, max_typos) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Prefilter;

    fn match_haystack(needle: &str, haystack: &str) -> bool {
        match_haystack_generic(needle, haystack, 0)
    }

    fn match_haystack_typos(needle: &str, haystack: &str, max_typos: u16) -> bool {
        match_haystack_generic(needle, haystack, max_typos)
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
    fn test_case_insensitivity() {
        assert!(match_haystack("foo", "FOO"));
        assert!(match_haystack("Foo", "foo"));
        assert!(match_haystack("ABC", "abc"));
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
        assert!(match_haystack_typos("BAR", "ba", 1));
        assert!(match_haystack_typos("Hello", "HLL", 2));
        assert!(match_haystack_typos("TeSt", "ES", 2));
        assert!(!match_haystack_typos("TeSt", "ES", 1));
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

    fn match_haystack_generic(needle: &str, haystack: &str, max_typos: u16) -> bool {
        let haystack = normalize_haystack(haystack);
        let haystack = haystack.as_bytes();

        let prefilter = Prefilter::new(needle.as_bytes());
        prefilter.match_haystack(haystack, max_typos).0
    }
}
