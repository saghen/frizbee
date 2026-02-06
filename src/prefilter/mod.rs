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

#[cfg(feature = "simd")]
pub mod bitmask;
pub mod scalar;
#[cfg(feature = "simd")]
pub mod simd;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[derive(Clone, Debug)]
pub struct Prefilter {
    needle: String,
    needle_cased: Vec<(u8, u8)>,
    #[cfg(target_arch = "x86_64")]
    needle_avx2: Option<Vec<std::arch::x86_64::__m256i>>,

    max_typos: u16,

    has_sse2: bool,
    has_avx2: bool,
}

impl Prefilter {
    pub fn new(needle: &str, max_typos: u16) -> Self {
        #[cfg(target_arch = "x86_64")]
        let has_sse2 = is_x86_feature_detected!("sse2");
        #[cfg(not(target_arch = "x86_64"))]
        let has_sse2 = false;

        #[cfg(target_arch = "x86_64")]
        let has_avx2 =
            has_sse2 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("avx");
        #[cfg(not(target_arch = "x86_64"))]
        let has_avx2 = false;

        let needle_cased = Self::case_needle(needle);
        Prefilter {
            needle: needle.to_string(),
            needle_cased: needle_cased.clone(),
            #[cfg(target_arch = "x86_64")]
            needle_avx2: has_avx2.then(|| unsafe { x86_64::needle_to_avx2(&needle_cased) }),

            max_typos,

            has_sse2,
            has_avx2,
        }
    }

    pub fn case_needle(needle: &str) -> Vec<(u8, u8)> {
        needle
            .as_bytes()
            .iter()
            .map(|&c| (c.to_ascii_uppercase(), c.to_ascii_lowercase()))
            .collect()
    }

    pub fn match_haystack(&self, haystack: &[u8]) -> bool {
        self.match_haystack_runtime_detection::<true, true, false>(haystack)
    }

    pub fn match_haystack_insensitive(&self, haystack: &[u8]) -> bool {
        self.match_haystack_runtime_detection::<true, false, false>(haystack)
    }

    pub fn match_haystack_unordered(&self, haystack: &[u8]) -> bool {
        self.match_haystack_runtime_detection::<false, true, false>(haystack)
    }

    pub fn match_haystack_unordered_insensitive(&self, haystack: &[u8]) -> bool {
        self.match_haystack_runtime_detection::<false, false, false>(haystack)
    }

    pub fn match_haystack_unordered_typos(&self, haystack: &[u8]) -> bool {
        self.match_haystack_runtime_detection::<false, true, true>(haystack)
    }

    pub fn match_haystack_unordered_typos_insensitive(&self, haystack: &[u8]) -> bool {
        self.match_haystack_runtime_detection::<false, false, true>(haystack)
    }

    #[inline(always)]
    fn match_haystack_runtime_detection<
        const ORDERED: bool,
        const CASE_SENSITIVE: bool,
        const TYPOS: bool,
    >(
        &self,
        haystack: &[u8],
    ) -> bool {
        match haystack.len() {
            0 => return true,
            1..8 => {
                return self.match_haystack_scalar::<ORDERED, CASE_SENSITIVE, TYPOS>(haystack);
            }
            _ => {}
        }

        match (self.has_avx2, self.has_sse2) {
            #[cfg(target_arch = "x86_64")]
            (true, _) => unsafe {
                self.match_haystack_avx2::<ORDERED, CASE_SENSITIVE, TYPOS>(haystack)
            },
            #[cfg(target_arch = "x86_64")]
            (_, true) => unsafe {
                self.match_haystack_sse2::<ORDERED, CASE_SENSITIVE, TYPOS>(haystack)
            },
            #[cfg(feature = "simd")]
            _ => self.match_haystack_simd::<ORDERED, CASE_SENSITIVE, TYPOS>(haystack),
            #[cfg(not(feature = "simd"))]
            _ => self.match_haystack_scalar::<ORDERED, CASE_SENSITIVE, TYPOS>(haystack),
        }
    }

    #[inline(always)]
    fn match_haystack_scalar<const ORDERED: bool, const CASE_SENSITIVE: bool, const TYPOS: bool>(
        &self,
        haystack: &[u8],
    ) -> bool {
        match (TYPOS, CASE_SENSITIVE) {
            (true, true) => {
                scalar::match_haystack_typos(self.needle.as_bytes(), haystack, self.max_typos)
            }
            (true, false) => scalar::match_haystack_typos_insensitive(
                &self.needle_cased,
                haystack,
                self.max_typos,
            ),
            (false, true) => scalar::match_haystack(self.needle.as_bytes(), haystack),
            (false, false) => scalar::match_haystack_insensitive(&self.needle_cased, haystack),
        }
    }

    #[inline(always)]
    #[cfg(feature = "simd")]
    fn match_haystack_simd<const ORDERED: bool, const CASE_SENSITIVE: bool, const TYPOS: bool>(
        &self,
        haystack: &[u8],
    ) -> bool {
        match (ORDERED, CASE_SENSITIVE, TYPOS) {
            (true, _, true) => panic!("ordered typos implementations are not yet available"),
            (true, true, false) => simd::match_haystack(self.needle.as_bytes(), haystack),
            (true, false, false) => simd::match_haystack_insensitive(&self.needle_cased, haystack),

            (false, true, false) => {
                simd::match_haystack_unordered(self.needle.as_bytes(), haystack)
            }
            (false, true, true) => simd::match_haystack_unordered_typos(
                self.needle.as_bytes(),
                haystack,
                self.max_typos,
            ),
            (false, false, false) => {
                simd::match_haystack_unordered_insensitive(&self.needle_cased, haystack)
            }
            (false, false, true) => simd::match_haystack_unordered_typos_insensitive(
                &self.needle_cased,
                haystack,
                self.max_typos,
            ),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn match_haystack_x86_64<
        const ORDERED: bool,
        const CASE_SENSITIVE: bool,
        const TYPOS: bool,
        const AVX2: bool,
    >(
        &self,
        haystack: &[u8],
    ) -> bool {
        unsafe {
            match (ORDERED, CASE_SENSITIVE, TYPOS) {
                (true, _, true) => panic!("ordered typos implementations are not yet available"),
                (true, true, false) => x86_64::match_haystack(self.needle.as_bytes(), haystack),
                (true, false, false) => {
                    x86_64::match_haystack_insensitive(&self.needle_cased, haystack)
                }

                (false, true, false) => {
                    x86_64::match_haystack_unordered(self.needle.as_bytes(), haystack)
                }
                (false, true, true) => x86_64::match_haystack_unordered_typos(
                    self.needle.as_bytes(),
                    haystack,
                    self.max_typos,
                ),
                (false, false, false) => {
                    if AVX2 {
                        x86_64::match_haystack_unordered_insensitive_avx2(
                            self.needle_avx2.as_ref().unwrap(),
                            haystack,
                        )
                    } else {
                        x86_64::match_haystack_unordered_insensitive(&self.needle_cased, haystack)
                    }
                }
                (false, false, true) => {
                    if AVX2 {
                        x86_64::match_haystack_unordered_typos_insensitive_avx2(
                            self.needle_avx2.as_ref().unwrap(),
                            haystack,
                            self.max_typos,
                        )
                    } else {
                        x86_64::match_haystack_unordered_typos_insensitive(
                            &self.needle_cased,
                            haystack,
                            self.max_typos,
                        )
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2,avx,avx2")]
    unsafe fn match_haystack_avx2<
        const ORDERED: bool,
        const CASE_SENSITIVE: bool,
        const TYPOS: bool,
    >(
        &self,
        haystack: &[u8],
    ) -> bool {
        unsafe { self.match_haystack_x86_64::<ORDERED, CASE_SENSITIVE, TYPOS, true>(haystack) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn match_haystack_sse2<
        const ORDERED: bool,
        const CASE_SENSITIVE: bool,
        const TYPOS: bool,
    >(
        &self,
        haystack: &[u8],
    ) -> bool {
        unsafe { self.match_haystack_x86_64::<ORDERED, CASE_SENSITIVE, TYPOS, false>(haystack) }
    }
}

#[cfg(test)]
mod tests {
    use super::Prefilter;

    /// Ensures both the ordered and unordered implementations return the same result
    fn match_haystack(needle: &str, haystack: &str) -> bool {
        let ordered = match_haystack_generic::<true, true>(needle, haystack, 0);
        let unordered = match_haystack_generic::<false, true>(needle, haystack, 0);
        assert_eq!(
            ordered, unordered,
            "ordered and unordered implementations produced different results for {needle} on {haystack}"
        );
        ordered
    }

    fn match_haystack_insensitive(needle: &str, haystack: &str) -> bool {
        let ordered = match_haystack_generic::<true, false>(needle, haystack, 0);
        let unordered = match_haystack_generic::<false, false>(needle, haystack, 0);
        assert_eq!(
            ordered, unordered,
            "ordered and unordered implementations produced different results for {needle} on {haystack}"
        );
        ordered
    }

    fn match_haystack_ordered(needle: &str, haystack: &str) -> bool {
        match_haystack_generic::<true, true>(needle, haystack, 0)
    }

    fn match_haystack_unordered(needle: &str, haystack: &str) -> bool {
        match_haystack_generic::<false, true>(needle, haystack, 0)
    }

    fn match_haystack_unordered_typos(needle: &str, haystack: &str, max_typos: u16) -> bool {
        match_haystack_generic::<false, true>(needle, haystack, max_typos)
    }

    fn match_haystack_unordered_typos_insensitive(
        needle: &str,
        haystack: &str,
        max_typos: u16,
    ) -> bool {
        match_haystack_generic::<false, false>(needle, haystack, max_typos)
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
        assert!(match_haystack_unordered("foo", "oof"));
        assert!(!match_haystack_ordered("foo", "oof"));

        assert!(match_haystack_unordered("abc", "cba"));
        assert!(!match_haystack_ordered("abc", "cba"));

        assert!(match_haystack_unordered("test", "tset"));
        assert!(!match_haystack_ordered("test", "tset"));

        assert!(match_haystack_unordered("hello", "olleh"));
        assert!(!match_haystack_ordered("hello", "olleh"));
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
        assert!(match_haystack_unordered("foo", haystack));
    }

    #[ignore = "overlapping loads are not yet masked on the ordered implementation"]
    #[test]
    fn test_overlapping_load() {
        // Because we load the last 16 bytes of the haystack in the final iteration,
        // when the haystack.len() % 16 != 0, we end up matching on the 'o' twice
        assert!(!match_haystack_ordered(
            "foo",
            "f_________________________o______"
        ));
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
        assert!(match_haystack_unordered("foo", "ofo"));
        assert!(match_haystack_unordered("foo", "fo")); // Missing one 'o'

        assert!(match_haystack_unordered("aaa", "aaa"));
        assert!(match_haystack_unordered("aaa", "aa"));
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
        assert!(!match_haystack_ordered("foo", haystack));
        assert!(match_haystack_unordered("foo", haystack));
    }

    #[test]
    #[should_panic]
    fn test_invalid_width_limit_panic() {
        // When W > 16, the function should panic when haystack.len() < 16
        assert!(match_haystack("abc", "cba"));
        assert!(match_haystack("abc", "cba"));
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
        assert!(match_haystack_unordered_typos("bar", "ba", 1));
        assert!(match_haystack_unordered_typos("bar", "ar", 1));
        assert!(match_haystack_unordered_typos("hello", "hllo", 1));
        assert!(match_haystack_unordered_typos("test", "tst", 1));

        // Should fail with 0 typos allowed
        assert!(!match_haystack_unordered_typos("bar", "ba", 0));
        assert!(!match_haystack_unordered_typos("hello", "hllo", 0));
    }

    #[test]
    fn test_typos_multiple_missing_characters() {
        assert!(match_haystack_unordered_typos("hello", "hll", 2));
        assert!(match_haystack_unordered_typos("testing", "tstng", 2));
        assert!(match_haystack_unordered_typos("abcdef", "abdf", 2));

        assert!(!match_haystack_unordered_typos("hello", "hll", 1));
        assert!(!match_haystack_unordered_typos("testing", "tstng", 1));
    }

    #[test]
    fn test_typos_with_gaps() {
        assert!(match_haystack_unordered_typos("bar", "b_r", 1));
        assert!(match_haystack_unordered_typos("test", "t__s_t", 1));
        assert!(match_haystack_unordered_typos("helo", "h_l_", 2));
    }

    #[test]
    fn test_typos_unordered_permutations() {
        assert!(match_haystack_unordered_typos("bar", "rb", 1));
        assert!(match_haystack_unordered_typos("abcdef", "fcda", 2));
    }

    #[test]
    fn test_typos_case_insensitive() {
        // Case insensitive with typos
        assert!(match_haystack_unordered_typos_insensitive("BAR", "ba", 1));
        assert!(match_haystack_unordered_typos_insensitive(
            "Hello", "HLL", 2
        ));
        assert!(match_haystack_unordered_typos_insensitive("TeSt", "ES", 2));
        assert!(!match_haystack_unordered_typos_insensitive("TeSt", "ES", 1));
    }

    #[test]
    fn test_typos_edge_cases() {
        // All characters missing (typos == needle length)
        assert!(match_haystack_unordered_typos("abc", "", 3));

        // More typos allowed than necessary
        assert!(match_haystack_unordered_typos("foo", "fo", 5));
    }

    #[test]
    fn test_typos_across_chunks() {
        assert!(match_haystack_unordered_typos(
            "abc",
            "a_______________b",
            1
        ));

        assert!(match_haystack_unordered_typos(
            "test",
            "t_______________s_______________t",
            1
        ));
    }

    #[test]
    fn test_typos_single_character_needle() {
        assert!(match_haystack_unordered_typos("a", "a", 0));
        assert!(match_haystack_unordered_typos("a", "", 1));
        assert!(!match_haystack_unordered_typos("a", "", 0));
    }

    fn normalize_haystack(haystack: &str) -> String {
        if haystack.len() < 8 {
            "_".repeat(8 - haystack.len()) + haystack
        } else {
            haystack.to_string()
        }
    }

    fn match_haystack_generic<const ORDERED: bool, const CASE_SENSITIVE: bool>(
        needle: &str,
        haystack: &str,
        max_typos: u16,
    ) -> bool {
        let prefilter = Prefilter::new(needle, max_typos);
        let haystack = normalize_haystack(haystack);
        let haystack = haystack.as_bytes();

        if max_typos > 0 {
            let typo_result =
                prefilter.match_haystack_simd::<ORDERED, CASE_SENSITIVE, true>(haystack);

            #[cfg(target_arch = "x86_64")]
            {
                let typo_result_x86_64 = unsafe {
                    prefilter
                        .match_haystack_x86_64::<ORDERED, CASE_SENSITIVE, true, false>(haystack)
                };
                assert_eq!(
                    typo_result, typo_result_x86_64,
                    "x86_64 typos (set to 0) and simd implementations produced different results"
                );
            }

            return typo_result;
        }

        let result = prefilter.match_haystack_simd::<ORDERED, CASE_SENSITIVE, false>(haystack);

        if !ORDERED {
            let typo_result =
                prefilter.match_haystack_simd::<ORDERED, CASE_SENSITIVE, true>(haystack);
            assert_eq!(
                result, typo_result,
                "regular and typos implementations (set to 0) produced different results"
            );
        }

        #[cfg(target_arch = "x86_64")]
        {
            let result_x86_64 = unsafe {
                prefilter.match_haystack_x86_64::<ORDERED, CASE_SENSITIVE, false, false>(haystack)
            };
            assert_eq!(
                result, result_x86_64,
                "x86_64 and simd implementations produced different results"
            );

            if !ORDERED {
                let typo_result_x86_64 = unsafe {
                    prefilter
                        .match_haystack_x86_64::<ORDERED, CASE_SENSITIVE, true, false>(haystack)
                };
                assert_eq!(
                    result, typo_result_x86_64,
                    "x86_64 typos (set to 0) and simd implementations produced different results"
                );
            }
        }

        result
    }
}
