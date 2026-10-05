#[cfg(target_arch = "x86_64")]
use crate::smith_waterman::avx512_u8_fits;
use crate::sort::radix_sort_matches;
use crate::{Config, Match, MatchIndices, Pattern};
use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

#[cfg(target_arch = "aarch64")]
use crate::literal::LiteralNEON;
#[cfg(scalar_fallback)]
use crate::literal::LiteralScalar;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::literal::LiteralWasm;
#[cfg(target_arch = "x86_64")]
use crate::literal::{LiteralAVX, LiteralAVX512, LiteralSSE};

pub(crate) mod algo;
mod backend;
mod iter;
// Threading requires `std`, and WASM does not support it at all
#[cfg(all(feature = "std", not(target_family = "wasm")))]
mod parallel;
use algo::{MANY_TYPOS, NO_PREFILTER, Specialized};
use backend::*;
pub use iter::{FuzzyMatch, FuzzyMatchExt, FuzzyMatchIndices};

/// Many variants so we use a macro to expand to the correct impl
macro_rules! dispatch {
    ($self:expr, $m:ident => $body:expr) => {
        match $self {
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX512U8($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX512($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::SSE($m) => $body,
            #[cfg(target_arch = "aarch64")]
            MatcherBackend::NEON($m) => $body,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            MatcherBackend::Wasm($m) => $body,
            #[cfg(scalar_fallback)]
            MatcherBackend::Scalar($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::LiteralAVX512($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::LiteralAVX($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::LiteralSSE($m) => $body,
            #[cfg(target_arch = "aarch64")]
            MatcherBackend::LiteralNEON($m) => $body,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            MatcherBackend::LiteralWasm($m) => $body,
            #[cfg(scalar_fallback)]
            MatcherBackend::LiteralScalar($m) => $body,
        }
    };
}

/// Each of these receives its own inline(always) hot loop, so a single function that branches
/// would be enormous (thousands of symbols). So instead, we dispatch to the correct implementation
/// which contains the `#[target_feature]` and hot loop (in backend.rs)
#[rustfmt::skip]
macro_rules! dispatch_typos {
    ($max_typos:expr, $needs_unicode:expr, |$typos:ident, $unicode:ident| $body:expr) => {
        match ($max_typos, $needs_unicode) {
            (None, false)     => { const $typos: u16 = NO_PREFILTER; const $unicode: bool = false; $body }
            (None, true)      => { const $typos: u16 = NO_PREFILTER; const $unicode: bool = true;  $body }
            (Some(0), false)  => { const $typos: u16 = 0;            const $unicode: bool = false; $body }
            (Some(0), true)   => { const $typos: u16 = 0;            const $unicode: bool = true;  $body }
            // Every typo budget shares one specialization: the prefilter's typo walk
            // reads the budget only for its head/tail bounds
            (Some(_), false)  => { const $typos: u16 = MANY_TYPOS;   const $unicode: bool = false; $body }
            (Some(_), true)   => { const $typos: u16 = MANY_TYPOS;   const $unicode: bool = true;  $body }
        }
    };
}

mod multi;
use multi::{CompiledPattern, CompiledPatterns};

/// A borrowed UTF-8 haystack for the bindings: `#[repr(C)]` pointer then
/// length (the pointer may be null when the length is 0), so C strings can be
/// viewed as `&[RawStr]`. Bindings matching through [`Matcher::match_list_raw`]
/// reuse frizbee's own copy of the matching code, which matches `RawStr`
#[doc(hidden)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RawStr<'a> {
    ptr: *const u8,
    len: usize,
    _str: PhantomData<&'a str>,
}

impl RawStr<'_> {
    /// # Safety
    /// Unless `len` is 0 (where `ptr` may be anything, including null), `ptr`
    /// must point to `len` bytes of valid UTF-8 that stay valid and unmodified
    /// for the lifetime
    pub const unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Self {
        Self {
            ptr,
            len,
            _str: PhantomData,
        }
    }
}

impl<'a> From<&'a str> for RawStr<'a> {
    fn from(s: &'a str) -> Self {
        unsafe { Self::from_raw_parts(s.as_ptr(), s.len()) }
    }
}

impl AsRef<str> for RawStr<'_> {
    #[inline(always)]
    fn as_ref(&self) -> &str {
        if self.len == 0 {
            return "";
        }
        // SAFETY: guaranteed by the constructor
        unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(self.ptr, self.len)) }
    }
}

// SAFETY: a shared view of immutable bytes, like `&str`
unsafe impl Send for RawStr<'_> {}
unsafe impl Sync for RawStr<'_> {}

/// Primary entrypoint for fuzzy matching
///
/// `Matcher` compiles the pattern once, allocates memory for the Smith Waterman
/// matrix, and reuses the selected SIMD backend.
///
/// Ideally, only construct these at most once per list. They're cheap to
/// construct, but end up being expensive if you construct them for each item in
/// your list.
#[derive(Debug, Clone)]
pub struct Matcher {
    config: Config,
    raw_patterns: Vec<Pattern>,
    patterns: CompiledPatterns,
}

impl Matcher {
    /// Creates a matcher from a single [`Pattern`]. Strings convert into a
    /// pattern that matches literally. Use [`Pattern::parse`] for query
    /// syntax, and [`Matcher::from_patterns`] or [`Matcher::from_query`]
    /// for multi-pattern queries.
    pub fn new(pattern: impl Into<Pattern>, config: &Config) -> Self {
        Self::from_patterns(&[pattern.into()], config)
    }

    /// Creates a matcher from a list of [`Pattern`]s (see
    /// [`Pattern::parse_query`]), matched independently. A haystack matches
    /// when all of the patterns match, where the score is the sum of each
    /// pattern's score.
    ///
    /// ```
    /// use frizbee::{Config, Matcher, Pattern};
    ///
    /// let mut matcher = Matcher::from_patterns(&Pattern::parse_query("foo !^bar"), &Config::default());
    /// let matches = matcher.match_list(&["foo", "barfoo", "foobar"]);
    /// assert_eq!(matches.len(), 2); // "barfoo" starts with "bar"
    /// ```
    pub fn from_patterns(patterns: &[Pattern], config: &Config) -> Self {
        Self {
            patterns: Self::build_patterns(patterns, config),
            raw_patterns: patterns.to_vec(),
            config: config.clone(),
        }
    }

    /// Shorthand for calling [`Matcher::from_patterns`] with
    /// [`Pattern::parse_query`].
    ///
    /// ```rust
    /// use frizbee::{Config, Matcher};
    ///
    /// let mut matcher = Matcher::from_query("foo !^bar", &Config::default());
    /// let matches = matcher.match_list(&["foo", "barfoo", "foobar"]);
    /// assert_eq!(matches.len(), 2); // "barfoo" starts with "bar"
    /// ```
    ///
    /// To override config per-pattern, call [`Pattern::parse_query`] directly
    /// and adjust each pattern's [`crate::PatternConfig`] before passing
    /// them to [`Matcher::from_patterns`]. For example, setting the max
    /// typos based on needle length:
    ///
    /// ```rust
    /// use frizbee::{Config, Matcher, Pattern};
    ///
    /// let patterns = Pattern::parse_query("foo !^bar")
    ///     .into_iter()
    ///     .map(|pattern| {
    ///         let max_typos = (pattern.needle.len() / 4) as u16;
    ///         pattern.max_typos(Some(max_typos))
    ///     })
    ///     .collect::<Vec<_>>();
    /// let mut matcher = Matcher::from_patterns(&patterns, &Config::default());
    /// let matches = matcher.match_list(&["foo", "barfoo", "foobar"]);
    /// ```
    pub fn from_query(query: &str, config: &Config) -> Self {
        Self::from_patterns(&Pattern::parse_query(query), config)
    }

    pub fn patterns(&self) -> &[Pattern] {
        &self.raw_patterns
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Updates the config and rebuilds the internal matcher backends.
    /// Skipped if the config is the same as the previous one.
    pub fn set_config(&mut self, config: Config) {
        if self.config == config {
            return;
        }
        self.config = config;
        self.patterns = Self::build_patterns(&self.raw_patterns, &self.config);
    }

    /// Updates the pattern, as in [`Matcher::new`], and rebuilds the internal
    /// matcher backend. Skipped if the pattern is the same as the previous
    /// one.
    pub fn set_pattern(&mut self, pattern: impl Into<Pattern>) {
        self.set_patterns(&[pattern.into()]);
    }

    /// Updates the patterns, as in [`Matcher::from_patterns`], and rebuilds the
    /// internal matcher backends. Skipped if the patterns are the same as
    /// the previous ones.
    pub fn set_patterns(&mut self, patterns: &[Pattern]) {
        if self.raw_patterns == patterns {
            return;
        }
        self.raw_patterns = patterns.to_vec();
        self.patterns = Self::build_patterns(&self.raw_patterns, &self.config);
    }

    fn build_patterns(sources: &[Pattern], config: &Config) -> CompiledPatterns {
        let mut compiled = sources
            .iter()
            .filter_map(|source| Self::compile(source, config))
            .collect::<Vec<_>>();
        match compiled.as_slice() {
            [] => CompiledPatterns::Empty,
            [single] if !single.negated => CompiledPatterns::Single(compiled.pop().unwrap()),
            _ => CompiledPatterns::Multi(compiled),
        }
    }

    /// Builds the backend for a pattern, resolving any per-pattern overrides
    /// against the matcher's config (see
    /// [`crate::PatternConfig::resolve`]). Returns `None` for empty
    /// needles.
    fn compile(source: &Pattern, config: &Config) -> Option<CompiledPattern> {
        if source.needle.is_empty() {
            return None;
        }
        let config = source.config.resolve(config);
        Some(CompiledPattern {
            negated: source.negated,
            needs_unicode: config.unicode.respects_unicode_for(&source.needle),
            max_typos: config.max_typos,
            backend: Self::get_backend(&source.needle, &config),
        })
    }

    /// Matches a list of haystacks, returning a list of [`Match`] values.
    /// This API provides the most performant path when matching on lists.
    ///
    /// This API should not be called with one item at a time as it performs
    /// dynamic dispatch to the underlying backend. Instead, consider using
    /// the [`Matcher::match_iter`], [`Matcher::match_one`] or
    /// [`iter::FuzzyMatchExt`] API.
    pub fn match_list<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<Match> {
        let mut matches = vec![];
        self.match_list_into(haystacks, 0, &mut matches);
        if self.config.sort.is_reversed() {
            matches.reverse();
        }
        if !self.patterns.is_empty() && self.config.sort.is_by_score() {
            radix_sort_matches(&mut matches);
        }
        matches
    }

    /// [`Matcher::match_list`] for the bindings (see [`RawStr`]). Never inlined,
    /// so the bindings call frizbee's copy instead of instantiating their own
    #[doc(hidden)]
    #[inline(never)]
    pub fn match_list_raw(&mut self, haystacks: &[RawStr<'_>]) -> Vec<Match> {
        self.match_list(haystacks)
    }

    /// Matches a list of haystacks, returning a list of [`MatchIndices`] which
    /// are equivalent to [`Match`] except they include the indices of the
    /// matched characters in the haystack.
    ///
    /// This API has not been optimized for performance, and should only be used
    /// on small lists or after matching a list of haystacks with
    /// [`Matcher::match_list`]. Useful for displaying matched indices in
    /// the UI.
    ///
    /// This API should not be called with one item at a time as it performs
    /// dynamic dispatch to the underlying backend. Instead, consider using
    /// the [`Matcher::match_iter_indices`] or [`iter::FuzzyMatchExt`] API.
    pub fn match_list_indices<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<MatchIndices> {
        Self::guard_against_haystack_overflow(haystacks.len(), 0);
        let mut matches: Vec<_> = haystacks
            .iter()
            .enumerate()
            .filter_map(|(index, haystack)| self.match_one_indices(haystack, index as u32))
            .collect();

        if self.config.sort.is_reversed() {
            matches.reverse();
        }
        if self.config.sort.is_by_score() {
            matches.sort_by_key(|m| core::cmp::Reverse(m.score));
        }
        matches
    }

    /// Returns an iterator over [`Match`] values for an iterator of strings.
    ///
    /// Haystacks are pulled from the source in batches and matched through the
    /// same path as [`Matcher::match_list`], so it performs about the same
    /// (minus the sort).
    ///
    /// You may also use the [`iter::FuzzyMatchExt`] API which provides a more
    /// convenient API for when re-using the [`Matcher`] isn't necessary.
    ///
    /// ```
    /// use frizbee::{Config, iter::FuzzyMatchExt};
    ///
    /// let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
    /// let matches: Vec<_> = haystacks
    ///     .iter()
    ///     .fuzzy_match("fBr", &Config::default())
    ///     .collect();
    /// ```
    pub fn match_iter<S: AsRef<str>>(
        &mut self,
        haystacks: impl IntoIterator<Item = S>,
    ) -> impl Iterator<Item = Match> {
        let mut batched = iter::Batched::new(haystacks.into_iter());
        core::iter::from_fn(move || batched.next_match(self))
    }

    /// Returns an iterator over [`MatchIndices`] values for an iterator of
    /// strings, which are equivalent to [`Match`] except they include the
    /// indices of the matched characters in the haystack.
    ///
    /// This API has not been optimized for performance, and should only be used
    /// on small lists or after matching a list of haystacks with
    /// [`Matcher::match_iter`]. Useful for displaying matched indices in
    /// the UI.
    ///
    /// You may also use the [`iter::FuzzyMatchExt`] API which provides a more
    /// convenient API for when re-using the [`Matcher`] isn't necessary.
    ///
    /// ```
    /// use frizbee::{Config, iter::FuzzyMatchExt};
    ///
    /// let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
    /// let matches: Vec<_> = haystacks
    ///     .iter()
    ///     .fuzzy_match_indices("fBr", &Config::default())
    ///     .collect();
    /// ```
    pub fn match_iter_indices<S: AsRef<str>>(
        &mut self,
        haystacks: impl IntoIterator<Item = S>,
    ) -> impl Iterator<Item = MatchIndices> {
        haystacks
            .into_iter()
            .enumerate()
            .filter_map(move |(index, haystack)| {
                let index = u32::try_from(index)
                    .expect("too many items in haystack, will overflow the u32 index");
                self.match_one_indices(haystack, index)
            })
    }

    /// Matches a single haystack, returning its [`Match`] if it passes. This
    /// API performs ~10% slower than the [`Matcher::match_list`] API.
    ///
    /// Consider using the [`Matcher::match_iter`] or [`Matcher::match_list`]
    /// APIs if you have more than one haystack to match, as they perform
    /// significantly better.
    pub fn match_one<S: AsRef<str>>(&mut self, haystack: S, index: u32) -> Option<Match> {
        let haystack = haystack.as_ref();
        match &mut self.patterns {
            CompiledPatterns::Empty => Some(Match::from_index(index as usize)),
            CompiledPatterns::Single(pattern) => {
                Self::dispatch_pattern_one(pattern, haystack, index)
            }
            CompiledPatterns::Multi(patterns) => Self::match_one_multi(patterns, haystack, index),
        }
    }

    /// Matches a single haystack, returning its [`MatchIndices`] if it passes,
    /// which is equivalent to [`Match`] except they include the indices of
    /// the matched characters in the haystack.
    ///
    /// This API has not been optimized for performance, and should only be used
    /// on small lists or after matching a list of haystacks with
    /// [`Matcher::match_one`], [`Matcher::match_iter`] or
    /// [`Matcher::match_list`]. Useful for displaying matched indices in the
    /// UI.
    pub fn match_one_indices<S: AsRef<str>>(
        &mut self,
        haystack: S,
        index: u32,
    ) -> Option<MatchIndices> {
        let haystack = haystack.as_ref();
        match &mut self.patterns {
            CompiledPatterns::Empty => Some(MatchIndices::from_index(index as usize)),
            CompiledPatterns::Single(pattern) => {
                Self::dispatch_pattern_one_indices(pattern, haystack, index)
            }
            CompiledPatterns::Multi(patterns) => {
                Self::match_one_indices_multi(patterns, haystack, index)
            }
        }
    }

    fn match_list_into<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        Self::guard_against_haystack_overflow(haystacks.len(), haystack_index_offset);
        match &mut self.patterns {
            CompiledPatterns::Empty => {
                let indices = (0..haystacks.len()).map(|i| i + haystack_index_offset as usize);
                matches.extend(indices.map(Match::from_index));
            }
            CompiledPatterns::Single(pattern) => {
                Self::dispatch_pattern_into(pattern, haystacks, haystack_index_offset, matches)
            }
            CompiledPatterns::Multi(patterns) => {
                Self::match_list_multi_into(patterns, haystacks, haystack_index_offset, matches)
            }
        }
    }

    fn dispatch_pattern_into<S: AsRef<str>>(
        pattern: &mut CompiledPattern,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        dispatch!(&mut pattern.backend, matcher => {
            dispatch_typos!(pattern.max_typos, pattern.needs_unicode, |TYPOS, UNICODE| {
                unsafe {
                    matcher.match_list::<TYPOS, UNICODE, S>(
                        haystacks,
                        haystack_index_offset,
                        matches,
                    )
                }
            })
        })
    }

    fn dispatch_pattern_one(
        pattern: &mut CompiledPattern,
        haystack: &str,
        index: u32,
    ) -> Option<Match> {
        let mut matches = Vec::new();
        Self::dispatch_pattern_into(pattern, &[RawStr::from(haystack)], index, &mut matches);
        matches.pop()
    }

    fn dispatch_pattern_one_indices(
        pattern: &mut CompiledPattern,
        haystack: &str,
        index: u32,
    ) -> Option<MatchIndices> {
        dispatch!(&mut pattern.backend, matcher => {
            if pattern.needs_unicode {
                unsafe { matcher.match_one_indices::<true>(haystack, index) }
            } else {
                unsafe { matcher.match_one_indices::<false>(haystack, index) }
            }
        })
    }

    #[inline(always)]
    fn guard_against_haystack_overflow(haystack_len: usize, haystack_index_offset: u32) {
        assert!(
            haystack_len.saturating_add(haystack_index_offset as usize) <= (u32::MAX as usize),
            "too many items in haystack, will overflow the u32 index: {} > {} (index offset: {})",
            haystack_len.saturating_add(haystack_index_offset as usize),
            u32::MAX,
            haystack_index_offset
        );
    }

    fn get_backend(needle: &str, config: &Config) -> MatcherBackend {
        if !config.matching.is_fuzzy() {
            return Self::get_literal_backend(needle, config);
        }

        #[cfg(target_arch = "x86_64")]
        {
            let unicode = config.unicode.respects_unicode_for(needle);
            let use_u8 = avx512_u8_fits(needle, unicode, &config.scoring);
            if use_u8 {
                if MatcherAVX512U8::is_available() {
                    return MatcherBackend::AVX512U8(unsafe {
                        MatcherAVX512U8::build(needle, config)
                    });
                }
            } else {
                if MatcherAVX512::is_available() {
                    return MatcherBackend::AVX512(unsafe { MatcherAVX512::build(needle, config) });
                }
            }
            if MatcherAVX::is_available() {
                return MatcherBackend::AVX(unsafe { MatcherAVX::build(needle, config) });
            }
            if MatcherSSE::is_available() {
                return MatcherBackend::SSE(unsafe { MatcherSSE::build(needle, config) });
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if MatcherNEON::is_available() {
                return MatcherBackend::NEON(unsafe { MatcherNEON::build(needle, config) });
            }
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if MatcherWasm::is_available() {
                return MatcherBackend::Wasm(unsafe { MatcherWasm::build(needle, config) });
            }
        }

        #[cfg(scalar_fallback)]
        {
            MatcherBackend::Scalar(unsafe { MatcherScalar::build(needle, config) })
        }
        #[cfg(not(scalar_fallback))]
        unreachable!("a SIMD backend is statically available")
    }

    fn get_literal_backend(needle: &str, config: &Config) -> MatcherBackend {
        #[cfg(target_arch = "x86_64")]
        {
            if LiteralAVX512::is_available() {
                return MatcherBackend::LiteralAVX512(unsafe {
                    LiteralAVX512::build(needle, config)
                });
            }
            if LiteralAVX::is_available() {
                return MatcherBackend::LiteralAVX(unsafe { LiteralAVX::build(needle, config) });
            }
            if LiteralSSE::is_available() {
                return MatcherBackend::LiteralSSE(unsafe { LiteralSSE::build(needle, config) });
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if LiteralNEON::is_available() {
                return MatcherBackend::LiteralNEON(unsafe { LiteralNEON::build(needle, config) });
            }
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if LiteralWasm::is_available() {
                return MatcherBackend::LiteralWasm(unsafe { LiteralWasm::build(needle, config) });
            }
        }

        #[cfg(scalar_fallback)]
        {
            MatcherBackend::LiteralScalar(unsafe { LiteralScalar::build(needle, config) })
        }
        #[cfg(not(scalar_fallback))]
        unreachable!("a SIMD backend is statically available")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CaseMatching, Scoring, SortStrategy};

    #[test]
    fn raw_str_matches_like_str() {
        let haystacks = ["fooBar", "", "foo_bar", "prelude", "bar/foo"];
        let mut raw = haystacks.map(RawStr::from);
        // C callers may pass NULL for empty strings
        raw[1] = unsafe { RawStr::from_raw_parts(core::ptr::null(), 0) };
        for query in ["fBr", "foo !bar", "!foo", ""] {
            let mut matcher = Matcher::from_query(query, &Config::default());
            let expected = matcher.match_list(&haystacks);
            assert_eq!(matcher.match_list_raw(&raw), expected, "query: {query:?}");
            #[cfg(all(feature = "std", not(target_family = "wasm")))]
            assert_eq!(matcher.match_list_parallel_raw(&raw, 2), expected);
        }
    }

    #[test]
    fn long_needles_saturate_instead_of_panicking() {
        let scoring = Scoring {
            match_score: u16::MAX / 4 + 1,
            ..Scoring::default()
        };
        let needle = "aaaa";
        assert!(needle.len() > scoring.max_needle_len());
        let config = Config::default().scoring(scoring);
        let mut matcher = Matcher::new(needle, &config);

        let haystacks = [needle, "aaa", "zzzz"];
        let matches = matcher.match_list(&haystacks);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[0].score, u16::MAX);
    }

    #[test]
    fn long_needles_saturate_on_the_literal_path() {
        let scoring = Scoring {
            match_score: u16::MAX / 4 + 1,
            ..Scoring::default()
        };
        let needle = "aaaa";
        assert!(needle.len() > scoring.max_needle_len());
        let config = Config::default()
            .matching(crate::Matching::Prefix)
            .scoring(scoring);
        let mut matcher = Matcher::new(needle, &config);

        let matches = matcher.match_list(&["aaaaa", "bbbb"]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].score, u16::MAX);
    }

    #[test]
    fn huge_scoring_configs_saturate() {
        let scoring = Scoring {
            match_score: u16::MAX,
            gap_extend_penalty: u16::MAX,
            gap_open_penalty: u16::MAX,
            ..Scoring::default()
        };
        let config = Config::default().scoring(scoring).max_typos(None);
        let mut matcher = Matcher::new("ab", &config);

        let matches = matcher.match_list(&["ab", "axb", "zz"]);
        assert_eq!(matches[0].score, u16::MAX);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_basic() {
        let needle = "deadbe";
        let haystack = vec!["deadbeef", "deadbf", "deadbeefg", "deadbe"];

        let config = Config::default().max_typos(None);
        let matches = Matcher::new(needle, &config).match_list(&haystack);

        println!("{:?}", matches);
        assert_eq!(matches.len(), 4);
        assert_eq!(matches[0].index, 3);
        assert_eq!(matches[1].index, 0);
        assert_eq!(matches[2].index, 2);
        assert_eq!(matches[3].index, 1);
    }

    #[test]
    fn test_no_typos() {
        let needle = "deadbe";
        let haystack = vec!["deadbeef", "deadbf", "deadbeefg", "deadbe"];

        let matches =
            Matcher::new(needle, &Config::default().max_typos(Some(0))).match_list(&haystack);
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_exact_match() {
        let needle = "deadbe";
        let haystack = vec!["deadbeef", "deadbf", "deadbeefg", "deadbe"];

        let matches = Matcher::new(needle, &Config::default()).match_list(&haystack);

        let exact_matches = matches.iter().filter(|m| m.exact).collect::<Vec<&Match>>();
        assert_eq!(exact_matches.len(), 1);
        assert_eq!(exact_matches[0].index, 3);
        for m in &exact_matches {
            assert_eq!(haystack[m.index as usize], needle)
        }
    }

    #[test]
    fn test_exact_matches() {
        let needle = "deadbe";
        let haystack = vec![
            "deadbe",
            "deadbeef",
            "deadbe",
            "deadbf",
            "deadbe",
            "deadbeefg",
            "deadbe",
        ];

        let matches = Matcher::new(needle, &Config::default()).match_list(&haystack);

        let exact_matches = matches.iter().filter(|m| m.exact).collect::<Vec<&Match>>();
        assert_eq!(exact_matches.len(), 4);
        for m in &exact_matches {
            assert_eq!(haystack[m.index as usize], needle)
        }
    }

    #[test]
    fn test_small_needle() {
        let config = Config::default().max_typos(Some(2));
        let matches = Matcher::new("1", &config).match_list(&["1"]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].index, 0);
        assert!(matches[0].exact);
    }

    #[test]
    fn multibyte_needle_indices_with_unicode_ignored() {
        let config = Config::default()
            .unicode(crate::UnicodeMatching::Ignore)
            .sort(SortStrategy::IndexAsc);

        let matches = Matcher::new("é", &config).match_list_indices(&["xxé"]);
        assert_eq!(matches.len(), 1);
        let mut indices = matches[0].indices.clone();
        indices.sort_unstable();
        // "é" is two bytes (0xC3 0xA9) at byte offsets 2 and 3 of "xxé"
        assert_eq!(indices, vec![2, 3]);
    }

    #[test]
    fn test_case_sensitive_matching() {
        let haystack = ["foo", "FOO", "fOo", "xxfooxx"];
        let config = Config::default().sort(SortStrategy::IndexAsc);

        let matches = Matcher::new("foo", &config).match_list(&haystack);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );

        let config = Config::default()
            .casing(CaseMatching::Respect)
            .sort(SortStrategy::IndexAsc);

        let matches = Matcher::new("foo", &config).match_list(&haystack);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 3]
        );

        let indices = Matcher::new("foo", &config).match_list_indices(&haystack);
        assert_eq!(
            indices.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 3]
        );

        let config = Config::default()
            .casing(CaseMatching::Smart)
            .sort(SortStrategy::IndexAsc);

        let matches = Matcher::new("FoO", &config).match_list(&["foo", "FOO", "FoO", "xxFoOxx"]);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn match_iter_matches_match_list() {
        let haystacks = [
            "deadbeef",
            "deadbf",
            "deadbeefg",
            "deadbe",
            "no-match",
            "DeAdBe",
            "é다😀dead__be",
        ];
        for needle in ["deadbe", "é다😀"] {
            for max_typos in [None, Some(0), Some(1), Some(2), Some(3)] {
                let config = Config::default()
                    .max_typos(max_typos)
                    .sort(SortStrategy::IndexAsc);
                let mut matcher = Matcher::new(needle, &config);
                let from_iter = matcher.match_iter(haystacks.iter()).collect::<Vec<_>>();
                let from_list = matcher.match_list(&haystacks);
                assert_eq!(
                    from_iter, from_list,
                    "needle: {needle:?}, max_typos: {max_typos:?}"
                );
            }
        }
    }

    #[test]
    fn match_iter_empty_needle_yields_all() {
        let mut matcher = Matcher::new("", &Config::default());
        let matches = matcher
            .match_iter(["foo", "bar"].iter())
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[1].index, 1);
    }

    #[test]
    fn match_iter_indices_matches_match_list_indices() {
        let haystacks = [
            "deadbeef",
            "deadbf",
            "deadbeefg",
            "deadbe",
            "no-match",
            "DeAdBe",
            "é다😀dead__be",
        ];
        for needle in ["deadbe", "é다😀"] {
            for max_typos in [None, Some(0), Some(1), Some(2), Some(3)] {
                let config = Config::default()
                    .max_typos(max_typos)
                    .sort(SortStrategy::IndexAsc);
                let mut matcher = Matcher::new(needle, &config);
                let from_iter = matcher
                    .match_iter_indices(haystacks.iter())
                    .collect::<Vec<_>>();
                let from_list = matcher.match_list_indices(&haystacks);
                assert_eq!(
                    from_iter, from_list,
                    "needle: {needle:?}, max_typos: {max_typos:?}"
                );
            }
        }
    }

    #[test]
    fn match_iter_indices_empty_needle_yields_all() {
        let mut matcher = Matcher::new("", &Config::default());
        let matches = matcher
            .match_iter_indices(["foo", "bar"].iter())
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[1].index, 1);
    }

    #[test]
    fn test_empty_needle() {
        let haystack = ["foo", "bar"];
        let mut matcher = Matcher::new("", &Config::default());
        assert!(matcher.patterns.is_empty());

        let matches = matcher.match_list(&haystack);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[1].index, 1);

        let indices = matcher.match_list_indices(&haystack);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].index, 0);
        assert_eq!(indices[1].index, 1);
    }

    fn is_u8(backend: &MatcherBackend) -> bool {
        use crate::smith_waterman::ScoreWidth;
        match backend {
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX512U8(_) => true,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX512(_) => false,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX(m) => matches!(m.smith_waterman, ScoreWidth::U8(_)),
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::SSE(m) => matches!(m.smith_waterman, ScoreWidth::U8(_)),
            #[cfg(target_arch = "aarch64")]
            MatcherBackend::NEON(m) => matches!(m.smith_waterman, ScoreWidth::U8(_)),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            MatcherBackend::Wasm(m) => matches!(m.smith_waterman, ScoreWidth::U8(_)),
            #[cfg(scalar_fallback)]
            MatcherBackend::Scalar(m) => matches!(m.smith_waterman, ScoreWidth::U8(_)),
            _ => panic!("expected a fuzzy backend"),
        }
    }

    #[test]
    fn u8_path_selected_for_short_needle() {
        let matcher = Matcher::new("abc", &Config::default());
        let CompiledPatterns::Single(pattern) = &matcher.patterns else {
            panic!("expected a single pattern");
        };
        assert!(is_u8(&pattern.backend));
    }

    /// AVX-512 runs ASCII needles on the u8 kernel only within its scan's
    /// ramp (9 bytes with the default scoring), the u16 kernel past it,
    /// and scores them alike
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_u8_kernel_needs_the_ramp_headroom() {
        if !MatcherAVX512U8::is_available() {
            return;
        }
        let haystacks = ["xcomponentsx/a/b", "c_o_m_p_o_n_e_n_t_s", "compon/ents"];
        for (needle, u8) in [("component", true), ("components", false)] {
            let config = Config::default()
                .max_typos(None)
                .sort(crate::SortStrategy::IndexAsc);
            let mut matcher = Matcher::new(needle, &config);
            let CompiledPatterns::Single(pattern) = &matcher.patterns else {
                panic!("expected a single pattern");
            };
            assert_eq!(matches!(pattern.backend, MatcherBackend::AVX512U8(_)), u8);
            let scores: Vec<_> = matcher
                .match_list(&haystacks)
                .iter()
                .map(|m| m.score)
                .collect();
            let want: Vec<_> = haystacks
                .iter()
                .map(|h| {
                    crate::smith_waterman::reference::score_with_exact_bonus(
                        needle,
                        h,
                        &config.scoring,
                        false,
                        false,
                    )
                })
                .collect();
            assert_eq!(scores, want, "{needle}");
        }
    }

    #[test]
    fn u16_path_selected_for_long_needle() {
        let matcher = Matcher::new("abcdefghijklmnopqrst", &Config::default());
        let CompiledPatterns::Single(pattern) = &matcher.patterns else {
            panic!("expected a single pattern");
        };
        assert!(!is_u8(&pattern.backend));
    }

    #[test]
    fn unicode_needle_width_counts_scalars() {
        // 8 scalars but 24 bytes: one row per scalar in the Unicode kernel,
        // one per byte in the ASCII kernel
        let needle = "그것은무엇입니까";
        let is_u8_with = |unicode| {
            let matcher = Matcher::new(needle, &Config::default().unicode(unicode));
            let CompiledPatterns::Single(pattern) = &matcher.patterns else {
                panic!("expected a single pattern");
            };
            is_u8(&pattern.backend)
        };
        assert!(is_u8_with(crate::UnicodeMatching::Smart));
        assert!(!is_u8_with(crate::UnicodeMatching::Ignore));
    }

    #[test]
    fn reuse_handles_state_changes() {
        let long_needle = "abcdefghijklmnopqrst";
        let first_haystacks = [
            "xxabcdefghijklmnopqrstxx".to_string(),
            "abcdefghijklmnopqrst".to_string(),
            "no-match".to_string(),
        ];
        let first_config = Config::default()
            .max_typos(None)
            .sort(SortStrategy::IndexAsc);
        let mut matcher = Matcher::new(long_needle, &first_config);

        let first = matcher.match_list(&first_haystacks);
        assert_eq!(
            &first,
            &Matcher::new(long_needle, &first_config).match_list(&first_haystacks)
        );

        let second_haystacks = [
            "fooBar".to_string(),
            "foo_bar".to_string(),
            "fbr".to_string(),
            "bar".to_string(),
        ];
        matcher.set_pattern("fB");
        let second_config = Config::default()
            .casing(CaseMatching::Smart)
            .sort(SortStrategy::IndexAsc);
        matcher.set_config(second_config.clone());
        let second = matcher.match_list(&second_haystacks);
        assert_eq!(
            &second,
            &Matcher::new("fB", &second_config).match_list(&second_haystacks)
        );

        let unicode_haystacks = [
            "é다😀".to_string(),
            "xxé__다__😀yy".to_string(),
            "é다".to_string(),
            "plain ascii".to_string(),
        ];
        matcher.set_pattern("é다😀");
        let unicode_config = Config::default()
            .max_typos(Some(0))
            .sort(SortStrategy::IndexAsc);
        matcher.set_config(unicode_config.clone());
        let unicode = matcher.match_list(&unicode_haystacks);
        assert_eq!(
            &unicode,
            &Matcher::new("é다😀", &unicode_config).match_list(&unicode_haystacks)
        );

        matcher.set_pattern("fB");
        let third_config = Config::default()
            .casing(CaseMatching::Ignore)
            .max_typos(Some(1));
        matcher.set_config(third_config.clone());
        let third = matcher.match_list(&first_haystacks);
        assert_eq!(
            &third,
            &Matcher::new("fB", &third_config).match_list(&first_haystacks)
        );
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn test_match_end_col_through_match_list() {
        let config = Config::default()
            .max_typos(None)
            .sort(SortStrategy::IndexAsc);
        let matches = Matcher::new("abc", &config).match_list(&["xabcx", "abcdef", "xxabc"]);
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].end_col, 3);
        assert_eq!(matches[1].end_col, 2);
        assert_eq!(matches[2].end_col, 4);
    }
}
