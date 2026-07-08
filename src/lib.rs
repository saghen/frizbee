//! Frizbee is a SIMD typo-resistant fuzzy string matcher written in Rust. The core of the algorithm uses Smith-Waterman with affine gaps, similar to FZF. In the included benchmark, with typo resistance disabled, it outperforms [Nucleo](https://github.com/helix-editor/nucleo) by ~4x and [FZF](https://github.com/junegunn/fzf) by ~5x and supports multithreading, see [benchmarks](./BENCHMARKS.md). When matching against unicode, it outperforms Nucleo and FZF by 20x.
//!
//! Used by [blink.cmp](https://github.com/saghen/blink.cmp), [skim](https://github.com/skim-rs/skim), and [fff](https://github.com/dmtrKovalenko/fff). Special thank you to [stefanboca](https://github.com/stefanboca) and [ii14](https://github.com/ii14)!
//!
//! For commercial support, please [contact me](mailto:frizbee@liam.super.fish). I'd be happy to work with you directly! Also, please consider [sponsoring me](https://github.com/sponsors/saghen).
//!
//! The core of the algorithm is Smith-Waterman with affine gaps and row-wise parallelism via SIMD. Besides the parallelism, this is the basis of other popular fuzzy matching algorithms like [FZF](https://github.com/junegunn/fzf) and [Nucleo](https://github.com/helix-editor/nucleo). The main properties of Smith-Waterman are:
//! - Always finds the best alignment
//! - Supports insertion (unmatched char in haystack, basis of fuzzy matching)
//! - Supports deletion (unmatched char in needle, basis of typo-resistance)
//! - Supports substitution (haystack and needle char mismatch, basis of typo-resistance)
//!
//! # Example: using `match_list`
//!
//! ```rust
//! use frizbee::{match_list, match_list_parallel, Config};
//!
//! let needle = "fBr";
//! let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
//!
//! let matches = match_list(needle, &haystacks, &Config::default());
//! // or in parallel (8 threads)
//! let matches = match_list_parallel(needle, &haystacks, &Config::default(), 8);
//! ```
//!
//! # Example: using `Matcher`
//!
//! Useful for when you want to match one needle against more than one haystack.
//!
//! ```rust
//! use frizbee::{Matcher, Config};
//!
//! let needle = "fBr";
//! let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
//!
//! let mut matcher = Matcher::new(needle, &Config::default());
//! // or use a matching mode (fuzzy, substring, prefix, suffix, exact) based on the query
//! // syntax, e.g. foo, 'foo, ^foo, foo$, ^foo$
//! let mut matcher = Matcher::from_query(needle);
//!
//! let matches = matcher.match_list(&haystacks);
//! ```
//!
//! # Example: using `FuzzyMatchExt`
//!
//! ```rust
//! use frizbee::{iter::FuzzyMatchExt, Config, radix_sort_matches};
//!
//! let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
//! let mut matches: Vec<_> = haystacks
//!     .iter()
//!     .fuzzy_match("fBr", &Config::default())
//!     .collect();
//! radix_sort_matches(&mut matches);
//! ```

use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod r#const;
mod k_merge;
mod literal;
mod matcher;
mod prefilter;
mod smith_waterman;
mod sort;

use r#const::*;

pub use k_merge::k_merge_matches;
pub use matcher::Matcher;
pub use sort::radix_sort_matches;

/// Iterator extension for fuzzy matching
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
pub mod iter {
    pub use crate::matcher::{FuzzyMatch, FuzzyMatchExt, FuzzyMatchIndices};
}

/// Matches a list of haystacks, returning a list of [`Match`] values.
/// This API provides the most performant path when matching on lists.
///
/// This API should not be called with one item at a time as it performs dynamic dispatch to
/// the underlying backend. Instead, consider using the [`Matcher::match_iter`],
/// [`Matcher::match_one`] or [`iter::FuzzyMatchExt`] API.
pub fn match_list<S1: AsRef<str>, S2: AsRef<str>>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
) -> Vec<Match> {
    Matcher::new(needle.as_ref(), config).match_list(haystacks)
}

/// Matches a list of haystacks, returning a list of [`MatchIndices`] which are equivalent
/// to [`Match`] except they include the indices of the matched characters in the haystack.
///
/// This API has not been optimized for performance, and should only be used on small lists or
/// after matching a list of haystacks with [`match_list`]. Useful for displaying
/// matched indices in the UI.
///
/// This API should not be called with one item at a time as it performs dynamic dispatch to
/// the underlying backend. Instead, consider using the [`Matcher::match_iter_indices`] or
/// [`iter::FuzzyMatchExt`] API.
pub fn match_list_indices<S1: AsRef<str>, S2: AsRef<str>>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
) -> Vec<MatchIndices> {
    Matcher::new(needle.as_ref(), config).match_list_indices(haystacks)
}

/// Matches a list of haystacks in parallel on multiple real threads, returning a list of
/// [`Match`] values. Threads work on 2048 item chunks, and the final result is ordered
/// according to [`Config::sort`]. The `threads` must be >0.
///
/// This API provides the most performant path when matching on lists.
pub fn match_list_parallel<S1: AsRef<str>, S2: AsRef<str> + Sync>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
    threads: usize,
) -> Vec<Match> {
    Matcher::new(needle.as_ref(), config).match_list_parallel(haystacks, threads)
}

/// Result of a fuzzy match, containing the score and index in the haystack
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Match {
    pub score: u16,
    /// Index of the match in the original list of haystacks
    pub index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    pub exact: bool,
    /// Column position (0-based haystack byte offset) where the best alignment ends.
    /// Only populated when the `match_end_col` feature is enabled.
    #[cfg(feature = "match_end_col")]
    pub end_col: u16,
}

impl Match {
    pub fn from_index(index: usize) -> Self {
        Self {
            score: 0,
            index: index as u32,
            exact: false,
            #[cfg(feature = "match_end_col")]
            end_col: 0,
        }
    }
}

impl PartialOrd for Match {
    fn partial_cmp(&self, other: &Match) -> Option<Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}
impl Ord for Match {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.score as u64)
            .cmp(&(other.score as u64))
            .reverse()
            .then_with(|| self.index.cmp(&other.index))
    }
}
impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index
    }
}
impl Eq for Match {}

/// Like [`Match`] but includes the indices of the chars in the haystack that matched the needle in
/// reverse order
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MatchIndices {
    pub score: u16,
    /// Index of the match in the original list of haystacks
    pub index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    pub exact: bool,
    /// Indices of the chars in the haystack that matched the needle in reverse order
    pub indices: Vec<usize>,
}

impl MatchIndices {
    pub fn from_index(index: usize) -> Self {
        Self {
            score: 0,
            index: index as u32,
            exact: false,
            indices: vec![],
        }
    }
}

impl PartialOrd for MatchIndices {
    fn partial_cmp(&self, other: &MatchIndices) -> Option<Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}
impl Ord for MatchIndices {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.score as u64)
            .cmp(&(other.score as u64))
            .reverse()
            .then_with(|| self.index.cmp(&other.index))
    }
}
impl PartialEq for MatchIndices {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index
    }
}
impl Eq for MatchIndices {}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Config {
    /// The maximum number of characters missing from the needle, before an item in the
    /// haystack is filtered out
    pub max_typos: Option<u16>,
    /// Controls how case sensitivity/insensitivity is handled while matching.
    #[cfg_attr(feature = "serde", serde(default))]
    pub casing: CaseMatching,
    /// Controls how unicode is handled while matching.
    #[cfg_attr(feature = "serde", serde(default))]
    pub unicode: UnicodeMatching,
    /// Selects the matching algorithm: fuzzy (Smith-Waterman) or one of the literal modes
    /// (exact, prefix, suffix, substring). Literal modes require the needle to appear as a
    /// contiguous run of characters and do not support typos (`max_typos` is ignored).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matching: Matching,
    /// Controls how results are ordered.
    #[cfg_attr(feature = "serde", serde(default))]
    pub sort: SortStrategy,
    /// Controls the scoring used by the smith waterman algorithm. You may tweak these but pay
    /// close attention to the documentation for each property, as small changes can lead to
    /// poor matching.
    pub scoring: Scoring,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_typos: Some(0),
            casing: CaseMatching::Ignore,
            unicode: UnicodeMatching::Smart,
            matching: Matching::Fuzzy,
            sort: SortStrategy::Score,
            scoring: Scoring::default(),
        }
    }
}

impl Config {
    /// Sets the matching mode
    pub fn matching(mut self, matching: Matching) -> Self {
        self.matching = matching;
        self
    }

    /// Sets the maximum number of typos allowed
    pub fn max_typos(mut self, max_typos: Option<u16>) -> Self {
        self.max_typos = max_typos;
        self
    }

    /// Sets the casing mode
    pub fn casing(mut self, casing: CaseMatching) -> Self {
        self.casing = casing;
        self
    }

    /// Sets the unicode mode
    pub fn unicode(mut self, unicode: UnicodeMatching) -> Self {
        self.unicode = unicode;
        self
    }

    /// Sets how results are ordered
    pub fn sort(mut self, sort: SortStrategy) -> Self {
        self.sort = sort;
        self
    }

    /// Sets the scoring
    pub fn scoring(mut self, scoring: Scoring) -> Self {
        self.scoring = scoring;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SortStrategy {
    /// Sort by descending score, using haystack index as a tie breaker
    #[default]
    Score,
    /// Preserve input order by haystack index
    Index,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CaseMatching {
    /// Ignore case while matching.
    #[default]
    Ignore,
    /// Ignore case unless the needle contains uppercase
    Smart,
    /// Require matching bytes to have the same case
    Respect,
}

impl CaseMatching {
    #[inline(always)]
    pub(crate) fn respects_case_for(self, needle: &str) -> bool {
        match self {
            CaseMatching::Ignore => false,
            CaseMatching::Smart => needle.chars().any(char::is_uppercase),
            CaseMatching::Respect => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UnicodeMatching {
    /// Always match against bytes directly
    Ignore,
    /// Ignore unicode unless the needle contains a multi-byte unicode char
    #[default]
    Smart,
    /// Always use expensive unicode Smith Waterman for correctness across
    /// multi-byte unicode chars in the haystack
    Always,
}

impl UnicodeMatching {
    #[inline(always)]
    pub(crate) fn respects_unicode_for(self, needle: &str) -> bool {
        match self {
            UnicodeMatching::Ignore => false,
            UnicodeMatching::Smart => !needle.is_ascii(),
            UnicodeMatching::Always => true,
        }
    }
}

/// Selects the matching algorithm
///
/// [`Matching::Fuzzy`] uses the Smith-Waterman algorithm (with typos, gaps and substitutions)
/// [`Matching::Exact`] matches the haystack exactly
/// [`Matching::Prefix`] matches the haystack if it starts with the needle
/// [`Matching::Suffix`] matches the haystack if it ends with the needle
/// [`Matching::Substring`] matches the haystack if it contains the needle
///
/// Only the [`Matching::Fuzzy`] mode supports typos
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Matching {
    /// Smith-Waterman fuzzy matching (the default)
    #[default]
    Fuzzy,
    /// The haystack must equal the needle
    Exact,
    /// The haystack must start with the needle
    Prefix,
    /// The haystack must end with the needle
    Suffix,
    /// The needle must appear somewhere in the haystack. When it appears more than once, the
    /// highest-scoring occurrence is used, preferring earlier matches on tie
    Substring,
}

impl Matching {
    #[inline(always)]
    pub(crate) fn is_fuzzy(self) -> bool {
        matches!(self, Matching::Fuzzy)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Scoring {
    /// Score for a matching character between needle and haystack
    pub match_score: u16,
    /// Penalty for a mismatch (substitution)
    pub mismatch_penalty: u16,
    /// Penalty for opening a gap (deletion/insertion)
    pub gap_open_penalty: u16,
    /// Penalty for extending a gap (deletion/insertion)
    pub gap_extend_penalty: u16,

    /// Bonus for matching the first character of the haystack (e.g. "h" on "hello_world")
    pub prefix_bonus: u16,
    /// Bonus for matching a capital letter after a lowercase letter
    /// (e.g. "b" on "fooBar" will receive a bonus on "B")
    pub capitalization_bonus: u16,
    /// Bonus for matching the case of the needle (e.g. "WorLd" on "WoRld" will receive a bonus on "W", "o", "d")
    pub matching_case_bonus: u16,
    /// Bonus for matching the exact needle (e.g. "foo" on "foo" will receive the bonus)
    pub exact_match_bonus: u16,
    /// Bonus for matching _after_ a delimiter character (e.g. "hw" on "hello_world",
    /// will give a bonus on "w") if "_" is included in the delimiters string
    pub delimiter_bonus: u16,
}

impl Default for Scoring {
    fn default() -> Self {
        Scoring {
            match_score: MATCH_SCORE,
            mismatch_penalty: MISMATCH_PENALTY,
            gap_open_penalty: GAP_OPEN_PENALTY,
            gap_extend_penalty: GAP_EXTEND_PENALTY,

            prefix_bonus: PREFIX_BONUS,
            capitalization_bonus: CAPITALIZATION_BONUS,
            matching_case_bonus: MATCHING_CASE_BONUS,
            exact_match_bonus: EXACT_MATCH_BONUS,
            delimiter_bonus: DELIMITER_BONUS,
        }
    }
}

impl Scoring {
    /// Panics if a needle of `needle_len` bytes could overflow the `u16` score. `max_bonus_per_char`
    /// is the largest bonus a single matched character can add on top of `match_score`; it differs
    /// between the fuzzy and literal scorers, so each passes its own.
    pub(crate) fn guard_against_score_overflow(&self, needle_len: usize, max_bonus_per_char: u16) {
        let max_per_char = self.match_score + max_bonus_per_char;
        let max_needle_len = (u16::MAX - self.prefix_bonus - self.exact_match_bonus) / max_per_char;
        assert!(
            needle_len <= max_needle_len as usize,
            "needle too long and could overflow the u16 score: {needle_len} > {max_needle_len}"
        );
    }
}
