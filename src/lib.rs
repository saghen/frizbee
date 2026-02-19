//! Frizbee is a SIMD typo-resistant fuzzy string matcher written in Rust. The core of the algorithm uses Smith-Waterman with affine gaps, similar to FZF, but with many of the scoring bonuses from FZY. In the included benchmark, with typo resistance disabled, it outperforms [nucleo](https://github.com/helix-editor/nucleo) by ~1.7x and supports multithreading, see [benchmarks](./BENCHMARKS.md). It matches against bytes directly, ignoring unicode. Used by [blink.cmp](https://github.com/saghen/blink.cmp), [skim](https://github.com/skim-rs/skim), [fff.nvim](https://github.com/dmtrKovalenko/fff.nvim). Special thank you to [stefanboca](https://github.com/stefanboca) and [ii14](https://github.com/ii14)!
//!
//! The core of the algorithm is Smith-Waterman with affine gaps and row-wise parallelism via SIMD. Besides the parallelism, this is the basis of other popular fuzzy matching algorithms like [FZF](https://github.com/junegunn/fzf) and [Nucleo](https://github.com/helix-editor/nucleo). The main properties of Smith-Waterman are:
//! - Always finds the best alignment
//! - Supports insertion, deletion and substitution
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
//! let matches = matcher.match_list(&haystacks);
//! ```
//!
//! # Example: custom scoring offsets
//!
//! If you want to apply a custom score before sorting the results, you may use the
//! `match_iter` API on the `Matcher`.
//!
//! ```rust
//! use frizbee::{Match, Matcher, Config};
//!
//! let needle = "fBr";
//! let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
//!
//! let mut matcher = Matcher::new(needle, &Config::default());
//! let mut matches = matcher.match_iter(&haystacks)
//!     .map(|m| Match { score: m.score + 10, ..m })
//!     .collect::<Vec<_>>();
//! matches.sort_unstable();
//! ```
//!
//! # Example: custom scoring based on alignment path
//!
//! If you want to apply a custom score based on which characters were matched in the, you may drop
//! down to the lower level
//!
//! ```rust
//! use frizbee::{Config, Match, Matcher, smith_waterman::Alignment};
//!
//! let needle = "fBr";
//! let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
//!
//! // Guarding against empty needle is required by `prefilter_iter()`
//! if needle.is_empty() {
//!       return
//!       // return (0..haystacks.len())
//!       //   .map(|index| Match {
//!       //       index: index as u32,
//!       //       score: 0,
//!       //       exact: false,
//!       //   })
//!       //   .collect();
//! }
//!
//!
//! let config = Config::default();
//! let mut matcher = Matcher::new(needle, &config);
//! let mut matches = matcher.prefilter_iter(&haystacks)
//!     .filter_map(|(index, haystack, skipped_chunks)| {
//!         let mut score = matcher
//!             .smith_waterman
//!             .score_haystack(haystack);
//!
//!         for alignment in matcher.iter_alignment_path(skipped_chunks, score) {
//!              // Return None if Alignment is None (max typos exceeded)
//!              match alignment? {
//!                 Alignment::Match((needle_idx, haystack_idx)) => {
//!                     // adjust score as desired
//!                 }
//!                 // optionally handle Left, Up and Mismatch
//!                 _ => {}
//!              }
//!         }
//!
//!         let exact = skipped_chunks == 0 && needle.as_bytes() == haystack;
//!         if exact {
//!             score += config.scoring.exact_match_bonus;
//!         }
//!
//!         Some(Match {
//!             index: index as u32,
//!             score,
//!             exact,
//!         })
//!     })
//!     .collect::<Vec<_>>();
//!
//! matches.sort_unstable();
//! ```

use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod r#const;
mod one_shot;
pub mod prefilter;
mod simd;
pub mod smith_waterman;

pub use one_shot::{Matcher, match_list, match_list_indices, match_list_parallel};

use r#const::*;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Match {
    pub score: u16,
    /// Index of the match in the original list of haystacks
    pub index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    pub exact: bool,
}

impl Match {
    pub fn from_index(index: usize) -> Self {
        Self {
            score: 0,
            index: index as u32,
            exact: false,
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

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config {
    /// The maximum number of characters missing from the needle, before an item in the
    /// haystack is filtered out
    pub max_typos: Option<u16>,
    /// Sort the results by score (descending)
    pub sort: bool,
    /// Controls the scoring used by the smith waterman algorithm. You may tweak these but pay
    /// close attention to the documentation for each property, as small changes can lead to
    /// poor matching.
    pub scoring: Scoring,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_typos: Some(0),
            sort: true,
            scoring: Scoring::default(),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
