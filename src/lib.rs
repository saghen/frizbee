#![allow(unsafe_op_in_unsafe_fn)]

use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod r#const;
mod one_shot;
pub mod prefilter;
mod simd;
pub mod smith_waterman;

pub use one_shot::{match_list, match_list_parallel};

use r#const::*;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Match {
    pub score: u16,
    /** Index of the match in the original list of haystacks */
    pub index: u32,
    /** Matched the needle exactly (i.e. "foo" on "foo") */
    pub exact: bool,
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
    pub indices: Vec<usize>,
    /** Matched the needle exactly (i.e. "foo" on "foo") */
    pub exact: bool,
}

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
