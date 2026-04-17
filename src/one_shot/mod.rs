use crate::{Config, Match, MatchIndices, Matchable};

mod matcher;
mod parallel;

pub use matcher::Matcher;
pub use parallel::{
    match_list_parallel, match_list_parallel_chunked, match_list_parallel_resolved,
};

pub fn match_list<S1: AsRef<str>, S2: Matchable>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
) -> Vec<Match> {
    Matcher::new(needle.as_ref(), config).match_list(haystacks)
}

pub fn match_list_indices<S1: AsRef<str>, S2: Matchable>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
) -> Vec<MatchIndices> {
    Matcher::new(needle.as_ref(), config).match_list_indices(haystacks)
}
