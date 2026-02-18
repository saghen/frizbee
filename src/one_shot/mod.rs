use crate::{Config, Match, MatchIndices};

mod matcher;
mod parallel;

pub use matcher::Matcher;
pub use parallel::match_list_parallel;

pub fn match_list<S1: AsRef<str>, S2: AsRef<str>>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
) -> Vec<Match> {
    Matcher::new(needle.as_ref(), config).match_list(haystacks)
}

pub fn match_list_indices<S1: AsRef<str>, S2: AsRef<str>>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
) -> Vec<MatchIndices> {
    Matcher::new(needle.as_ref(), config).match_list_indices(haystacks)
}
