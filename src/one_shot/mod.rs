use crate::{Config, Match};

mod matcher;
mod parallel;

use matcher::Matcher;
pub use parallel::match_list_parallel;

pub fn match_list<S1: AsRef<str>, S2: AsRef<str>>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
) -> Vec<Match> {
    let mut matches = vec![];
    let mut matcher = Matcher::new(needle.as_ref(), config);
    matcher.match_list_impl(haystacks, 0, &mut matches);

    if config.sort {
        matches.sort_unstable();
    }

    matches
}
