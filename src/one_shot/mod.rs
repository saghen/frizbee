#[cfg(feature = "parallel_sort")]
use rayon::prelude::*;

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
    assert!(
        haystacks.len() < (u32::MAX as usize),
        "haystack index overflow"
    );

    // Matching
    let mut matches = vec![];
    let mut matcher = Matcher::new(needle.as_ref(), config);
    matcher.match_list_impl(haystacks, 0, &mut matches);

    // Sorting
    if config.sort {
        #[cfg(feature = "parallel_sort")]
        matches.par_sort_unstable();
        #[cfg(not(feature = "parallel_sort"))]
        matches.sort_unstable();
    }

    matches
}
