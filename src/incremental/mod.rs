#[cfg(feature = "parallel_sort")]
use rayon::prelude::*;

use crate::{
    Config,
    prefilter::Prefilter,
    smith_waterman::{greedy::match_greedy, x86_64::SmithWatermanMatcher},
};

mod aligned_string;
mod match_;
pub(crate) use aligned_string::AlignedBytes;
use bumpalo::{Bump, collections::Vec as BumpVec};
pub use match_::IncrementalMatch;

// TODO: implement clone
#[derive(Debug)]
pub struct IncrementalMatcher {
    allocator: Bump,
    haystacks: Vec<AlignedBytes>,
    config: Config,
}

impl IncrementalMatcher {
    pub fn new(config: Config) -> Self {
        Self {
            allocator: Bump::with_capacity(160 * 1024 * 1024), // 160 MiB
            haystacks: vec![],
            config,
        }
    }

    pub fn add_haystack(&mut self, haystack: &str) {
        self.haystacks.push(AlignedBytes::new(haystack.as_bytes()));
    }

    pub fn add_haystacks(&mut self, haystacks: &[&str]) {
        self.haystacks
            .extend(haystacks.iter().map(|h| AlignedBytes::new(h.as_bytes())));
    }

    pub fn match_list(&mut self, needle: &str) -> Vec<IncrementalMatch<'_>> {
        // Guard against empty needle or empty haystacks
        if needle.is_empty() {
            return (0..self.haystacks.len())
                .map(|i| IncrementalMatch {
                    index: i as u32,
                    score: 0,
                    exact: false,
                    score_matrix: BumpVec::new_in(&self.allocator),
                })
                .collect();
        }
        if self.haystacks.is_empty() {
            return vec![];
        }

        let needle = needle.as_bytes();

        // If max_typos is set, we can ignore any haystacks that are shorter than the needle
        // minus the max typos, since it's impossible for them to match
        let min_haystack_len = self
            .config
            .max_typos
            .map(|max| needle.len() - (max as usize))
            .unwrap_or(0);

        let mut matches = vec![];
        let prefilter = Prefilter::<true>::new(needle);
        let mut smith_waterman = SmithWatermanMatcher::<true>::new(needle, &self.config.scoring);

        for (i, haystack, skipped_chunks) in self
            .haystacks
            .iter()
            .enumerate()
            .filter(|(_, h)| h.len() >= min_haystack_len)
            // Prefiltering
            .filter_map(|(i, haystack)| {
                let (matched, skipped_chunks) =
                    self.config.max_typos.map_or((true, 0), |max_typos| {
                        prefilter.match_haystack(haystack.as_slice(), max_typos)
                    });
                // Skip any chunks where we know the needle doesn't match
                matched.then_some((i, haystack, skipped_chunks))
            })
        {
            // Haystack too large, fallback to greedy matching
            let match_result = if haystack.len() > 512 {
                match_greedy(
                    needle,
                    &haystack.as_slice()[(skipped_chunks * 16)..],
                    &self.config.scoring,
                )
                .map(|(score, _)| (score, BumpVec::new_in(&self.allocator)))
            }
            // Smith waterman matching
            else {
                let capacity = (needle.len() + 1) * (haystack.len().div_ceil(16) + 1);
                let mut score_matrix = BumpVec::with_capacity_in(capacity, &self.allocator);
                unsafe {
                    std::ptr::write_bytes(score_matrix.as_mut_ptr(), 0, capacity);
                    score_matrix.set_len(capacity);
                }

                smith_waterman
                    .match_haystack(
                        &haystack.as_slice()[(skipped_chunks * 16)..],
                        self.config.max_typos,
                        &mut score_matrix,
                    )
                    .map(|score| (score, score_matrix))
            };

            if let Some((mut score, score_matrix)) = match_result {
                // Exact match bonus
                let exact = skipped_chunks == 0 && needle == haystack.as_slice();
                if exact {
                    score += self.config.scoring.exact_match_bonus;
                }

                matches.push(IncrementalMatch {
                    index: i as u32,
                    score,
                    exact,
                    score_matrix,
                });
            }
        }

        if self.config.sort {
            #[cfg(feature = "parallel_sort")]
            matches.par_sort_unstable();
            #[cfg(not(feature = "parallel_sort"))]
            matches.sort_unstable();
        }

        matches
    }
}
