use std::pin::Pin;

#[cfg(feature = "parallel_sort")]
use rayon::prelude::*;

use crate::{
    Config, Match,
    prefilter::Prefilter,
    simd::{AVXVector, SSEVector},
    smith_waterman::{
        greedy::match_greedy,
        simd::{SmithWatermanMatcher, typos_from_score_matrix},
    },
};

mod aligned_string;
mod match_;
pub(crate) use aligned_string::AlignedBytes;
use bumpalo::{Bump, collections::Vec as BumpVec};
pub use match_::IncrementalMatch;

// TODO: implement clone
#[derive(Debug)]
pub struct IncrementalMatcher {
    allocator: Pin<Box<Bump>>,
    // SAFETY: matches always dropped before allocator,
    // allocator is pinned and never moves
    matches: Vec<IncrementalMatch<'static>>, // actually borrows from allocator
    pub needle: String,
    pub haystacks: Vec<AlignedBytes>,
    pub config: Config,
}

impl IncrementalMatcher {
    pub fn new(config: Config) -> Self {
        Self {
            allocator: Pin::new(Box::new(Bump::with_capacity(160 * 1024 * 1024))), // 160 MiB
            needle: String::new(),
            matches: vec![],
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

    pub fn match_list(&mut self, needle: &str) -> Vec<Match> {
        // Guard against empty needle or empty haystacks
        if needle.is_empty() {
            self.reset();
            return (0..self.haystacks.len())
                .map(|i| Match {
                    index: i as u32,
                    score: 0,
                    exact: false,
                })
                .collect();
        }
        if self.haystacks.is_empty() {
            self.reset();
            return vec![];
        }

        // Return existing matches since the needle has not changed
        if needle == self.needle {
            return self.matches.iter().map(Into::into).collect();
        }

        // If max_typos is set, we can ignore any haystacks that are shorter than the needle
        // minus the max typos, since it's impossible for them to match
        let min_haystack_len = self
            .config
            .max_typos
            .map(|max| needle.len() - (max as usize))
            .unwrap_or(0);

        let matches = if !(self.needle.is_empty() && self.matches.is_empty())
            && needle.starts_with(&self.needle)
        {
            let needle = needle.as_bytes();
            let prev_matches = std::mem::take(&mut self.matches);
            self.match_incremental_suffix(
                prev_matches,
                needle,
                &needle[self.needle.len()..],
                min_haystack_len,
            )
        } else {
            self.reset();
            IncrementalMatcher::match_all(
                needle.as_bytes(),
                &self.haystacks,
                min_haystack_len,
                &self.config,
                &self.allocator,
            )
        };
        self.needle = needle.to_string();
        self.matches = matches;

        let mut matches = self.matches.iter().map(Into::into).collect::<Vec<_>>();
        if self.config.sort {
            #[cfg(feature = "parallel_sort")]
            matches.par_sort_unstable();
            #[cfg(not(feature = "parallel_sort"))]
            matches.sort_unstable();
        }
        matches
    }

    // /// Removed one or more characters from the needle. We can use and truncate the existing matches
    // /// but we must also search for new matches in the entire haystack.
    // fn match_incremental_prefix(
    //     &mut self,
    //     needle: &str,
    //     min_haystack_len: usize,
    //     prefilter: Prefilter<true>,
    //     smith_waterman: SmithWatermanMatcher<SSEVector, AVXVector, true>,
    // ) -> Vec<IncrementalMatch> {
    //     vec![]
    // }

    /// Added one or more characters to the needle. We can use the existing matches, matching only
    /// the new characters.
    fn match_incremental_suffix(
        &self,
        matches: Vec<IncrementalMatch<'static>>,
        needle: &[u8],
        needle_suffix: &[u8],
        min_haystack_len: usize,
    ) -> Vec<IncrementalMatch<'static>> {
        let prefilter = Prefilter::<true>::new(needle);
        let smith_waterman = SmithWatermanMatcher::<SSEVector, AVXVector, true>::new(
            needle_suffix,
            &self.config.scoring,
        );

        matches
            .into_iter()
            .map(|match_| (self.haystacks[match_.index as usize].as_slice(), match_))
            .filter(|(haystack, _)| haystack.len() >= min_haystack_len)
            // Prefiltering
            .filter_map(|(haystack, match_)| {
                let (matched, skipped_chunks) =
                    self.config.max_typos.map_or((true, 0), |max_typos| {
                        prefilter.match_haystack(haystack, max_typos)
                    });
                // Skip any chunks where we know the needle doesn't match
                matched.then_some((haystack, match_, skipped_chunks))
            })
            .filter_map(|(haystack, mut match_, skipped_chunks)| {
                let haystack_skipped = &haystack[skipped_chunks * 16..];

                let mut score = if haystack.len() > 512 {
                    match_greedy(needle_suffix, haystack_skipped, &self.config.scoring)
                        .map(|(score, _)| score)
                } else {
                    match_.extend(needle_suffix.len(), haystack_skipped.len());
                    // We only want the smith waterman to fill in the new rows, so we slice from the
                    // previous row
                    let score_matrix_start = match_.score_matrix.len()
                        - (needle_suffix.len() + 1) * (haystack_skipped.len().div_ceil(16) + 1);
                    let score = smith_waterman.score_haystack(
                        haystack_skipped,
                        &mut match_.score_matrix[score_matrix_start..],
                    );

                    if let Some(max_typos) = self.config.max_typos {
                        let typos = typos_from_score_matrix(
                            &match_.score_matrix,
                            score,
                            max_typos,
                            haystack.len(),
                        );
                        if typos > max_typos {
                            return None;
                        }
                    }
                    Some(score)
                }?;

                let exact = needle == haystack;
                if exact {
                    score += self.config.scoring.exact_match_bonus;
                }
                match_.score = score;
                match_.exact = exact;
                Some(match_)
            })
            .collect()
    }

    /// No existing matches so match everything as if we're performing a one shot match, except we
    /// store the score matrices for each match.
    fn match_all(
        needle: &[u8],
        haystacks: &[AlignedBytes],
        min_haystack_len: usize,
        config: &Config,
        allocator: &Bump,
    ) -> Vec<IncrementalMatch<'static>> {
        let prefilter = Prefilter::<true>::new(needle);
        let smith_waterman =
            SmithWatermanMatcher::<SSEVector, AVXVector, true>::new(needle, &config.scoring);

        let mut matches = vec![];
        for (i, haystack, skipped_chunks) in haystacks
            .iter()
            .enumerate()
            .filter(|(_, h)| h.len() >= min_haystack_len)
            // Prefiltering
            .filter_map(|(i, haystack)| {
                let (matched, skipped_chunks) = config.max_typos.map_or((true, 0), |max_typos| {
                    prefilter.match_haystack(haystack.as_slice(), max_typos)
                });
                // Skip any chunks where we know the needle doesn't match
                matched.then_some((i, haystack.as_slice(), skipped_chunks))
            })
        {
            let haystack_skipped = &haystack[skipped_chunks * 16..];

            // Haystack too large, fallback to greedy matching
            let match_result = if haystack.len() > 512 {
                let score_matrix: BumpVec<'_, AVXVector> = BumpVec::new_in(allocator);
                let score_matrix: BumpVec<'static, AVXVector> =
                    unsafe { std::mem::transmute(score_matrix) };

                match_greedy(needle, haystack_skipped, &config.scoring)
                    .map(|(score, _)| (score, score_matrix))
            }
            // Smith waterman matching
            else {
                let mut score_matrix =
                    IncrementalMatcher::generate_score_matrix(needle, haystack_skipped, allocator);
                smith_waterman
                    .match_haystack(haystack_skipped, config.max_typos, &mut score_matrix)
                    .map(|score| (score, score_matrix))
            };

            if let Some((mut score, score_matrix)) = match_result {
                // Exact match bonus
                let exact = skipped_chunks == 0 && needle == haystack;
                if exact {
                    score += config.scoring.exact_match_bonus;
                }

                matches.push(IncrementalMatch {
                    index: i as u32,
                    score,
                    exact,
                    score_matrix,
                });
            }
        }
        matches
    }

    fn generate_score_matrix(
        needle: &[u8],
        haystack: &[u8],
        allocator: &Bump,
    ) -> BumpVec<'static, AVXVector> {
        let capacity = (needle.len() + 1) * (haystack.len().div_ceil(16) + 1);
        // let vec = unsafe { bumpalo::vec![in allocator; AVXVector::zero(); capacity] };
        let mut score_matrix: BumpVec<'_, AVXVector> =
            BumpVec::with_capacity_in(capacity, allocator);
        unsafe {
            std::ptr::write_bytes(score_matrix.as_mut_ptr(), 0, capacity);
            score_matrix.set_len(capacity);
        }
        unsafe { std::mem::transmute(score_matrix) }
    }

    pub fn reset(&mut self) {
        self.needle = String::new();
        self.matches = vec![];
        self.allocator.reset();
    }
}

impl Drop for IncrementalMatcher {
    fn drop(&mut self) {
        // Clear matches first to drop references before allocator
        self.matches.clear();
    }
}
