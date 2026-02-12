use std::sync::Arc;

use super::match_list;
use crate::one_shot::matcher::match_list_impl;
use crate::{Config, Match};

mod thread_slice;
mod threaded_vec;

use thread_slice::ThreadSlice;
use threaded_vec::ThreadedVec;

/// Computes the Smith-Waterman score with affine gaps for a needle against a list of haystacks
/// with multithreading.
///
/// You should call this function with as many haystacks as you have available as it will
/// automatically chunk the haystacks based on string length to avoid unnecessary computation
/// due to SIMD
pub fn match_list_parallel<S1: AsRef<str>, S2: AsRef<str> + Sync + Send>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
    max_threads: usize,
) -> Vec<Match> {
    let thread_count = choose_thread_count(haystacks.len(), config.max_typos).clamp(1, max_threads);
    if thread_count == 1 {
        return match_list(needle, haystacks, config);
    }

    let mut matches = match config.max_typos {
        None => match_list_parallel_fixed(needle, haystacks, config, thread_count),
        _ => match_list_parallel_expandable(needle, haystacks, config, thread_count),
    };

    if config.sort {
        #[cfg(feature = "parallel_sort")]
        {
            use rayon::prelude::*;
            matches.par_sort();
        }
        #[cfg(not(feature = "parallel_sort"))]
        matches.sort_unstable();
    }

    matches
}

/// Since max_typos is None, we may use an unitialized vector to store the matches and provide a slice
/// to each thread based on the number of items it will process, since all items will be returned
fn match_list_parallel_fixed<S1: AsRef<str>, S2: AsRef<str> + Sync + Send>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
    thread_count: usize,
) -> Vec<Match> {
    assert!(config.max_typos.is_none(), "max_typos must be None");

    let mut matches = Vec::with_capacity(haystacks.len());
    #[allow(clippy::uninit_vec)]
    unsafe {
        matches.set_len(haystacks.len())
    };
    let mut matches_remaining_slice = matches.as_mut_slice();

    let items_per_thread = haystacks.len().div_ceil(thread_count);
    std::thread::scope(|s| {
        for (thread_idx, haystacks) in haystacks.chunks(items_per_thread).enumerate() {
            debug_assert!(thread_idx < thread_count, "thread index out of bounds");

            let (matches_slice, remaining_slice) =
                matches_remaining_slice.split_at_mut(haystacks.len());
            matches_remaining_slice = remaining_slice;

            let needle = needle.as_ref().to_owned();
            let mut thread_slice = ThreadSlice::new(matches_slice);
            let opts = *config;
            s.spawn(move || {
                match_list_impl(
                    needle,
                    haystacks,
                    (thread_idx * items_per_thread) as u32,
                    &opts,
                    &mut thread_slice,
                )
            });

            // TODO: assert that thread_slice.pos == haystaks.len()
        }
    });

    matches
}

/// Since max_typos is Some, we'll receive an unknown number of matches, so we use an thread safe
/// batched expandable vec to store the matches. In the typical case (<20% matching), there
/// shouldn't be a bottleneck when adding items to the vector.
fn match_list_parallel_expandable<S1: AsRef<str>, S2: AsRef<str> + Sync + Send>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
    thread_count: usize,
) -> Vec<Match> {
    assert!(config.max_typos.is_some(), "max_typos must be Some");

    let batch_size = 1024;
    let matches = Arc::new(ThreadedVec::new(batch_size, thread_count));

    let items_per_thread = haystacks.len().div_ceil(thread_count);
    std::thread::scope(|s| {
        for (thread_idx, haystacks) in haystacks.chunks(items_per_thread).enumerate() {
            assert!(thread_idx < thread_count, "thread index out of bounds");

            let needle = needle.as_ref().to_owned();
            let mut matches = matches.clone();
            let opts = *config;
            s.spawn(move || {
                match_list_impl(
                    needle,
                    haystacks,
                    (thread_idx * items_per_thread) as u32,
                    &opts,
                    &mut matches,
                )
            });
        }
    });

    Arc::try_unwrap(matches).unwrap().into_vec()
}

fn choose_thread_count(haystacks_len: usize, max_typos: Option<u16>) -> usize {
    // TODO: ideally, we'd change this based on the average length of items in the haystack and the
    // length of the needle. Perhaps random sampling would work well?
    let min_items_per_thread = match max_typos {
        Some(0) => 5000,
        // Slower prefilter
        Some(1) => 4000,
        Some(_) => 3000,
        None => 2500,
    };

    haystacks_len / min_items_per_thread
}
