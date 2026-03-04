use itertools::Itertools;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::Matcher;
use crate::{Config, Match};

pub fn match_list_parallel<S1: AsRef<str>, S2: AsRef<str> + Sync>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
    threads: usize,
) -> Vec<Match> {
    assert!(
        haystacks.len() < (u32::MAX as usize),
        "haystack index overflow"
    );

    if needle.as_ref().is_empty() {
        return (0..haystacks.len())
            .map(|index| Match {
                index: index as u32,
                score: 0,
                exact: false,
            })
            .collect();
    }

    if haystacks.is_empty() {
        return vec![];
    }

    // Smaller chunks enable better load balancing via stealing
    // but too small increases atomic contention
    let chunk_size = 512;
    let num_chunks = haystacks.len().div_ceil(chunk_size);
    let next_chunk = AtomicUsize::new(0);

    let needle = needle.as_ref();
    let matcher = Matcher::new(needle, config);

    thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut local_matches = Vec::new();
                    let mut matcher = matcher.clone();

                    loop {
                        // Claim next available chunk
                        let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                        if chunk_idx >= num_chunks {
                            break;
                        }

                        let start = chunk_idx * chunk_size;
                        let end = (start + chunk_size).min(haystacks.len());
                        let haystacks_chunk = &haystacks[start..end];

                        matcher.match_list_into(haystacks_chunk, start as u32, &mut local_matches);
                    }

                    // Each thread sorts so that we can perform k-way merge
                    if config.sort {
                        local_matches.sort_unstable();
                    }

                    local_matches
                })
            })
            .collect();

        if config.sort {
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .kmerge()
                .collect()
        } else {
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        }
    })
}
