use itertools::Itertools;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::Matcher;
use crate::sort::radix_sort_matches;
use crate::{Config, Match, Matchable, MatchableChunked};

pub fn match_list_parallel<S1: AsRef<str>, S2: Matchable + Sync>(
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
        return haystacks
            .iter()
            .enumerate()
            .filter(|(_, item)| item.match_str().is_some())
            .map(|(index, _)| Match {
                index: index as u32,
                score: 0,
                exact: false,
                #[cfg(feature = "match_end_col")]
                end_col: 0,
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
                        radix_sort_matches(&mut local_matches);
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

/// Parallel chunked matching with a resolver callback.
///
/// For each item, `resolve` fills a stack buffer with chunk pointers and returns
/// `Some((chunk_count, byte_len))` or `None` to skip the item.
pub fn match_list_parallel_resolved<
    S1: AsRef<str>,
    T: Sync,
    F: Fn(&T, &mut [*const u8; 32]) -> Option<(usize, u16)> + Sync,
>(
    needle: S1,
    items: &[T],
    resolve: &F,
    config: &Config,
    threads: usize,
) -> Vec<Match> {
    assert!(items.len() < (u32::MAX as usize), "item index overflow");

    if needle.as_ref().is_empty() {
        let mut ptrs_buf = [core::ptr::null::<u8>(); 32];
        return items
            .iter()
            .enumerate()
            .filter(|(_, item)| resolve(item, &mut ptrs_buf).is_some())
            .map(|(index, _)| Match {
                index: index as u32,
                score: 0,
                exact: false,
                #[cfg(feature = "match_end_col")]
                end_col: 0,
            })
            .collect();
    }

    if items.is_empty() {
        return vec![];
    }

    let chunk_size = 512;
    let num_chunks = items.len().div_ceil(chunk_size);
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
                        let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                        if chunk_idx >= num_chunks {
                            break;
                        }

                        let start = chunk_idx * chunk_size;
                        let end = (start + chunk_size).min(items.len());
                        let items_chunk = &items[start..end];

                        matcher.match_list_resolved_into(
                            items_chunk,
                            start as u32,
                            resolve,
                            &mut local_matches,
                        );
                    }

                    if config.sort {
                        radix_sort_matches(&mut local_matches);
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

pub fn match_list_parallel_chunked<S1: AsRef<str>, C: MatchableChunked + Sync>(
    needle: S1,
    haystacks: &[C],
    ctx: &C::Ctx,
    config: &Config,
    threads: usize,
) -> Vec<Match>
where
    C::Ctx: Sync,
{
    assert!(
        haystacks.len() < (u32::MAX as usize),
        "haystack index overflow"
    );

    if needle.as_ref().is_empty() {
        return haystacks
            .iter()
            .enumerate()
            .filter(|(_, item)| item.haystack_info(ctx).is_some())
            .map(|(index, _)| Match {
                index: index as u32,
                score: 0,
                exact: false,
                #[cfg(feature = "match_end_col")]
                end_col: 0,
            })
            .collect();
    }

    if haystacks.is_empty() {
        return vec![];
    }

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
                        let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                        if chunk_idx >= num_chunks {
                            break;
                        }

                        let start = chunk_idx * chunk_size;
                        let end = (start + chunk_size).min(haystacks.len());
                        let haystacks_chunk = &haystacks[start..end];

                        matcher.match_list_chunked_into(
                            haystacks_chunk,
                            ctx,
                            start as u32,
                            &mut local_matches,
                        );
                    }

                    if config.sort {
                        radix_sort_matches(&mut local_matches);
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
