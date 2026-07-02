use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::Matcher;
use crate::k_merge::k_merge_matches;
use crate::sort::radix_sort_matches;
use crate::{Config, Match, match_list};

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
    assert!(threads > 0, "threads must be positive");
    if threads == 1 {
        return match_list(needle, haystacks, config);
    }

    if needle.as_ref().is_empty() {
        return (0..haystacks.len())
            .map(|index| Match {
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
    let chunk_size = 2048;
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
            k_merge_matches(
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>(),
            )
        } else {
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thread_counts() -> &'static [usize] {
        if cfg!(miri) {
            &[1, 2, 8]
        } else {
            &[1, 2, 3, 4, 5, 6, 7, 8]
        }
    }

    #[test]
    fn sorted_matches_sequential_across_chunk_boundaries() {
        let mut haystacks = (0..4101)
            .map(|index| format!("nomatch-{index}"))
            .collect::<Vec<_>>();
        for (index, value) in [
            (0, "abc"),
            (2047, "xabc"),
            (2048, "abxc"),
            (2049, "alpha/beta/abc"),
            (4095, "ABC"),
            (4096, "a_b_c"),
            (4100, "zabc"),
        ] {
            haystacks[index] = value.to_string();
        }

        let config = Config {
            sort: true,
            ..Config::default()
        };
        let sequential = match_list("abc", &haystacks, &config);
        assert!(sequential.is_sorted());

        for &threads in thread_counts() {
            let parallel = match_list_parallel("abc", &haystacks, &config, threads);
            assert_eq!(&parallel, &sequential, "threads={threads}");
            assert!(parallel.is_sorted(), "threads={threads}");
        }
    }

    #[test]
    #[should_panic(expected = "threads must be positive")]
    fn zero_threads_panics() {
        let _ = match_list_parallel("a", &["a"], &Config::default(), 0);
    }
}
