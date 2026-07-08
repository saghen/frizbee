use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::Matcher;
use crate::Match;
use crate::k_merge::k_merge_matches;
use crate::sort::radix_sort_matches;

impl Matcher {
    /// Matches a list of haystacks in parallel on multiple real threads, returning a list of
    /// [`Match`] values. Threads work on 2048 item chunks, which are sorted and merged into a
    /// single sorted `Vec` at the end. The `threads` must be >0.
    ///
    /// This API provides the most performant path when matching on lists.
    pub fn match_list_parallel<S: AsRef<str> + Sync>(
        &mut self,
        haystacks: &[S],
        threads: usize,
    ) -> Vec<Match> {
        Self::guard_against_haystack_overflow(haystacks.len(), 0);
        assert!(threads > 0, "threads must be positive");

        if haystacks.is_empty() || self.needle.is_empty() || threads == 1 {
            return self.match_list(haystacks);
        }

        // Smaller chunks enable better load balancing via stealing
        // but too small increases atomic contention
        let chunk_size = 2048;
        let num_chunks = haystacks.len().div_ceil(chunk_size);
        let next_chunk = AtomicUsize::new(0);

        let matcher = &*self;

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

                            matcher.match_list_into(
                                haystacks_chunk,
                                start as u32,
                                &mut local_matches,
                            );
                        }

                        // Each thread sorts so that we can perform k-way merge
                        if matcher.config.sort {
                            radix_sort_matches(&mut local_matches);
                        }

                        local_matches
                    })
                })
                .collect();

            if matcher.config.sort == SortStrategy::Score {
                k_merge_matches(handles.into_iter().map(|h| h.join().unwrap()).collect())
            } else {
                k_merge_matches_by_index(handles.into_iter().map(|h| h.join().unwrap()).collect())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Config, match_list, match_list_parallel};

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

        let config = Config::default().sort(true);
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
