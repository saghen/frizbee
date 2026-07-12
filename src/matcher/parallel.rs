use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::Matcher;
use crate::k_merge::{
    k_merge_matches_by_index_asc, k_merge_matches_by_index_desc,
    k_merge_matches_by_score_then_index_asc, k_merge_matches_by_score_then_index_desc,
};
use crate::sort::radix_sort_matches;
use crate::{Match, SortStrategy};

impl Matcher {
    /// Matches a list of haystacks in parallel on multiple real threads, returning a list of
    /// [`Match`] values. Threads work on 2048 item chunks, and the final result is ordered
    /// according to [`crate::Config::sort`]. The `threads` must be >0.
    ///
    /// This API provides the most performant path when matching on lists.
    pub fn match_list_parallel<S: AsRef<str> + Sync>(
        &mut self,
        haystacks: &[S],
        threads: usize,
    ) -> Vec<Match> {
        Self::guard_against_haystack_overflow(haystacks.len(), 0);
        assert!(threads > 0, "threads must be positive");

        // Limit threads based on the number of haystacks
        let threads = threads.min(haystacks.len().div_ceil(2000)).max(1);

        if haystacks.is_empty() || self.patterns.is_empty() || threads == 1 {
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
                        if matcher.config.sort.is_reversed() {
                            local_matches.reverse();
                        }
                        if matcher.config.sort.is_by_score() {
                            radix_sort_matches(&mut local_matches);
                        }

                        local_matches
                    })
                })
                .collect();

            let matches = handles.into_iter().map(|h| h.join().unwrap()).collect();
            match matcher.config.sort {
                SortStrategy::ScoreThenIndexAsc => k_merge_matches_by_score_then_index_asc(matches),
                SortStrategy::ScoreThenIndexDesc => {
                    k_merge_matches_by_score_then_index_desc(matches)
                }
                SortStrategy::IndexAsc => k_merge_matches_by_index_asc(matches),
                SortStrategy::IndexDesc => k_merge_matches_by_index_desc(matches),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Config, Matcher};

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

        let config = Config::default();
        let sequential = Matcher::new("abc", &config).match_list(&haystacks);
        assert!(sequential.is_sorted());

        for &threads in thread_counts() {
            let parallel = Matcher::new("abc", &config).match_list_parallel(&haystacks, threads);
            assert_eq!(&parallel, &sequential, "threads={threads}");
            assert!(parallel.is_sorted(), "threads={threads}");
        }
    }

    #[test]
    #[should_panic(expected = "threads must be positive")]
    fn zero_threads_panics() {
        let _ = Matcher::new("a", &Config::default()).match_list_parallel(&["a"], 0);
    }

    #[test]
    fn multi_pattern_matches_sequential_across_chunk_boundaries() {
        use crate::{Matcher, Pattern, SortStrategy};

        let mut haystacks = (0..4101)
            .map(|index| format!("nomatch-{index}"))
            .collect::<Vec<_>>();
        for (index, value) in [
            (0, "abc"),
            (1, "abcxyz"),
            (2047, "xabc"),
            (2048, "abxc"),
            (2049, "alpha/beta/abc"),
            (2050, "xyz/abc"),
            (4095, "ABC"),
            (4096, "a_b_c"),
            (4100, "zabc"),
        ] {
            haystacks[index] = value.to_string();
        }

        for query in ["abc !xyz", "abc a", "!abc !xyz"] {
            for sort in [SortStrategy::ScoreThenIndexAsc, SortStrategy::IndexAsc] {
                let config = Config::default().sort(sort);
                let mut matcher = Matcher::from_patterns(&Pattern::parse_query(query), &config);
                let sequential = matcher.match_list(&haystacks);

                for &threads in thread_counts() {
                    let parallel = matcher.match_list_parallel(&haystacks, threads);
                    assert_eq!(
                        &parallel, &sequential,
                        "query={query:?}, sort={sort:?}, threads={threads}"
                    );
                }
            }
        }
    }
}
