//! Merges multiple pre-sorted runs of [`crate::Match`]es into a single sorted [`Vec`]
//! using the k-way merge algorithm specialized for [`crate::Match`]es.

use crate::Match;

/// Ordering policy for merging. Implementations must define a total order
/// consistent with the pre-sorted runs.
pub trait MergeOrder {
    /// Returns true if `left` should be emitted before `right`.
    fn less(left: &Match, right: &Match) -> bool;
}

/// Order by score (desc), tie-broken by index (asc)
pub struct ByScoreThenIndexAsc;

impl MergeOrder for ByScoreThenIndexAsc {
    #[inline(always)]
    fn less(left: &Match, right: &Match) -> bool {
        left.score > right.score || (left.score == right.score && left.index < right.index)
    }
}

/// Order by score (desc), tie-broken by index (desc)
pub struct ByScoreThenIndexDesc;

impl MergeOrder for ByScoreThenIndexDesc {
    #[inline(always)]
    fn less(left: &Match, right: &Match) -> bool {
        left.score > right.score || (left.score == right.score && left.index > right.index)
    }
}

/// Order by index (asc)
pub struct ByIndexAsc;

impl MergeOrder for ByIndexAsc {
    #[inline(always)]
    fn less(left: &Match, right: &Match) -> bool {
        left.index < right.index
    }
}

/// Order by index (desc)
pub struct ByIndexDesc;

impl MergeOrder for ByIndexDesc {
    #[inline(always)]
    fn less(left: &Match, right: &Match) -> bool {
        left.index > right.index
    }
}

/// Merge by score (desc), tie-broken by index (asc)
///
/// See [`k_merge_matches_by`] for docs.
pub fn k_merge_matches_by_score_then_index_asc(runs: Vec<Vec<Match>>) -> Vec<Match> {
    k_merge_matches_by::<ByScoreThenIndexAsc>(runs)
}

/// Merge by score (desc), tie-broken by index (desc)
///
/// See [`k_merge_matches_by`] for docs.
pub fn k_merge_matches_by_score_then_index_desc(runs: Vec<Vec<Match>>) -> Vec<Match> {
    k_merge_matches_by::<ByScoreThenIndexDesc>(runs)
}

/// Merge by index (asc)
///
/// See [`k_merge_matches_by`] for docs.
pub fn k_merge_matches_by_index_asc(runs: Vec<Vec<Match>>) -> Vec<Match> {
    k_merge_matches_by::<ByIndexAsc>(runs)
}

/// Merge by index (desc)
///
/// See [`k_merge_matches_by`] for docs.
pub fn k_merge_matches_by_index_desc(runs: Vec<Vec<Match>>) -> Vec<Match> {
    k_merge_matches_by::<ByIndexDesc>(runs)
}

/// Merges multiple pre-sorted runs of `Match`es into a single sorted `Vec`,
/// using the ordering policy `O`.
///
/// The input runs must already be sorted according to the same order `O`.
///
/// [`k_merge_matches_by_score_then_index_asc`] to sort by score (desc), tie-broken
/// by index (asc)
/// [`k_merge_matches_by_index_asc`] to sort by index
/// [`k_merge_matches_by_score_then_index_desc`] to sort by score (desc)
pub fn k_merge_matches_by<O: MergeOrder>(runs: Vec<Vec<Match>>) -> Vec<Match> {
    let total_matches = runs.iter().map(Vec::len).sum();

    let mut merged = Vec::with_capacity(total_matches);
    let mut heap = Vec::with_capacity(runs.len());

    // One cursor per non-empty sorted run; heap root is the next globally best match.
    for (run_idx, run) in runs.iter().enumerate() {
        if let Some(&head) = run.first() {
            heap.push(MergeCursor {
                run_idx,
                match_idx: 0,
                head,
            });
        }
    }

    heapify_merge_cursors::<O>(&mut heap);

    while heap.len() > 1 {
        let run_idx = heap[0].run_idx;
        let next_match_idx = heap[0].match_idx + 1;
        merged.push(heap[0].head);

        // Advance the winning run in place; only remove it when it is exhausted.
        if let Some(&head) = runs[run_idx].get(next_match_idx) {
            heap[0].match_idx = next_match_idx;
            heap[0].head = head;
        } else {
            heap.swap_remove(0);
        }

        sift_down_merge_cursor::<O>(&mut heap, 0);
    }

    if let Some(cursor) = heap.pop() {
        // Once one run remains, it is already sorted after the emitted prefix.
        merged.extend_from_slice(&runs[cursor.run_idx][cursor.match_idx..]);
    }

    merged
}

#[inline(always)]
fn heapify_merge_cursors<O: MergeOrder>(heap: &mut [MergeCursor]) {
    for idx in (0..heap.len() / 2).rev() {
        sift_down_merge_cursor::<O>(heap, idx);
    }
}

#[inline(always)]
fn sift_down_merge_cursor<O: MergeOrder>(heap: &mut [MergeCursor], index: usize) {
    let mut pos = index;
    let mut child = 2 * pos + 1;

    while child + 1 < heap.len() {
        child += merge_cursor_less::<O>(&heap[child + 1], &heap[child]) as usize;
        if !merge_cursor_less::<O>(&heap[child], &heap[pos]) {
            return;
        }

        heap.swap(pos, child);
        pos = child;
        child = 2 * pos + 1;
    }

    if child < heap.len() && merge_cursor_less::<O>(&heap[child], &heap[pos]) {
        heap.swap(pos, child);
    }
}

#[inline(always)]
fn merge_cursor_less<O: MergeOrder>(left: &MergeCursor, right: &MergeCursor) -> bool {
    O::less(&left.head, &right.head)
}

#[derive(Clone, Copy)]
struct MergeCursor {
    run_idx: usize,
    match_idx: usize,
    head: Match,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mtch(score: u16, index: u32) -> Match {
        Match {
            score,
            index,
            exact: false,
            #[cfg(feature = "match_end_col")]
            end_col: 0,
        }
    }

    #[test]
    fn merges_two_sorted_match_runs() {
        let runs = vec![
            vec![mtch(100, 1), mtch(80, 3), mtch(20, 4)],
            vec![mtch(100, 0), mtch(90, 2), mtch(80, 5)],
        ];

        let merged = k_merge_matches_by_score_then_index_asc(runs);

        assert_eq!(
            merged,
            vec![
                mtch(100, 0),
                mtch(100, 1),
                mtch(90, 2),
                mtch(80, 3),
                mtch(80, 5),
                mtch(20, 4),
            ]
        );
        assert!(merged.is_sorted());
    }

    #[test]
    fn heap_merge_skips_empty_runs() {
        let runs = vec![
            vec![mtch(90, 2)],
            Vec::new(),
            vec![mtch(100, 0), mtch(80, 4)],
            Vec::new(),
            vec![mtch(95, 1), mtch(80, 3)],
        ];

        let merged = k_merge_matches_by_score_then_index_asc(runs);

        assert_eq!(
            merged,
            vec![
                mtch(100, 0),
                mtch(95, 1),
                mtch(90, 2),
                mtch(80, 3),
                mtch(80, 4),
            ]
        );
        assert!(merged.is_sorted());
    }

    #[test]
    fn heap_merge_handles_many_runs() {
        let runs = (0..=16)
            .map(|run_idx| vec![mtch(100 - run_idx as u16, run_idx as u32)])
            .rev()
            .collect::<Vec<_>>();

        let merged = k_merge_matches_by_score_then_index_asc(runs);

        assert_eq!(
            merged,
            (0..=16)
                .map(|idx| mtch(100 - idx as u16, idx as u32))
                .collect::<Vec<_>>()
        );
        assert!(merged.is_sorted());
    }

    #[test]
    fn merges_by_index_only() {
        let runs = vec![
            vec![mtch(100, 1), mtch(80, 3), mtch(20, 5)],
            vec![mtch(100, 0), mtch(90, 2), mtch(80, 4)],
        ];

        let merged = k_merge_matches_by::<ByIndexAsc>(runs);

        assert_eq!(
            merged.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
    }
}
