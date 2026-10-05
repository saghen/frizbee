use crate::{Match, MatchIndices};
use alloc::{vec, vec::Vec};

/// Sorts a [`Vec`] of [`Match`] values in-place by descending `score` using a
/// stable radix sort. This assumes that the matches are already sorted by
/// index.
#[inline]
pub fn radix_sort_matches(matches: &mut Vec<Match>) {
    let (matches_b, high_radices) = radix_sort_first_pass(matches);

    if high_radices == 0 {
        *matches = matches_b;
        return;
    }

    radix_sort_second_pass(matches_b, matches);
}

#[inline(always)]
fn radix_sort_first_pass(matches: &[Match]) -> (Vec<Match>, u8) {
    // pass 1
    let mut histogram = [0u32; 256];
    let mut high_radices = 0u8;
    for m in matches.iter() {
        let radix = m.score & 0xFF;
        histogram[radix as usize] += 1;
        high_radices |= (m.score >> 8) as u8;
    }
    let mut offsets = [0u32; 256];
    for idx in (1..256).rev() {
        offsets[idx - 1] = offsets[idx] + histogram[idx];
    }

    let mut matches_b = vec![Match::default(); matches.len()];
    for m in matches.iter() {
        let radix = m.score & 0xFF;
        let output_idx = offsets[radix as usize] as usize;
        matches_b[output_idx] = *m;
        offsets[radix as usize] += 1;
    }

    (matches_b, high_radices)
}

#[inline(always)]
fn radix_sort_second_pass(matches_b: Vec<Match>, matches: &mut [Match]) {
    // pass 2
    let mut histogram = [0u32; 256];
    for m in matches_b.iter() {
        let radix = (m.score >> 8) & 0xFF;
        histogram[radix as usize] += 1;
    }
    let mut offsets = [0u32; 256];
    for idx in (1..256).rev() {
        offsets[idx - 1] = offsets[idx] + histogram[idx];
    }
    for m in matches_b {
        let radix = (m.score >> 8) & 0xFF;
        matches[offsets[radix as usize] as usize] = m;
        offsets[radix as usize] += 1;
    }
}

/// Stable sort of [`MatchIndices`] by descending score, through
/// [`radix_sort_matches`] (std's stable sort would add a copy of driftsort
/// to wasm builds)
pub(crate) fn sort_indices_by_score(matches: &mut Vec<MatchIndices>) {
    let mut order: Vec<Match> = matches
        .iter()
        .enumerate()
        .map(|(i, m)| Match {
            score: m.score,
            index: i as u32,
            exact: false,
            #[cfg(feature = "match_end_col")]
            end_col: 0,
        })
        .collect();
    radix_sort_matches(&mut order);
    let mut unsorted = core::mem::take(matches);
    matches.extend(
        order.iter().map(|m| {
            core::mem::replace(&mut unsorted[m.index as usize], MatchIndices::from_index(0))
        }),
    );
}

/// Sorts `indices` descending with an insertion sort, as the indices of a
/// match are few
pub(crate) fn sort_indices_desc(indices: &mut [u32]) {
    for i in 1..indices.len() {
        let index = indices[i];
        let mut j = i;
        while j > 0 && indices[j - 1] < index {
            indices[j] = indices[j - 1];
            j -= 1;
        }
        indices[j] = index;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn test_sorted() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        // Miri tests are too slow for a full sized test, bound to 4096
        let count = if cfg!(miri) { 1 << 12 } else { 1 << 24 };
        let mut matches = (0u32..count)
            .map(|index| Match {
                score: rng.random::<u16>(),
                index,
                exact: rng.random_bool(0.5),
                #[cfg(feature = "match_end_col")]
                end_col: 0,
            })
            .collect::<Vec<_>>();

        radix_sort_matches(&mut matches);

        assert!(matches.is_sorted());
    }
}
