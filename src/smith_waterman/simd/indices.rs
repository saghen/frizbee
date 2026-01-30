use std::collections::HashSet;
use std::simd::cmp::*;
use std::simd::{Select, Simd};

#[inline]
pub fn char_indices_from_score_matrix<const W: usize, const L: usize>(
    score_matrices: &[[Simd<u16, L>; W]],
) -> Vec<Vec<usize>> {
    // Find the maximum score row/col for each haystack
    let mut max_scores = Simd::splat(0);
    let mut max_rows = Simd::splat(0);
    let mut max_cols = Simd::splat(0);

    for (col, col_scores) in score_matrices.iter().enumerate() {
        for (row, row_scores) in col_scores.iter().enumerate() {
            let scores_mask = row_scores.simd_ge(max_scores);

            max_rows = scores_mask.select(Simd::splat(row as u16), max_rows);
            max_cols = scores_mask.select(Simd::splat(col as u16), max_cols);

            max_scores = max_scores.simd_max(*row_scores);
        }
    }

    let max_score_positions = max_rows.to_array().into_iter().zip(max_cols.to_array());

    // Traceback and store the matched indices
    let mut indices = vec![HashSet::new(); L];

    for (idx, (row_idx, col_idx)) in max_score_positions.enumerate() {
        let indices = &mut indices[idx];

        let mut row_idx: usize = row_idx.into();
        let mut col_idx: usize = col_idx.into();
        let mut score = score_matrices[col_idx][row_idx][idx];

        // NOTE: row_idx = 0 or col_idx = 0 will always have a score of 0
        while score > 0 {
            // Gather up the scores for all possible paths
            let diag = if col_idx == 0 || row_idx == 0 {
                0
            } else {
                score_matrices[col_idx - 1][row_idx - 1][idx]
            };
            let left = if col_idx == 0 {
                0
            } else {
                score_matrices[col_idx - 1][row_idx][idx]
            };
            let up = if row_idx == 0 {
                0
            } else {
                score_matrices[col_idx][row_idx - 1][idx]
            };

            // Diagonal (match/mismatch)
            if diag >= left && diag >= up {
                // Check if the score decreases (remember we're going backwards)
                // to see if we've found a match
                if diag < score {
                    indices.insert(row_idx);
                }

                row_idx = row_idx.saturating_sub(1);
                col_idx = col_idx.saturating_sub(1);

                score = diag;
            }
            // Up (gap in haystack)
            else if up >= left {
                // Finished crossing a gap, remove any previous rows
                if up > score && up > 0 {
                    indices.remove(&(row_idx));
                    indices.insert(row_idx.saturating_sub(1));
                }

                row_idx = row_idx.saturating_sub(1);

                score = up;
            }
            // Left (gap in needle)
            else {
                col_idx = col_idx.saturating_sub(1);
                score = left;
            }
        }
    }

    indices
        .iter()
        .map(|indices| {
            let mut indices = indices.iter().copied().collect::<Vec<_>>();
            indices.sort();
            indices
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{Scoring, smith_waterman::simd::smith_waterman};

    use super::*;

    fn get_indices(needle: &str, haystack: &str) -> Vec<usize> {
        let haystacks = [haystack; 1];
        let (_, score_matrices, _) =
            smith_waterman::<16, 1>(needle, &haystacks, None, &Scoring::default());
        let indices = char_indices_from_score_matrix(&score_matrices);
        indices[0].clone()
    }

    #[test]
    fn test_leaking() {
        let needle = "t";
        let haystacks = [
            "true",
            "toDate",
            "toString",
            "transpose",
            "testing",
            "to",
            "toRgba",
            "toolbar",
            "true",
            "toDate",
            "toString",
            "transpose",
            "testing",
            "to",
            "toRgba",
            "toolbar",
        ];

        let (_, score_matrices, _) =
            smith_waterman::<16, 16>(needle, &haystacks, None, &Scoring::default());
        let indices = char_indices_from_score_matrix(&score_matrices);
        for indices in indices.into_iter() {
            assert_eq!(indices, [0])
        }
    }

    #[test]
    fn test_basic_indices() {
        assert_eq!(get_indices("b", "abc"), vec![1]);
        assert_eq!(get_indices("c", "abc"), vec![2]);
    }

    #[test]
    fn test_prefix_indices() {
        assert_eq!(get_indices("a", "abc"), vec![0]);
        assert_eq!(get_indices("a", "aabc"), vec![0]);
        assert_eq!(get_indices("a", "babc"), vec![1]);
    }

    #[test]
    fn test_exact_match_indices() {
        assert_eq!(get_indices("a", "a"), vec![0]);
        assert_eq!(get_indices("abc", "abc"), vec![0, 1, 2]);
        assert_eq!(get_indices("ab", "abc"), vec![0, 1]);
    }

    #[test]
    fn test_delimiter_indices() {
        assert_eq!(get_indices("b", "a-b"), vec![2]);
        assert_eq!(get_indices("a", "a-b-c"), vec![0]);
        assert_eq!(get_indices("b", "a--b"), vec![3]);
        assert_eq!(get_indices("c", "a--bc"), vec![4]);
    }

    #[test]
    fn test_affine_gap_indices() {
        assert_eq!(get_indices("test", "Uterst"), vec![1, 2, 4, 5]);
        assert_eq!(get_indices("test", "Uterrst"), vec![1, 2, 5, 6]);
        assert_eq!(get_indices("test", "Uterrs t"), vec![1, 2, 5, 7]);
    }

    #[test]
    fn test_capital_indices() {
        assert_eq!(get_indices("a", "A"), vec![0]);
        assert_eq!(get_indices("A", "Aa"), vec![0]);
        assert_eq!(get_indices("D", "forDist"), vec![3]);
    }

    #[test]
    fn test_typo_indices() {
        assert_eq!(get_indices("b", "a"), vec![]);
        assert_eq!(get_indices("reba", "repack"), vec![0, 1, 3]);
        assert_eq!(get_indices("bbb", "abc"), vec![1]);
    }
}
