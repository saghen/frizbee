use std::collections::HashSet;

pub fn char_indices_from_score_matrix(score_matrix: &[&[u16]]) -> Vec<usize> {
    if score_matrix.is_empty() || score_matrix[0].is_empty() {
        return vec![];
    }

    // Find the maximum score row/col
    let mut max_score_position = (0, 0);
    let mut max_score = 0;
    for (col, col_scores) in score_matrix.iter().enumerate() {
        for (row, score) in col_scores.iter().enumerate() {
            if *score > max_score {
                max_score = *score;
                max_score_position = (col, row);
            }
        }
    }

    // Traceback and store the matched indices
    let mut indices = HashSet::new();
    let (mut col_idx, mut row_idx) = max_score_position;
    let mut score = score_matrix[col_idx][row_idx];

    // NOTE: row_idx = 0 or col_idx = 0 will always have a score of 0
    while score > 0 {
        // Gather up the scores for all possible paths
        let diag = if col_idx == 0 || row_idx == 0 {
            0
        } else {
            score_matrix[col_idx - 1][row_idx - 1]
        };
        let left = if col_idx == 0 {
            0
        } else {
            score_matrix[col_idx - 1][row_idx]
        };
        let up = if row_idx == 0 {
            0
        } else {
            score_matrix[col_idx][row_idx - 1]
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

    let mut indices = indices.iter().copied().collect::<Vec<_>>();
    indices.sort();
    indices
}

#[cfg(test)]
mod tests {
    use super::super::smith_waterman;
    use super::char_indices_from_score_matrix;
    use crate::Scoring;

    fn get_indices(needle: &str, haystack: &str) -> Vec<usize> {
        let (_, score_matrix, _) = smith_waterman(needle, haystack, &Scoring::default());
        let score_matrix_ref = score_matrix
            .iter()
            .map(|v| v.as_slice())
            .collect::<Vec<_>>();
        char_indices_from_score_matrix(&score_matrix_ref)
    }

    #[test]
    fn test_basic_indices() {
        assert_eq!(get_indices("", "abc"), vec![]);
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
