pub fn typos_from_score_matrix(score_matrix: &[&[u16]]) -> u16 {
    let mut typo_count = 0;
    let mut score = 0;
    let mut positions = 0;

    // Get the starting position by looking at the last column
    // (last character of the needle)
    let last_column = score_matrix.last().unwrap();
    for (idx, &row_score) in last_column.iter().enumerate() {
        if row_score > score {
            score = row_score;
            positions = idx;
        }
    }

    // Traceback and store the matched indices
    // for (idx, &row_idx) in positions.to_array().iter().enumerate() {
    let mut col_idx = score_matrix.len() - 1;
    let mut row_idx: usize = positions;

    // NOTE: row_idx = 0 or col_idx = 0 will always have a score of 0
    while col_idx > 0 {
        // Must be moving left
        if row_idx == 0 {
            typo_count += 1;
            col_idx -= 1;
            continue;
        }

        // Gather up the scores for all possible paths
        let diag = score_matrix[col_idx - 1][row_idx - 1];
        let left = score_matrix[col_idx - 1][row_idx];
        let up = score_matrix[col_idx][row_idx - 1];

        // Match or mismatch
        if diag >= left && diag >= up {
            // Must be a mismatch
            if diag >= score {
                typo_count += 1;
            }
            row_idx -= 1;
            col_idx -= 1;
            score = diag;
        // Skipped character in needle
        } else if left >= up {
            typo_count += 1;
            col_idx -= 1;
            score = left;
        // Skipped character in haystack
        } else {
            row_idx -= 1;
            score = up;
        }
    }

    // HACK: Compensate for the last column being a typo
    if col_idx == 0 && score == 0 {
        typo_count += 1;
    }

    typo_count
}

#[cfg(test)]
mod tests {
    use super::super::smith_waterman;
    use super::typos_from_score_matrix;

    fn get_typos(needle: &str, haystack: &str) -> u16 {
        let (_, score_matrix) = smith_waterman(needle, haystack);
        let score_matrix_ref = score_matrix
            .iter()
            .map(|v| v.as_slice())
            .collect::<Vec<_>>();
        typos_from_score_matrix(&score_matrix_ref)
    }

    #[test]
    fn test_typos_basic() {
        assert_eq!(get_typos("a", "abc"), 0);
        assert_eq!(get_typos("b", "abc"), 0);
        assert_eq!(get_typos("c", "abc"), 0);
        assert_eq!(get_typos("ac", "abc"), 0);

        assert_eq!(get_typos("d", "abc"), 1);
        assert_eq!(get_typos("da", "abc"), 1);
        assert_eq!(get_typos("dc", "abc"), 1);
        assert_eq!(get_typos("ad", "abc"), 1);
        assert_eq!(get_typos("adc", "abc"), 1);
        assert_eq!(get_typos("add", "abc"), 2);
        assert_eq!(get_typos("ddd", "abc"), 3);
        assert_eq!(get_typos("ddd", ""), 3);
        assert_eq!(get_typos("d", ""), 1);
    }
}
