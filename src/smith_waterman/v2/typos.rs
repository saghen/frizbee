use std::arch::x86_64::*;

use super::ops::argmax_epu16;

pub unsafe fn typos_from_score_matrix(score_matrix: &[__m256i]) -> u16 {
    let mut typo_count = 0;

    let mut row_idx = score_matrix.len() - 1;
    let mut col_idx = argmax_epu16(score_matrix[row_idx]);

    let score_matrix = std::mem::transmute::<_, &[std::simd::Simd<u16, 16>]>(score_matrix);
    let mut score = score_matrix[row_idx][col_idx];

    while col_idx > 0 {
        // Must be moving left
        if row_idx == 0 {
            typo_count += 1;
            col_idx -= 1;
            continue;
        }

        // Gather up the scores for all possible paths
        let diag = score_matrix[row_idx - 1][col_idx - 1];
        let left = score_matrix[row_idx][col_idx];
        let up = score_matrix[row_idx - 1][col_idx];

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

    typo_count
}
