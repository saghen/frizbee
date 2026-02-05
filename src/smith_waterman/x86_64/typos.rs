use std::arch::x86_64::*;

use super::ops::_mm256_idx_epu16;

pub fn typos_from_score_matrix(
    score_matrix: &[__m256i],
    max_score: u16,
    max_typos: u16,
    haystack_len: usize,
) -> u16 {
    let haystack_chunks = haystack_len.div_ceil(16) + 1;
    let mut row_idx = score_matrix.len() / haystack_chunks - 1;
    let mut col_idx = (1..haystack_chunks)
        .find_map(|chunk_idx| {
            let index = unsafe {
                _mm256_idx_epu16(
                    score_matrix[row_idx * haystack_chunks + chunk_idx],
                    max_score,
                )
            };
            (index != 8).then(|| chunk_idx * 16 + index)
        })
        .expect("could not find max score in score matrix final row");

    let score_matrix = unsafe { std::mem::transmute::<&[__m256i], &[[u16; 16]]>(score_matrix) };

    let mut typo_count = 0;
    let mut score = max_score;

    while row_idx > 0 {
        if typo_count >= max_typos {
            return typo_count;
        }

        // Must be moving up
        if col_idx == 0 {
            return typo_count + row_idx as u16;
        }

        // Gather up the scores for all possible paths
        let diag =
            score_matrix[(row_idx - 1) * haystack_chunks + (col_idx - 1) / 16][(col_idx - 1) % 16];
        let left = score_matrix[row_idx * haystack_chunks + (col_idx - 1) / 16][(col_idx - 1) % 16];
        let up = score_matrix[(row_idx - 1) * haystack_chunks + col_idx / 16][col_idx % 16];

        // Match or mismatch
        if diag >= left && diag >= up {
            // Must be a mismatch
            if diag >= score {
                typo_count += 1;
            }
            row_idx -= 1;
            col_idx -= 1;
            score = diag;
        // Skipped character in haystack
        } else if left >= up {
            col_idx -= 1;
            score = left;
        // Skipped character in needle
        } else {
            typo_count += 1;
            row_idx -= 1;
            score = up;
        }
    }

    typo_count
}
