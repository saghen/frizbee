use super::matrix::Matrix;
use crate::simd::Vector256;

pub enum Alignment {
    Left((usize, usize)),
    Up((usize, usize)),
    Match((usize, usize)),
    Mismatch((usize, usize)),
}

impl Alignment {
    pub fn pos(&self) -> (usize, usize) {
        match self {
            Alignment::Left(pos) | Alignment::Up(pos) => *pos,
            Alignment::Match(pos) | Alignment::Mismatch(pos) => *pos,
        }
    }

    pub fn col(&self) -> usize {
        match self {
            Alignment::Left((_, col)) | Alignment::Up((_, col)) => *col,
            Alignment::Match((_, col)) | Alignment::Mismatch((_, col)) => *col,
        }
    }

    pub fn row(&self) -> usize {
        match self {
            Alignment::Left((row, _)) | Alignment::Up((row, _)) => *row,
            Alignment::Match((row, _)) | Alignment::Mismatch((row, _)) => *row,
        }
    }
}

/// Iterator over alignment path positions with support for max typos.
///
/// Yields `Some((needle_idx, haystack_idx))` for each position in the path,
/// or `None` to signal that max_typos was exceeded.
pub struct AlignmentPathIter<'a> {
    score_matrix: &'a [[u16; 16]],
    match_masks: &'a [[u16; 16]],
    haystack_chunks: usize,
    row_idx: usize,
    col_idx: usize,
    skipped_chunks: usize,
    max_typos: Option<u16>,
    typo_count: u16,
    score: u16,
    finished: bool,
}

impl<'a> AlignmentPathIter<'a> {
    #[inline(always)]
    pub fn new<Simd256: Vector256>(
        score_matrix: &'a Matrix<Simd256>,
        match_masks: &'a Matrix<Simd256>,
        needle_len: usize,
        skipped_chunks: usize,
        score: u16,
        max_typos: Option<u16>,
    ) -> Self {
        let col_idx = Self::get_col_idx(
            score_matrix,
            needle_len,
            score_matrix.haystack_chunks,
            score,
        );

        Self {
            score_matrix: score_matrix.as_slice(),
            match_masks: match_masks.as_slice(),
            haystack_chunks: score_matrix.haystack_chunks,
            row_idx: needle_len,
            col_idx,
            skipped_chunks,
            max_typos,
            typo_count: 0,
            score,
            finished: false,
        }
    }

    #[inline(always)]
    fn get_col_idx<Simd256: Vector256>(
        score_matrix: &Matrix<Simd256>,
        needle_len: usize,
        haystack_chunks: usize,
        score: u16,
    ) -> usize {
        for chunk_idx in 1..haystack_chunks {
            let chunk = &score_matrix.get(needle_len, chunk_idx);
            let idx = unsafe { chunk.idx_u16(score) };
            if idx != 16 {
                return chunk_idx * 16 + idx;
            }
        }
        panic!("could not find max score in score matrix final row");
    }

    #[inline(always)]
    fn get_score(&self, row: usize, col: usize) -> u16 {
        self.score_matrix[row * self.haystack_chunks + col / 16][col % 16]
    }

    #[inline(always)]
    fn get_is_match(&self, row: usize, col: usize) -> bool {
        self.match_masks[row * self.haystack_chunks + col / 16][col % 16] != 0
    }
}

impl<'a> Iterator for AlignmentPathIter<'a> {
    type Item = Option<Alignment>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.row_idx == 0 || self.finished {
            return None;
        }

        if let Some(max_typos) = self.max_typos {
            if self.typo_count > max_typos {
                self.finished = true;
                return Some(None);
            }
        }

        // Must be moving up only (at left edge), or lost alignment
        if self.col_idx < 16 || self.score == 0 {
            if let Some(max_typos) = self.max_typos {
                if (self.typo_count + self.row_idx as u16) > max_typos {
                    self.finished = true;
                    return Some(None);
                }
            }
            return None;
        }

        // Capture current position to yield (adjusted to 0-indexed)
        let current_pos = (
            self.row_idx - 1,
            self.col_idx - 16 + self.skipped_chunks * 16,
        );

        if self.get_is_match(self.row_idx, self.col_idx) {
            self.row_idx -= 1;
            self.col_idx -= 1;
            self.score = self.get_score(self.row_idx, self.col_idx);
            return Some(Some(Alignment::Match(current_pos)));
        }

        // Gather scores for all possible paths
        let diag = self.get_score(self.row_idx - 1, self.col_idx - 1);
        let left = self.get_score(self.row_idx, self.col_idx - 1);
        let up = self.get_score(self.row_idx - 1, self.col_idx);

        // Match or mismatch (diagonal)
        if diag >= left && diag >= up {
            self.row_idx -= 1;
            self.col_idx -= 1;
            // Must be a mismatch if score didn't increase
            if diag >= self.score {
                self.typo_count += 1;
                self.score = diag;
                return Some(Some(Alignment::Mismatch(current_pos)));
            }
            self.score = diag;
            Some(Some(Alignment::Match(current_pos)))
        // Skipped character in haystack (left)
        } else if left >= up {
            self.col_idx -= 1;
            self.score = left;
            Some(Some(Alignment::Left(current_pos)))
        // Skipped character in needle (up)
        } else {
            self.typo_count += 1;
            self.row_idx -= 1;
            self.score = up;
            Some(Some(Alignment::Up(current_pos)))
        }
    }
}
