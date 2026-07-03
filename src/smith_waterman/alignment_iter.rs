use super::backend::{Backend, ScoreVec};
use super::matrix::Matrix;

pub(crate) enum Alignment {
    Left,
    Up,
    Match((usize, usize)),
    Mismatch,
}

/// Iterator over alignment path positions with support for max typos.
///
/// Yields `Some(Alignment::*)` for each step taken, or `None` to signal that
/// max_typos was exceeded.
///
/// The iterator is non-generic over `Backend`: the backend's lane count and
/// element width (1 or 2 bytes) are resolved at construction so the body
/// monomorphizes once instead of per backend. The element-width branch in
/// `get_score` / `get_is_match` is a single field load + compare
pub(crate) struct AlignmentPathIter<'a> {
    score_matrix: &'a [u8],
    match_masks: &'a [u8],
    /// Number of byte-positions in one row = chunks_per_row * LANES * LANE_BYTES.
    row_stride: usize,
    lanes_per_chunk: usize,
    /// 1 for u8 scoring, 2 for u16 scoring.
    lane_bytes: usize,
    row_idx: usize,
    col_idx: usize,
    haystack_start_pos: usize,
    unicode_haystack: Option<&'a [u8]>,
    max_typos: Option<u16>,
    typo_count: u16,
    score: u16,
    finished: bool,
}

impl<'a> AlignmentPathIter<'a> {
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn new<B: Backend>(
        score_matrix: &'a Matrix<B>,
        match_masks: &'a Matrix<B>,
        needle_len: usize,
        haystack_chunks: usize,
        haystack_start_pos: usize,
        unicode_haystack: Option<&'a [u8]>,
        score: u16,
        max_typos: Option<u16>,
    ) -> Self {
        let col_idx = Self::get_col_idx::<B>(score_matrix, needle_len, haystack_chunks, score);

        Self {
            score_matrix: score_matrix.as_byte_slice(),
            match_masks: match_masks.as_byte_slice(),
            row_stride: score_matrix.haystack_chunks * B::LANES * B::LANE_BYTES,
            lanes_per_chunk: B::LANES,
            lane_bytes: B::LANE_BYTES,
            row_idx: needle_len,
            col_idx,
            haystack_start_pos,
            unicode_haystack,
            max_typos,
            typo_count: 0,
            score,
            finished: false,
        }
    }

    #[inline(always)]
    fn get_col_idx<B: Backend>(
        score_matrix: &Matrix<B>,
        needle_len: usize,
        haystack_chunks: usize,
        score: u16,
    ) -> usize {
        for chunk_idx in 1..haystack_chunks {
            let chunk = score_matrix.get(needle_len, chunk_idx);
            let idx = unsafe { chunk.find_lane(score) };
            if idx != B::LANES {
                return chunk_idx * B::LANES + idx;
            }
        }
        panic!("could not find max score in score matrix final row");
    }

    #[inline(always)]
    fn get_score(&self, row: usize, col: usize) -> u16 {
        let offset = row * self.row_stride + col * self.lane_bytes;
        if self.lane_bytes == 2 {
            u16::from_ne_bytes([self.score_matrix[offset], self.score_matrix[offset + 1]])
        } else {
            self.score_matrix[offset] as u16
        }
    }

    #[inline(always)]
    fn get_is_match(&self, row: usize, col: usize) -> bool {
        let offset = row * self.row_stride + col * self.lane_bytes;
        if self.lane_bytes == 2 {
            self.match_masks[offset] != 0 || self.match_masks[offset + 1] != 0
        } else {
            self.match_masks[offset] != 0
        }
    }
}

impl<'a> Iterator for AlignmentPathIter<'a> {
    type Item = Option<Alignment>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.row_idx == 0 || self.finished {
            return None;
        }

        if let Some(max_typos) = self.max_typos
            && self.typo_count > max_typos
        {
            self.finished = true;
            return Some(None);
        }

        // Must be moving up only (at left edge), or lost alignment
        if self.col_idx < self.lanes_per_chunk || self.score == 0 {
            if let Some(max_typos) = self.max_typos
                && (self.typo_count + self.row_idx as u16) > max_typos
            {
                self.finished = true;
                return Some(None);
            }
            return None;
        }

        let current_pos = (
            self.row_idx - 1,
            self.col_idx - self.lanes_per_chunk + self.haystack_start_pos,
        );

        // Cannot move up or left when on a continuation byte (multi-byte unicode char)
        // so walk left
        if let Some(haystack) = self.unicode_haystack
            && haystack
                .get(current_pos.1)
                .is_some_and(|byte| byte & 0xC0 == 0x80)
        {
            self.col_idx -= 1;
            self.score = self.get_score(self.row_idx, self.col_idx);
            return Some(Some(Alignment::Left));
        }

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
                return Some(Some(Alignment::Mismatch));
            }
            self.score = diag;
            Some(Some(Alignment::Match(current_pos)))
        // Skipped character in haystack (left)
        } else if left >= up {
            self.col_idx -= 1;
            self.score = left;
            Some(Some(Alignment::Left))
        // Skipped character in needle (up)
        } else {
            self.typo_count += 1;
            self.row_idx -= 1;
            self.score = up;
            Some(Some(Alignment::Up))
        }
    }
}
