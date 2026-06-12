use crate::{
    Scoring,
    prefilter::case_needle,
    simd::{Backend, BytesVec, MaskVec, ScoreVec},
    smith_waterman::greedy::match_greedy,
};

use super::alignment_iter::Alignment;
use super::matrix::Matrix;

const MAX_HAYSTACK_LEN: usize = 512;

#[derive(Debug, Clone)]
pub struct SmithWatermanMatcherInternal<B: Backend> {
    pub needle: String,
    pub needle_simd: Vec<(B::Bytes, B::Bytes)>,
    pub scoring: Scoring,
    pub score_matrix: Matrix<B>,
    pub match_masks: Matrix<B>,
    /// Number of LANES-wide chunks (incl. the leading zero column) actually
    /// consumed by the most recent `score_haystack` call. The matrix stride is
    /// always sized for `MAX_HAYSTACK_LEN` for zero-free reuse.
    pub haystack_chunks: usize,
}

impl<B: Backend> SmithWatermanMatcherInternal<B> {
    pub fn new(needle: &[u8], scoring: &Scoring) -> Self {
        Self {
            needle: String::from_utf8_lossy(needle).to_string(),
            needle_simd: Self::broadcast_needle(needle),
            scoring: scoring.clone(),
            score_matrix: Matrix::new(needle.len(), MAX_HAYSTACK_LEN),
            match_masks: Matrix::new(needle.len(), MAX_HAYSTACK_LEN),
            haystack_chunks: 0,
        }
    }

    fn broadcast_needle(needle: &[u8]) -> Vec<(B::Bytes, B::Bytes)> {
        let needle_cased = case_needle(needle);
        needle_cased
            .iter()
            .map(|(c1, c2)| unsafe { (B::Bytes::splat(*c1), B::Bytes::splat(*c2)) })
            .collect()
    }

    #[inline(always)]
    pub fn match_haystack(&mut self, haystack: &[u8], max_typos: Option<u16>) -> Option<u16> {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(self.needle.as_bytes(), haystack, &self.scoring)
                .map(|(score, _)| score);
        }

        let score = self.score_haystack(haystack);
        match max_typos {
            Some(max_typos) if !self.has_alignment_path(score, max_typos) => None,
            _ => Some(score),
        }
    }

    #[inline(always)]
    pub fn match_haystack_indices(
        &mut self,
        haystack: &[u8],
        skipped_chunks: usize,
        max_typos: Option<u16>,
    ) -> Option<(u16, Vec<usize>)> {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(self.needle.as_bytes(), haystack, &self.scoring);
        }

        let score = self.score_haystack(haystack);

        let mut indices = Vec::with_capacity(self.needle.len());
        let mut prev_haystack_idx = usize::MAX;
        for pos in self.iter_alignment_path(skipped_chunks, score, max_typos) {
            match pos {
                Some(Alignment::Match((_, haystack_idx))) => {
                    if prev_haystack_idx != haystack_idx {
                        indices.push(haystack_idx);
                        prev_haystack_idx = haystack_idx;
                    }
                }
                Some(_) => {}
                None => return None,
            }
        }

        Some((score, indices))
    }

    #[inline(always)]
    pub fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(self.needle.as_bytes(), haystack, &self.scoring)
                .map(|(score, _)| score)
                .unwrap_or(0);
        }

        let scoring = &self.scoring;
        let haystack_chunks = haystack.len().div_ceil(B::LANES) + 1;
        self.haystack_chunks = haystack_chunks;

        // Matrix stride is fixed at construction. Row 0 and column 0 are
        // always zero (never written by the inner loop), so no re-zeroing is
        // needed between calls.
        let score_matrix = &mut self.score_matrix;
        let match_masks = &mut self.match_masks;

        unsafe {
            // Constants
            let gap_extend_penalty = B::Score::splat(scoring.gap_extend_penalty);
            let gap_open_penalty =
                B::Score::splat(scoring.gap_open_penalty - scoring.gap_extend_penalty);
            let match_score = B::Score::splat(scoring.match_score + scoring.mismatch_penalty);
            let mismatch_penalty = B::Score::splat(scoring.mismatch_penalty);
            let matching_case_bonus = B::Score::splat(scoring.matching_case_bonus);
            let capitalization_bonus = B::Score::splat(scoring.capitalization_bonus);
            let delimiter_bonus = B::Score::splat(scoring.delimiter_bonus);

            // State
            // TODO: have prefix bonus scale based on distance
            let mut prefix_bonus_masked = B::Score::first_lane(scoring.prefix_bonus);
            let mut prev_chunk_char_is_delimiter_mask = B::Mask::zero();
            let mut prev_chunk_is_lower_mask = B::Mask::zero();
            let mut max_scores = B::Score::zero();

            // TODO: try doing N needle chars per haystack chunk for better cache locality
            for (col_idx, haystack_chunk) in (0..(haystack_chunks - 1)).map(|col_idx| {
                let haystack_chunk =
                    B::Bytes::load_partial(haystack.as_ptr(), col_idx * B::LANES, haystack.len());
                (col_idx + 1, haystack_chunk)
            }) {
                // Bonus for matching a capital letter after a lowercase letter
                let is_upper_mask = haystack_chunk
                    .lt(B::Bytes::splat(b'Z' + 1))
                    .and(haystack_chunk.gt(B::Bytes::splat(b'A' - 1)));
                let is_lower_mask = haystack_chunk
                    .lt(B::Bytes::splat(b'z' + 1))
                    .and(haystack_chunk.gt(B::Bytes::splat(b'a' - 1)));
                let is_letter_mask = is_upper_mask.or(is_lower_mask);

                // Bonus when an uppercase byte follows a lowercase byte
                let capitalization_mask = B::widen_mask(
                    is_upper_mask.and(is_lower_mask.shift_right_padded_1(prev_chunk_is_lower_mask)),
                );
                let capitalization_bonus_masked = capitalization_mask.and(capitalization_bonus);
                prev_chunk_is_lower_mask = is_lower_mask;

                // Bonus for matching after a delimiter character. We consider
                // anything that isn't a digit or a letter and is within ASCII
                // range to be a delimiter.
                let is_digit_mask = haystack_chunk
                    .gt(B::Bytes::splat(b'0' - 1))
                    .and(haystack_chunk.lt(B::Bytes::splat(b'9' + 1)));
                let char_is_delimiter_mask = is_letter_mask
                    .or(is_digit_mask)
                    .or(haystack_chunk.gt(B::Bytes::splat(127)))
                    .not();
                let prev_char_is_delimiter_mask =
                    char_is_delimiter_mask.shift_right_padded_1(prev_chunk_char_is_delimiter_mask);
                let delimiter_mask =
                    B::widen_mask(prev_char_is_delimiter_mask.and(char_is_delimiter_mask.not()));
                let delimiter_bonus_masked = delimiter_mask.and(delimiter_bonus);
                prev_chunk_char_is_delimiter_mask = char_is_delimiter_mask;

                // Match-conditional bonuses (delimiter, capitalization, prefix)
                let match_and_masked_bonuses = delimiter_bonus_masked
                    .add(capitalization_bonus_masked)
                    .add(prefix_bonus_masked)
                    .add(match_score);

                let mut up_gap_mask = B::Score::zero();
                let mut prev_row_scores = B::Score::zero();
                let mut row_scores = B::Score::zero();

                for (row_idx, (needle_char, flipped_case_needle_char)) in
                    self.needle_simd.iter().enumerate().map(|(i, c)| (i + 1, c))
                {
                    // Match needle char against the chunk (case insensitive)
                    let exact_case_match_mask = needle_char.eq(haystack_chunk);
                    let flipped_case_match_mask = flipped_case_needle_char.eq(haystack_chunk);
                    let match_mask =
                        B::widen_mask(exact_case_match_mask.or(flipped_case_match_mask));
                    let exact_case_match_mask = B::widen_mask(exact_case_match_mask);

                    // Diagonal — typical match/mismatch, advancing one cell.
                    let diag_scores = {
                        let diag = prev_row_scores
                            .shift_right_padded::<1>(score_matrix.get(row_idx - 1, col_idx - 1));
                        // Add bonuses for matches
                        let diag = diag.add(match_mask.and(match_and_masked_bonuses));
                        // Always pay the mismatch penalty
                        let diag = diag.subs(mismatch_penalty);
                        // Reward matching the needle's exact case
                        diag.add(exact_case_match_mask.and(matching_case_bonus))
                    };

                    // Up — skipping a char in the needle.
                    let up_scores = {
                        let after_extend = prev_row_scores.subs(gap_extend_penalty);
                        after_extend.subs(up_gap_mask.and(gap_open_penalty))
                    };

                    // Max of diagonal, up, and left (after gap extension).
                    row_scores = B::propagate_horizontal_gaps(
                        diag_scores.max(up_scores),
                        score_matrix.get(row_idx, col_idx - 1),
                        match_mask,
                        match_masks.get(row_idx, col_idx - 1),
                        gap_open_penalty,
                        gap_extend_penalty,
                    );

                    score_matrix.set(row_idx, col_idx, row_scores);
                    match_masks.set(row_idx, col_idx, match_mask);
                    prev_row_scores = row_scores;
                    up_gap_mask = match_mask;
                }

                // Last row of the matrix at this chunk
                max_scores = max_scores.max(row_scores);
                prefix_bonus_masked = B::Score::zero();
            }

            max_scores.horizontal_max()
        }
    }

    pub fn match_end_col(&self, haystack: &[u8]) -> u16 {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(self.needle.as_bytes(), haystack, &self.scoring)
                .and_then(|(_, indices)| indices.last().copied())
                .unwrap_or(0) as u16;
        }

        let mut match_end_col: u16 = 0;
        let mut max_score = 0;
        for col_idx in 1..(haystack.len().div_ceil(B::LANES) + 1) {
            let chunk_scores = self.score_matrix.get(self.needle.len(), col_idx);
            let chunk_max_score = unsafe { chunk_scores.horizontal_max() };
            if chunk_max_score > max_score {
                max_score = chunk_max_score;
                let lane = unsafe { chunk_scores.find_lane(chunk_max_score) };
                match_end_col = ((col_idx - 1) * B::LANES + lane) as u16;
            }
        }
        match_end_col
    }

    #[cfg(test)]
    pub fn print_score_matrix(&self, haystack: &str) {
        let haystack_chunks = haystack.len().div_ceil(B::LANES) + 1;
        let stride = self.score_matrix.haystack_chunks;
        let bytes = self.score_matrix.as_byte_slice();
        let lanes = B::LANES;
        let lane_bytes = B::LANE_BYTES;

        print!("     ");
        for char in haystack.chars() {
            print!("{:<4} ", char);
        }
        println!();

        for row in 1..=self.needle.len() {
            print!("{:<4} ", self.needle.chars().nth(row - 1).unwrap_or(' '));
            for chunk in 1..haystack_chunks {
                for lane in 0..lanes {
                    let offset = (row * stride * lanes + chunk * lanes + lane) * lane_bytes;
                    let value: u16 = if lane_bytes == 2 {
                        u16::from_ne_bytes([bytes[offset], bytes[offset + 1]])
                    } else {
                        bytes[offset] as u16
                    };
                    print!("{:<4} ", value);
                }
            }
            println!();
        }
        println!();
    }
}
