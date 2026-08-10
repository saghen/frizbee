use std::marker::PhantomData;

use crate::{
    Scoring,
    prefilter::case_needle,
    simd::{Vector128Expansion, Vector256},
    smith_waterman::greedy::match_greedy,
};

use super::alignment_iter::Alignment;
use super::gaps::propagate_horizontal_gaps;
use super::matrix::Matrix;

const MAX_HAYSTACK_LEN: usize = 512;

pub const PREFIX_MASK: [u8; 32] = [
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

#[derive(Debug, Clone)]
pub struct SmithWatermanMatcherInternal<Simd128: Vector128Expansion<Simd256>, Simd256: Vector256> {
    pub needle: String,
    pub needle_simd: Vec<(Simd128, Simd128)>,
    pub scoring: Scoring,
    pub score_matrix: Matrix<Simd256>,
    pub match_masks: Matrix<Simd256>,
    /// Actual haystack chunks for the most recent score_haystack call.
    pub haystack_chunks: usize,
    phantom: PhantomData<Simd256>,
}

impl<Simd128: Vector128Expansion<Simd256>, Simd256: Vector256>
    SmithWatermanMatcherInternal<Simd128, Simd256>
{
    pub fn new(needle: &[u8], scoring: &Scoring) -> Self {
        Self {
            needle: String::from_utf8_lossy(needle).to_string(),
            needle_simd: Self::broadcast_needle(needle),
            scoring: *scoring,
            score_matrix: Matrix::new(needle.len(), 0),
            match_masks: Matrix::new(needle.len(), 0),
            haystack_chunks: 0,
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn reserve_haystack_len(&mut self, haystack_len: usize) {
        let haystack_len = haystack_len.min(MAX_HAYSTACK_LEN);
        self.score_matrix.reserve_haystack_len(haystack_len);
        self.match_masks.reserve_haystack_len(haystack_len);
    }

    fn broadcast_needle(needle: &[u8]) -> Vec<(Simd128, Simd128)> {
        let needle_cased = case_needle(needle);
        needle_cased
            .iter()
            .map(|(c1, c2)| unsafe { (Simd128::splat_u8(*c1), Simd128::splat_u8(*c2)) })
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

    /// Like `match_haystack` but also returns the end column of the best alignment.
    /// For the SIMD path, scans the last row of the score matrix to find where the max landed.
    /// For the greedy fallback, uses the last matched index.
    #[cfg(feature = "match_end_col")]
    #[inline(always)]
    pub fn match_haystack_with_end_col(
        &mut self,
        haystack: &[u8],
        max_typos: Option<u16>,
    ) -> Option<(u16, u16)> {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(self.needle.as_bytes(), haystack, &self.scoring)
                .map(|(score, indices)| (score, indices.last().copied().unwrap_or(0) as u16));
        }

        let score = self.score_haystack(haystack);
        match max_typos {
            Some(max_typos) if !self.has_alignment_path(score, max_typos) => None,
            _ => {
                let end_col = self.get_match_end_col(score);
                Some((score, end_col))
            }
        }
    }

    /// Find the haystack byte position where the max score occurs in the last needle row.
    /// Must be called after `score_haystack` which populates the matrix.
    #[cfg(feature = "match_end_col")]
    #[inline(always)]
    pub fn get_match_end_col(&self, max_score: u16) -> u16 {
        let needle_len = self.needle.len();
        for chunk_idx in 1..self.haystack_chunks {
            let chunk = self.score_matrix.get(needle_len, chunk_idx);
            let idx = unsafe { chunk.idx_u16(max_score) };
            if idx != 16 {
                return ((chunk_idx - 1) * 16 + idx) as u16;
            }
        }
        0
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
        if haystack.is_empty() {
            return 0;
        }
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(self.needle.as_bytes(), haystack, &self.scoring)
                .map(|(score, _)| score)
                .unwrap_or(0);
        }

        self.reserve_haystack_len(haystack.len());
        let scoring = &self.scoring;
        let haystack_chunks = haystack.len().div_ceil(16) + 1;
        self.haystack_chunks = haystack_chunks;

        let score_matrix = &mut self.score_matrix;
        let match_masks = &mut self.match_masks;

        unsafe {
            // Constants
            let gap_extend_penalty = Simd256::splat_u16(scoring.gap_extend_penalty);
            let gap_open_penalty =
                Simd256::splat_u16(scoring.gap_open_penalty - scoring.gap_extend_penalty);
            let match_score = Simd256::splat_u16(scoring.match_score + scoring.mismatch_penalty);
            let mismatch_penalty = Simd256::splat_u16(scoring.mismatch_penalty);
            let matching_case_bonus = Simd256::splat_u16(scoring.matching_case_bonus);
            let capitalization_bonus = Simd256::splat_u16(scoring.capitalization_bonus);
            let delimiter_bonus = Simd256::splat_u16(scoring.delimiter_bonus);

            // State
            let mut prefix_bonus_masked =
                Simd256::splat_u16(scoring.prefix_bonus).and(Simd256::load_unaligned(PREFIX_MASK));
            let mut prev_chunk_char_is_delimiter_mask = Simd128::zero();
            let mut prev_chunk_is_lower_mask = Simd128::zero();
            let mut max_scores = Simd256::zero();

            for (col_idx, haystack) in (0..(haystack_chunks - 1)).map(|col_idx| {
                let haystack =
                    Simd128::load_partial(haystack.as_ptr(), col_idx * 16, haystack.len());
                (col_idx + 1, haystack)
            }) {
                // Bonus for matching a capital letter after a lowercase letter
                let is_upper_mask = Simd128::and(
                    haystack.lt_u8(Simd128::splat_u8(b'Z' + 1)),
                    haystack.gt_u8(Simd128::splat_u8(b'A' - 1)),
                );
                let is_lower_mask = Simd128::and(
                    haystack.lt_u8(Simd128::splat_u8(b'z' + 1)),
                    haystack.gt_u8(Simd128::splat_u8(b'a' - 1)),
                );
                let is_letter_mask = is_upper_mask.or(is_lower_mask);

                // Give the bonus if the character is uppercase and the previous character was lowercase
                let capitalization_mask = Simd128::and(
                    is_upper_mask,
                    is_lower_mask.shift_right_padded_u8::<1>(prev_chunk_is_lower_mask),
                )
                .cast_i8_to_i16();
                let capitalization_bonus_masked = capitalization_mask.and(capitalization_bonus);

                prev_chunk_is_lower_mask = is_lower_mask;

                // Bonus for matching after a delimiter character
                let is_digit_mask = Simd128::and(
                    haystack.gt_u8(Simd128::splat_u8(b'0' - 1)),
                    haystack.lt_u8(Simd128::splat_u8(b'9' + 1)),
                );
                let char_is_delimiter_mask = is_letter_mask
                    .or(is_digit_mask)
                    .or(haystack.gt_u8(Simd128::splat_u8(127)))
                    .not();
                let prev_char_is_delimiter_mask = char_is_delimiter_mask
                    .shift_right_padded_u8::<1>(prev_chunk_char_is_delimiter_mask);
                let delimiter_mask = prev_char_is_delimiter_mask
                    .and(char_is_delimiter_mask.not())
                    .cast_i8_to_i16();
                let delimiter_bonus_masked = delimiter_mask.and(delimiter_bonus);
                prev_chunk_char_is_delimiter_mask = char_is_delimiter_mask;

                // Delimiter, capitalization and prefix bonuses
                let match_and_masked_bonuses = delimiter_bonus_masked
                    .add_u16(capitalization_bonus_masked)
                    .add_u16(prefix_bonus_masked)
                    .add_u16(match_score);

                let mut up_gap_mask = Simd256::zero();
                let mut prev_row_scores = Simd256::zero();
                let mut row_scores = Simd256::zero();

                let needle_chunks = self.needle_simd.chunks_exact(2);
                let needle_rem = needle_chunks.remainder();
                let mut row_idx = 1;

                for pair in needle_chunks {
                    let (n0, fn0) = &pair[0];
                    let (n1, fn1) = &pair[1];

                    let exact0 = n0.eq_u8(haystack);
                    let flipped0 = fn0.eq_u8(haystack);
                    let exact1 = n1.eq_u8(haystack);
                    let flipped1 = fn1.eq_u8(haystack);

                    let match_mask0 = exact0.or(flipped0).cast_i8_to_i16();
                    let exact0_cast = exact0.cast_i8_to_i16();
                    let match_mask1 = exact1.or(flipped1).cast_i8_to_i16();
                    let exact1_cast = exact1.cast_i8_to_i16();

                    let diag0 = prev_row_scores
                        .shift_right_padded_u16::<1>(score_matrix.get(row_idx - 1, col_idx - 1))
                        .add_u16(match_mask0.and(match_and_masked_bonuses))
                        .subs_u16(mismatch_penalty)
                        .add_u16(exact0_cast.and(matching_case_bonus));

                    let up0 = prev_row_scores
                        .subs_u16(gap_extend_penalty)
                        .subs_u16(up_gap_mask.and(gap_open_penalty));

                    let r0_scores = propagate_horizontal_gaps::<Simd256>(
                        diag0.max_u16(up0),
                        score_matrix.get(row_idx, col_idx - 1),
                        match_mask0,
                        match_masks.get(row_idx, col_idx - 1),
                        gap_open_penalty,
                        gap_extend_penalty,
                    );
                    score_matrix.set(row_idx, col_idx, r0_scores);
                    match_masks.set(row_idx, col_idx, match_mask0);

                    let diag1 = r0_scores
                        .shift_right_padded_u16::<1>(score_matrix.get(row_idx, col_idx - 1))
                        .add_u16(match_mask1.and(match_and_masked_bonuses))
                        .subs_u16(mismatch_penalty)
                        .add_u16(exact1_cast.and(matching_case_bonus));

                    let up1 = r0_scores
                        .subs_u16(gap_extend_penalty)
                        .subs_u16(match_mask0.and(gap_open_penalty));

                    row_scores = propagate_horizontal_gaps::<Simd256>(
                        diag1.max_u16(up1),
                        score_matrix.get(row_idx + 1, col_idx - 1),
                        match_mask1,
                        match_masks.get(row_idx + 1, col_idx - 1),
                        gap_open_penalty,
                        gap_extend_penalty,
                    );
                    score_matrix.set(row_idx + 1, col_idx, row_scores);
                    match_masks.set(row_idx + 1, col_idx, match_mask1);

                    prev_row_scores = row_scores;
                    up_gap_mask = match_mask1;
                    row_idx += 2;
                }

                for (needle_char, flipped_case_needle_char) in needle_rem {
                    let exact_case_match_mask = (*needle_char).eq_u8(haystack);
                    let flipped_case_match_mask = (*flipped_case_needle_char).eq_u8(haystack);
                    let match_mask = exact_case_match_mask
                        .or(flipped_case_match_mask)
                        .cast_i8_to_i16();
                    let exact_case_match_mask = exact_case_match_mask.cast_i8_to_i16();

                    let diag_scores = prev_row_scores
                        .shift_right_padded_u16::<1>(score_matrix.get(row_idx - 1, col_idx - 1))
                        .add_u16(match_mask.and(match_and_masked_bonuses))
                        .subs_u16(mismatch_penalty)
                        .add_u16(exact_case_match_mask.and(matching_case_bonus));

                    let up_scores = prev_row_scores
                        .subs_u16(gap_extend_penalty)
                        .subs_u16(up_gap_mask.and(gap_open_penalty));

                    row_scores = propagate_horizontal_gaps::<Simd256>(
                        diag_scores.max_u16(up_scores),
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
                    row_idx += 1;
                }

                // because we do this after the loop, we're guaranteed to be on the last row
                max_scores = max_scores.max_u16(row_scores);
                prefix_bonus_masked = Simd256::zero();
            }

            max_scores.smax_u16()
        }
    }

    /// Score a haystack provided as pre-chunked, 16-byte aligned pointers.
    ///
    /// Each pointer in `chunk_ptrs` must point to exactly 16 bytes of aligned
    /// data (a `SimdChunk`). The last chunk is zero-padded at build time.
    /// `byte_len` is the actual path length (for DP dimensioning).
    ///
    /// This is the fastest scoring path: each column is a single aligned SIMD
    /// load — no `load_partial`, no segment lookup, no bridge copies.
    #[inline(always)]
    pub fn score_haystack_chunked(&mut self, chunk_ptrs: &[*const u8], byte_len: u16) -> u16 {
        let total_len = byte_len as usize;
        if total_len == 0 {
            return 0;
        }
        if total_len > MAX_HAYSTACK_LEN {
            return self
                .match_greedy_chunked(chunk_ptrs, total_len)
                .map(|(score, _)| score)
                .unwrap_or(0);
        }

        self.reserve_haystack_len(total_len);
        let scoring = &self.scoring;
        let haystack_chunks = total_len.div_ceil(16) + 1;
        self.haystack_chunks = haystack_chunks;

        let score_matrix = &mut self.score_matrix;
        let match_masks = &mut self.match_masks;

        unsafe {
            let gap_extend_penalty = Simd256::splat_u16(scoring.gap_extend_penalty);
            let gap_open_penalty =
                Simd256::splat_u16(scoring.gap_open_penalty - scoring.gap_extend_penalty);
            let match_score = Simd256::splat_u16(scoring.match_score + scoring.mismatch_penalty);
            let mismatch_penalty = Simd256::splat_u16(scoring.mismatch_penalty);
            let matching_case_bonus = Simd256::splat_u16(scoring.matching_case_bonus);
            let capitalization_bonus = Simd256::splat_u16(scoring.capitalization_bonus);
            let delimiter_bonus = Simd256::splat_u16(scoring.delimiter_bonus);

            let mut prefix_bonus_masked =
                Simd256::splat_u16(scoring.prefix_bonus).and(Simd256::load_unaligned(PREFIX_MASK));
            let mut prev_chunk_char_is_delimiter_mask = Simd128::zero();
            let mut prev_chunk_is_lower_mask = Simd128::zero();
            let mut max_scores = Simd256::zero();

            for (raw_col_idx, &chunk_ptr) in chunk_ptrs.iter().enumerate() {
                let col_idx = raw_col_idx + 1;

                // Direct aligned load — no load_partial, no branching
                let haystack = Simd128::load_aligned_16(chunk_ptr);

                let is_upper_mask = Simd128::and(
                    haystack.lt_u8(Simd128::splat_u8(b'Z' + 1)),
                    haystack.gt_u8(Simd128::splat_u8(b'A' - 1)),
                );
                let is_lower_mask = Simd128::and(
                    haystack.lt_u8(Simd128::splat_u8(b'z' + 1)),
                    haystack.gt_u8(Simd128::splat_u8(b'a' - 1)),
                );
                let is_letter_mask = is_upper_mask.or(is_lower_mask);

                let capitalization_mask = Simd128::and(
                    is_upper_mask,
                    is_lower_mask.shift_right_padded_u8::<1>(prev_chunk_is_lower_mask),
                )
                .cast_i8_to_i16();
                let capitalization_bonus_masked = capitalization_mask.and(capitalization_bonus);
                prev_chunk_is_lower_mask = is_lower_mask;

                let is_digit_mask = Simd128::and(
                    haystack.gt_u8(Simd128::splat_u8(b'0' - 1)),
                    haystack.lt_u8(Simd128::splat_u8(b'9' + 1)),
                );
                let char_is_delimiter_mask = is_letter_mask
                    .or(is_digit_mask)
                    .or(haystack.gt_u8(Simd128::splat_u8(127)))
                    .not();
                let prev_char_is_delimiter_mask = char_is_delimiter_mask
                    .shift_right_padded_u8::<1>(prev_chunk_char_is_delimiter_mask);
                let delimiter_mask = prev_char_is_delimiter_mask
                    .and(char_is_delimiter_mask.not())
                    .cast_i8_to_i16();
                let delimiter_bonus_masked = delimiter_mask.and(delimiter_bonus);
                prev_chunk_char_is_delimiter_mask = char_is_delimiter_mask;

                let match_and_masked_bonuses = delimiter_bonus_masked
                    .add_u16(capitalization_bonus_masked)
                    .add_u16(prefix_bonus_masked)
                    .add_u16(match_score);

                let mut up_gap_mask = Simd256::zero();
                let mut prev_row_scores = Simd256::zero();
                let mut row_scores = Simd256::zero();

                let needle_chunks = self.needle_simd.chunks_exact(2);
                let needle_rem = needle_chunks.remainder();
                let mut row_idx = 1;

                for pair in needle_chunks {
                    let (n0, fn0) = &pair[0];
                    let (n1, fn1) = &pair[1];

                    let exact0 = n0.eq_u8(haystack);
                    let flipped0 = fn0.eq_u8(haystack);
                    let exact1 = n1.eq_u8(haystack);
                    let flipped1 = fn1.eq_u8(haystack);

                    let match_mask0 = exact0.or(flipped0).cast_i8_to_i16();
                    let exact0_cast = exact0.cast_i8_to_i16();
                    let match_mask1 = exact1.or(flipped1).cast_i8_to_i16();
                    let exact1_cast = exact1.cast_i8_to_i16();

                    let diag0 = prev_row_scores
                        .shift_right_padded_u16::<1>(score_matrix.get(row_idx - 1, col_idx - 1))
                        .add_u16(match_mask0.and(match_and_masked_bonuses))
                        .subs_u16(mismatch_penalty)
                        .add_u16(exact0_cast.and(matching_case_bonus));

                    let up0 = prev_row_scores
                        .subs_u16(gap_extend_penalty)
                        .subs_u16(up_gap_mask.and(gap_open_penalty));

                    let r0_scores = propagate_horizontal_gaps::<Simd256>(
                        diag0.max_u16(up0),
                        score_matrix.get(row_idx, col_idx - 1),
                        match_mask0,
                        match_masks.get(row_idx, col_idx - 1),
                        gap_open_penalty,
                        gap_extend_penalty,
                    );
                    score_matrix.set(row_idx, col_idx, r0_scores);
                    match_masks.set(row_idx, col_idx, match_mask0);

                    let diag1 = r0_scores
                        .shift_right_padded_u16::<1>(score_matrix.get(row_idx, col_idx - 1))
                        .add_u16(match_mask1.and(match_and_masked_bonuses))
                        .subs_u16(mismatch_penalty)
                        .add_u16(exact1_cast.and(matching_case_bonus));

                    let up1 = r0_scores
                        .subs_u16(gap_extend_penalty)
                        .subs_u16(match_mask0.and(gap_open_penalty));

                    row_scores = propagate_horizontal_gaps::<Simd256>(
                        diag1.max_u16(up1),
                        score_matrix.get(row_idx + 1, col_idx - 1),
                        match_mask1,
                        match_masks.get(row_idx + 1, col_idx - 1),
                        gap_open_penalty,
                        gap_extend_penalty,
                    );
                    score_matrix.set(row_idx + 1, col_idx, row_scores);
                    match_masks.set(row_idx + 1, col_idx, match_mask1);

                    prev_row_scores = row_scores;
                    up_gap_mask = match_mask1;
                    row_idx += 2;
                }

                for (needle_char, flipped_case_needle_char) in needle_rem {
                    let exact_case_match_mask = (*needle_char).eq_u8(haystack);
                    let flipped_case_match_mask = (*flipped_case_needle_char).eq_u8(haystack);
                    let match_mask = exact_case_match_mask
                        .or(flipped_case_match_mask)
                        .cast_i8_to_i16();
                    let exact_case_match_mask = exact_case_match_mask.cast_i8_to_i16();

                    let diag_scores = prev_row_scores
                        .shift_right_padded_u16::<1>(score_matrix.get(row_idx - 1, col_idx - 1))
                        .add_u16(match_mask.and(match_and_masked_bonuses))
                        .subs_u16(mismatch_penalty)
                        .add_u16(exact_case_match_mask.and(matching_case_bonus));

                    let up_scores = prev_row_scores
                        .subs_u16(gap_extend_penalty)
                        .subs_u16(up_gap_mask.and(gap_open_penalty));

                    row_scores = propagate_horizontal_gaps::<Simd256>(
                        diag_scores.max_u16(up_scores),
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
                    row_idx += 1;
                }

                max_scores = max_scores.max_u16(row_scores);
                prefix_bonus_masked = Simd256::zero();
            }

            max_scores.smax_u16()
        }
    }

    /// Greedy fallback for haystacks that exceed the preallocated DP matrix.
    /// Reconstructs a contiguous buffer from the chunk pointers.
    fn match_greedy_chunked(
        &self,
        chunk_ptrs: &[*const u8],
        total_len: usize,
    ) -> Option<(u16, Vec<usize>)> {
        let mut buf = vec![0u8; total_len];
        for (i, &ptr) in chunk_ptrs.iter().enumerate() {
            let start = i * 16;
            let take = 16.min(total_len - start);
            unsafe {
                core::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr().add(start), take);
            }
        }
        match_greedy(self.needle.as_bytes(), &buf, &self.scoring)
    }

    #[inline(always)]
    pub fn match_haystack_chunked(
        &mut self,
        chunk_ptrs: &[*const u8],
        byte_len: u16,
        max_typos: Option<u16>,
    ) -> Option<u16> {
        let total_len = byte_len as usize;
        if total_len > MAX_HAYSTACK_LEN {
            return self
                .match_greedy_chunked(chunk_ptrs, total_len)
                .map(|(score, _)| score);
        }

        let score = self.score_haystack_chunked(chunk_ptrs, byte_len);
        match max_typos {
            Some(max_typos) if !self.has_alignment_path(score, max_typos) => None,
            _ => Some(score),
        }
    }

    #[cfg(feature = "match_end_col")]
    #[inline(always)]
    pub fn match_haystack_chunked_with_end_col(
        &mut self,
        chunk_ptrs: &[*const u8],
        byte_len: u16,
        max_typos: Option<u16>,
    ) -> Option<(u16, u16)> {
        let total_len = byte_len as usize;
        if total_len > MAX_HAYSTACK_LEN {
            return self
                .match_greedy_chunked(chunk_ptrs, total_len)
                .map(|(score, indices)| (score, indices.last().copied().unwrap_or(0) as u16));
        }

        let score = self.score_haystack_chunked(chunk_ptrs, byte_len);
        if score == 0 {
            return None;
        }
        match max_typos {
            Some(max_typos) if !self.has_alignment_path(score, max_typos) => None,
            _ => {
                let end_col = self.get_match_end_col(score);
                Some((score, end_col))
            }
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
        for col_idx in 1..(haystack.len().div_ceil(16) + 1) {
            let chunk_scores = self.score_matrix.get(self.needle.len(), col_idx);
            let chunk_max_score = unsafe { chunk_scores.smax_u16() };
            if chunk_max_score > max_score {
                max_score = chunk_max_score;
                let lane = unsafe { chunk_scores.idx_u16(chunk_max_score) };
                match_end_col = ((col_idx - 1) * 16 + lane) as u16;
            }
        }
        match_end_col
    }

    #[cfg(test)]
    pub fn print_score_matrix(&self, haystack: &str) {
        let haystack_chunks = haystack.len().div_ceil(16) + 1;

        print!("     ");
        for char in haystack.chars() {
            print!("{:<4} ", char);
        }
        println!();

        for row in 1..=self.needle.len() {
            print!("{:<4} ", self.needle.chars().nth(row - 1).unwrap_or(' '));
            for col in 1..haystack_chunks {
                for value in self.score_matrix.get(row, col).to_array_256_u16() {
                    print!("{:<4} ", value);
                }
            }
            println!();
        }
        println!();
    }
}
