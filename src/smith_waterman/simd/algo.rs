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
    phantom: PhantomData<Simd256>,
}

impl<Simd128: Vector128Expansion<Simd256>, Simd256: Vector256>
    SmithWatermanMatcherInternal<Simd128, Simd256>
{
    pub fn new(needle: &[u8], scoring: &Scoring) -> Self {
        Self {
            needle: String::from_utf8_lossy(needle).to_string(),
            needle_simd: Self::broadcast_needle(needle),
            scoring: scoring.clone(),
            score_matrix: Matrix::new(needle.len(), MAX_HAYSTACK_LEN),
            match_masks: Matrix::new(needle.len(), MAX_HAYSTACK_LEN),
            phantom: PhantomData,
        }
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
        let haystack_chunks = haystack.len().div_ceil(16) + 1;

        let score_matrix = &mut self.score_matrix;
        score_matrix.set_haystack_chunks(haystack_chunks);
        score_matrix.zero();
        let match_masks = &mut self.match_masks;
        match_masks.set_haystack_chunks(haystack_chunks);
        match_masks.zero();

        unsafe {
            // Constants
            let gap_extend = Simd256::splat_u16(scoring.gap_extend_penalty);
            let gap_open =
                Simd256::splat_u16(scoring.gap_open_penalty - scoring.gap_extend_penalty);
            let match_score = Simd256::splat_u16(scoring.match_score + scoring.mismatch_penalty);
            let mismatch_penalty = Simd256::splat_u16(scoring.mismatch_penalty);
            let matching_case_bonus = Simd256::splat_u16(scoring.matching_case_bonus);
            let capitalization_bonus = Simd256::splat_u16(scoring.capitalization_bonus);
            let delimiter_bonus = Simd256::splat_u16(scoring.delimiter_bonus);

            // State
            // TODO: have prefix bonus scale based on distance
            let mut prefix_bonus_masked =
                Simd256::splat_u16(scoring.prefix_bonus).and(Simd256::load_unaligned(PREFIX_MASK));
            let mut prev_chunk_char_is_delimiter_mask = Simd128::zero();
            let mut prev_chunk_is_lower_mask = Simd128::zero();
            let mut max_scores = Simd256::zero();

            // TODO: try doing N needle chars per haystack chunk for better cache locality
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
                // We consider anything that isn't a digit or a letter, and within ASCII range, to
                // be a delimiter
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

                for (row_idx, (needle_char, flipped_case_needle_char)) in
                    self.needle_simd.iter().enumerate().map(|(i, c)| (i + 1, c))
                {
                    // Match needle chars against the haystack (case insensitive)
                    let exact_case_match_mask = (*needle_char).eq_u8(haystack);
                    let flipped_case_match_mask = (*flipped_case_needle_char).eq_u8(haystack);
                    let match_mask = exact_case_match_mask
                        .or(flipped_case_match_mask)
                        .cast_i8_to_i16();
                    let exact_case_match_mask = exact_case_match_mask.cast_i8_to_i16();

                    // Diagonal - typical match/mismatch, moving along one haystack and needle char
                    let diag_scores = {
                        let diag = prev_row_scores.shift_right_padded_u16::<1>(
                            score_matrix.get(row_idx - 1, col_idx - 1),
                        );

                        // Add match score (+ mismatch penalty) with bonuses for matches, avoiding blendv
                        // Bonuses are delimiter, capitalization and prefix
                        let diag = diag.add_u16(match_mask.and(match_and_masked_bonuses));
                        // Always add mismatch penalty
                        let diag = diag.subs_u16(mismatch_penalty);
                        // Add matching case bonus
                        diag.add_u16(exact_case_match_mask.and(matching_case_bonus))
                    };

                    // Up - skipping char in needle
                    let up_scores = {
                        // Always apply gap extend penalty
                        let score_after_gap_extend = prev_row_scores.subs_u16(gap_extend);
                        // Apply gap open penalty - gap extend penalty for opened gaps, avoiding blendv
                        score_after_gap_extend.subs_u16(up_gap_mask.and(gap_open))
                    };

                    // Max of diagonal, up and left (after gap extension)
                    row_scores = propagate_horizontal_gaps::<Simd256>(
                        diag_scores.max_u16(up_scores),         // Current
                        score_matrix.get(row_idx, col_idx - 1), // Left
                        match_mask,                             // Current
                        match_masks.get(row_idx, col_idx - 1),  // Left
                        scoring.gap_open_penalty,
                        scoring.gap_extend_penalty,
                    );

                    // Store results
                    score_matrix.set(row_idx, col_idx, row_scores);
                    match_masks.set(row_idx, col_idx, match_mask);
                    prev_row_scores = row_scores;
                    up_gap_mask = match_mask;
                }

                // because we do this after the loop, we're guaranteed to be on the last row
                max_scores = max_scores.max_u16(row_scores);
                prefix_bonus_masked = Simd256::zero();
            }

            max_scores.smax_u16()
        }
    }

    #[cfg(test)]
    pub fn print_score_matrix(&self, haystack: &str) {
        let haystack_chunks = haystack.len().div_ceil(16) + 1;
        let score_matrix = self.score_matrix.as_slice();

        print!("     ");
        for char in haystack.chars() {
            print!("{:<4} ", char);
        }
        println!();

        for (i, row) in score_matrix
            .chunks_exact(haystack_chunks)
            .enumerate()
            .skip(1)
            .take(self.needle.len())
        {
            print!("{:<4} ", self.needle.chars().nth(i - 1).unwrap_or(' '));
            for col in row.iter().skip(1).flatten() {
                print!("{:<4} ", col);
            }
            println!();
        }
        println!();
    }
}
