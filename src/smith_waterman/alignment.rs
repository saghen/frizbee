//! Typo-aware Smith-Waterman alignment

use super::{SmithWaterman, backend::Backend};
use crate::{Scoring, prefilter::UnicodeChar};
use alloc::{vec, vec::Vec};

/// With the given number of typos, this cell is unreachable
const UNREACHABLE: i32 = -1;

/// The cell's score came from the cell above, rather than the diagonal
const UP: u8 = 1;
/// The cell's score came from a gap along the row
const LEFT: u8 = 2;
/// The gap leaving the cell came from further left, rather than the cell
const LEFT_CONTINUES: u8 = 4;

impl<B: Backend> SmithWaterman<B> {
    /// The haystack byte indices (offset by `haystack_start_pos`) of the chars
    /// matched by the best alignment with at most `max_typos`
    /// needle chars unmatched, empty when there is none
    #[inline(always)]
    pub(crate) fn alignment_indices(
        &self,
        haystack: &[u8],
        haystack_start_pos: usize,
        unicode: bool,
        max_typos: Option<u16>,
    ) -> Vec<u32> {
        Aligner {
            needle: self.needle.as_bytes(),
            needle_unicode: unicode.then_some(self.needle_unicode.as_slice()),
            case_sensitive: self.case_sensitive,
            scoring: &self.scoring,
            haystack,
            haystack_start_pos,
        }
        .align(max_typos)
        .1
    }
}

struct Aligner<'a> {
    needle: &'a [u8],
    /// The rows in Unicode mode, where only scalars' lead bytes are columns
    needle_unicode: Option<&'a [UnicodeChar]>,
    case_sensitive: bool,
    scoring: &'a Scoring,
    haystack: &'a [u8],
    haystack_start_pos: usize,
}

impl Aligner<'_> {
    /// Whether the needle char of `row` matches at haystack byte `start`,
    /// and whether in its exact case, as the kernels compare them
    fn matches(&self, row: usize, start: usize) -> (bool, bool) {
        let haystack = &self.haystack[start..];

        // Unicode
        if let Some(needle) = self.needle_unicode {
            let c = &needle[row - 1];
            let exact = haystack.starts_with(&c.chars[..c.len]);
            let flipped = c.has_flip && haystack.starts_with(&c.flipped_chars[..c.len]);
            return (exact || flipped, exact);
        }

        // Ascii
        let (haystack_char, needle_char) = (haystack[0], self.needle[row - 1]);
        let exact = haystack_char == needle_char;
        let flipped = !self.case_sensitive
            && haystack_char ^ 0x20 == needle_char
            && haystack_char.is_ascii_alphabetic();
        (exact || flipped, exact)
    }

    /// The best alignment's score and indices
    #[inline(never)]
    fn align(&self, max_typos: Option<u16>) -> (i32, Vec<u32>) {
        let (scoring, haystack) = (self.scoring, self.haystack);

        // Haystack byte for each column, skipping Unicode continuation bytes
        let unicode = self.needle_unicode.is_some();
        let starts: Vec<usize> = (0..haystack.len())
            .filter(|&start| !unicode || haystack[start] & 0xC0 != 0x80)
            .collect();

        // Smith waterman uses 1 extra row and column as padding at the start
        let rows = self.needle_unicode.map_or(self.needle.len(), <[_]>::len) + 1;
        let cols = starts.len() + 1;

        // On typo, step up from the layer below
        // When max_typos == Some(N), layer 0 stays unreachable, and the layer
        // above it has no typos. Without one, typos stay in the only layer
        let typo_step = max_typos.is_some() as usize;
        let layers =
            typo_step + max_typos.map_or(1, |max_typos| (rows - 1).min(max_typos as usize) + 1);

        let at = |row: usize, layer: usize, col: usize| (row * layers + layer) * cols + col;

        let gap_extend = -(scoring.gap_extend_penalty as i32);
        let gap_open = -(scoring
            .gap_open_penalty
            .saturating_sub(scoring.gap_extend_penalty) as i32);

        // Smith Waterman scores and alignment directions
        let mut score_matrix = vec![UNREACHABLE; rows * layers * cols];
        score_matrix[at(0, typo_step, 0)..][..cols].fill(0);
        let mut direction_matrix = vec![0u8; rows * layers * cols];

        // Scalar Smith Waterman, filling the scoring matrix (unlike SIMD impl)
        for row in 1..rows {
            for layer in typo_step..layers {
                let typo_layer = layer - typo_step;

                // Before the haystack every needle char so far is skipped
                score_matrix[at(row, layer, 0)] =
                    advance(score_matrix[at(row - 1, typo_layer, 0)], gap_extend);
                // The best gap into the column, and what a gap from the cell
                // to its left starts at
                let (mut gap, mut gap_source) = (UNREACHABLE, score_matrix[at(row, layer, 0)]);

                for col in 1..cols {
                    let start = starts[col - 1];
                    let (is_match, exact_case) = self.matches(row, start);

                    // Diagonal: match or mismatch
                    let diag = if is_match {
                        let mut bonus = scoring.match_score as i32;
                        if exact_case {
                            bonus += scoring.matching_case_bonus as i32;
                        }
                        if start > 0 {
                            bonus += byte_bonus(haystack[start - 1], haystack[start], scoring);
                        } else if self.haystack_start_pos == 0 {
                            bonus += scoring.prefix_bonus as i32;
                        }
                        advance(score_matrix[at(row - 1, layer, col - 1)], bonus)
                    } else {
                        let mismatch = -(scoring.mismatch_penalty as i32);
                        advance(score_matrix[at(row - 1, typo_layer, col - 1)], mismatch)
                    };

                    // Up: needle char skipped
                    let above_is_match = row > 1 && self.matches(row - 1, start).0;
                    let up_penalty = gap_extend + if above_is_match { gap_open } else { 0 };
                    let up = advance(score_matrix[at(row - 1, typo_layer, col)], up_penalty);

                    // Left: haystack char skipped
                    gap = advance(gap.max(gap_source), gap_extend);

                    let cell = diag.max(up);
                    gap_source = advance(cell, if is_match { gap_open } else { 0 });
                    score_matrix[at(row, layer, col)] = cell.max(gap);
                    direction_matrix[at(row, layer, col)] = ((up > diag) as u8 * UP)
                        | ((gap > cell) as u8 * LEFT)
                        | ((gap > gap_source) as u8 * LEFT_CONTINUES);
                }
            }
        }

        // Find layer/row/col of the highest score,
        // preferring lower layers (less typos)
        let (mut best, mut end) = (UNREACHABLE, None);
        for layer in typo_step..layers {
            for col in 1..cols {
                let score = score_matrix[at(rows - 1, layer, col)];
                if score > best {
                    (best, end) = (score, Some((layer, col)));
                }
            }
        }
        let Some((mut layer, mut col)) = end else {
            return (best, Vec::new());
        };

        // Perform alignment and store indices
        let mut indices: Vec<u32> = Vec::with_capacity(self.needle.len());
        let (mut row, mut in_gap) = (rows - 1, false);
        while row > 0 && col > 0 {
            let step = direction_matrix[at(row, layer, col)];
            // Left: along the gap to the cell it left from
            if step & if in_gap { LEFT_CONTINUES } else { LEFT } != 0 {
                (in_gap, col) = (true, col - 1);
                continue;
            }
            in_gap = false;

            let up = step & UP != 0;
            let start = starts[col - 1];
            if !up && self.matches(row, start).0 {
                let start = (self.haystack_start_pos + start) as u32;
                let char_len = self.needle_unicode.map_or(1, |needle| needle[row - 1].len);
                indices.extend((start..start + char_len as u32).rev());
            } else {
                // Typo: needle char was skipped or mismatched
                layer -= typo_step;
            }
            row -= 1;
            if !up {
                col -= 1;
            }
        }

        // Alignment starts from the end, so reverse the indices such that it's
        // in ascending order
        indices.reverse();

        (best, indices)
    }
}

/// A reachable score after a move worth `delta`, floored at zero
fn advance(score: i32, delta: i32) -> i32 {
    if score == UNREACHABLE {
        return UNREACHABLE;
    }
    (score + delta).max(0)
}

/// The bonus for a match on byte `c` after byte `prev`, aside from the
/// matching case (see `score_haystack`)
fn byte_bonus(prev: u8, c: u8, scoring: &Scoring) -> i32 {
    // Any ASCII byte that isn't a letter or a digit
    let is_delimiter = |c: u8| c.is_ascii() && !c.is_ascii_alphanumeric();
    let mut bonus = 0;
    if is_delimiter(prev) && !is_delimiter(c) {
        bonus += scoring.delimiter_bonus as i32;
    }
    if prev.is_ascii_lowercase() && c.is_ascii_uppercase() {
        bonus += scoring.capitalization_bonus as i32;
    }
    bonus
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smith_waterman::backend::BackendScalar8;
    use alloc::string::String;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    fn random_string(rng: &mut SmallRng, max_len: usize, alphabet: &[&str]) -> String {
        (0..rng.random_range(1..=max_len))
            .map(|_| alphabet[rng.random_range(0..alphabet.len())])
            .collect()
    }

    /// Without a budget the alignment scores as the kernels do, so a wrong
    /// move here shows up as a different score. With one, it stays within it
    #[test]
    fn alignment_matches_the_kernel_score_and_the_budget() {
        let mut rng = SmallRng::seed_from_u64(42);
        for case in 0..if cfg!(miri) { 20 } else { 4000 } {
            let unicode = case % 2 == 0;
            let alphabet: &[&str] = if unicode {
                &["a", "A", "é", "É", "😀", "_", "x"]
            } else {
                &["a", "b", "A", "B", "_", "/", "x"]
            };
            let needle = random_string(&mut rng, 5, alphabet);
            let haystack = random_string(&mut rng, 40, alphabet);
            let haystack = haystack.as_bytes();
            let start_pos = case % 3;

            let mut matcher =
                SmithWaterman::<BackendScalar8>::new(&needle, &Scoring::default(), case % 5 == 0);
            let score = if unicode {
                matcher.score_haystack_unicode(haystack, start_pos == 0)
            } else {
                matcher.score_haystack(haystack, start_pos == 0)
            };
            let alignment = Aligner {
                needle: needle.as_bytes(),
                needle_unicode: unicode.then_some(matcher.needle_unicode.as_slice()),
                case_sensitive: matcher.case_sensitive,
                scoring: &matcher.scoring,
                haystack,
                haystack_start_pos: start_pos,
            };
            assert_eq!(
                alignment.align(None).0,
                score as i32,
                "{needle:?} on {haystack:?}"
            );

            let rows = needle.chars().count();
            for max_typos in 0..=rows {
                let (score, indices) = alignment.align(Some(max_typos as u16));
                let matched = indices
                    .iter()
                    .filter(|&&index| haystack[index as usize - start_pos] & 0xC0 != 0x80)
                    .count();
                assert!(
                    score == UNREACHABLE || rows - matched <= max_typos,
                    "{needle:?} on {haystack:?} with {max_typos} typos: {indices:?}"
                );
            }
        }
    }
}
