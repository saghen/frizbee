//! Traces the optimal alignment back through the stored score matrix. Every
//! step reproduces the cell's score exactly from the diagonal, the cell above
//! or an earlier cell of the row (one always does: the cell is their max), so
//! the path never loses alignment and its matches score the reported score.

use super::{
    SmithWaterman,
    backend::{Backend, ScoreVec},
};
use crate::{Scoring, prefilter::UnicodeChar};
use alloc::vec::Vec;

impl<B: Backend> SmithWaterman<B> {
    /// The haystack byte indices (offset by `haystack_start_pos`, in reverse)
    /// of the chars matched by the optimal alignment scoring `score`, from the
    /// score matrix of the last `score_haystack*_matrix` call on `haystack`
    #[inline(always)]
    pub(crate) fn alignment_indices(
        &self,
        haystack: &[u8],
        haystack_start_pos: usize,
        unicode: bool,
        score: u16,
    ) -> Vec<u32> {
        // Rows of `stride` chunks
        let scores = self.score_matrix.as_byte_slice();
        let stride = self.score_matrix.haystack_chunks;
        let needle_unicode = unicode.then_some(self.needle_unicode.as_slice());
        let rows = needle_unicode.map_or(self.needle.len(), <[_]>::len);
        // The match ends at the first cell of the last row holding the score
        let Some(end) = (1..haystack.len().div_ceil(B::LANES) + 1).find_map(|chunk| {
            let offset = (rows * stride + chunk) * B::LANES * B::LANE_BYTES;
            debug_assert!(offset + B::LANES * B::LANE_BYTES <= scores.len());
            // SAFETY: inside the stored rows
            let lane = unsafe {
                scores
                    .as_ptr()
                    .add(offset)
                    .cast::<B::Score>()
                    .read_unaligned()
                    .find_lane(score)
            };
            (lane != B::LANES).then_some((chunk - 1) * B::LANES + lane)
        }) else {
            debug_assert!(score == 0, "score {score} not in the last row");
            return Vec::new();
        };
        Traceback {
            scores,
            row_lanes: stride * B::LANES,
            lanes: B::LANES,
            wide: B::LANE_BYTES == 2,
            needle: self.needle.as_bytes(),
            needle_unicode,
            case_sensitive: self.case_sensitive,
            scoring: &self.scoring,
            haystack,
            haystack_start_pos,
        }
        .indices(rows, end)
    }
}

/// The stored score matrix of one scored haystack and what its scores were made
/// of, without the backend type so the traceback is built once. The
/// haystack's byte `col` is lane `col + lanes`, after the leading zero chunk
struct Traceback<'a> {
    scores: &'a [u8],
    /// Lanes per row: `haystack_chunks * LANES`
    row_lanes: usize,
    lanes: usize,
    /// u16 scores, rather than u8
    wide: bool,
    needle: &'a [u8],
    /// The rows in Unicode mode, where only scalars' lead bytes are columns
    needle_unicode: Option<&'a [UnicodeChar]>,
    case_sensitive: bool,
    scoring: &'a Scoring,
    haystack: &'a [u8],
    haystack_start_pos: usize,
}

impl Traceback<'_> {
    /// The indices of the alignment ending at the cell of `row` and haystack
    /// byte `end` (see [`SmithWaterman::alignment_indices`])
    #[inline(never)]
    fn indices(&self, row: usize, end: usize) -> Vec<u32> {
        if self.needle_unicode.is_some() {
            self.walk::<true>(row, end)
        } else {
            self.walk::<false>(row, end)
        }
    }

    /// Whether the needle char of `row` (from 1) matches at haystack byte
    /// `col`, holding `h`, and whether in its exact case, as the kernels
    /// compare them
    #[inline(always)]
    fn matches<const UNICODE: bool>(&self, row: usize, col: usize, h: u8) -> (bool, bool) {
        if UNICODE {
            let c = &self.needle_unicode.unwrap_or_default()[row - 1];
            let haystack = &self.haystack[col..];
            if haystack.len() < c.len {
                return (false, false);
            }
            // At most 4 bytes, compared inline rather than by `memcmp`
            let starts_with = |chars: &[u8; 4]| (0..c.len).all(|i| haystack[i] == chars[i]);
            let exact = starts_with(&c.chars);
            let flipped = c.has_flip && starts_with(&c.flipped_chars);
            (exact || flipped, exact)
        } else {
            let n = self.needle[row - 1];
            let flip = if self.case_sensitive { 0 } else { 0x20 };
            let flipped = h ^ flip == n && CLASSES[h as usize] & LETTER != 0;
            (h == n || flipped, h == n)
        }
    }

    /// See the module docs
    #[inline(always)]
    fn walk<const UNICODE: bool>(&self, mut row: usize, end: usize) -> Vec<u32> {
        let (scoring, lanes, stride) = (self.scoring, self.lanes, self.row_lanes);
        let extend = scoring.gap_extend_penalty as i32;
        // Gaps from a matched cell pay the rest of the open penalty
        let open = scoring
            .gap_open_penalty
            .saturating_sub(scoring.gap_extend_penalty) as i32;
        let mismatch = scoring.mismatch_penalty as i32;
        let prefix_bonus = if self.haystack_start_pos == 0 {
            scoring.prefix_bonus as i32
        } else {
            0
        };

        let score = |pos: usize| -> i32 {
            debug_assert!((pos + 1) * (self.wide as usize + 1) <= self.scores.len());
            // SAFETY: the walk stays inside the stored rows and lanes
            unsafe {
                if self.wide {
                    let ptr = self.scores.as_ptr().add(2 * pos);
                    ptr.cast::<u16>().read_unaligned() as i32
                } else {
                    *self.scores.get_unchecked(pos) as i32
                }
            }
        };
        let byte = |col: usize| -> u8 {
            debug_assert!(col < self.haystack.len());
            // SAFETY: the walk's columns are inside the haystack
            unsafe { *self.haystack.get_unchecked(col) }
        };
        // The indices fit the needle's length: written through a pointer, the
        // vector stays out of memory in the loop
        let mut indices: Vec<u32> = Vec::with_capacity(self.needle.len());
        let out = indices.as_mut_ptr();
        let mut len = 0;
        // `lane` is the cell's lane in its row (the haystack's byte `col` is
        // lane `col + lanes`) and `pos` its offset in the matrix
        let mut lane = lanes + end;
        let mut pos = row * stride + lane;
        'walk: loop {
            // Row 0 and the zero chunk's lanes before the haystack score zero
            let cell = score(pos);
            if cell == 0 {
                break;
            }
            let col = lane - lanes;
            let h = byte(col);
            // Unicode continuation bytes aren't columns
            if UNICODE && h & 0xC0 == 0x80 {
                lane -= 1;
                pos -= 1;
                continue;
            }

            // Diagonal: a match with its bonuses, or a mismatch
            let diag = score(pos - stride - 1);
            let (is_match, exact_case) = self.matches::<UNICODE>(row, col, h);
            let diag_gain = if is_match {
                let mut gain = scoring.match_score as i32;
                if exact_case {
                    gain += scoring.matching_case_bonus as i32;
                }
                gain + if col == 0 {
                    prefix_bonus
                } else {
                    byte_bonus(byte(col - 1), h, scoring)
                }
            } else {
                -mismatch
            };
            if cell == diag + diag_gain {
                if is_match {
                    let start = (self.haystack_start_pos + col) as u32;
                    let char_len = if UNICODE {
                        self.needle_unicode.unwrap_or_default()[row - 1].len as u32
                    } else {
                        1
                    };
                    for index in (start..start + char_len).rev() {
                        debug_assert!(len < indices.capacity());
                        // SAFETY: within the capacity (see above)
                        unsafe { out.add(len).write(index) };
                        len += 1;
                    }
                }
                row -= 1;
                lane -= 1;
                pos -= stride + 1;
                continue;
            }

            // Up: a needle char skipped, paying the open penalty below a match
            let mut up = score(pos - stride) - extend;
            if row > 1 && self.matches::<UNICODE>(row - 1, col, h).0 {
                up -= open;
            }
            if cell == up {
                row -= 1;
                pos -= stride;
                continue;
            }

            // Left: haystack chars skipped from an earlier column, paying the
            // extend penalty per column and the open penalty at a matched
            // source
            let mut target = cell;
            loop {
                if lane == lanes {
                    break 'walk;
                }
                lane -= 1;
                pos -= 1;
                let col = lane - lanes;
                let h = byte(col);
                if UNICODE && h & 0xC0 == 0x80 {
                    continue;
                }
                target += extend;
                let source = score(pos);
                if source >= target
                    && source
                        == target
                            + if self.matches::<UNICODE>(row, col, h).0 {
                                open
                            } else {
                                0
                            }
                {
                    break;
                }
            }
        }
        // SAFETY: the first `len` entries were written
        unsafe { indices.set_len(len) };
        indices
    }
}

/// [`CLASSES`] bits
const DELIMITER: u8 = 1;
const LOWER: u8 = 2;
const UPPER: u8 = 4;
const LETTER: u8 = LOWER | UPPER;

/// Each byte's class as the kernels see it (see `BytesVec::classes`)
static CLASSES: [u8; 256] = {
    let mut classes = [0; 256];
    let mut b = 0;
    while b < 128 {
        let c = b as u8;
        classes[b] = if c.is_ascii_lowercase() {
            LOWER
        } else if c.is_ascii_uppercase() {
            UPPER
        } else if c.is_ascii_digit() {
            0
        } else {
            DELIMITER
        };
        b += 1;
    }
    classes
};

/// The bonus for a match on byte `c` after byte `prev`, aside from the
/// matching case (see `ChunkBonuses`)
#[inline(always)]
fn byte_bonus(prev: u8, c: u8, scoring: &Scoring) -> i32 {
    let (prev, c) = (CLASSES[prev as usize], CLASSES[c as usize]);
    let mut bonus = 0;
    if prev & DELIMITER != 0 && c & DELIMITER == 0 {
        bonus += scoring.delimiter_bonus as i32;
    }
    if prev & LOWER != 0 && c & UPPER != 0 {
        bonus += scoring.capitalization_bonus as i32;
    }
    bonus
}
