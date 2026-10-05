//! A plain scalar Smith-Waterman, written for readability rather than speed,
//! that defines the score every kernel is meant to compute. The SIMD kernels
//! are checked against it in [`tests`].
//!
//! # Scoring
//!
//! The needle is aligned locally against the haystack, and must be consumed
//! entirely: the score is the best cell of the needle's last row, never below
//! zero. Only the haystack's own chars are columns, so an alignment can't run
//! past its end (e.g. into a kernel's zeroed lanes after the last chunk).
//!
//! In ASCII mode each needle byte is a row and each haystack byte a column. In
//! Unicode mode each needle scalar is a row and each haystack scalar a column;
//! the kernels' UTF-8 continuation bytes are transport lanes that neither score
//! nor cost anything, so they don't appear here at all.
//!
//! Each cell `H[i][j]` is the max of zero and:
//!
//! - **diagonal**: `H[i-1][j-1]` plus, when needle char `i` matches haystack
//!   char `j` (itself or, unless case sensitive, its [`opposite_case`]),
//!   `match_score`, `matching_case_bonus` for the needle's exact case and the
//!   column's bonus, or minus `mismatch_penalty` otherwise. The bonus is
//!   `prefix_bonus` on the haystack's first char (with `include_prefix`),
//!   `delimiter_bonus` on a non-delimiter after a delimiter (an ASCII char
//!   that isn't a letter or digit), or `capitalization_bonus` on an ASCII
//!   uppercase letter after an ASCII lowercase one; non-ASCII chars never
//!   earn or trigger one, as the kernels only look at bytes.
//! - **up** (a needle char skipped): `H[i-1][j]` minus `gap_open_penalty` when
//!   needle char `i - 1` matches haystack char `j`, or `gap_extend_penalty`
//!   otherwise. There is no vertical gap state: every step down pays on its
//!   own, as the kernels do.
//! - **left** (haystack chars skipped): for any earlier column `k`,
//!   `H[i][k] - gap_extend_penalty * (j - k)`, minus another
//!   `gap_open_penalty - gap_extend_penalty` when needle char `i` matches
//!   haystack char `k`. So a gap opened after a matched cell pays the open
//!   penalty once, at its source, and the chars it skips pay the extend penalty
//!   whether or not they match.
//!
//! A `gap_open_penalty` below `gap_extend_penalty` counts as
//! `gap_extend_penalty`: the kernels charge the open penalty as an extra
//! `gap_open_penalty - gap_extend_penalty` (clamped at zero) on top of the
//! extend penalty.
//!
//! The matcher adds `exact_match_bonus` when the haystack equals the needle
//! byte for byte (see [`score_with_exact_bonus`]), the kernels don't.
//!
//! The kernels only run on haystacks up to `MAX_HAYSTACK_LEN` bytes (longer
//! ones are scored greedily), and the u8 kernels only for needles whose every
//! cell fits in a u8. Within those limits a kernel must equal this reference.

use super::{backend::Backend, score_fits_in_u8, u8_scan_ramp_fits};
use crate::Scoring;
use alloc::{string::String, vec, vec::Vec};

mod tests;

/// Scores the haystack's bytes like the ASCII kernels
/// (`SmithWaterman::score_haystack`)
pub(crate) fn score(
    needle: &str,
    haystack: &[u8],
    scoring: &Scoring,
    case_sensitive: bool,
    include_prefix: bool,
) -> u16 {
    let needle: Vec<(u32, u32)> = needle
        .bytes()
        .map(|c| {
            let other = if case_sensitive {
                c
            } else if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else {
                c.to_ascii_lowercase()
            };
            (c as u32, other as u32)
        })
        .collect();
    let haystack: Vec<u32> = haystack.iter().map(|&c| c as u32).collect();
    align(&needle, &haystack, scoring, include_prefix)
}

/// Scores the haystack's scalars like the Unicode kernels
/// (`SmithWaterman::score_haystack_unicode`)
pub(crate) fn score_unicode(
    needle: &str,
    haystack: &str,
    scoring: &Scoring,
    case_sensitive: bool,
    include_prefix: bool,
) -> u16 {
    let needle: Vec<(u32, u32)> = needle
        .chars()
        .map(|c| {
            let other = if case_sensitive { c } else { opposite_case(c) };
            (c as u32, other as u32)
        })
        .collect();
    let haystack: Vec<u32> = haystack.chars().map(|c| c as u32).collect();
    align(&needle, &haystack, scoring, include_prefix)
}

/// The score the matcher reports for a whole haystack: the kernel's score plus
/// the exact match bonus
pub(crate) fn score_with_exact_bonus(
    needle: &str,
    haystack: &str,
    scoring: &Scoring,
    case_sensitive: bool,
    unicode: bool,
) -> u16 {
    let score = if unicode {
        score_unicode(needle, haystack, scoring, case_sensitive, true)
    } else {
        score(needle, haystack.as_bytes(), scoring, case_sensitive, true)
    };
    if needle == haystack {
        score.saturating_add(scoring.exact_match_bonus)
    } else {
        score
    }
}

/// The char a needle char also matches when case insensitive: its single
/// opposite case char of the same UTF-8 length, or itself when there's none
/// (e.g. `ß`, whose uppercase is `SS`)
pub(crate) fn opposite_case(c: char) -> char {
    let other: Vec<char> = if c.is_uppercase() {
        c.to_lowercase().collect()
    } else if c.is_lowercase() {
        c.to_uppercase().collect()
    } else {
        return c;
    };
    match other[..] {
        [o] if o.len_utf8() == c.len_utf8() => o,
        _ => c,
    }
}

/// The DP over needle chars (each with the opposite case char it also matches)
/// and haystack chars, both as code points (bytes in ASCII mode)
fn align(needle: &[(u32, u32)], haystack: &[u32], s: &Scoring, include_prefix: bool) -> u16 {
    // Opening a gap never costs less than extending it, like in the kernels
    let open = s.gap_open_penalty.max(s.gap_extend_penalty) as i32;
    let extend = s.gap_extend_penalty as i32;
    // The cost of a gap's first step away from a cell
    let gap_from = |matched: bool| if matched { open } else { extend };

    let bonuses: Vec<i32> = (0..haystack.len())
        .map(|j| column_bonus(haystack, j, s, include_prefix))
        .collect();

    // Row `i - 1` of the matrix and its match mask, with a zero column 0
    let mut above = vec![0i32; haystack.len() + 1];
    let mut above_matched = vec![false; haystack.len() + 1];
    for &(c, other_case) in needle {
        let mut row = vec![0i32; haystack.len() + 1];
        let mut row_matched = vec![false; haystack.len() + 1];
        // The best horizontal gap reaching the current column
        let mut left = 0i32;
        for j in 1..=haystack.len() {
            let h = haystack[j - 1];
            let exact_case = h == c;
            let matched = exact_case || h == other_case;

            let diag = if matched {
                let case_bonus = if exact_case { s.matching_case_bonus } else { 0 };
                above[j - 1] + s.match_score as i32 + bonuses[j - 1] + case_bonus as i32
            } else {
                above[j - 1] - s.mismatch_penalty as i32
            };
            let up = above[j] - gap_from(above_matched[j]);
            let cell = 0.max(diag).max(up).max(left);

            row[j] = cell;
            row_matched[j] = matched;
            // Extend the gaps reaching this column, or open one from it
            left = (left - extend).max(cell - gap_from(matched));
        }
        above = row;
        above_matched = row_matched;
    }

    let best = above.into_iter().max().unwrap_or(0);
    best.min(u16::MAX as i32) as u16
}

/// The bonus for a match on haystack char `j`, aside from the matching case
fn column_bonus(haystack: &[u32], j: usize, s: &Scoring, include_prefix: bool) -> i32 {
    let is_delimiter = |c: u32| c < 128 && !(c as u8).is_ascii_alphanumeric();
    let is_lower = |c: u32| c < 128 && (c as u8).is_ascii_lowercase();
    let is_upper = |c: u32| c < 128 && (c as u8).is_ascii_uppercase();

    let c = haystack[j];
    let mut bonus = 0;
    if j == 0 {
        if include_prefix {
            bonus += s.prefix_bonus;
        }
    } else {
        let prev = haystack[j - 1];
        if is_delimiter(prev) && !is_delimiter(c) {
            bonus += s.delimiter_bonus;
        }
        if is_lower(prev) && is_upper(c) {
            bonus += s.capitalization_bonus;
        }
    }
    bonus as i32
}

// ---------------------------------------------------------------------------
// Alignment paths
// ---------------------------------------------------------------------------

/// The needle's chars (with the opposite case char each also matches) and the
/// haystack's chars with their byte offsets, as rows and columns of the DP:
/// bytes in ASCII mode, scalars in Unicode mode
fn rows_and_columns(
    needle: &str,
    haystack: &[u8],
    unicode: bool,
    case_sensitive: bool,
) -> (Vec<(u32, u32)>, Vec<u32>, Vec<usize>) {
    if unicode {
        let needle = needle
            .chars()
            .map(|c| {
                let other = if case_sensitive { c } else { opposite_case(c) };
                (c as u32, other as u32)
            })
            .collect();
        let haystack = core::str::from_utf8(haystack).expect("Unicode haystack must be UTF-8");
        let (columns, offsets) = haystack.char_indices().map(|(i, c)| (c as u32, i)).unzip();
        (needle, columns, offsets)
    } else {
        let needle = needle
            .bytes()
            .map(|c| {
                let other = if case_sensitive {
                    c
                } else if c.is_ascii_lowercase() {
                    c.to_ascii_uppercase()
                } else {
                    c.to_ascii_lowercase()
                };
                (c as u32, other as u32)
            })
            .collect();
        let columns = haystack.iter().map(|&c| c as u32).collect();
        (needle, columns, (0..haystack.len()).collect())
    }
}

/// The score a diagonal step onto haystack char `j` adds for needle char `c`,
/// and whether it's a match
fn diagonal(c: (u32, u32), haystack: &[u32], j: usize, s: &Scoring, prefix: bool) -> (i64, bool) {
    let h = haystack[j];
    if h == c.0 || h == c.1 {
        let case_bonus = if h == c.0 { s.matching_case_bonus } else { 0 };
        let gain =
            s.match_score as i64 + column_bonus(haystack, j, s, prefix) as i64 + case_bonus as i64;
        (gain, true)
    } else {
        (-(s.mismatch_penalty as i64), false)
    }
}

/// How a DP cell was reached, see [`optimal_indices`]
#[derive(Clone, Copy, PartialEq)]
enum Step {
    /// The alignment starts here
    Start,
    Diagonal,
    /// A needle char skipped
    Up,
    /// Haystack chars skipped: see the cell's `E` pointer
    Left,
}

/// An optimal alignment by the same DP as [`score`] / [`score_unicode`], with
/// each cell's best predecessor recorded and traced back from the first
/// column of the needle's last row holding the score (like the kernels):
/// `H[i][j]` is the best alignment ending at needle char `i` and haystack char
/// `j`, `E[i][j]` the best one ending in a horizontal gap that skips haystack
/// char `j`, and an up step `H[i - 1][j]` minus its cost (there's no vertical
/// gap state, see the module docs). Returns the score and the haystack byte
/// offsets of the matched chars, in reverse like the kernels' indices.
pub(crate) fn optimal_indices(
    needle: &str,
    haystack: &[u8],
    s: &Scoring,
    case_sensitive: bool,
    include_prefix: bool,
    unicode: bool,
) -> (u16, Vec<u32>) {
    let (needle, columns, offsets) = rows_and_columns(needle, haystack, unicode, case_sensitive);
    let (n, m) = (needle.len(), columns.len());
    let open = s.gap_open_penalty.max(s.gap_extend_penalty) as i64;
    let extend = s.gap_extend_penalty as i64;
    let matches = |i: usize, j: usize| diagonal(needle[i - 1], &columns, j - 1, s, false).1;
    let gap_from = |i: usize, j: usize| {
        if i > 0 && j > 0 && matches(i, j) {
            open
        } else {
            extend
        }
    };

    // Row and column 0 are the empty alignment
    let mut h = vec![vec![0i64; m + 1]; n + 1];
    let mut e = vec![vec![i64::MIN / 4; m + 1]; n + 1];
    let mut step = vec![vec![Step::Start; m + 1]; n + 1];
    // Whether `E[i][j]` extends `E[i][j - 1]` rather than opening from `H[i][j - 1]`
    let mut e_extends = vec![vec![false; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            let (gain, _) = diagonal(needle[i - 1], &columns, j - 1, s, include_prefix);
            let (opened, extended) = (h[i][j - 1] - gap_from(i, j - 1), e[i][j - 1] - extend);
            e[i][j] = opened.max(extended);
            e_extends[i][j] = extended > opened;

            let candidates = [
                (h[i - 1][j - 1] + gain, Step::Diagonal),
                (h[i - 1][j] - gap_from(i - 1, j), Step::Up),
                (e[i][j], Step::Left),
            ];
            for (value, from) in candidates {
                if value > h[i][j] {
                    h[i][j] = value;
                    step[i][j] = from;
                }
            }
        }
    }

    let score = h[n].iter().copied().max().unwrap_or(0);
    let mut j = h[n].iter().position(|&v| v == score).unwrap_or(0);
    let mut i = n;
    let mut indices = Vec::new();
    while i > 0 && j > 0 && h[i][j] > 0 {
        match step[i][j] {
            Step::Start => break,
            Step::Diagonal => {
                if matches(i, j) {
                    let end = offsets.get(j).copied().unwrap_or(haystack.len());
                    indices.extend((offsets[j - 1]..end).rev().map(|b| b as u32));
                }
                i -= 1;
                j -= 1;
            }
            Step::Up => i -= 1,
            Step::Left => {
                while e_extends[i][j] {
                    j -= 1;
                }
                j -= 1;
            }
        }
    }
    (score.min(u16::MAX as i64) as u16, indices)
}

/// The best score of an alignment whose matched haystack chars are exactly
/// `indices` (byte offsets in reverse, like the kernels' indices), by the same
/// DP as [`score`] / [`score_unicode`] restricted to such alignments. An error
/// describes indices that aren't strictly decreasing whole chars of the
/// haystack, or that no alignment matches.
///
/// An alignment is a path of diagonal (match or mismatch), up and left steps
/// from a start cell to the needle's last row, scoring the sum of its steps.
/// Every column `j` in `indices` must be entered by a diagonal match, and it
/// can't be skipped by a left gap, entered by a mismatch, or precede the
/// start. Every other column the path crosses is a mismatch or skipped.
/// Unlike the DP, the restricted path doesn't restart at zero, so a valid
/// traceback (one stopping where the DP restarted) scores exactly the DP's
/// score, and an incomplete one (stopping early) or a wrong one less.
pub(crate) fn indices_score(
    needle: &str,
    haystack: &[u8],
    s: &Scoring,
    case_sensitive: bool,
    include_prefix: bool,
    unicode: bool,
    indices: &[u32],
) -> Result<i64, String> {
    use alloc::format;

    let (needle, columns, offsets) = rows_and_columns(needle, haystack, unicode, case_sensitive);
    let (n, m) = (needle.len(), columns.len());
    if !indices.windows(2).all(|w| w[0] > w[1]) {
        return Err(format!("indices {indices:?} aren't strictly decreasing"));
    }
    // Columns (1-based like the DP's) of the matched chars, each of whose
    // bytes must all be in `indices`
    let mut matched = vec![false; m + 1];
    let mut bytes = indices.iter().rev().map(|&b| b as usize).peekable();
    for (j, &start) in offsets.iter().enumerate() {
        let end = offsets.get(j + 1).copied().unwrap_or(haystack.len());
        let count = (start..end)
            .filter(|b| bytes.next_if_eq(b).is_some())
            .count();
        if count != 0 && count != end - start {
            return Err(format!(
                "indices {indices:?} split the char at byte {start}"
            ));
        }
        matched[j + 1] = count != 0;
    }
    if let Some(b) = bytes.next() {
        return Err(format!("index {b} is past the haystack's end"));
    }
    let first = matched.iter().position(|&c| c).unwrap_or(m + 1);
    let last = matched.iter().rposition(|&c| c).unwrap_or(0);

    let open = s.gap_open_penalty.max(s.gap_extend_penalty) as i64;
    let extend = s.gap_extend_penalty as i64;
    let matches = |i: usize, j: usize| diagonal(needle[i - 1], &columns, j - 1, s, false).1;
    let gap_from = |i: usize, j: usize| {
        if i > 0 && j > 0 && matches(i, j) {
            open
        } else {
            extend
        }
    };
    const NONE: i64 = i64::MIN / 4;
    // The path may start left of the first matched column, at any row
    let start = |j: usize| if j < first { 0 } else { NONE };

    let mut above: Vec<i64> = (0..=m).map(start).collect();
    for i in 1..=n {
        let mut row = vec![NONE; m + 1];
        row[0] = 0;
        // The best path ending in a left gap that skips the current column
        let mut left = NONE;
        for j in 1..=m {
            let (gain, is_match) = diagonal(needle[i - 1], &columns, j - 1, s, include_prefix);
            let diag = if is_match == matched[j] {
                above[j - 1] + gain
            } else {
                NONE
            };
            let up = above[j] - gap_from(i - 1, j);
            let skipped = if matched[j] { NONE } else { left };
            row[j] = diag.max(up).max(skipped).max(start(j)).max(NONE);
            let opened = row[j] - gap_from(i, j);
            left = if matched[j] {
                opened
            } else {
                opened.max(left - extend)
            }
            .max(NONE);
        }
        above = row;
    }
    match above[last.max(1).min(m)..].iter().copied().max() {
        Some(best) if best > NONE / 2 => Ok(best),
        _ if n == 0 || m == 0 => Ok(0),
        _ => Err(format!("no alignment matches exactly {indices:?}")),
    }
}

/// xorshift64*, so the randomized tests reproduce from their seed
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub(crate) fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }

    pub(crate) fn string(&mut self, alphabet: &[char], len: usize) -> String {
        (0..len).map(|_| self.pick(alphabet)).collect()
    }
}

/// [`score_fits_in_u8`] for backend `B`'s u8 kernel, whose ASCII (not
/// `unicode`) scan may also need its ramp's headroom (see
/// [`Backend::ASCII_SCAN_RAMP`])
pub(crate) fn u8_kernel_fits<B: Backend>(rows: usize, scoring: &Scoring, unicode: bool) -> bool {
    score_fits_in_u8(rows, scoring)
        && (unicode
            || B::ASCII_SCAN_RAMP == 0
            || u8_scan_ramp_fits(rows, scoring, B::ASCII_SCAN_RAMP))
}
