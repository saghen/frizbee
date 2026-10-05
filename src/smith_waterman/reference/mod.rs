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

use super::score_fits_in_u8;
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
