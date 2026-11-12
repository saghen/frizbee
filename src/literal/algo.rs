use super::rank::rare_byte_offsets;
use crate::prefilter::algo::can_overread;
use crate::prefilter::backend::{Backend, BitMaskOps};
use crate::prefilter::{case_byte, case_needle_unicode};
use crate::smith_waterman::byte_bonus;
use crate::{Config, Match, MatchIndices, Matching, Scoring};
use alloc::vec::Vec;

/// Literal matching: exact / prefix / suffix / substring
/// Specialized for one SIMD [`crate::prefilter::backend::Backend`] supporting
/// both ASCII and Unicode Identical scoring to Smith-Waterman
#[derive(Debug, Clone)]
pub(crate) struct LiteralImpl<B: Backend> {
    mode: Matching,
    scoring: Scoring,
    needle_len: usize,
    chunks: Vec<NeedleChunk<B>>,
    /// The part of the score that only depends on the needle: the match
    /// scores, and the delimiter bonuses past each chunk's first byte
    base_score: u64,
    /// The two rarest needle bytes (see [`rare_byte_offsets`]), checked before
    /// the whole needle. `seed_a.offset <= seed_b.offset`
    seed_a: Seed<B>,
    seed_b: Seed<B>,
}

/// The needle byte at `offset` as `(original, opposite-case)`, plain and
/// splatted
#[derive(Debug, Clone)]
struct Seed<B: Backend> {
    offset: usize,
    bytes: (u8, u8),
    splat: (B::Chunk, B::Chunk),
}

impl<B: Backend> Seed<B> {
    #[inline(always)]
    fn hits(&self, byte: u8) -> bool {
        (byte == self.bytes.0) | (byte == self.bytes.1)
    }
}

/// Up to `LANES` needle bytes, compared against a haystack window at once. A
/// scalar (a codepoint on the unicode path, a byte on the ASCII path) is never
/// split between chunks. The masks have one bit per byte
#[derive(Debug, Clone)]
struct NeedleChunk<B: Backend> {
    /// Byte offset in the needle
    offset: usize,
    /// The bytes and their opposite-case bytes
    orig: B::Chunk,
    flipped: B::Chunk,
    /// The chunk's bytes
    bytes: u64,
    /// The bytes continuing a multi-byte scalar
    continuations: u64,
    /// The bytes that are an ASCII letter, and the uppercase ones
    letters: u64,
    upper: u64,
}

impl<B: Backend> LiteralImpl<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub(crate) unsafe fn new(needle: &str, config: &Config) -> Self {
        let case_sensitive = config.casing.respects_case_for(needle);
        // Per needle byte: the original, its opposite case, and whether it
        // starts a scalar
        let bytes: Vec<(u8, u8, bool)> = if config.unicode.respects_unicode_for(needle) {
            case_needle_unicode(needle, case_sensitive)
                .iter()
                .flat_map(|c| (0..c.len).map(move |i| (c.chars[i], c.flipped_chars[i], i == 0)))
                .collect()
        } else {
            needle
                .bytes()
                .map(|byte| (byte, case_byte(byte, case_sensitive).1, true))
                .collect()
        };

        let scoring = &config.scoring;
        let scalars = bytes.iter().filter(|byte| byte.2).count();
        let mut base_score = scalars as u64 * scoring.match_score as u64;
        let mut chunks = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let mut end = (offset + B::LANES).min(bytes.len());
            while end < bytes.len() && !bytes[end].2 {
                end -= 1;
            }
            let chunk = &bytes[offset..end];

            let (mut orig, mut flipped) = ([0u8; 64], [0u8; 64]);
            let (mut continuations, mut letters, mut upper) = (0u64, 0u64, 0u64);
            for (k, &(byte, flipped_byte, starts_scalar)) in chunk.iter().enumerate() {
                (orig[k], flipped[k]) = (byte, flipped_byte);
                continuations |= (!starts_scalar as u64) << k;
                letters |= (byte.is_ascii_alphabetic() as u64) << k;
                upper |= (byte.is_ascii_uppercase() as u64) << k;
            }
            let delimiter_bonuses = chunk
                .windows(2)
                .filter(|pair| is_delimiter(pair[0].0) && !is_delimiter(pair[1].0))
                .count();
            base_score += delimiter_bonuses as u64 * scoring.delimiter_bonus as u64;
            chunks.push(NeedleChunk {
                offset,
                orig: unsafe { B::load(orig.as_ptr()) },
                flipped: unsafe { B::load(flipped.as_ptr()) },
                bytes: u64::first_n(chunk.len()),
                continuations,
                letters,
                upper,
            });
            offset = end;
        }

        // A single-byte needle has no pair, so both seeds collapse onto offset
        // 0
        let (seed_a_off, seed_b_off) = if bytes.len() >= 2 {
            rare_byte_offsets(needle.as_bytes())
        } else {
            (0, 0)
        };
        let seed = |offset: usize| {
            let (orig, flipped, _) = bytes[offset];
            Seed {
                offset,
                bytes: (orig, flipped),
                splat: unsafe { (B::splat(orig), B::splat(flipped)) },
            }
        };

        Self {
            mode: config.matching,
            scoring: config.scoring.clone(),
            needle_len: needle.len(),
            chunks,
            base_score,
            seed_a: seed(seed_a_off),
            seed_b: seed(seed_b_off),
        }
    }

    pub(crate) fn is_available() -> bool {
        B::is_available()
    }

    #[inline(always)]
    pub(super) unsafe fn match_list_impl<H: AsRef<str>>(
        &self,
        haystacks: &[H],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        // One loop for the substring scan and one for the anchored modes, each
        // with its own code alone
        unsafe {
            if self.mode == Matching::Substring {
                self.match_list_mode(true, haystacks, haystack_index_offset, matches)
            } else {
                self.match_list_mode(false, haystacks, haystack_index_offset, matches)
            }
        }
    }

    #[inline(always)]
    unsafe fn match_list_mode<H: AsRef<str>>(
        &self,
        substring: bool,
        haystacks: &[H],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        for (index, haystack) in (haystack_index_offset..).zip(haystacks.iter()) {
            if let Some(m) = unsafe { self.match_one_impl(substring, haystack.as_ref(), index) } {
                matches.push(m);
            }
        }
    }

    #[inline(always)]
    unsafe fn match_one_impl(&self, substring: bool, haystack: &str, index: u32) -> Option<Match> {
        let haystack = haystack.as_bytes();
        let (pos, score) = unsafe { self.find(substring, haystack) }?;
        let exact = pos == 0 && self.needle_len == haystack.len();
        Some(Match {
            index,
            score,
            exact,
            #[cfg(feature = "match_end_col")]
            end_col: (pos + self.needle_len)
                .saturating_sub(1)
                .min(u16::MAX as usize) as u16,
        })
    }

    #[inline(always)]
    pub(super) unsafe fn match_one_indices_impl(
        &self,
        haystack: &str,
        index: u32,
    ) -> Option<MatchIndices> {
        let haystack = haystack.as_bytes();
        let substring = self.mode == Matching::Substring;
        let (pos, score) = unsafe { self.find(substring, haystack) }?;
        let exact = pos == 0 && self.needle_len == haystack.len();
        // Every byte of the matched run is a matched index, but add in reverse
        // order to match the fuzzy matcher implementation
        let indices = (pos..pos + self.needle_len)
            .rev()
            .map(|i| i as u32)
            .collect();
        Some(MatchIndices {
            index,
            score,
            exact,
            indices,
        })
    }

    /// Scores the needle at byte `pos`, or `None` when it doesn't match there.
    /// Each chunk is compared against its haystack window in both cases, and
    /// the compare masks give the match and the bonuses.
    ///
    /// # Safety
    /// The backend's target features must be enabled, and the needle must fit
    /// in the haystack at `pos`.
    #[inline(always)]
    unsafe fn score_at(&self, haystack: &[u8], pos: usize) -> Option<u16> {
        let scoring = &self.scoring;
        let mut score = self.base_score;
        for chunk in &self.chunks {
            let start = pos + chunk.offset;
            let window = unsafe { load::<B>(haystack, start) };
            let orig: u64 = unsafe { B::eq(window, chunk.orig) }.into();
            let flipped: u64 = unsafe { B::eq(window, chunk.flipped) }.into();
            // A scalar matches when all of its bytes match in the same case
            let orig_misses = scalar_starts(!orig & chunk.bytes, chunk.continuations);
            let flipped_misses = scalar_starts(!flipped & chunk.bytes, chunk.continuations);
            if orig_misses & flipped_misses != 0 {
                return None;
            }

            let matching_case = chunk.bytes & !chunk.continuations & !orig_misses;
            // A haystack letter has the needle's case where the original
            // matched, and the opposite case otherwise
            let lower = (orig ^ chunk.upper) & chunk.letters;
            let upper = lower ^ chunk.letters;
            let capitalizations = upper & (lower << 1);
            score += matching_case.count_ones() as u64 * scoring.matching_case_bonus as u64
                + capitalizations.count_ones() as u64 * scoring.capitalization_bonus as u64;

            // The first byte's bonuses depend on the haystack byte before it
            score += if start == 0 {
                scoring.prefix_bonus as u64
            } else {
                byte_bonus(haystack[start - 1], haystack[start], scoring) as u64
            };
        }

        if pos == 0 && self.needle_len == haystack.len() {
            score += scoring.exact_match_bonus as u64;
        }
        Some(score.min(u16::MAX as u64) as u16)
    }

    /// Returns the matched byte position (start) if the haystack matches under
    /// the configured mode as well as the score.
    /// For substring, it checks all positions to find the best-score,
    /// preferring earlier matches when tied.
    #[inline(always)]
    unsafe fn find(&self, substring: bool, haystack: &[u8]) -> Option<(usize, u16)> {
        let needle_len = self.needle_len;
        if haystack.len() < needle_len {
            return None;
        }
        if substring {
            return unsafe { self.find_substring(haystack) };
        }

        // The anchored modes have one position
        let pos = match self.mode {
            Matching::Exact if haystack.len() != needle_len => return None,
            Matching::Suffix => haystack.len() - needle_len,
            _ => 0,
        };
        // The seeds reject most haystacks without loading a window
        if !(self.seed_a.hits(haystack[pos + self.seed_a.offset])
            & self.seed_b.hits(haystack[pos + self.seed_b.offset]))
        {
            return None;
        }
        Some((pos, unsafe { self.score_at(haystack, pos) }?))
    }

    /// Two-byte SIMD prefilter (similar to `memchr::memmem`)
    ///
    /// Scan the string for both seed bytes (the two rarest bytes from the
    /// needle), and on a match, compare the whole needle.
    #[inline(always)]
    unsafe fn find_substring(&self, haystack: &[u8]) -> Option<(usize, u16)> {
        let len = haystack.len();
        let needle_len = self.needle_len;
        debug_assert!(needle_len > 0, "empty needles are handled by the caller");
        let last_start = len - needle_len + 1;

        let mut best: Option<(usize, u16)> = None;

        let mut start = 0usize;
        while start < last_start {
            // Mask out any over-read lanes and any lanes that can't possibly
            // match the needle
            let usable = last_start - start;
            let valid = if usable >= B::LANES {
                B::Mask::all()
            } else {
                B::Mask::first_n(usable)
            };

            // Seed window lane `k` corresponds to position `pos = start + k`:
            // it reads `haystack[start + seed_off + k] = haystack[pos +
            // seed_off]`.
            let chunk_a = unsafe { load::<B>(haystack, start + self.seed_a.offset) };
            let hits_a = unsafe { B::occ(chunk_a, self.seed_a.splat) }.and(valid);
            // Skip the second load entirely when the first seed matches nowhere
            // in this window.
            if hits_a.is_zero() {
                start += B::LANES;
                continue;
            }

            // A single-byte needle has only one seed (both offsets will be 0),
            // so skip Branch predictor will predict this perfectly, so there's
            // no performance cost
            let mut hits = if self.seed_a.offset != self.seed_b.offset {
                let chunk_b = unsafe { load::<B>(haystack, start + self.seed_b.offset) };
                hits_a.and(unsafe { B::occ(chunk_b, self.seed_b.splat) })
            } else {
                hits_a
            };
            while !hits.is_zero() {
                let pos = start + unsafe { B::first_hit_pos(hits) };
                hits = hits.clear_through_lowest(hits);
                // The seeds hit, the rest of the needle may not
                if let Some(score) = unsafe { self.score_at(haystack, pos) }
                    && best.is_none_or(|(_, best_score)| score > best_score)
                {
                    best = Some((pos, score));
                }
            }
            start += B::LANES;
        }
        best
    }
}

/// Loads the `LANES` bytes at `start`. The ones past the haystack are
/// unspecified
///
/// # Safety
/// The backend's target features must be enabled, and `start` must be inside
/// the haystack.
#[inline(always)]
unsafe fn load<B: Backend>(haystack: &[u8], start: usize) -> B::Chunk {
    unsafe {
        let ptr = haystack.as_ptr().add(start);
        let remaining = haystack.len() - start;
        // One rarely taken branch: haystack lengths vary around `LANES`, so
        // `||` would mispredict
        if (remaining >= B::LANES) | can_overread(ptr, B::LANES) {
            B::load(ptr)
        } else {
            B::load_partial(ptr, remaining, B::Mask::first_n(remaining))
        }
    }
}

/// Moves the bits on a scalar's continuation bytes down to its first byte
#[inline(always)]
fn scalar_starts(mut bits: u64, continuations: u64) -> u64 {
    while bits & continuations != 0 {
        bits = (bits & !continuations) | ((bits & continuations) >> 1);
    }
    bits
}

#[inline(always)]
fn is_delimiter(byte: u8) -> bool {
    byte <= 127 && !byte.is_ascii_alphanumeric()
}
