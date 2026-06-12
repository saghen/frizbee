use std::arch::x86_64::*;

use crate::prefilter::case_needle;

/// Highest typo count supported by `match_haystack_with_typos`.
///
/// Exists only because stable Rust can't size arrays with `MAX_TYPOS + 1`
/// (that needs `generic_const_exprs`), so lane state lives in fixed-capacity
/// arrays and only the first `MAX_TYPOS + 1` entries are used. Loops are
/// bounded by the const parameter, so the unused tail costs nothing.
pub const MAX_SUPPORTED_TYPOS: usize = 7;

#[derive(Debug, Clone)]
pub struct PrefilterAVX512 {
    /// (case_a, case_b) broadcast to 512 bits per needle char
    needle_simd: Vec<(__m512i, __m512i)>,
}

impl PrefilterAVX512 {
    /// # Safety
    /// Caller must ensure that AVX-512F + AVX-512BW are available at runtime
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn new(needle: &[u8]) -> Self {
        let needle_simd = case_needle(needle)
            .iter()
            .map(|&(c1, c2)| (_mm512_set1_epi8(c1 as i8), _mm512_set1_epi8(c2 as i8)))
            .collect();
        Self { needle_simd }
    }

    pub fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_extended_feature_info()
            .is_some_and(|info| info.has_avx512f() && info.has_avx512bw())
    }

    /// Checks if the needle is wholly contained in the haystack.
    ///
    /// Returns `(matched, start_pos, end_pos)` where `end_pos` is the
    /// exclusive byte offset just past the rightmost occurrence of the final
    /// needle char in `haystack[start_pos..end_pos]`.
    ///
    /// # Safety
    /// The caller must ensure that AVX-512F + AVX-512BW are available.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        if len == 0 {
            return (true, 0, 0);
        }

        let mut can_skip_chunks = true;
        let mut match_start_pos = 0usize;
        let mut needle_iter = self.needle_simd.iter();
        let mut needle_char = *needle_iter.next().unwrap();

        let mut start = 0usize;
        while start < len {
            let (haystack_chunk, mut haystack_chunk_mask) =
                unsafe { load_window(haystack, start, len) };

            loop {
                let mask_a = _mm512_cmpeq_epi8_mask(needle_char.0, haystack_chunk);
                let mask_b = _mm512_cmpeq_epi8_mask(needle_char.1, haystack_chunk);
                let mask = (mask_a | mask_b) & haystack_chunk_mask;

                if mask == 0 {
                    break;
                }
                // mask ^ (mask - 1) = all bits from 0..=lowest_set_bit
                haystack_chunk_mask &= !(mask ^ mask.wrapping_sub(1));

                if can_skip_chunks {
                    match_start_pos = mask.trailing_zeros() as usize;
                    can_skip_chunks = false;
                }

                if let Some(next_needle_char) = needle_iter.next() {
                    needle_char = *next_needle_char;
                }
                // on the last chunk and no more needle chars, reuse mask
                // for getting last needle char's last position in haystack
                else if start + 64 >= len {
                    return (
                        true,
                        match_start_pos,
                        start + 64 - (mask.leading_zeros() as usize),
                    );
                }
                // not on the last chunk, find the position of the last
                // needle char from the end of the haystack
                else {
                    let last_needle_char = *self.needle_simd.last().unwrap();
                    let end_pos =
                        unsafe { start + find_last_char_pos(last_needle_char, &haystack[start..]) };
                    return (true, match_start_pos, end_pos);
                }
            }

            start += 64;
        }

        (false, match_start_pos, len)
    }

    /// Occurrence mask: positions in `chunk` holding needle char `idx` (either
    /// case variant).
    ///
    /// Cost: 2x vpcmpeqb (p5, lat 3-4) + mask combine. The compares are the only
    /// port-5 uops; everything downstream is GPR work. Latency to a usable GPR
    /// value is ~7c (cmp -> kmov -> or), so issue probes as early as possible
    /// relative to their use. Check codegen: `lo | hi` should become
    /// kmovq+kmovq+or (p0, p0, p0156), not korq+kmovq (korq is another p5 uop on
    /// Intel and p5 is the bottleneck; use `_cvtmask64_u64` to force it if LLVM
    /// picks korq).
    #[inline(always)]
    unsafe fn occ(&self, chunk: __m512i, idx: usize) -> u64 {
        // SAFETY: callers guarantee idx < needle_len
        let pair = unsafe { self.needle_simd.get_unchecked(idx) };
        let lo = unsafe { _mm512_cmpeq_epi8_mask(pair.0, chunk) };
        let hi = unsafe { _mm512_cmpeq_epi8_mask(pair.1, chunk) };
        lo | hi
    }

    /// Cold success epilogue, outlined so the hot loops stay short and dense
    /// (one fewer branch target per return site, better I-cache/BTB behavior).
    #[cold]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn found_1_typo(&self, haystack: &[u8], match_start_pos: usize) -> (bool, usize, usize) {
        debug_assert!(match_start_pos != usize::MAX);
        let end_pos = unsafe { self.find_end_pos_with_typos::<1>(haystack) };
        (true, match_start_pos, end_pos)
    }

    /// 1-typo specialization of `match_haystack_typos::<1>` — identical results,
    /// restructured for throughput. See `match_haystack_typos` for semantics.
    ///
    /// # Safety
    /// AVX-512F + AVX-512BW required (BMI1/BMI2 are implied by every AVX-512 CPU).
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,bmi1,bmi2")]
    pub unsafe fn match_haystack_1_typo(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle_simd.len();
        if needle_len <= 1 {
            return (true, 0, 0);
        }
        if len == 0 {
            return (false, 0, 0);
        }

        // Lane 0: zero deletions spent, needs needle[i0] next.
        // Lane 1: one deletion spent, needs needle[i1] next.
        // Invariant: i1 >= i0 + 1. Established here by pre-spending lane 1's
        // deletion on needle[0] (exactly what the generic code's first deletion
        // edge produces), and preserved by every edge below.
        let mut i0 = 0usize;
        let mut i1 = 1usize;
        let mut match_start_pos = usize::MAX;

        let mut start = 0usize;
        while start < len {
            let (chunk, chunk_mask) = unsafe { load_window(haystack, start, len) };

            // The only probes on the miss path. Independent: 4x vpcmpeqb issue
            // back-to-back on p5, two kmov chains retire in parallel.
            let mut occ0 = unsafe { self.occ(chunk, i0) };
            let mut occ1 = unsafe { self.occ(chunk, i1) };

            // Fast rejection. At chunk entry m0 == m1 == chunk_mask and
            // i1 >= i0 + 1, so no deletion edge can fire; if neither char occurs
            // the fixpoint is the identity. This is the only branch executed per
            // chunk on misses and predicts near-perfectly on long miss runs.
            if (occ0 | occ1) & chunk_mask != 0 {
                let mut m0 = chunk_mask;
                let mut m1 = chunk_mask;
                // one-slot probe cache: occurrence mask of needle[prev_idx],
                // saved when lane 1 advances past it. Catches the case where
                // lane 0 (one char behind) needs that same mask next iteration.
                // Chunk-local: occurrence masks are meaningless across chunks.
                let mut prev_idx = usize::MAX;
                let mut occ_prev = 0u64;

                loop {
                    let mut advanced = false;

                    // -- lane 0 match edge --
                    let hit0 = occ0 & m0;
                    if hit0 != 0 {
                        match_start_pos = match_start_pos.min(start + _tzcnt_u64(hit0) as usize);
                        // consume through the matched byte. blsmsk = bits up to
                        // and including the lowest set bit; `& !` fuses to ANDN.
                        // This 2-uop, 2-cycle chain is the loop-carried
                        // dependency for the lane -- entirely p15/p0156.
                        m0 &= !_blsmsk_u64(hit0);
                        i0 += 1;
                        // i0 == needle_len is unreachable: i0 only reaches
                        // needle_len - 1 here, because the deletion edge below
                        // fires (cand = needle_len > i1) and returns before the
                        // next lane-0 match could complete the needle.
                        debug_assert!(i0 < needle_len);
                        occ0 = if i0 == i1 {
                            occ1 // attached regime: lane 1 already probed this char
                        } else if i0 == prev_idx {
                            occ_prev // lane 1 probed it earlier this chunk
                        } else {
                            unsafe { self.occ(chunk, i0) }
                        };
                        advanced = true;
                    }

                    // -- deletion edge (lane 0 -> lane 1) --
                    // Placed between the match edges so a lane-0 advance cascades
                    // into lane 1 in the *same* iteration (the generic ordering
                    // needs an extra fixpoint iteration for the same transfer).
                    let cand = i0 + 1;
                    if cand > i1 {
                        // invariant => cand == i1 + 1
                        if cand == needle_len {
                            // trailing needle char deleted; needle_len >= 2 means
                            // lane 0 matched at least once, so start_pos is set
                            return unsafe { self.found_1_typo(haystack, match_start_pos) };
                        }
                        i1 = cand;
                        m1 = m0;
                        occ1 = unsafe { self.occ(chunk, i1) };
                        advanced = true;
                    } else if cand == i1 && m0 > m1 {
                        // same progress, strictly earlier position (masks share a
                        // chunk mask and differ only in a cleared low prefix, so
                        // numerically-greater == superset == earlier). Same char,
                        // so occ1 stays valid: mask-only adoption, no probe.
                        m1 = m0;
                        advanced = true;
                    }

                    // -- lane 1 match edge --
                    let hit1 = occ1 & m1;
                    if hit1 != 0 {
                        match_start_pos = match_start_pos.min(start + _tzcnt_u64(hit1) as usize);
                        m1 &= !_blsmsk_u64(hit1);
                        prev_idx = i1; // save: lane 0 may need this char's mask next
                        occ_prev = occ1;
                        i1 += 1;
                        if i1 == needle_len {
                            return unsafe { self.found_1_typo(haystack, match_start_pos) };
                        }
                        occ1 = unsafe { self.occ(chunk, i1) };
                        advanced = true;
                    }

                    if !advanced {
                        break;
                    }
                }
            }

            start += 64;
            // Length pruning, collapsed to one test: i1 >= i0 + 1 implies
            // needle_len - i1 <= (needle_len - i0) - 1, so lane 0 viable
            // (with its one deletion left) implies lane 1 viable. Checking
            // lane 1 alone is exact.
            if start < len && needle_len - i1 > len - start {
                break;
            }
        }

        let reported_start = if match_start_pos == usize::MAX {
            0
        } else {
            match_start_pos
        };
        (false, reported_start, len)
    }

    /// Checks if the needle, with at most `MAX_TYPOS` needle chars deleted, is
    /// wholly contained in the haystack.
    ///
    /// Deletion is a relaxation of substitution: if a needle char is mismatched
    /// in the final alignment, the prefilter can simply delete it. So this never
    /// rejects a haystack that Smith-Waterman could match with `<= MAX_TYPOS`
    /// typos (no false negatives), though it may accept haystacks where the
    /// downstream score ends up poor (false positives are fine for a prefilter).
    ///
    /// Returns `(matched, start_pos, end_pos)` where the bounds are widened to
    /// cover every feasible alignment:
    /// - `start_pos` is the earliest occurrence of any of the first
    ///   `MAX_TYPOS + 1` needle chars (any of them could be the first matched
    ///   char if the chars before it are deleted)
    /// - `end_pos` is just past the rightmost occurrence of any of the last
    ///   `MAX_TYPOS + 1` needle chars, for the symmetric reason
    ///
    /// # Algorithm
    /// One streaming pass over the haystack with `MAX_TYPOS + 1` lanes. Lane
    /// `typos` holds the single dominant state reachable using at most `typos`
    /// deletions: `(needle_idx, position)`, where `needle_idx` counts needle
    /// chars accounted for (matched or deleted) and position is encoded as the
    /// set of still-available bytes in the current chunk. States are compared
    /// lexicographically (more needle chars accounted for wins; ties broken by
    /// earlier haystack position, i.e. a superset mask). Within each chunk we
    /// iterate to a fixpoint, alternating deletion edges (lane `typos` may adopt
    /// lane `typos - 1`'s state with one extra needle char skipped, consuming no
    /// haystack) and match edges (each lane greedily consumes the lowest
    /// available occurrence of its next needle char).
    ///
    /// # Safety
    /// The caller must ensure that AVX-512F + AVX-512BW are available.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn match_haystack_typos<const MAX_TYPOS: usize>(
        &self,
        haystack: &[u8],
    ) -> (bool, usize, usize) {
        const { assert!(MAX_TYPOS <= MAX_SUPPORTED_TYPOS) };

        let len = haystack.len();
        let needle_len = self.needle_simd.len();

        // every needle char can be deleted: matches anything, including ""
        if needle_len <= MAX_TYPOS {
            return (true, 0, 0);
        }
        if len == 0 {
            return (false, 0, 0);
        }

        // lane state; only indices 0..=MAX_TYPOS are used
        let mut needle_idx = [0usize; MAX_SUPPORTED_TYPOS + 1];
        let mut lane_chunk_mask = [0u64; MAX_SUPPORTED_TYPOS + 1];

        let mut match_start_pos = usize::MAX;

        let mut start = 0usize;
        while start < len {
            let (haystack_chunk, haystack_chunk_mask) =
                unsafe { load_window(haystack, start, len) };
            // entering a new chunk, every lane may consume any byte of it
            for typos in 0..=MAX_TYPOS {
                lane_chunk_mask[typos] = haystack_chunk_mask;
            }

            loop {
                let mut advanced = false;

                // deletion edges: lane `typos` may account for one more needle
                // char than lane `typos - 1` at lane `typos - 1`'s haystack
                // position (a deletion consumes no haystack). Ascending order
                // cascades the closure through all lanes in a single pass.
                for typos in 1..=MAX_TYPOS {
                    let candidate_idx = needle_idx[typos - 1] + 1;
                    // lexicographic improvement: strictly more needle chars
                    // accounted for, or the same amount at a strictly earlier
                    // position (masks are the chunk mask with a low prefix
                    // cleared, so superset == numerically greater)
                    if candidate_idx > needle_idx[typos]
                        || (candidate_idx == needle_idx[typos]
                            && lane_chunk_mask[typos - 1] > lane_chunk_mask[typos])
                    {
                        needle_idx[typos] = candidate_idx;
                        lane_chunk_mask[typos] = lane_chunk_mask[typos - 1];
                        advanced = true;
                        // trailing needle chars deleted
                        if candidate_idx == needle_len {
                            let end_pos =
                                unsafe { self.find_end_pos_with_typos::<MAX_TYPOS>(haystack) };
                            return (true, match_start_pos, end_pos);
                        }
                    }
                }

                // match edges: after the deletion pass, lanes sit at strictly
                // increasing needle indices, so each lane compares a distinct
                // needle char against the chunk
                for typos in 0..=MAX_TYPOS {
                    let needle_char = self.needle_simd[needle_idx[typos]];
                    let mask_a = _mm512_cmpeq_epi8_mask(needle_char.0, haystack_chunk);
                    let mask_b = _mm512_cmpeq_epi8_mask(needle_char.1, haystack_chunk);
                    let mask = (mask_a | mask_b) & lane_chunk_mask[typos];
                    if mask == 0 {
                        continue;
                    }
                    // consume through the lowest matching byte
                    lane_chunk_mask[typos] &= !(mask ^ mask.wrapping_sub(1));
                    needle_idx[typos] += 1;
                    advanced = true;
                    match_start_pos = match_start_pos.min(start + mask.trailing_zeros() as usize);
                    if needle_idx[typos] == needle_len {
                        let end_pos =
                            unsafe { self.find_end_pos_with_typos::<MAX_TYPOS>(haystack) };
                        return (true, match_start_pos, end_pos);
                    }
                }

                if !advanced {
                    break;
                }
            }

            start += 64;

            // length pruning: a lane needs `needle_len - needle_idx` more chars,
            // of which at most `MAX_TYPOS - typos` can be deletions; the rest
            // must be matched against remaining haystack bytes
            if start < len {
                let remaining_haystack = len - start;
                let mut any_lane_viable = false;
                for typos in 0..=MAX_TYPOS {
                    let unaccounted_chars = needle_len - needle_idx[typos];
                    let deletions_left = MAX_TYPOS - typos;
                    if unaccounted_chars <= deletions_left + remaining_haystack {
                        any_lane_viable = true;
                        break;
                    }
                }
                if !any_lane_viable {
                    break;
                }
            }
        }

        let reported_start = if match_start_pos == usize::MAX {
            0
        } else {
            match_start_pos
        };
        (false, reported_start, len)
    }

    /// Finds the exclusive end bound for a typo-tolerant match: just past the
    /// rightmost occurrence of any of the last `MAX_TYPOS + 1` needle chars.
    ///
    /// At most `MAX_TYPOS` of the trailing needle chars can be deleted, so every
    /// feasible alignment ends on one of them; the rightmost occurrence of any
    /// of them is therefore a safe (conservative) upper bound. Scans backward
    /// chunk by chunk, so it typically touches only the last chunk.
    ///
    /// Must only be called after a successful match with a non-empty haystack
    /// and `needle_len > MAX_TYPOS`.
    ///
    /// # Safety
    /// The caller must ensure that AVX-512F + AVX-512BW are available.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn find_end_pos_with_typos<const MAX_TYPOS: usize>(&self, haystack: &[u8]) -> usize {
        let len = haystack.len();
        let needle_len = self.needle_simd.len();
        let last_needle_chars = &self.needle_simd[needle_len - 1 - MAX_TYPOS..];

        let mut start = (len - 1) / 64 * 64;
        loop {
            let (haystack_chunk, haystack_chunk_mask) =
                unsafe { load_window(haystack, start, len) };
            let mut mask = 0u64;
            for needle_char in last_needle_chars {
                mask |= _mm512_cmpeq_epi8_mask(needle_char.0, haystack_chunk)
                    | _mm512_cmpeq_epi8_mask(needle_char.1, haystack_chunk);
            }
            mask &= haystack_chunk_mask;
            if mask != 0 {
                return start + 64 - mask.leading_zeros() as usize;
            }
            if start == 0 {
                break;
            }
            start -= 64;
        }
        // unreachable after a successful match, but stay conservative
        len
    }
}

/// Scans `haystack[min_pos..]` backwards for the rightmost occurrence of
/// `last_needle_char` (case-insensitive via the broadcast pair) and returns
/// the exclusive end offset (`pos + 1`) into the original `haystack`.
///
/// The forward pass already proved the `needle_char` can be found somewhere within the provided
/// `haystack` slice.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512BW availability and `min_pos <= haystack.len()`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn find_last_char_pos(needle_char: (__m512i, __m512i), haystack: &[u8]) -> usize {
    let len = haystack.len();
    let mut start = len.saturating_sub(64);
    loop {
        let (haystack_chunk, haystack_chunk_mask) = unsafe { load_window(haystack, start, len) };
        let mask_a = _mm512_cmpeq_epi8_mask(needle_char.0, haystack_chunk);
        let mask_b = _mm512_cmpeq_epi8_mask(needle_char.1, haystack_chunk);
        let mask = (mask_a | mask_b) & haystack_chunk_mask;
        if mask != 0 {
            return start + 64 - (mask.leading_zeros() as usize);
        }
        // looping infinitely is safe, since the caller already ensured that
        // the needle char matches somewhere in the haystack
        start = start.saturating_sub(64);
    }
}

/// Loads a 64-byte window from the haystack, returning the byte offset of the
/// loaded data, a validity mask over the lanes (covers the full register when
/// the load is complete), and the loaded chunk.
///
/// - `start + 64 <= len`: aligned 64-byte load at `start`
/// - `len < 64`: masked load covering only the valid bytes (page-safe)
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512BW availability.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load_window(haystack: &[u8], start: usize, len: usize) -> (__m512i, u64) {
    unsafe {
        if start + 64 <= len {
            (
                _mm512_loadu_si512(haystack.as_ptr().add(start) as *const __m512i),
                u64::MAX,
            )
        } else if can_overread_64(haystack.as_ptr().add(start)) {
            (
                _mm512_loadu_si512(haystack.as_ptr().add(start) as *const __m512i),
                (1u64 << (len - start)) - 1,
            )
        } else {
            // len - start < 64: masked load, zero-fills inactive lanes
            let mask: u64 = (1u64 << (len - start)).wrapping_sub(1);
            (
                _mm512_maskz_loadu_epi8(mask, haystack.as_ptr().add(start) as *const i8),
                u64::MAX,
            )
        }
    }
}

#[inline(always)]
fn can_overread_64(ptr: *const u8) -> bool {
    (ptr as usize & 0xFFF) <= (4096 - 64)
}
