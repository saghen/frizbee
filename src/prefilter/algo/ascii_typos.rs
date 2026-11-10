use super::{PathState, Prefilter, load_window};
use crate::prefilter::backend::{Backend, BitMaskOps};

impl<B: Backend> Prefilter<B> {
    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub unsafe fn match_haystack_many_typos(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> (bool, usize, usize) {
        unsafe { self.match_haystack_many_typos_impl(haystack, max_typos as usize) }
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub unsafe fn match_haystack_1_typo(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle_ascii.len();
        if needle_len <= 1 {
            return (true, 0, len);
        }
        if len == 0 {
            return (false, 0, 0);
        }

        let mut first_path_needle_idx = 0usize;
        let mut second_path_needle_idx = 1usize;
        let mut match_start_pos = usize::MAX;

        for start in (0..len).step_by(B::LANES) {
            // compare needle chars of both paths against the current chunk
            let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
            let mut first_path_mask =
                unsafe { B::occ(chunk, self.needle_unchecked(first_path_needle_idx)) };
            let mut second_path_mask =
                unsafe { B::occ(chunk, self.needle_unchecked(second_path_needle_idx)) };

            let mut first_path_chunk_mask = chunk_mask;
            let mut second_path_chunk_mask = chunk_mask;

            loop {
                let mut advanced = false;

                let candidate_needle_idx = first_path_needle_idx + 1;
                if candidate_needle_idx > second_path_needle_idx {
                    // first path is on the second last char
                    // and since this path allows a typo, we've matched
                    if candidate_needle_idx == needle_len {
                        return unsafe { self.found_with_typos(haystack, match_start_pos, 1) };
                    }

                    // first path caught up to or passed the second path
                    // skip the next needle char
                    second_path_needle_idx = candidate_needle_idx;
                    second_path_chunk_mask = first_path_chunk_mask;
                    second_path_mask =
                        unsafe { B::occ(chunk, self.needle_unchecked(second_path_needle_idx)) };
                } else if candidate_needle_idx == second_path_needle_idx
                    && first_path_chunk_mask > second_path_chunk_mask
                {
                    second_path_chunk_mask = first_path_chunk_mask;
                }

                // match first path against current chunk
                let first_path_matches = first_path_mask.and(first_path_chunk_mask);
                if !first_path_matches.is_zero() {
                    match_start_pos = match_start_pos
                        .min(start + unsafe { B::first_hit_pos(first_path_matches) });

                    first_path_needle_idx += 1;

                    first_path_chunk_mask = unsafe {
                        B::clear_through_lowest(first_path_chunk_mask, first_path_matches)
                    };
                    first_path_mask =
                        unsafe { B::occ(chunk, self.needle_unchecked(first_path_needle_idx)) };
                    advanced = true;
                }

                // match second path against current chunk
                let second_path_matches = second_path_mask.and(second_path_chunk_mask);
                if !second_path_matches.is_zero() {
                    match_start_pos = match_start_pos
                        .min(start + unsafe { B::first_hit_pos(second_path_matches) });

                    second_path_needle_idx += 1;
                    if second_path_needle_idx >= needle_len {
                        return unsafe { self.found_with_typos(haystack, match_start_pos, 1) };
                    }

                    second_path_chunk_mask = unsafe {
                        B::clear_through_lowest(second_path_chunk_mask, second_path_matches)
                    };
                    second_path_mask =
                        unsafe { B::occ(chunk, self.needle_unchecked(second_path_needle_idx)) };
                    advanced = true;
                }

                if !advanced {
                    break;
                }
            }
        }

        let match_start_pos = if match_start_pos == usize::MAX {
            0
        } else {
            match_start_pos
        };
        (false, match_start_pos, len)
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub unsafe fn match_haystack_2_typos(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle_ascii.len();
        if needle_len <= 2 {
            return (true, 0, len);
        }
        if len == 0 {
            return (false, 0, 0);
        }

        let mut first_path_needle_idx = 0usize;
        let mut second_path_needle_idx = 1usize;
        let mut third_path_needle_idx = 2usize;
        let mut match_start_pos = usize::MAX;

        for start in (0..len).step_by(B::LANES) {
            // compare needle chars of all paths against the current chunk
            let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
            let mut first_path_mask =
                unsafe { B::occ(chunk, self.needle_unchecked(first_path_needle_idx)) };
            let mut second_path_mask =
                unsafe { B::occ(chunk, self.needle_unchecked(second_path_needle_idx)) };
            let mut third_path_mask =
                unsafe { B::occ(chunk, self.needle_unchecked(third_path_needle_idx)) };

            let mut first_path_chunk_mask = chunk_mask;
            let mut second_path_chunk_mask = chunk_mask;
            let mut third_path_chunk_mask = chunk_mask;

            loop {
                let mut advanced = false;

                let second_path_candidate_needle_idx = first_path_needle_idx + 1;
                if second_path_candidate_needle_idx > second_path_needle_idx {
                    // first path is on the second last char
                    // and since this path allows typos, we've matched
                    if second_path_candidate_needle_idx == needle_len {
                        return unsafe { self.found_with_typos(haystack, match_start_pos, 2) };
                    }

                    // first path caught up to or passed the second path
                    // skip the next needle char
                    second_path_needle_idx = second_path_candidate_needle_idx;
                    second_path_chunk_mask = first_path_chunk_mask;
                    second_path_mask =
                        unsafe { B::occ(chunk, self.needle_unchecked(second_path_needle_idx)) };
                } else if second_path_candidate_needle_idx == second_path_needle_idx
                    && first_path_chunk_mask > second_path_chunk_mask
                {
                    second_path_chunk_mask = first_path_chunk_mask;
                }

                let third_path_candidate_needle_idx = second_path_needle_idx + 1;
                if third_path_candidate_needle_idx > third_path_needle_idx {
                    // second path is on the second last char
                    // and since this path allows typos, we've matched
                    if third_path_candidate_needle_idx == needle_len {
                        return unsafe { self.found_with_typos(haystack, match_start_pos, 2) };
                    }

                    // second path caught up to or passed the third path
                    // skip the next needle char
                    third_path_needle_idx = third_path_candidate_needle_idx;
                    third_path_chunk_mask = second_path_chunk_mask;
                    third_path_mask =
                        unsafe { B::occ(chunk, self.needle_unchecked(third_path_needle_idx)) };
                } else if third_path_candidate_needle_idx == third_path_needle_idx
                    && second_path_chunk_mask > third_path_chunk_mask
                {
                    third_path_chunk_mask = second_path_chunk_mask;
                }

                // match first path against current chunk
                let first_path_matches = first_path_mask.and(first_path_chunk_mask);
                if !first_path_matches.is_zero() {
                    match_start_pos = match_start_pos
                        .min(start + unsafe { B::first_hit_pos(first_path_matches) });

                    first_path_needle_idx += 1;

                    first_path_chunk_mask = unsafe {
                        B::clear_through_lowest(first_path_chunk_mask, first_path_matches)
                    };
                    first_path_mask =
                        unsafe { B::occ(chunk, self.needle_unchecked(first_path_needle_idx)) };
                    advanced = true;
                }

                // match second path against current chunk
                let second_path_matches = second_path_mask.and(second_path_chunk_mask);
                if !second_path_matches.is_zero() {
                    match_start_pos = match_start_pos
                        .min(start + unsafe { B::first_hit_pos(second_path_matches) });

                    second_path_needle_idx += 1;
                    if second_path_needle_idx >= needle_len {
                        return unsafe { self.found_with_typos(haystack, match_start_pos, 2) };
                    }

                    second_path_chunk_mask = unsafe {
                        B::clear_through_lowest(second_path_chunk_mask, second_path_matches)
                    };
                    second_path_mask =
                        unsafe { B::occ(chunk, self.needle_unchecked(second_path_needle_idx)) };
                    advanced = true;
                }

                // match third path against current chunk
                let third_path_matches = third_path_mask.and(third_path_chunk_mask);
                if !third_path_matches.is_zero() {
                    match_start_pos = match_start_pos
                        .min(start + unsafe { B::first_hit_pos(third_path_matches) });

                    third_path_needle_idx += 1;
                    if third_path_needle_idx >= needle_len {
                        return unsafe { self.found_with_typos(haystack, match_start_pos, 2) };
                    }

                    third_path_chunk_mask = unsafe {
                        B::clear_through_lowest(third_path_chunk_mask, third_path_matches)
                    };
                    third_path_mask =
                        unsafe { B::occ(chunk, self.needle_unchecked(third_path_needle_idx)) };
                    advanced = true;
                }

                if !advanced {
                    break;
                }
            }
        }

        let match_start_pos = if match_start_pos == usize::MAX {
            0
        } else {
            match_start_pos
        };
        (false, match_start_pos, len)
    }

    #[inline(always)]
    unsafe fn match_haystack_many_typos_impl(
        &mut self,
        haystack: &[u8],
        max_typos: usize,
    ) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle_ascii.len();
        if needle_len <= max_typos {
            return (true, 0, len);
        }
        if len == 0 {
            return (false, 0, 0);
        }
        if !self.lcs_masks.is_empty() {
            return unsafe { self.match_haystack_lcs(haystack, max_typos) };
        }

        let path_count = max_typos + 1;
        self.paths.resize(
            path_count,
            PathState {
                needle_idx: 0,
                needle_mask: B::Mask::zero(),
            },
        );
        for path in &mut self.paths {
            path.needle_idx = 0;
            path.needle_mask = B::Mask::zero();
        }

        let paths = self.paths.as_mut_ptr();
        let needle = self.needle_ascii.as_slice();
        let mut match_start_pos = usize::MAX;

        for start in (0..len).step_by(B::LANES) {
            let (chunk, mut chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
            // compare needle chars of all paths against the current chunk
            for path_idx in 0..path_count {
                unsafe {
                    let path = paths.add(path_idx);
                    (*path).needle_mask = B::occ(chunk, *needle.get_unchecked((*path).needle_idx));
                }
            }

            loop {
                for path_idx in 1..path_count {
                    unsafe {
                        let prev = *paths.add(path_idx - 1);
                        let path = paths.add(path_idx);
                        let candidate_needle_idx = prev.needle_idx + 1;
                        if candidate_needle_idx > (*path).needle_idx {
                            // previous path is on the second last char
                            // and since this path allows typos, we've matched
                            if candidate_needle_idx == needle_len {
                                return self.found_with_typos(haystack, match_start_pos, max_typos);
                            }

                            // previous path caught up to or passed this path
                            // skip the next needle char
                            (*path).needle_idx = candidate_needle_idx;
                            (*path).needle_mask =
                                B::occ(chunk, *needle.get_unchecked(candidate_needle_idx));
                        }
                    }
                }

                // match all paths against current chunk
                let mut match_mask = B::Mask::zero();
                for path_idx in 0..path_count {
                    unsafe {
                        match_mask = match_mask.or((*paths.add(path_idx)).needle_mask);
                    }
                }
                let matches = match_mask.and(chunk_mask);
                if matches.is_zero() {
                    break;
                }

                let hit_pos = unsafe { B::first_hit_pos(matches) };
                let hit = matches.and(B::Mask::first_n(hit_pos + 1));
                match_start_pos = match_start_pos.min(start + hit_pos);

                // advance every path that matched the first available hit
                for path_idx in 0..path_count {
                    unsafe {
                        let path = paths.add(path_idx);
                        if (*path).needle_mask.and(hit).is_zero() {
                            continue;
                        }

                        (*path).needle_idx += 1;
                        if (*path).needle_idx == needle_len {
                            return self.found_with_typos(haystack, match_start_pos, max_typos);
                        }
                        (*path).needle_mask =
                            B::occ(chunk, *needle.get_unchecked((*path).needle_idx));
                    }
                }

                chunk_mask = unsafe { B::clear_through_lowest(chunk_mask, hit) };
            }
        }

        let match_start_pos = if match_start_pos == usize::MAX {
            0
        } else {
            match_start_pos
        };
        (false, match_start_pos, len)
    }

    // Bit-parallel LCS (Hyyro): at most `max_typos` needle bytes may be deleted
    // iff LCS(needle, haystack) >= needle_len - max_typos
    #[inline(always)]
    unsafe fn match_haystack_lcs(&self, haystack: &[u8], max_typos: usize) -> (bool, usize, usize) {
        let mut rows = [!0u64];
        unsafe {
            lcs_advance::<1>(
                self.lcs_masks.as_ptr(),
                &[haystack],
                &mut rows,
                0,
                haystack.len(),
            );
            self.lcs_window(haystack, rows[0], max_typos)
        }
    }

    #[inline(always)]
    pub fn many_typos_x4_available(&self, max_typos: u16) -> bool {
        !self.lcs_masks.is_empty() && self.needle_ascii.len() > max_typos as usize
    }

    // Four interleaved LCS rows: one row is a serial add chain, four fill the core
    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub unsafe fn match_haystack_many_typos_x4(
        &self,
        haystacks: [&[u8]; 4],
        max_typos: u16,
    ) -> [(bool, usize, usize); 4] {
        let masks = self.lcs_masks.as_ptr();

        // longest first, so lanes that run out of bytes drop off the end
        let mut order = [0usize, 1, 2, 3];
        order.sort_unstable_by_key(|&lane| core::cmp::Reverse(haystacks[lane].len()));
        let sorted = order.map(|lane| haystacks[lane]);

        let mut rows = [!0u64; 4];
        let (len3, len2, len1, len0) = (
            sorted[3].len(),
            sorted[2].len(),
            sorted[1].len(),
            sorted[0].len(),
        );
        unsafe {
            lcs_advance::<4>(masks, &sorted, &mut rows, 0, len3);
            lcs_advance::<3>(masks, &sorted, &mut rows, len3, len2);
            lcs_advance::<2>(masks, &sorted, &mut rows, len2, len1);
            lcs_advance::<1>(masks, &sorted, &mut rows, len1, len0);
        }

        let mut windows = [(false, 0, 0); 4];
        for (slot, &lane) in order.iter().enumerate() {
            windows[lane] =
                unsafe { self.lcs_window(sorted[slot], rows[slot], max_typos as usize) };
        }
        windows
    }

    #[inline(always)]
    unsafe fn lcs_window(
        &self,
        haystack: &[u8],
        row: u64,
        max_typos: usize,
    ) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle_ascii.len();
        let needle_bits = !0u64 >> (64 - needle_len);
        if ((!row & needle_bits).count_ones() as usize) + max_typos < needle_len {
            return (false, 0, len);
        }

        // the window starts at the first byte any typo path could consume
        let masks = self.lcs_masks.as_ptr();
        let leading = !0u64 >> (63 - max_typos);
        let start_pos = haystack
            .iter()
            .position(|&byte| unsafe { *masks.add(byte as usize) } & leading != 0)
            .unwrap_or(0);
        let end_pos = unsafe { self.find_end_pos_with_typos(haystack, max_typos) };
        (true, start_pos, end_pos)
    }

    #[inline(always)]
    unsafe fn found_with_typos(
        &self,
        haystack: &[u8],
        match_start_pos: usize,
        max_typos: usize,
    ) -> (bool, usize, usize) {
        debug_assert!(match_start_pos != usize::MAX);
        let end_pos = unsafe { self.find_end_pos_with_typos(haystack, max_typos) };
        (true, match_start_pos, end_pos)
    }

    #[inline(always)]
    unsafe fn find_end_pos_with_typos(&self, haystack: &[u8], max_typos: usize) -> usize {
        let len = haystack.len();
        let needle_len = self.needle_ascii.len();
        let first = needle_len - 1 - max_typos;

        let mut start = (len - 1) / B::LANES * B::LANES;
        loop {
            let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
            let mut mask = B::Mask::zero();
            for &needle in &self.needle_ascii[first..] {
                mask = mask.or(unsafe { B::occ(chunk, needle) });
            }
            mask = mask.and(chunk_mask);
            if !mask.is_zero() {
                return start + B::LANES - mask.leading_zeros();
            }
            if start == 0 {
                break;
            }
            start -= B::LANES;
        }
        len
    }
}

// Advances the first `N` LCS rows over bytes `from..to` of their haystacks
#[inline(always)]
unsafe fn lcs_advance<const N: usize>(
    masks: *const u64,
    haystacks: &[&[u8]],
    rows: &mut [u64],
    from: usize,
    to: usize,
) {
    let mut local = [0u64; N];
    local.copy_from_slice(&rows[..N]);
    for pos in from..to {
        for (lane, row) in local.iter_mut().enumerate() {
            let byte = unsafe { *haystacks.get_unchecked(lane).get_unchecked(pos) };
            let hits = *row & unsafe { *masks.add(byte as usize) };
            *row = row.wrapping_add(hits) | row.wrapping_sub(hits);
        }
    }
    rows[..N].copy_from_slice(&local);
}
