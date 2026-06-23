use super::{
    UnicodeChar,
    backend::{Backend, BitMaskOps},
    case_needle, case_needle_unicode,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct PathState<M> {
    pub needle_idx: usize,
    pub needle_mask: M,
}

#[derive(Debug, Clone)]
pub(crate) struct Prefilter<B: Backend> {
    needle_ascii: Vec<(B::Chunk, B::Chunk)>,
    needle_unicode: Vec<UnicodeChar<B::Chunk>>,
    paths: Vec<PathState<B::Mask>>,
}

impl<B: Backend> Prefilter<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub unsafe fn new(needle: &str, case_sensitive: bool) -> Self {
        let needle_ascii = case_needle(needle.as_bytes(), case_sensitive)
            .into_iter()
            .map(|c| unsafe { B::broadcast(c) })
            .collect();
        let needle_unicode = case_needle_unicode(needle, case_sensitive)
            .into_iter()
            .map(|c| unsafe { c.broadcast::<B>() })
            .collect();

        Self {
            needle_ascii,
            needle_unicode,
            paths: Vec::new(),
        }
    }

    #[inline(always)]
    unsafe fn needle_unchecked(&self, idx: usize) -> (B::Chunk, B::Chunk) {
        unsafe { *self.needle_ascii.get_unchecked(idx) }
    }

    #[inline(always)]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        if len == 0 {
            return (false, 0, 0);
        }

        let mut can_skip_chunks = true;
        let mut match_start_pos = 0usize;
        let needle = self.needle_ascii.as_slice();
        let mut needle_iter = needle.iter();
        let mut needle_char = *needle_iter.next().unwrap();
        let mut start = 0usize;

        while start < len {
            let (chunk, mut chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };

            loop {
                let mask = unsafe { B::occ(chunk, needle_char) }.and(chunk_mask);
                if mask.is_zero() {
                    break;
                }

                chunk_mask = chunk_mask.clear_through_lowest(mask);
                if can_skip_chunks {
                    match_start_pos = start + mask.trailing_zeros();
                    can_skip_chunks = false;
                }

                if let Some(&next_needle_char) = needle_iter.next() {
                    needle_char = next_needle_char;
                } else if start + B::LANES >= len {
                    return (
                        true,
                        match_start_pos,
                        start + B::LANES - mask.leading_zeros(),
                    );
                } else {
                    let last = *needle.last().unwrap();
                    let end_pos =
                        start + unsafe { find_last_char_pos::<B>(last, &haystack[start..]) };
                    return (true, match_start_pos, end_pos);
                }
            }

            start += B::LANES;
        }

        (false, match_start_pos, len)
    }

    #[inline(always)]
    pub unsafe fn match_haystack_unicode(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        if len == 0 {
            return (false, 0, 0);
        }

        let mut can_skip_chunks = true;
        let mut match_start_pos = 0usize;
        let needle = self.needle_unicode.as_slice();
        let mut needle_iter = needle.iter();
        let mut needle_char = needle_iter.next().unwrap();
        let mut last_needle_char_byte = needle_char.chars[needle_char.len - 1];
        let mut start = 0usize;

        let mut prev_chunk = unsafe { B::zero() };
        while start < len {
            let (chunk, mut chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };

            loop {
                // check the last byte first since it's the most discriminating
                // since the prefix bytes indentify the script/block, not the char
                // and nearby chars are likely to be all in the same script/block
                let mut mask = unsafe { B::occ(chunk, last_needle_char_byte) }.and(chunk_mask);
                if mask.is_zero() {
                    break;
                }

                // check that the rest of the bytes in the char match
                if needle_char.len == 2 {
                    let shifted_chunk = unsafe { B::shift_left::<1>(chunk, prev_chunk) };
                    let occ =
                        unsafe { B::occ(shifted_chunk, needle_char.chars[needle_char.len - 2]) };
                    mask = mask.and(occ);
                    if mask.is_zero() {
                        break;
                    }
                } else if needle_char.len == 3 {
                    let shifted_chunk_1 = unsafe { B::shift_left::<1>(chunk, prev_chunk) };
                    let shifted_chunk_2 = unsafe { B::shift_left::<2>(chunk, prev_chunk) };
                    let occ_1 =
                        unsafe { B::occ(shifted_chunk_1, needle_char.chars[needle_char.len - 2]) };
                    let occ_2 =
                        unsafe { B::occ(shifted_chunk_2, needle_char.chars[needle_char.len - 3]) };
                    mask = mask.and(occ_1).and(occ_2);
                    if mask.is_zero() {
                        break;
                    }
                } else if needle_char.len == 4 {
                    let shifted_chunk_1 = unsafe { B::shift_left::<1>(chunk, prev_chunk) };
                    let shifted_chunk_2 = unsafe { B::shift_left::<2>(chunk, prev_chunk) };
                    let shifted_chunk_3 = unsafe { B::shift_left::<3>(chunk, prev_chunk) };
                    let occ_1 =
                        unsafe { B::occ(shifted_chunk_1, needle_char.chars[needle_char.len - 2]) };
                    let occ_2 =
                        unsafe { B::occ(shifted_chunk_2, needle_char.chars[needle_char.len - 3]) };
                    let occ_3 =
                        unsafe { B::occ(shifted_chunk_3, needle_char.chars[needle_char.len - 4]) };
                    mask = mask.and(occ_1).and(occ_2).and(occ_3);
                    if mask.is_zero() {
                        break;
                    }
                }

                chunk_mask = chunk_mask.clear_through_lowest(mask);
                if can_skip_chunks {
                    // since we match on the final byte, subtract the byte length of the char
                    match_start_pos = start + mask.trailing_zeros() + 1 - needle_char.len;
                    can_skip_chunks = false;
                }

                if let Some(next_needle_char) = needle_iter.next() {
                    needle_char = next_needle_char;
                    last_needle_char_byte = needle_char.chars[needle_char.len - 1];
                } else if start + B::LANES > len - needle_char.len {
                    return (
                        true,
                        match_start_pos,
                        start + B::LANES - mask.leading_zeros(),
                    );
                } else {
                    // this still works for multi-byte unicode, since on a false positive, we
                    // end up including extra suffix which doesn't affect correctness
                    let last = *needle.last().unwrap();
                    let last_needle_char_byte = last.chars[last.len - 1];
                    let end_pos = start
                        + unsafe {
                            find_last_char_pos::<B>(last_needle_char_byte, &haystack[start..])
                        };
                    return (true, match_start_pos, end_pos);
                }
            }

            start += B::LANES;
            prev_chunk = chunk;
        }

        (false, match_start_pos, len)
    }

    #[inline(always)]
    pub unsafe fn match_haystack_many_typos(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> (bool, usize, usize) {
        unsafe { self.match_haystack_many_typos_impl(haystack, max_typos as usize) }
    }

    #[inline(always)]
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

    #[inline(always)]
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
                            if candidate_needle_idx == needle_len {
                                return self.found_with_typos(haystack, match_start_pos, max_typos);
                            }
                            (*path).needle_idx = candidate_needle_idx;
                            (*path).needle_mask =
                                B::occ(chunk, *needle.get_unchecked(candidate_needle_idx));
                        }
                    }
                }

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

#[inline(always)]
pub(crate) unsafe fn find_last_char_pos<B: Backend>(
    needle: (B::Chunk, B::Chunk),
    haystack: &[u8],
) -> usize {
    let len = haystack.len();
    let mut start = len.saturating_sub(B::LANES);
    loop {
        let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
        let mask = unsafe { B::occ(chunk, needle) }.and(chunk_mask);
        if !mask.is_zero() {
            return start + B::LANES - mask.leading_zeros();
        }
        start = start.saturating_sub(B::LANES);
    }
}

#[inline(always)]
pub(crate) unsafe fn load_window<B: Backend>(
    haystack: &[u8],
    start: usize,
    len: usize,
) -> (B::Chunk, B::Mask) {
    unsafe {
        debug_assert!(B::LANES <= 64);
        let remaining = len - start;
        if remaining >= B::LANES {
            return (B::load(haystack.as_ptr().add(start)), B::Mask::all());
        }

        let mask = B::Mask::first_n(remaining);
        let ptr = haystack.as_ptr().add(start);
        if can_overread(ptr, B::LANES) {
            (B::load(ptr), mask)
        } else {
            (B::load_partial(ptr, remaining, mask), mask)
        }
    }
}

#[inline(always)]
pub(crate) fn can_overread(ptr: *const u8, bytes: usize) -> bool {
    if cfg!(feature = "safe_read") || cfg!(miri) {
        return false;
    }
    (ptr as usize & 0xFFF) <= (4096 - bytes)
}
