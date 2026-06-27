use super::{PathState, Prefilter};
use crate::prefilter::backend::{Backend, BitMaskOps};

impl<B: Backend> Prefilter<B> {
    #[inline(always)]
    pub unsafe fn match_haystack_unicode_many_typos(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> (bool, usize, usize) {
        unsafe { self.match_haystack_unicode_many_typos_impl(haystack, max_typos as usize) }
    }

    #[inline(always)]
    pub unsafe fn match_haystack_unicode_1_typo(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle_unicode.len();
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
            let mut first_path_mask = unsafe {
                Self::unicode_char_mask(
                    start,
                    len,
                    haystack,
                    self.unicode_needle_unchecked(first_path_needle_idx),
                )
            };
            let mut second_path_mask = unsafe {
                Self::unicode_char_mask(
                    start,
                    len,
                    haystack,
                    self.unicode_needle_unchecked(second_path_needle_idx),
                )
            };

            let mut first_path_chunk_mask = B::Mask::all();
            let mut second_path_chunk_mask = B::Mask::all();

            loop {
                let mut advanced = false;

                let candidate_needle_idx = first_path_needle_idx + 1;
                if candidate_needle_idx > second_path_needle_idx {
                    // first path is on the second last char
                    // and since this path allows a typo, we've matched
                    if candidate_needle_idx == needle_len {
                        return unsafe {
                            self.found_with_unicode_typos(haystack, match_start_pos, 1)
                        };
                    }

                    // first path caught up to or passed the second path
                    // skip the next needle char
                    second_path_needle_idx = candidate_needle_idx;
                    second_path_chunk_mask = first_path_chunk_mask;
                    second_path_mask = unsafe {
                        Self::unicode_char_mask(
                            start,
                            len,
                            haystack,
                            self.unicode_needle_unchecked(second_path_needle_idx),
                        )
                    };
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
                    first_path_mask = unsafe {
                        Self::unicode_char_mask(
                            start,
                            len,
                            haystack,
                            self.unicode_needle_unchecked(first_path_needle_idx),
                        )
                    };
                    advanced = true;
                }

                // match second path against current chunk
                let second_path_matches = second_path_mask.and(second_path_chunk_mask);
                if !second_path_matches.is_zero() {
                    match_start_pos = match_start_pos
                        .min(start + unsafe { B::first_hit_pos(second_path_matches) });

                    second_path_needle_idx += 1;
                    if second_path_needle_idx >= needle_len {
                        return unsafe {
                            self.found_with_unicode_typos(haystack, match_start_pos, 1)
                        };
                    }

                    second_path_chunk_mask = unsafe {
                        B::clear_through_lowest(second_path_chunk_mask, second_path_matches)
                    };
                    second_path_mask = unsafe {
                        Self::unicode_char_mask(
                            start,
                            len,
                            haystack,
                            self.unicode_needle_unchecked(second_path_needle_idx),
                        )
                    };
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
    pub unsafe fn match_haystack_unicode_2_typos(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle_unicode.len();
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
            let mut first_path_mask = unsafe {
                Self::unicode_char_mask(
                    start,
                    len,
                    haystack,
                    self.unicode_needle_unchecked(first_path_needle_idx),
                )
            };
            let mut second_path_mask = unsafe {
                Self::unicode_char_mask(
                    start,
                    len,
                    haystack,
                    self.unicode_needle_unchecked(second_path_needle_idx),
                )
            };
            let mut third_path_mask = unsafe {
                Self::unicode_char_mask(
                    start,
                    len,
                    haystack,
                    self.unicode_needle_unchecked(third_path_needle_idx),
                )
            };

            let mut first_path_chunk_mask = B::Mask::all();
            let mut second_path_chunk_mask = B::Mask::all();
            let mut third_path_chunk_mask = B::Mask::all();

            loop {
                let mut advanced = false;

                let second_path_candidate_needle_idx = first_path_needle_idx + 1;
                if second_path_candidate_needle_idx > second_path_needle_idx {
                    // first path is on the second last char
                    // and since this path allows typos, we've matched
                    if second_path_candidate_needle_idx == needle_len {
                        return unsafe {
                            self.found_with_unicode_typos(haystack, match_start_pos, 2)
                        };
                    }

                    // first path caught up to or passed the second path
                    // skip the next needle char
                    second_path_needle_idx = second_path_candidate_needle_idx;
                    second_path_chunk_mask = first_path_chunk_mask;
                    second_path_mask = unsafe {
                        Self::unicode_char_mask(
                            start,
                            len,
                            haystack,
                            self.unicode_needle_unchecked(second_path_needle_idx),
                        )
                    };
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
                        return unsafe {
                            self.found_with_unicode_typos(haystack, match_start_pos, 2)
                        };
                    }

                    // second path caught up to or passed the third path
                    // skip the next needle char
                    third_path_needle_idx = third_path_candidate_needle_idx;
                    third_path_chunk_mask = second_path_chunk_mask;
                    third_path_mask = unsafe {
                        Self::unicode_char_mask(
                            start,
                            len,
                            haystack,
                            self.unicode_needle_unchecked(third_path_needle_idx),
                        )
                    };
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
                    first_path_mask = unsafe {
                        Self::unicode_char_mask(
                            start,
                            len,
                            haystack,
                            self.unicode_needle_unchecked(first_path_needle_idx),
                        )
                    };
                    advanced = true;
                }

                // match second path against current chunk
                let second_path_matches = second_path_mask.and(second_path_chunk_mask);
                if !second_path_matches.is_zero() {
                    match_start_pos = match_start_pos
                        .min(start + unsafe { B::first_hit_pos(second_path_matches) });

                    second_path_needle_idx += 1;
                    if second_path_needle_idx >= needle_len {
                        return unsafe {
                            self.found_with_unicode_typos(haystack, match_start_pos, 2)
                        };
                    }

                    second_path_chunk_mask = unsafe {
                        B::clear_through_lowest(second_path_chunk_mask, second_path_matches)
                    };
                    second_path_mask = unsafe {
                        Self::unicode_char_mask(
                            start,
                            len,
                            haystack,
                            self.unicode_needle_unchecked(second_path_needle_idx),
                        )
                    };
                    advanced = true;
                }

                // match third path against current chunk
                let third_path_matches = third_path_mask.and(third_path_chunk_mask);
                if !third_path_matches.is_zero() {
                    match_start_pos = match_start_pos
                        .min(start + unsafe { B::first_hit_pos(third_path_matches) });

                    third_path_needle_idx += 1;
                    if third_path_needle_idx >= needle_len {
                        return unsafe {
                            self.found_with_unicode_typos(haystack, match_start_pos, 2)
                        };
                    }

                    third_path_chunk_mask = unsafe {
                        B::clear_through_lowest(third_path_chunk_mask, third_path_matches)
                    };
                    third_path_mask = unsafe {
                        Self::unicode_char_mask(
                            start,
                            len,
                            haystack,
                            self.unicode_needle_unchecked(third_path_needle_idx),
                        )
                    };
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
    unsafe fn match_haystack_unicode_many_typos_impl(
        &mut self,
        haystack: &[u8],
        max_typos: usize,
    ) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle_unicode.len();
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
        let needle = self.needle_unicode.as_slice();
        let mut match_start_pos = usize::MAX;

        for start in (0..len).step_by(B::LANES) {
            let mut chunk_mask = B::Mask::all();
            // compare needle chars of all paths against the current chunk
            for path_idx in 0..path_count {
                unsafe {
                    let path = paths.add(path_idx);
                    (*path).needle_mask = Self::unicode_char_mask(
                        start,
                        len,
                        haystack,
                        needle.get_unchecked((*path).needle_idx),
                    );
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
                                return self.found_with_unicode_typos(
                                    haystack,
                                    match_start_pos,
                                    max_typos,
                                );
                            }

                            // previous path caught up to or passed this path
                            // skip the next needle char
                            (*path).needle_idx = candidate_needle_idx;
                            (*path).needle_mask = Self::unicode_char_mask(
                                start,
                                len,
                                haystack,
                                needle.get_unchecked(candidate_needle_idx),
                            );
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
                            return self.found_with_unicode_typos(
                                haystack,
                                match_start_pos,
                                max_typos,
                            );
                        }
                        (*path).needle_mask = Self::unicode_char_mask(
                            start,
                            len,
                            haystack,
                            needle.get_unchecked((*path).needle_idx),
                        );
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
    unsafe fn found_with_unicode_typos(
        &self,
        haystack: &[u8],
        match_start_pos: usize,
        max_typos: usize,
    ) -> (bool, usize, usize) {
        debug_assert!(match_start_pos != usize::MAX);
        let end_pos = unsafe { self.find_end_pos_with_unicode_typos(haystack, max_typos) };
        (true, match_start_pos, end_pos)
    }

    #[inline(always)]
    unsafe fn find_end_pos_with_unicode_typos(&self, haystack: &[u8], max_typos: usize) -> usize {
        let len = haystack.len();
        let needle_len = self.needle_unicode.len();
        let first = needle_len - 1 - max_typos;

        let mut start = len.saturating_sub(B::LANES);
        loop {
            let mut end_pos = 0usize;
            for needle_char in &self.needle_unicode[first..] {
                let mask = unsafe { Self::unicode_char_mask(start, len, haystack, needle_char) };
                if !mask.is_zero() {
                    end_pos =
                        end_pos.max(start + B::LANES - mask.leading_zeros() + needle_char.len - 1);
                }
            }
            if end_pos != 0 {
                return end_pos;
            }
            if start == 0 {
                break;
            }
            start = start.saturating_sub(B::LANES);
        }
        len
    }
}
