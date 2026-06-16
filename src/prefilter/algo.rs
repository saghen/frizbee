use crate::prefilter::{
    backend::{Backend, BitMaskOps},
    case_needle,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct LaneState<M> {
    pub needle_idx: usize,
    pub chunk_mask: M,
}

#[derive(Debug, Clone)]
pub(crate) struct Prefilter<B: Backend> {
    needle: Vec<B::Needle>,
    lanes: Vec<LaneState<B::Mask>>,
}

impl<B: Backend> Prefilter<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub unsafe fn new(needle: &[u8]) -> Self {
        let needle = case_needle(needle)
            .iter()
            .map(|&(c1, c2)| unsafe { B::broadcast(c1, c2) })
            .collect();
        Self {
            needle,
            lanes: Vec::new(),
        }
    }

    #[inline(always)]
    unsafe fn needle_unchecked(&self, idx: usize) -> B::Needle {
        unsafe { *self.needle.get_unchecked(idx) }
    }

    #[inline(always)]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        if len == 0 {
            return (false, 0, 0);
        }

        let mut can_skip_chunks = true;
        let mut match_start_pos = 0usize;
        let needle = self.needle.as_slice();
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
        let needle_len = self.needle.len();
        if needle_len <= 1 {
            return (true, 0, len);
        }
        if len == 0 {
            return (false, 0, 0);
        }

        let mut i0 = 0usize;
        let mut i1 = 1usize;
        let mut match_start_pos = usize::MAX;

        for start in (0..len).step_by(B::LANES) {
            let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
            let mut occ0 = unsafe { B::occ(chunk, self.needle_unchecked(i0)) };
            let mut occ1 = unsafe { B::occ(chunk, self.needle_unchecked(i1)) };

            if !occ0.or(occ1).and(chunk_mask).is_zero() {
                let mut m0 = chunk_mask;
                let mut m1 = chunk_mask;
                let mut prev_idx = usize::MAX;
                let mut occ_prev = B::Mask::zero();

                loop {
                    let mut advanced = false;

                    let hit0 = occ0.and(m0);
                    if !hit0.is_zero() {
                        match_start_pos =
                            match_start_pos.min(start + unsafe { B::first_hit_pos(hit0) });
                        m0 = unsafe { B::clear_through_lowest(m0, hit0) };
                        i0 += 1;
                        debug_assert!(i0 < needle_len);
                        occ0 = if i0 == i1 {
                            occ1
                        } else if i0 == prev_idx {
                            occ_prev
                        } else {
                            unsafe { B::occ(chunk, self.needle_unchecked(i0)) }
                        };
                        advanced = true;
                    }

                    let candidate_idx = i0 + 1;
                    if candidate_idx > i1 {
                        if candidate_idx == needle_len {
                            return unsafe { self.found_with_typos(haystack, match_start_pos, 1) };
                        }
                        i1 = candidate_idx;
                        m1 = m0;
                        occ1 = unsafe { B::occ(chunk, self.needle_unchecked(i1)) };
                        advanced = true;
                    } else if candidate_idx == i1 && m0 > m1 {
                        m1 = m0;
                        advanced = true;
                    }

                    let hit1 = occ1.and(m1);
                    if !hit1.is_zero() {
                        match_start_pos =
                            match_start_pos.min(start + unsafe { B::first_hit_pos(hit1) });
                        m1 = unsafe { B::clear_through_lowest(m1, hit1) };
                        prev_idx = i1;
                        occ_prev = occ1;
                        i1 += 1;
                        if i1 == needle_len {
                            return unsafe { self.found_with_typos(haystack, match_start_pos, 1) };
                        }
                        occ1 = unsafe { B::occ(chunk, self.needle_unchecked(i1)) };
                        advanced = true;
                    }

                    if !advanced {
                        break;
                    }
                }
            }

            let next_start = start + B::LANES;
            if next_start < len && needle_len - i1 > len - next_start {
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

    #[inline(always)]
    pub unsafe fn match_haystack_2_typos(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle.len();
        if needle_len <= 2 {
            return (true, 0, len);
        }
        if len == 0 {
            return (false, 0, 0);
        }

        let mut i0 = 0usize;
        let mut i1 = 1usize;
        let mut i2 = 2usize;
        let mut match_start_pos = usize::MAX;

        for start in (0..len).step_by(B::LANES) {
            let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
            let mut occ0 = unsafe { B::occ(chunk, self.needle_unchecked(i0)) };
            let mut occ1 = unsafe { B::occ(chunk, self.needle_unchecked(i1)) };
            let mut occ2 = unsafe { B::occ(chunk, self.needle_unchecked(i2)) };

            if !occ0.or(occ1).or(occ2).and(chunk_mask).is_zero() {
                let mut m0 = chunk_mask;
                let mut m1 = chunk_mask;
                let mut m2 = chunk_mask;

                loop {
                    let mut advanced = false;

                    let hit0 = occ0.and(m0);
                    if !hit0.is_zero() {
                        match_start_pos =
                            match_start_pos.min(start + unsafe { B::first_hit_pos(hit0) });
                        m0 = unsafe { B::clear_through_lowest(m0, hit0) };
                        i0 += 1;
                        debug_assert!(i0 < needle_len);
                        occ0 = unsafe { B::occ(chunk, self.needle_unchecked(i0)) };
                        advanced = true;
                    }

                    let cand1 = i0 + 1;
                    if cand1 > i1 {
                        if cand1 == needle_len {
                            return unsafe { self.found_with_typos(haystack, match_start_pos, 2) };
                        }
                        i1 = cand1;
                        m1 = m0;
                        occ1 = unsafe { B::occ(chunk, self.needle_unchecked(i1)) };
                        advanced = true;
                    } else if cand1 == i1 && m0 > m1 {
                        m1 = m0;
                        advanced = true;
                    }

                    let hit1 = occ1.and(m1);
                    if !hit1.is_zero() {
                        match_start_pos =
                            match_start_pos.min(start + unsafe { B::first_hit_pos(hit1) });
                        m1 = unsafe { B::clear_through_lowest(m1, hit1) };
                        i1 += 1;
                        if i1 == needle_len {
                            return unsafe { self.found_with_typos(haystack, match_start_pos, 2) };
                        }
                        occ1 = unsafe { B::occ(chunk, self.needle_unchecked(i1)) };
                        advanced = true;
                    }

                    let cand2 = i1 + 1;
                    if cand2 > i2 {
                        if cand2 == needle_len {
                            return unsafe { self.found_with_typos(haystack, match_start_pos, 2) };
                        }
                        i2 = cand2;
                        m2 = m1;
                        occ2 = unsafe { B::occ(chunk, self.needle_unchecked(i2)) };
                        advanced = true;
                    } else if cand2 == i2 && m1 > m2 {
                        m2 = m1;
                        advanced = true;
                    }

                    let hit2 = occ2.and(m2);
                    if !hit2.is_zero() {
                        match_start_pos =
                            match_start_pos.min(start + unsafe { B::first_hit_pos(hit2) });
                        m2 = unsafe { B::clear_through_lowest(m2, hit2) };
                        i2 += 1;
                        if i2 == needle_len {
                            return unsafe { self.found_with_typos(haystack, match_start_pos, 2) };
                        }
                        occ2 = unsafe { B::occ(chunk, self.needle_unchecked(i2)) };
                        advanced = true;
                    }

                    if !advanced {
                        break;
                    }
                }
            }

            let next_start = start + B::LANES;
            if next_start < len && needle_len - i2 > len - next_start {
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

    #[inline(always)]
    unsafe fn match_haystack_many_typos_impl(
        &mut self,
        haystack: &[u8],
        max_typos: usize,
    ) -> (bool, usize, usize) {
        let len = haystack.len();
        let needle_len = self.needle.len();
        if needle_len <= max_typos {
            return (true, 0, len);
        }
        if len == 0 {
            return (false, 0, 0);
        }

        let lane_count = max_typos + 1;
        self.lanes.resize(
            lane_count,
            LaneState {
                needle_idx: 0,
                chunk_mask: B::Mask::zero(),
            },
        );
        for lane in &mut self.lanes {
            lane.needle_idx = 0;
            lane.chunk_mask = B::Mask::zero();
        }

        let lanes = self.lanes.as_mut_ptr();
        let needle = self.needle.as_slice();
        let mut match_start_pos = usize::MAX;

        for start in (0..len).step_by(B::LANES) {
            let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
            for typos in 0..lane_count {
                unsafe {
                    (*lanes.add(typos)).chunk_mask = chunk_mask;
                }
            }

            loop {
                let mut advanced = false;

                for typos in 1..lane_count {
                    unsafe {
                        let prev = *lanes.add(typos - 1);
                        let lane = lanes.add(typos);
                        let candidate_idx = prev.needle_idx + 1;
                        if candidate_idx > (*lane).needle_idx
                            || (candidate_idx == (*lane).needle_idx
                                && prev.chunk_mask > (*lane).chunk_mask)
                        {
                            (*lane).needle_idx = candidate_idx;
                            (*lane).chunk_mask = prev.chunk_mask;
                            advanced = true;
                            if candidate_idx == needle_len {
                                return self.found_with_typos(haystack, match_start_pos, max_typos);
                            }
                        }
                    }
                }

                for typos in 0..lane_count {
                    unsafe {
                        let lane = lanes.add(typos);
                        let mask = B::occ(chunk, *needle.get_unchecked((*lane).needle_idx))
                            .and((*lane).chunk_mask);
                        if mask.is_zero() {
                            continue;
                        }

                        (*lane).chunk_mask = B::clear_through_lowest((*lane).chunk_mask, mask);
                        (*lane).needle_idx += 1;
                        advanced = true;
                        match_start_pos = match_start_pos.min(start + B::first_hit_pos(mask));
                        if (*lane).needle_idx == needle_len {
                            return self.found_with_typos(haystack, match_start_pos, max_typos);
                        }
                    }
                }

                if !advanced {
                    break;
                }
            }

            let next_start = start + B::LANES;
            if next_start < len {
                let remaining_haystack = len - next_start;
                let mut any_lane_viable = false;
                for typos in 0..lane_count {
                    unsafe {
                        let unaccounted_chars = needle_len - (*lanes.add(typos)).needle_idx;
                        let deletions_left = max_typos - typos;
                        if unaccounted_chars <= deletions_left + remaining_haystack {
                            any_lane_viable = true;
                            break;
                        }
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
        let needle_len = self.needle.len();
        let first = needle_len - 1 - max_typos;

        let mut start = (len - 1) / B::LANES * B::LANES;
        loop {
            let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start, len) };
            let mut mask = B::Mask::zero();
            for &needle in &self.needle[first..] {
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
pub(crate) unsafe fn find_last_char_pos<B: Backend>(needle: B::Needle, haystack: &[u8]) -> usize {
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
        #[cfg(feature = "safe_read")]
        {
            (B::load_partial(ptr, remaining, mask), mask)
        }
        #[cfg(not(feature = "safe_read"))]
        {
            if can_overread(ptr, B::LANES) {
                (B::load(ptr), mask)
            } else {
                (B::load_partial(ptr, remaining, mask), mask)
            }
        }
    }
}

#[cfg(not(feature = "safe_read"))]
#[inline(always)]
pub(crate) fn can_overread(ptr: *const u8, bytes: usize) -> bool {
    (ptr as usize & 0xFFF) <= (4096 - bytes)
}
