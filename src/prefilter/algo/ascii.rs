use super::{Prefilter, load_window};
use crate::prefilter::backend::{Backend, BitMaskOps};

impl<B: Backend> Prefilter<B> {
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
