use super::{Prefilter, load::load_window_maskless, load_window};
use crate::prefilter::{
    UnicodeChar,
    backend::{Backend, BitMaskOps},
};

impl<B: Backend> Prefilter<B> {
    #[inline(always)]
    unsafe fn match_unicode_char_prefix(
        start: usize,
        len: usize,
        haystack: &[u8],
        needle_char: &UnicodeChar,
    ) -> B::Mask {
        debug_assert!(needle_char.len <= 4 && needle_char.len > 0);
        match needle_char.len {
            2 => unsafe {
                B::eq(
                    load_window_maskless::<B>(haystack, start, len),
                    B::splat(needle_char.chars[0]),
                )
            },
            3 => unsafe {
                let occ_1 = B::eq(
                    load_window_maskless::<B>(haystack, start + 1, len),
                    B::splat(needle_char.chars[1]),
                );
                let occ_2 = B::eq(
                    load_window_maskless::<B>(haystack, start, len),
                    B::splat(needle_char.chars[0]),
                );
                occ_1.and(occ_2)
            },
            4 => unsafe {
                let occ_1 = B::eq(
                    load_window_maskless::<B>(haystack, start + 2, len),
                    B::splat(needle_char.chars[2]),
                );
                let occ_2 = B::eq(
                    load_window_maskless::<B>(haystack, start + 1, len),
                    B::splat(needle_char.chars[1]),
                );
                let occ_3 = B::eq(
                    load_window_maskless::<B>(haystack, start, len),
                    B::splat(needle_char.chars[0]),
                );
                occ_1.and(occ_2).and(occ_3)
            },
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub(super) unsafe fn unicode_char_mask(
        start: usize,
        len: usize,
        haystack: &[u8],
        needle_char: &UnicodeChar,
    ) -> B::Mask {
        let char_len = needle_char.len;
        debug_assert!(char_len <= 4 && char_len > 0);
        if start + char_len > len {
            return B::Mask::zero();
        }

        let (chunk, chunk_mask) = unsafe { load_window::<B>(haystack, start + char_len - 1, len) };
        let mut mask =
            unsafe { B::eq(chunk, B::splat(needle_char.chars[char_len - 1])) }.and(chunk_mask);
        if !mask.is_zero() && char_len > 1 {
            mask = mask
                .and(unsafe { Self::match_unicode_char_prefix(start, len, haystack, needle_char) });
        }
        mask
    }

    #[inline(always)]
    pub unsafe fn match_haystack_unicode(&self, haystack: &[u8]) -> (bool, usize, usize) {
        let len = haystack.len();
        if len == 0 {
            return (false, 0, 0);
        }

        let mut can_skip_chunks = true;
        let mut match_start_pos = 0usize;
        let mut needle_iter = self.needle_unicode.iter();
        let mut needle_char = needle_iter.next().unwrap();
        let mut last_needle_char_byte = unsafe { B::splat(needle_char.chars[needle_char.len - 1]) };
        let mut start = 0usize;

        while start + needle_char.len <= len {
            let (chunk, mut chunk_mask) =
                unsafe { load_window::<B>(haystack, start + needle_char.len - 1, len) };

            loop {
                // check the last byte first since it's the most discriminating
                // since the prefix bytes indentify the script/block, not the char
                // and nearby chars are likely to be all in the same script/block
                let mut mask = unsafe { B::eq(chunk, last_needle_char_byte) }.and(chunk_mask);
                if mask.is_zero() {
                    break;
                }

                // check that the rest of the bytes in the char match
                if needle_char.len > 1 {
                    mask = mask.and(unsafe {
                        Self::match_unicode_char_prefix(start, len, haystack, needle_char)
                    });
                    if mask.is_zero() {
                        break;
                    }
                }

                chunk_mask = chunk_mask.clear_through_lowest(mask);
                if can_skip_chunks {
                    // since we match on the final byte, subtract the byte length of the char
                    match_start_pos = start + mask.trailing_zeros();
                    can_skip_chunks = false;
                }

                if let Some(next_needle_char) = needle_iter.next() {
                    needle_char = next_needle_char;
                    last_needle_char_byte =
                        unsafe { B::splat(needle_char.chars[needle_char.len - 1]) };
                } else if start + needle_char.len - 1 + B::LANES >= len {
                    return (
                        true,
                        match_start_pos,
                        start + B::LANES - mask.leading_zeros() + needle_char.len - 1,
                    );
                } else {
                    let end_pos = start
                        + unsafe {
                            Self::find_last_unicode_char_pos(needle_char, &haystack[start..])
                        };
                    return (true, match_start_pos, end_pos);
                }
            }

            start += B::LANES;
        }

        (false, match_start_pos, len)
    }

    #[inline(always)]
    unsafe fn find_last_unicode_char_pos(needle_char: &UnicodeChar, haystack: &[u8]) -> usize {
        let len = haystack.len();
        let char_len = needle_char.len;
        debug_assert!(char_len <= 4 && char_len > 0);
        debug_assert!(len >= char_len);

        let last_byte = unsafe { B::splat(needle_char.chars[char_len - 1]) };
        let mut start = len.saturating_sub(B::LANES + char_len - 1);
        loop {
            let (chunk, chunk_mask) =
                unsafe { load_window::<B>(haystack, start + char_len - 1, len) };
            let mut mask = unsafe { B::eq(chunk, last_byte) }.and(chunk_mask);
            if char_len > 1 {
                mask = mask.and(unsafe {
                    Self::match_unicode_char_prefix(start, len, haystack, needle_char)
                });
            }

            if !mask.is_zero() {
                return start + B::LANES - mask.leading_zeros() + char_len - 1;
            }

            if start == 0 {
                break;
            }
            start = start.saturating_sub(B::LANES);
        }

        len
    }
}
