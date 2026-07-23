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
        char_len: usize,
        chars: [u8; 4],
    ) -> B::Mask {
        debug_assert!(char_len <= 4 && char_len > 1);
        match char_len {
            2 => unsafe {
                B::eq(
                    load_window_maskless::<B>(haystack, start, len),
                    B::splat(chars[0]),
                )
            },
            3 => unsafe {
                let occ_1 = B::eq(
                    load_window_maskless::<B>(haystack, start + 1, len),
                    B::splat(chars[1]),
                );
                let occ_2 = B::eq(
                    load_window_maskless::<B>(haystack, start, len),
                    B::splat(chars[0]),
                );
                occ_1.and(occ_2)
            },
            4 => unsafe {
                let occ_1 = B::eq(
                    load_window_maskless::<B>(haystack, start + 2, len),
                    B::splat(chars[2]),
                );
                let occ_2 = B::eq(
                    load_window_maskless::<B>(haystack, start + 1, len),
                    B::splat(chars[1]),
                );
                let occ_3 = B::eq(
                    load_window_maskless::<B>(haystack, start, len),
                    B::splat(chars[0]),
                );
                occ_1.and(occ_2).and(occ_3)
            },
            _ => unreachable!(),
        }
    }

    /// Occurrence mask for one case variant of a needle char: lanes where `last_byte` matches,
    /// verified against the char's remaining UTF-8 bytes
    #[inline(always)]
    unsafe fn char_variant_mask(
        (chunk, chunk_mask): (B::Chunk, B::Mask),
        last_byte: B::Chunk,
        start: usize,
        len: usize,
        haystack: &[u8],
        char_len: usize,
        chars: [u8; 4],
    ) -> B::Mask {
        let mut mask = unsafe { B::eq(chunk, last_byte) }.and(chunk_mask);
        if !mask.is_zero() && char_len > 1 {
            mask = mask.and(unsafe {
                Self::match_unicode_char_prefix(start, len, haystack, char_len, chars)
            });
        }
        mask
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

        // check the regular needle
        let mut mask = unsafe {
            Self::char_variant_mask(
                (chunk, chunk_mask),
                B::splat(needle_char.chars[char_len - 1]),
                start,
                len,
                haystack,
                char_len,
                needle_char.chars,
            )
        };

        // check the case flipped version
        mask = mask.or(unsafe {
            Self::char_variant_mask(
                (chunk, chunk_mask),
                B::splat(needle_char.flipped_chars[char_len - 1]),
                start,
                len,
                haystack,
                char_len,
                needle_char.flipped_chars,
            )
        });
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
        let mut last_needle_char_bytes = unsafe {
            (
                B::splat(needle_char.chars[needle_char.len - 1]),
                B::splat(needle_char.flipped_chars[needle_char.len - 1]),
            )
        };
        let mut start = 0usize;

        while start + needle_char.len <= len {
            // keep the subsequence mask (`available`) separate from the load's width-dependent
            // bounds (`valid`) so a width change can reload without dropping consumed lanes
            let mut char_len = needle_char.len;
            let (mut chunk, mut valid) =
                unsafe { load_window::<B>(haystack, start + char_len - 1, len) };
            let mut available = B::Mask::all();

            loop {
                let chunk_mask = available.and(valid);
                // check the regular needle first
                let mut mask = unsafe {
                    Self::char_variant_mask(
                        (chunk, chunk_mask),
                        last_needle_char_bytes.0,
                        start,
                        len,
                        haystack,
                        needle_char.len,
                        needle_char.chars,
                    )
                };

                // check the case flipped version
                mask = mask.or(unsafe {
                    Self::char_variant_mask(
                        (chunk, chunk_mask),
                        last_needle_char_bytes.1,
                        start,
                        len,
                        haystack,
                        needle_char.len,
                        needle_char.flipped_chars,
                    )
                });

                if mask.is_zero() {
                    break;
                }

                available = available.clear_through_lowest(mask);
                if can_skip_chunks {
                    match_start_pos = start + mask.trailing_zeros();
                    can_skip_chunks = false;
                }

                if let Some(next_needle_char) = needle_iter.next() {
                    needle_char = next_needle_char;
                    last_needle_char_bytes = unsafe {
                        (
                            B::splat(needle_char.chars[needle_char.len - 1]),
                            B::splat(needle_char.flipped_chars[needle_char.len - 1]),
                        )
                    };
                    // reload the window when the char width changes to realign the lanes
                    if needle_char.len != char_len {
                        if start + needle_char.len > len {
                            break;
                        }
                        char_len = needle_char.len;
                        (chunk, valid) =
                            unsafe { load_window::<B>(haystack, start + char_len - 1, len) };
                    }
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

        let last_bytes = unsafe {
            (
                B::splat(needle_char.chars[char_len - 1]),
                B::splat(needle_char.flipped_chars[char_len - 1]),
            )
        };
        let mut start = len.saturating_sub(B::LANES + char_len - 1);
        loop {
            let (chunk, chunk_mask) =
                unsafe { load_window::<B>(haystack, start + char_len - 1, len) };

            // we check both the original and flipped case bytes simultaneously since either
            // could have matched. this can result in a false positive, but that's fine for
            // this stage since this just controls bounds, and bounds being too large hurt
            // performance, not correctness.
            let mut mask = unsafe { B::eq(chunk, last_bytes.0).or(B::eq(chunk, last_bytes.1)) }
                .and(chunk_mask);

            if !mask.is_zero() && char_len > 1 {
                mask = mask.and(unsafe {
                    Self::match_unicode_char_prefix(
                        start,
                        len,
                        haystack,
                        char_len,
                        needle_char.chars,
                    )
                    .or(Self::match_unicode_char_prefix(
                        start,
                        len,
                        haystack,
                        char_len,
                        needle_char.flipped_chars,
                    ))
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
