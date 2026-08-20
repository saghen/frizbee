mod ascii;
mod ascii_typos;
mod load;
mod rare;
mod unicode;
mod unicode_typos;

#[cfg(target_arch = "x86_64")]
pub(crate) use ascii::find_last_char_pos;
pub(crate) use load::{can_overread, load_window};

use super::{UnicodeChar, backend::Backend, case_needle, case_needle_unicode};
use alloc::vec::Vec;
use rare::RareByte;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PathState<M> {
    pub needle_idx: usize,
    pub needle_mask: M,
}

#[derive(Debug, Clone)]
pub(crate) struct Prefilter<B: Backend> {
    needle_ascii: Vec<(B::Chunk, B::Chunk)>,
    needle_unicode: Vec<UnicodeChar>,
    paths: Vec<PathState<B::Mask>>,
    /// Adaptive pre-rejection for the ASCII 0-typo path
    rare_ascii: RareByte<B>,
    /// Adaptive pre-rejection for the Unicode 0-typo path
    rare_unicode: RareByte<B>,
}

impl<B: Backend> Prefilter<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub unsafe fn new(needle: &str, case_sensitive: bool) -> Self {
        let needle_ascii: Vec<(B::Chunk, B::Chunk)> =
            case_needle(needle.as_bytes(), case_sensitive)
                .into_iter()
                .map(|(c1, c2)| unsafe { (B::splat(c1), B::splat(c2)) })
                .collect();
        let needle_unicode = case_needle_unicode(needle, case_sensitive);
        let rare_ascii = RareByte::new(needle_ascii.clone());
        let rare_unicode = RareByte::new(
            needle_unicode
                .iter()
                .map(|c| unsafe {
                    (
                        B::splat(c.chars[c.len - 1]),
                        B::splat(c.flipped_chars[c.len - 1]),
                    )
                })
                .collect(),
        );

        Self {
            needle_ascii,
            needle_unicode,
            paths: Vec::new(),
            rare_ascii,
            rare_unicode,
        }
    }

    /// Adaptive pre-rejection for backends with their own 0-typo ASCII walk
    /// (see [`RareByte::rejects`])
    ///
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub(crate) unsafe fn rare_ascii_rejects(&mut self, haystack: &[u8]) -> bool {
        unsafe { self.rare_ascii.rejects(haystack) }
    }

    #[inline(always)]
    unsafe fn needle_unchecked(&self, idx: usize) -> (B::Chunk, B::Chunk) {
        unsafe { *self.needle_ascii.get_unchecked(idx) }
    }

    #[inline(always)]
    unsafe fn unicode_needle_unchecked(&self, idx: usize) -> &UnicodeChar {
        unsafe { self.needle_unicode.get_unchecked(idx) }
    }
}
