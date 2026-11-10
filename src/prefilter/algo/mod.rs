mod ascii;
mod ascii_typos;
mod load;
mod unicode;
mod unicode_typos;

#[cfg(target_arch = "x86_64")]
pub(crate) use ascii::find_last_char_pos;
pub(crate) use load::{can_overread, load_window};

use super::{UnicodeChar, backend::Backend, case_needle, case_needle_unicode};
use alloc::vec::Vec;

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
    // byte -> bitset of needle positions holding that byte; empty when needle > 64 bytes
    lcs_masks: Vec<u64>,
}

impl<B: Backend> Prefilter<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub unsafe fn new(needle: &str, case_sensitive: bool) -> Self {
        let needle_ascii = case_needle(needle.as_bytes(), case_sensitive)
            .into_iter()
            .map(|(c1, c2)| unsafe { (B::splat(c1), B::splat(c2)) })
            .collect();
        let needle_unicode = case_needle_unicode(needle, case_sensitive);

        let mut lcs_masks = Vec::new();
        if (1..=64).contains(&needle.len()) {
            lcs_masks.resize(256, 0u64);
            for (i, (c1, c2)) in case_needle(needle.as_bytes(), case_sensitive)
                .into_iter()
                .enumerate()
            {
                lcs_masks[c1 as usize] |= 1 << i;
                lcs_masks[c2 as usize] |= 1 << i;
            }
        }

        Self {
            needle_ascii,
            needle_unicode,
            paths: Vec::new(),
            lcs_masks,
        }
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
