mod ascii;
mod block;
mod load;
mod rare;
mod unicode;

#[cfg(target_arch = "x86_64")]
pub(crate) use ascii::find_last_char_pos;
pub(crate) use load::{can_overread, load_window};

use super::{UnicodeChar, backend::Backend, case_needle, case_needle_unicode};
use alloc::vec::Vec;
use rare::RareByte;

#[derive(Debug, Clone)]
pub(crate) struct Prefilter<B: Backend> {
    needle_ascii: Vec<(B::Chunk, B::Chunk)>,
    needle_unicode: Vec<UnicodeChar>,
    /// Needle bytes splatted, lowercased unless case sensitive (the block
    /// walk folds the haystack to match, see [`block`])
    needle_folded: Vec<B::Chunk>,
    /// Needle scalars as rows of the block walk ([`block`])
    needle_scalars: Vec<block::ScalarRow<B>>,
    case_sensitive: bool,
    /// Block walk state carried across blocks ([`block`])
    ended: Vec<u64>,
    /// Walk state for runtime typo budgets ([`block`])
    state: Vec<B::Block>,
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
        let needle_cased = case_needle(needle.as_bytes(), case_sensitive);
        let needle_ascii: Vec<(B::Chunk, B::Chunk)> = needle_cased
            .iter()
            .map(|&(c1, c2)| unsafe { (B::splat(c1), B::splat(c2)) })
            .collect();
        let needle_folded = needle_cased
            .iter()
            .map(|&(c, _)| unsafe {
                B::splat(if case_sensitive {
                    c
                } else {
                    c.to_ascii_lowercase()
                })
            })
            .collect();
        let needle_unicode = case_needle_unicode(needle, case_sensitive);
        let needle_scalars: Vec<block::ScalarRow<B>> = needle_unicode
            .iter()
            .map(|c| unsafe { block::ScalarRow::new(c) })
            .collect();
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
            needle_folded,
            needle_scalars,
            case_sensitive,
            ended: Vec::new(),
            state: Vec::new(),
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
}
