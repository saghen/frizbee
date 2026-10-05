mod exact;
mod load;
mod rows;
mod typos;

pub(crate) use load::{can_overread, load_window};

use super::{backend::Backend, case_needle_unicode};
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub(crate) struct Prefilter<B: Backend> {
    /// Needle bytes splatted, lowercased unless case sensitive (the block
    /// walk folds the haystack to match, see [`rows`]), with their
    /// case-flipped variants for the chunk walks
    needle_ascii: Vec<(B::Chunk, B::Chunk)>,
    /// Needle scalars as rows of the block walk ([`rows`])
    needle_scalars: Vec<rows::ScalarRow<B>>,
    case_sensitive: bool,
    /// Per row carry (0 or 1) across blocks of the block walk ([`rows`])
    carries: Vec<u64>,
    /// The typo walk's budget, see [`Prefilter::match_haystack_typos`]
    #[cfg(target_arch = "wasm32")]
    max_typos: usize,
}

impl<B: Backend> Prefilter<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub unsafe fn new(needle: &str, case_sensitive: bool) -> Self {
        let needle_ascii = needle
            .bytes()
            .map(|c| unsafe {
                if case_sensitive {
                    (B::splat(c), B::splat(c))
                } else {
                    (
                        B::splat(c.to_ascii_lowercase()),
                        B::splat(c.to_ascii_uppercase()),
                    )
                }
            })
            .collect();
        let needle_scalars = case_needle_unicode(needle, case_sensitive)
            .iter()
            .map(|c| unsafe { rows::ScalarRow::new(c) })
            .collect();
        Self {
            needle_ascii,
            needle_scalars,
            case_sensitive,
            carries: Vec::new(),
            #[cfg(target_arch = "wasm32")]
            max_typos: 0,
        }
    }
}
