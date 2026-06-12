use crate::prefilter::algo::PrefilterImpl;

use super::Backend;

#[derive(Debug, Clone)]
pub struct PrefilterScalar(PrefilterImpl<PrefilterScalarBackend>);

impl PrefilterScalar {
    pub fn new(needle: &[u8]) -> Self {
        Self(unsafe { PrefilterImpl::new(needle) })
    }

    pub fn is_available() -> bool {
        PrefilterImpl::<PrefilterScalarBackend>::is_available()
    }

    #[inline(never)]
    pub fn match_haystack(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.0.match_haystack(haystack) }
    }

    #[inline(never)]
    pub fn match_haystack_1_typo(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.0.match_haystack_1_typo(haystack) }
    }

    #[inline(never)]
    pub fn match_haystack_2_typos(&self, haystack: &[u8]) -> (bool, usize, usize) {
        unsafe { self.0.match_haystack_2_typos(haystack) }
    }

    #[inline(never)]
    pub fn match_haystack_typos(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> (bool, usize, usize) {
        match max_typos {
            0 => self.match_haystack(haystack),
            1 => self.match_haystack_1_typo(haystack),
            2 => self.match_haystack_2_typos(haystack),
            _ => unsafe { self.0.match_haystack_typos(haystack, max_typos) },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterScalarBackend;

impl Backend for PrefilterScalarBackend {
    const LANES: usize = 16;

    type Chunk = [u8; 16];
    type Needle = (u8, u8);
    type Mask = u16;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn broadcast(c1: u8, c2: u8) -> Self::Needle {
        (c1, c2)
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe {
            let mut chunk = [0u8; 16];
            std::ptr::copy_nonoverlapping(ptr, chunk.as_mut_ptr(), 16);
            chunk
        }
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const u8, remaining: usize, _mask: Self::Mask) -> Self::Chunk {
        unsafe {
            let mut chunk = [0u8; 16];
            std::ptr::copy_nonoverlapping(ptr, chunk.as_mut_ptr(), remaining);
            chunk
        }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: Self::Needle) -> Self::Mask {
        let mut mask = 0u16;
        for (idx, &byte) in chunk.iter().enumerate() {
            if byte == needle.0 || byte == needle.1 {
                mask |= 1u16 << idx;
            }
        }
        mask
    }
}
