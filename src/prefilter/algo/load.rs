use crate::prefilter::backend::{Backend, BitMaskOps};

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
        // if the read wouldn't pass a page boundary, we can read past the end
        // of the haystack and mask out the invalid bytes
        if can_overread(ptr, B::LANES) {
            (B::load(ptr), mask)
        } else {
            (B::load_partial(ptr, remaining, mask), mask)
        }
    }
}

#[inline(always)]
pub(super) unsafe fn load_window_maskless<B: Backend>(
    haystack: &[u8],
    start: usize,
    len: usize,
) -> B::Chunk {
    unsafe {
        debug_assert!(B::LANES <= 64);
        let remaining = len - start;
        if remaining >= B::LANES {
            return B::load(haystack.as_ptr().add(start));
        }

        let ptr = haystack.as_ptr().add(start);
        if can_overread(ptr, B::LANES) {
            B::load(ptr)
        } else {
            let mask = B::Mask::first_n(remaining);
            B::load_partial(ptr, remaining, mask)
        }
    }
}

#[inline(always)]
pub(crate) fn can_overread(ptr: *const u8, bytes: usize) -> bool {
    if cfg!(feature = "safe_read") || cfg!(miri) {
        return false;
    }
    (ptr as usize & 0xFFF) <= (4096 - bytes)
}
