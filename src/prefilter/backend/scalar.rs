use super::Backend;

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
