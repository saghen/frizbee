use super::Backend;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterScalarBackend;

impl Backend for PrefilterScalarBackend {
    const LANES: usize = 16;

    type Chunk = [u8; 16];
    type Mask = u16;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> Self::Chunk {
        [c; 16]
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        let mut mask = 0u16;
        for (idx, &byte) in a.iter().enumerate() {
            if byte == b[idx] {
                mask |= 1u16 << idx;
            }
        }
        mask
    }

    #[inline(always)]
    unsafe fn broadcast(c: (u8, u8)) -> (Self::Chunk, Self::Chunk) {
        ([c.0; 16], [c.1; 16])
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
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        let mut mask = 0u16;
        for (idx, &byte) in chunk.iter().enumerate() {
            if byte == needle.0[idx] || byte == needle.1[idx] {
                mask |= 1u16 << idx;
            }
        }
        mask
    }
}
