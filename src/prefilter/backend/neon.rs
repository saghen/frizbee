use std::arch::aarch64::*;

use super::Backend;

const BIT_WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterNEONBackend;

impl Backend for PrefilterNEONBackend {
    const LANES: usize = 16;

    type Chunk = uint8x16_t;
    type Mask = u16;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn broadcast(c: (u8, u8)) -> (Self::Chunk, Self::Chunk) {
        unsafe { (vdupq_n_u8(c.0), vdupq_n_u8(c.1)) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { vld1q_u8(ptr) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        unsafe {
            let mask = vorrq_u8(vceqq_u8(needle.0, chunk), vceqq_u8(needle.1, chunk));
            movemask_u8(mask)
        }
    }
}

#[inline(always)]
unsafe fn movemask_u8(mask: uint8x16_t) -> u16 {
    unsafe {
        let bits = vandq_u8(vshrq_n_u8::<7>(mask), vld1q_u8(BIT_WEIGHTS.as_ptr()));
        let lo = vaddv_u8(vget_low_u8(bits)) as u16;
        let hi = vaddv_u8(vget_high_u8(bits)) as u16;
        lo | (hi << 8)
    }
}
