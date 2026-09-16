use core::arch::aarch64::*;

use super::{Backend, BitMaskOps};

/// Four bits per lane, so NEON comparisons only need SHRN to pack their lanes
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct NeonMask(u64);

impl BitMaskOps for NeonMask {
    #[inline(always)]
    fn zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    fn all() -> Self {
        Self(u64::MAX)
    }

    #[inline(always)]
    fn first_n(n: usize) -> Self {
        Self(u64::first_n(n.min(16) * 4))
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    fn trailing_zeros(self) -> usize {
        self.0.trailing_zeros() as usize / 4
    }

    #[inline(always)]
    fn leading_zeros(self) -> usize {
        self.0.leading_zeros() as usize / 4
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    fn clear_through_lowest(self, hit: Self) -> Self {
        // Clear through the matched nibble's top bit to consume the whole lane
        Self(self.0.clear_through_lowest(hit.0 & 0x8888_8888_8888_8888))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterNEONBackend;

impl Backend for PrefilterNEONBackend {
    const LANES: usize = 16;

    type Chunk = uint8x16_t;
    type Mask = NeonMask;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> Self::Chunk {
        unsafe { vdupq_n_u8(c) }
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        unsafe { movemask_u8(vceqq_u8(a, b)) }
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
unsafe fn movemask_u8(mask: uint8x16_t) -> NeonMask {
    unsafe {
        let nibbles = vshrn_n_u16::<4>(vreinterpretq_u16_u8(mask));
        NeonMask(vget_lane_u64::<0>(vreinterpret_u64_u8(nibbles)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefilter::backend::PrefilterScalarBackend;

    #[test]
    fn masks_match_scalar_exhaustively() {
        for bits in 0..=u16::MAX {
            let bytes: [u8; 16] =
                core::array::from_fn(|i| if bits & (1 << i) != 0 { b'a' } else { b'b' });
            let mask = unsafe {
                let chunk = PrefilterNEONBackend::load(bytes.as_ptr());
                let a = PrefilterNEONBackend::splat(b'a');
                let uppercase_a = PrefilterNEONBackend::splat(b'A');
                let mask = PrefilterNEONBackend::eq(chunk, a);
                assert_eq!(mask, PrefilterNEONBackend::occ(chunk, (a, uppercase_a)));
                assert_eq!(bits, PrefilterScalarBackend::eq(bytes, [b'a'; 16]));
                mask
            };
            assert_eq!(mask.is_zero(), bits == 0);
            assert_eq!(mask.trailing_zeros(), bits.trailing_zeros() as usize);
            assert_eq!(mask.leading_zeros(), bits.leading_zeros() as usize);
            assert_eq!(mask.or(NeonMask::zero()), mask);
            assert_eq!(mask.or(NeonMask::all()), NeonMask::all());

            let mut remaining = mask;
            let mut scalar = bits;
            while !remaining.is_zero() {
                assert_eq!(remaining.trailing_zeros(), scalar.trailing_zeros() as usize);
                remaining = remaining.clear_through_lowest(remaining);
                scalar &= scalar - 1;
            }
            assert_eq!(scalar, 0);

            for n in 0..=17 {
                let prefix = NeonMask::first_n(n);
                let scalar_prefix = u16::first_n(n);
                let hits = mask.and(prefix);
                let scalar_hits = bits & scalar_prefix;
                assert_eq!(hits.is_zero(), scalar_hits == 0);
                assert_eq!(hits.trailing_zeros(), scalar_hits.trailing_zeros() as usize);
                assert_eq!(hits.leading_zeros(), scalar_hits.leading_zeros() as usize);
                let cleared = prefix.clear_through_lowest(hits);
                let scalar_cleared = scalar_prefix.clear_through_lowest(scalar_hits);
                assert_eq!(cleared.is_zero(), scalar_cleared == 0);
                assert_eq!(
                    cleared.trailing_zeros(),
                    scalar_cleared.trailing_zeros() as usize
                );
                assert_eq!(
                    cleared.leading_zeros(),
                    scalar_cleared.leading_zeros() as usize
                );
                assert_eq!(
                    cleared.partial_cmp(&prefix),
                    scalar_cleared.partial_cmp(&scalar_prefix)
                );
            }
        }
    }
}
