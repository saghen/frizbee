use core::arch::aarch64::*;

use super::{Backend, BitMaskOps};
use crate::prefilter::{Kernel, Window, algo::Prefilter};

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

/// Four bits per lane over 32 lanes: two SHRN-packed comparisons in one u128,
/// halving the chunk iterations (and mispredicted branches) of the 16-lane mask
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct NeonWideMask(u128);

const WIDE_LANE_TOP_BITS: u128 = 0x8888_8888_8888_8888_8888_8888_8888_8888;

impl BitMaskOps for NeonWideMask {
    #[inline(always)]
    fn zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    fn all() -> Self {
        Self(u128::MAX)
    }

    #[inline(always)]
    fn first_n(n: usize) -> Self {
        if n >= 32 {
            Self(u128::MAX)
        } else {
            Self((1u128 << (n * 4)) - 1)
        }
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
        let hit = hit.0 & WIDE_LANE_TOP_BITS;
        Self(self.0 & !(hit ^ hit.wrapping_sub(1)))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterNEONWideBackend;

impl Backend for PrefilterNEONWideBackend {
    const LANES: usize = 32;

    type Chunk = uint8x16x2_t;
    type Mask = NeonWideMask;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> Self::Chunk {
        unsafe {
            let v = vdupq_n_u8(c);
            uint8x16x2_t(v, v)
        }
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        unsafe { movemask_u8_wide(vceqq_u8(a.0, b.0), vceqq_u8(a.1, b.1)) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { vld1q_u8_x2(ptr) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        unsafe {
            let lo = vorrq_u8(vceqq_u8(needle.0.0, chunk.0), vceqq_u8(needle.1.0, chunk.0));
            let hi = vorrq_u8(vceqq_u8(needle.0.1, chunk.1), vceqq_u8(needle.1.1, chunk.1));
            movemask_u8_wide(lo, hi)
        }
    }
}

/// Packs two 16-lane byte masks into one nibble-per-lane u128 (`lo` first)
#[inline(always)]
unsafe fn movemask_u8_wide(lo: uint8x16_t, hi: uint8x16_t) -> NeonWideMask {
    unsafe {
        let nibbles = vshrn_n_u16::<4>(vreinterpretq_u16_u8(lo));
        let nibbles = vshrn_high_n_u16::<4>(nibbles, vreinterpretq_u16_u8(hi));
        NeonWideMask(core::mem::transmute::<uint8x16_t, u128>(nibbles))
    }
}

/// Plain ASCII matching on 32-byte windows, where the unpredictable per-chunk
/// branch dominates. Unicode and typo paths keep several masks live, so u128
/// masks spill there; they stay on 16-byte windows.
#[derive(Debug, Clone)]
pub(crate) struct PrefilterNEON {
    wide: Prefilter<PrefilterNEONWideBackend>,
    narrow: Prefilter<PrefilterNEONBackend>,
}

impl Kernel for PrefilterNEON {
    #[inline(always)]
    fn new(needle: &str, case_sensitive: bool) -> Self {
        Self {
            wide: Kernel::new(needle, case_sensitive),
            narrow: Kernel::new(needle, case_sensitive),
        }
    }

    #[inline(always)]
    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    fn match_haystack(&self, haystack: &[u8]) -> Window {
        Kernel::match_haystack(&self.wide, haystack)
    }

    #[inline(always)]
    fn match_haystack_unicode(&self, haystack: &[u8]) -> Window {
        Kernel::match_haystack_unicode(&self.narrow, haystack)
    }

    #[inline(always)]
    fn match_haystack_1_typo(&self, haystack: &[u8]) -> Window {
        Kernel::match_haystack_1_typo(&self.narrow, haystack)
    }

    #[inline(always)]
    fn match_haystack_unicode_1_typo(&self, haystack: &[u8]) -> Window {
        Kernel::match_haystack_unicode_1_typo(&self.narrow, haystack)
    }

    #[inline(always)]
    fn match_haystack_2_typos(&self, haystack: &[u8]) -> Window {
        Kernel::match_haystack_2_typos(&self.narrow, haystack)
    }

    #[inline(always)]
    fn match_haystack_unicode_2_typos(&self, haystack: &[u8]) -> Window {
        Kernel::match_haystack_unicode_2_typos(&self.narrow, haystack)
    }

    #[inline(always)]
    fn match_haystack_many_typos(&mut self, haystack: &[u8], max_typos: u16) -> Window {
        Kernel::match_haystack_many_typos(&mut self.narrow, haystack, max_typos)
    }

    #[inline(always)]
    fn match_haystack_unicode_many_typos(&mut self, haystack: &[u8], max_typos: u16) -> Window {
        Kernel::match_haystack_unicode_many_typos(&mut self.narrow, haystack, max_typos)
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

    fn wide_reference_first_n(n: usize) -> u32 {
        if n >= 32 { u32::MAX } else { (1u32 << n) - 1 }
    }

    fn check_wide(bits: u32) {
        let bytes: [u8; 32] =
            core::array::from_fn(|i| if bits & (1 << i) != 0 { b'a' } else { b'b' });
        let mask = unsafe {
            let chunk = PrefilterNEONWideBackend::load(bytes.as_ptr());
            let a = PrefilterNEONWideBackend::splat(b'a');
            let uppercase_a = PrefilterNEONWideBackend::splat(b'A');
            let mask = PrefilterNEONWideBackend::eq(chunk, a);
            assert_eq!(mask, PrefilterNEONWideBackend::occ(chunk, (a, uppercase_a)));
            assert_eq!(mask, PrefilterNEONWideBackend::occ(chunk, (uppercase_a, a)));
            mask
        };
        assert_eq!(mask.is_zero(), bits == 0);
        assert_eq!(mask.trailing_zeros(), bits.trailing_zeros() as usize);
        assert_eq!(mask.leading_zeros(), bits.leading_zeros() as usize);
        assert_eq!(mask.or(NeonWideMask::zero()), mask);
        assert_eq!(mask.or(NeonWideMask::all()), NeonWideMask::all());
        assert_eq!(mask.and(NeonWideMask::all()), mask);

        let mut remaining = mask;
        let mut scalar = bits;
        while !remaining.is_zero() {
            assert_eq!(remaining.trailing_zeros(), scalar.trailing_zeros() as usize);
            remaining = remaining.clear_through_lowest(remaining);
            scalar &= scalar - 1;
        }
        assert_eq!(scalar, 0);

        for n in 0..=33 {
            let prefix = NeonWideMask::first_n(n);
            let scalar_prefix = wide_reference_first_n(n);
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

    #[test]
    fn wide_masks_match_scalar_half_exhaustively() {
        for bits in 0..=u16::MAX {
            check_wide(bits as u32);
            check_wide((bits as u32) << 16);
            check_wide((bits as u32) | ((bits as u32) << 16));
            check_wide((bits as u32) | ((!bits as u32) << 16));
        }
    }

    #[test]
    fn wide_masks_match_scalar_pseudo_random() {
        // xorshift32 covers patterns that touch both halves independently
        let mut x = 0x9E37_79B9u32;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            check_wide(x);
        }
    }

    #[test]
    fn wide_ordering_matches_lane_order() {
        // masks compare like the lane sets they represent, lowest lane first
        for n in 0..32 {
            assert!(NeonWideMask::first_n(n) < NeonWideMask::first_n(n + 1));
        }
    }
}
