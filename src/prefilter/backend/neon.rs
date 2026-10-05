use core::arch::aarch64::*;

use super::{Backend, BitMaskOps, ChunkBlock, Register, load_block_chunks, pair::PairChunks};

/// Four bits per lane, so NEON comparisons only need SHRN to pack their lanes
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NeonMask(u64);

impl From<NeonMask> for u64 {
    /// One bit per byte instead of four
    #[inline(always)]
    fn from(mask: NeonMask) -> u64 {
        let bits = mask.0 & 0x1111_1111_1111_1111;
        let bits = (bits | (bits >> 3)) & 0x0303_0303_0303_0303;
        let bits = (bits | (bits >> 6)) & 0x000F_000F_000F_000F;
        let bits = (bits | (bits >> 12)) & 0x0000_00FF_0000_00FF;
        (bits | (bits >> 24)) & 0xFFFF
    }
}

impl BitMaskOps for NeonMask {
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
    type Block = u64;
    type Chunks = ChunkBlock<uint8x16_t, 4>;

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

    #[inline(always)]
    unsafe fn load_block(haystack: &[u8]) -> (Self::Chunks, Self::Block) {
        unsafe { load_block_chunks::<Self, 4>(haystack) }
    }

    #[inline(always)]
    unsafe fn fold_block(chunks: &mut Self::Chunks) {
        unsafe {
            for chunk in chunks.chunks.iter_mut().take(chunks.count) {
                *chunk = Self::fold(*chunk);
            }
        }
    }

    /// All four chunks at once (see [`pack_bits`])
    #[inline(always)]
    unsafe fn eq_block(chunks: &Self::Chunks, needle: Self::Chunk) -> Self::Block {
        unsafe {
            let [a, b, c, d] = chunks.chunks;
            pack_bits([
                vceqq_u8(a, needle),
                vceqq_u8(b, needle),
                vceqq_u8(c, needle),
                vceqq_u8(d, needle),
            ])
        }
    }
}

impl Register for PrefilterNEONBackend {
    #[inline(always)]
    unsafe fn cmpeq(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vceqq_u8(a, b) }
    }

    #[inline(always)]
    unsafe fn or(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe { vorrq_u8(a, b) }
    }

    /// One bit per byte rather than [`NeonMask`]'s four, so masks of two
    /// registers fit a u32 and need no compaction (see [`pack_bits`])
    #[inline(always)]
    unsafe fn movemask_pair(lo: uint8x16_t, hi: uint8x16_t) -> u32 {
        unsafe {
            let bits = vld1q_u8(BITS.as_ptr());
            let sums = vpaddq_u8(vminq_u8(lo, bits), vminq_u8(hi, bits));
            let sums = vget_low_u8(vpaddq_u8(sums, sums));
            vget_lane_u32::<0>(vreinterpret_u32_u8(vpadd_u8(sums, sums)))
        }
    }

    #[inline(always)]
    unsafe fn fold(chunk: uint8x16_t) -> uint8x16_t {
        unsafe {
            let upper = vcltq_u8(vsubq_u8(chunk, vdupq_n_u8(b'A')), vdupq_n_u8(26));
            vorrq_u8(chunk, vandq_u8(upper, vdupq_n_u8(0x20)))
        }
    }

    /// Two chunks at a time (see [`pack_bits`]), the last two only when
    /// loaded
    #[inline(always)]
    unsafe fn eq_block_pair(chunks: &PairChunks<Self>, needle: uint8x16_t) -> u128 {
        unsafe {
            let [[a, b], [c, d], [e, f], [g, h]] = chunks.chunks;
            let low = pack_bits([
                vceqq_u8(a, needle),
                vceqq_u8(b, needle),
                vceqq_u8(c, needle),
                vceqq_u8(d, needle),
            ]);
            if chunks.count <= 2 {
                return low as u128;
            }
            let high = pack_bits([
                vceqq_u8(e, needle),
                vceqq_u8(f, needle),
                vceqq_u8(g, needle),
                vceqq_u8(h, needle),
            ]);
            ((high as u128) << 64) | low as u128
        }
    }
}

/// Each byte's bit within its 8-byte group
const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// One bit per byte of four compare results: weighting each byte by its bit
/// within its 8-byte group and summing pairwise packs them into one u64 with
/// a single transfer, where per-register masks need a nibble compaction
/// each. UMIN rather than AND keeps LLVM from proving the sums disjoint,
/// which turns each ADDP into UZP1 + UZP2 + ORR.
#[inline(always)]
unsafe fn pack_bits([a, b, c, d]: [uint8x16_t; 4]) -> u64 {
    unsafe {
        let bits = vld1q_u8(BITS.as_ptr());
        let (a, b, c, d) = (
            vminq_u8(a, bits),
            vminq_u8(b, bits),
            vminq_u8(c, bits),
            vminq_u8(d, bits),
        );
        let sums = vpaddq_u8(vpaddq_u8(a, b), vpaddq_u8(c, d));
        let sums = vpadd_u8(vget_low_u8(sums), vget_high_u8(sums));
        vget_lane_u64::<0>(vreinterpret_u64_u8(sums))
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
            assert_eq!(u64::from(mask), bits as u64);
            assert_eq!(mask.is_zero(), bits == 0);
            assert_eq!(mask.trailing_zeros(), bits.trailing_zeros() as usize);
            assert_eq!(mask.leading_zeros(), bits.leading_zeros() as usize);

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
            }
        }
    }
}
