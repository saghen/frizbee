use std::arch::x86_64::*;

use crate::simd::Aligned32;

#[derive(Debug, Clone, Copy)]
pub struct AVXVector(pub __m256i);

impl super::Vector for AVXVector {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(_mm256_setzero_si256())
    }

    #[inline(always)]
    unsafe fn splat_u8(value: u8) -> Self {
        Self(_mm256_set1_epi8(value as i8))
    }

    #[inline(always)]
    unsafe fn splat_u16(value: u16) -> Self {
        Self(_mm256_set1_epi16(value as i16))
    }

    #[inline(always)]
    unsafe fn load_aligned(data: *const u8) -> Self {
        Self(_mm256_load_si256(data as *const __m256i))
    }

    #[inline(always)]
    unsafe fn load_unaligned(data: *const u8) -> Self {
        Self(_mm256_loadu_si256(data as *const __m256i))
    }

    #[inline(always)]
    unsafe fn eq_u8(self, other: Self) -> Self {
        Self(_mm256_cmpeq_epi8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn gt_u8(self, other: Self) -> Self {
        Self(_mm256_cmpgt_epi8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn lt_u8(self, other: Self) -> Self {
        Self(_mm256_cmpgt_epi8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn max_u16(self, other: Self) -> Self {
        Self(_mm256_max_epu16(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn smax_u16(self) -> u16 {
        let high = _mm256_extracti128_si256(self.0, 1);
        let low = _mm256_castsi256_si128(self.0);

        let max128 = _mm_max_epu16(low, high);
        let max64 = _mm_max_epu16(max128, _mm_srli_si128(max128, 8));
        let max32 = _mm_max_epu16(max64, _mm_srli_si128(max64, 4));
        let max16 = _mm_max_epu16(max32, _mm_srli_si128(max32, 2));

        _mm_extract_epi16(max16, 0) as u16
    }

    #[inline(always)]
    unsafe fn add_u16(self, other: Self) -> Self {
        Self(_mm256_add_epi16(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn subs_u16(self, other: Self) -> Self {
        Self(_mm256_subs_epu16(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(_mm256_and_si256(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        Self(_mm256_or_si256(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        Self(_mm256_xor_si256(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self(_mm256_xor_si256(self.0, _mm256_set1_epi32(-1)))
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u16<const N: i32>(self, other: Self) -> Self {
        // Permute: [padding_low, padding_high] + [low, high] -> [padding_high, low]
        let shifted_lanes = _mm256_permute2x128_si256(other.0, self.0, 0x21);

        // alignr to shift right by 2 bytes within each 128-bit lane pair
        // alignr(b, combined, 2) does:
        //   low lane:  takes from [a_high : b_low] and shifts right by 2
        //   high lane: takes from [b_low : b_high] and shifts right by 2
        Self(match N {
            1 => _mm256_alignr_epi8(self.0, shifted_lanes, 14),
            2 => _mm256_alignr_epi8(self.0, shifted_lanes, 12),
            3 => _mm256_alignr_epi8(self.0, shifted_lanes, 8),
            4 => _mm256_alignr_epi8(self.0, shifted_lanes, 6),
            5 => _mm256_alignr_epi8(self.0, shifted_lanes, 4),
            6 => _mm256_alignr_epi8(self.0, shifted_lanes, 2),
            7 => _mm256_alignr_epi8(self.0, shifted_lanes, 0),
            _ => unreachable!(),
        })
    }
}

impl super::Vector256 for AVXVector {
    #[inline(always)]
    unsafe fn idx_u16(self, search: u16) -> usize {
        // compare all elements with max, get mask
        let cmp = _mm256_cmpeq_epi16(self.0, _mm256_set1_epi16(search as i16));
        let mask = _mm256_movemask_epi8(cmp) as u32;

        // find first set bit
        // divide by 2 to get element index (since u16 = 2 bytes)
        mask.trailing_zeros() as usize / 2
    }

    #[inline(always)]
    unsafe fn from_aligned(data: Aligned32<[u8; 32]>) -> Self {
        Self(std::mem::transmute::<[u8; 32], __m256i>(data.0))
    }

    #[inline(always)]
    unsafe fn blendv(self, other: Self, mask: Self) -> Self {
        Self(_mm256_blendv_epi8(self.0, other.0, mask.0))
    }

    #[inline(always)]
    unsafe fn shift_right_u16<const N: i32>(self) -> Self {
        // Permute: [low, high] -> [zeros, low]
        let shifted_lanes = _mm256_permute2x128_si256(self.0, self.0, 0x08);

        // alignr shifts within 128-bit lanes
        // We need to shift by 2 bytes (one u16) to the right
        // alignr(a, b, n) = (a:b) >> (n*8) for each 128-bit lane
        Self(match N {
            1 => _mm256_alignr_epi8(self.0, shifted_lanes, 14),
            2 => _mm256_alignr_epi8(self.0, shifted_lanes, 12),
            3 => _mm256_alignr_epi8(self.0, shifted_lanes, 8),
            4 => _mm256_alignr_epi8(self.0, shifted_lanes, 6),
            5 => _mm256_alignr_epi8(self.0, shifted_lanes, 4),
            6 => _mm256_alignr_epi8(self.0, shifted_lanes, 2),
            7 => _mm256_alignr_epi8(self.0, shifted_lanes, 0),
            _ => unreachable!(),
        })
    }
}
