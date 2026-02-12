use std::arch::x86_64::*;

use raw_cpuid::{CpuId, CpuIdReader};

use crate::simd::Aligned32;

#[derive(Debug, Clone, Copy)]
pub struct SSE256Vector(pub(crate) (__m128i, __m128i));

impl super::Vector for SSE256Vector {
    fn is_available<R: CpuIdReader>(cpuid: &CpuId<R>) -> bool {
        cpuid
            .get_feature_info()
            .is_some_and(|info| info.has_sse41())
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        Self((_mm_setzero_si128(), _mm_setzero_si128()))
    }

    #[inline(always)]
    unsafe fn splat_u8(value: u8) -> Self {
        Self((_mm_set1_epi8(value as i8), _mm_set1_epi8(value as i8)))
    }

    #[inline(always)]
    unsafe fn splat_u16(value: u16) -> Self {
        Self((_mm_set1_epi16(value as i16), _mm_set1_epi16(value as i16)))
    }

    #[inline(always)]
    unsafe fn load_aligned(data: *const u8) -> Self {
        Self((
            _mm_load_si128(data as *const __m128i),
            _mm_load_si128(data as *const __m128i),
        ))
    }

    #[inline(always)]
    unsafe fn load_unaligned(data: *const u8) -> Self {
        Self((
            _mm_loadu_si128(data as *const __m128i),
            _mm_loadu_si128(data as *const __m128i),
        ))
    }
    #[inline(always)]
    unsafe fn eq_u8(self, other: Self) -> Self {
        Self((
            _mm_cmpeq_epi8(self.0.0, other.0.0),
            _mm_cmpeq_epi8(self.0.1, other.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn gt_u8(self, other: Self) -> Self {
        Self((
            _mm_cmpgt_epi8(self.0.0, other.0.0),
            _mm_cmpgt_epi8(self.0.1, other.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn lt_u8(self, other: Self) -> Self {
        Self((
            _mm_cmplt_epi8(self.0.0, other.0.0),
            _mm_cmplt_epi8(self.0.1, other.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn max_u16(self, other: Self) -> Self {
        Self((
            _mm_max_epu16(self.0.0, other.0.0),
            _mm_max_epu16(self.0.1, other.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn smax_u16(self) -> u16 {
        // PHMINPOSUW finds minimum, so we invert to find maximum
        let all_ones = _mm_set1_epi16(-1); // 0xFFFF
        let inverted = (
            _mm_xor_si128(self.0.0, all_ones),
            _mm_xor_si128(self.0.1, all_ones),
        ); // ~v

        // Find minimum of inverted values (= maximum of original)
        let min_pos = (_mm_minpos_epu16(inverted.0), _mm_minpos_epu16(inverted.1));
        let min_pos = _mm_min_epu8(min_pos.0, min_pos.1);

        // Extract and invert back
        let min_val = _mm_extract_epi16(min_pos, 0) as u16;
        !min_val // Invert to get original max
    }

    #[inline(always)]
    unsafe fn add_u16(self, other: Self) -> Self {
        Self((
            _mm_add_epi16(self.0.0, other.0.0),
            _mm_add_epi16(self.0.1, other.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn subs_u16(self, other: Self) -> Self {
        Self((
            _mm_subs_epu16(self.0.0, other.0.0),
            _mm_subs_epu16(self.0.1, other.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self((
            _mm_and_si128(self.0.0, other.0.0),
            _mm_and_si128(self.0.1, other.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        Self((
            _mm_or_si128(self.0.0, other.0.0),
            _mm_or_si128(self.0.1, other.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        Self((
            _mm_xor_si128(self.0.0, other.0.0),
            _mm_xor_si128(self.0.1, other.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self((
            _mm_xor_si128(self.0.0, _mm_set1_epi32(-1)),
            _mm_xor_si128(self.0.1, _mm_set1_epi32(-1)),
        ))
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u16<const L: i32>(self, other: Self) -> Self {
        const { assert!(L >= 0 && L <= 8) };

        macro_rules! impl_shift {
            ($l:expr) => {
                Self((
                    _mm_alignr_epi8::<{ $l * 2 }>(self.0.0, other.0.1),
                    _mm_alignr_epi8::<{ $l * 2 }>(self.0.1, self.0.0),
                ))
            };
        }

        match L {
            0 => self,
            1 => impl_shift!(1),
            2 => impl_shift!(2),
            3 => impl_shift!(3),
            4 => impl_shift!(4),
            5 => impl_shift!(5),
            6 => impl_shift!(6),
            7 => impl_shift!(7),
            8 => Self((other.0.1, self.0.0)),
            _ => unreachable!(),
        }
    }
}

impl super::Vector256 for SSE256Vector {
    #[inline(always)]
    unsafe fn idx_u16(self, search: u16) -> usize {
        // compare all elements with max, get mask
        let (cmp_low, cmp_high) = (
            _mm_cmpeq_epi16(self.0.0, _mm_set1_epi16(search as i16)),
            _mm_cmpeq_epi16(self.0.1, _mm_set1_epi16(search as i16)),
        );
        let (mask_low, mask_high) = (
            _mm_movemask_epi8(cmp_low) as u32,
            _mm_movemask_epi8(cmp_high) as u32,
        );

        // find first set bit
        // divide by 2 to get element index (since u16 = 2 bytes)
        let low_trailing = mask_low.trailing_zeros() as usize / 2;
        let high_trailing = mask_high.trailing_zeros() as usize / 2 + 16;
        low_trailing.min(high_trailing)
    }

    #[inline(always)]
    unsafe fn from_aligned(data: Aligned32<[u8; 32]>) -> Self {
        Self(std::mem::transmute::<[u8; 32], (__m128i, __m128i)>(data.0))
    }

    #[inline(always)]
    unsafe fn blendv(self, other: Self, mask: Self) -> Self {
        Self((
            _mm_blendv_epi8(self.0.0, other.0.0, mask.0.0),
            _mm_blendv_epi8(self.0.1, other.0.1, mask.0.1),
        ))
    }

    #[inline(always)]
    unsafe fn shift_right_u16<const N: i32>(self) -> Self {
        const { assert!(N >= 0 && N <= 8) };

        Self(match N {
            0 => (self.0.0, self.0.1),
            1 => (
                _mm_srli_si128(self.0.0, 2),
                _mm_alignr_epi8(self.0.1, self.0.0, 2),
            ),
            2 => (
                _mm_srli_si128(self.0.0, 4),
                _mm_alignr_epi8(self.0.1, self.0.0, 4),
            ),
            3 => (
                _mm_srli_si128(self.0.0, 6),
                _mm_alignr_epi8(self.0.1, self.0.0, 6),
            ),
            4 => (
                _mm_srli_si128(self.0.0, 8),
                _mm_alignr_epi8(self.0.1, self.0.0, 8),
            ),
            5 => (
                _mm_srli_si128(self.0.0, 10),
                _mm_alignr_epi8(self.0.1, self.0.0, 10),
            ),
            6 => (
                _mm_srli_si128(self.0.0, 12),
                _mm_alignr_epi8(self.0.1, self.0.0, 12),
            ),
            7 => (
                _mm_srli_si128(self.0.0, 14),
                _mm_alignr_epi8(self.0.1, self.0.0, 14),
            ),
            _ => unreachable!(),
        })
    }
}
