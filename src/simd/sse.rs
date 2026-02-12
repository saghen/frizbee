use std::arch::x86_64::*;

use raw_cpuid::{CpuId, CpuIdReader};

use crate::simd::{AVXVector, SSE256Vector};

#[derive(Debug, Clone, Copy)]
pub struct SSEVector(__m128i);

impl SSEVector {
    #[inline(always)]
    unsafe fn load_partial_safe(ptr: *const u8, len: usize) -> __m128i {
        debug_assert!(len < 8);

        let val: u64 = match len {
            0 => 0,
            1 => *ptr as u64,
            2 => (ptr as *const u16).read_unaligned() as u64,
            3 => {
                let lo = (ptr as *const u16).read_unaligned() as u64;
                let hi = *ptr.add(2) as u64;
                lo | (hi << 16)
            }
            4 => (ptr as *const u32).read_unaligned() as u64,
            5 => {
                let lo = (ptr as *const u32).read_unaligned() as u64;
                let hi = *ptr.add(4) as u64;
                lo | (hi << 32)
            }
            6 => {
                let lo = (ptr as *const u32).read_unaligned() as u64;
                let hi = (ptr.add(4) as *const u16).read_unaligned() as u64;
                lo | (hi << 32)
            }
            7 => {
                let lo = (ptr as *const u32).read_unaligned() as u64;
                let hi = (ptr.add(4) as *const u32).read_unaligned() as u64;
                // Mask hi to only 3 bytes (24 bits)
                lo | ((hi & 0xFFFFFF) << 32)
            }
            _ => std::hint::unreachable_unchecked(),
        };

        _mm_cvtsi64_si128(val as i64)
    }

    #[inline(always)]
    fn can_overread_8(ptr: *const u8) -> bool {
        // Safe if not in last 7 bytes of a page
        (ptr as usize & 0xFFF) <= (4096 - 8)
    }
}

impl super::Vector for SSEVector {
    #[inline]
    fn is_available<R: CpuIdReader>(cpuid: &CpuId<R>) -> bool {
        cpuid
            .get_feature_info()
            .is_some_and(|info| info.has_sse41())
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(_mm_setzero_si128())
    }

    #[inline(always)]
    unsafe fn splat_u8(value: u8) -> Self {
        Self(_mm_set1_epi8(value as i8))
    }

    #[inline(always)]
    unsafe fn splat_u16(value: u16) -> Self {
        Self(_mm_set1_epi16(value as i16))
    }

    #[inline(always)]
    unsafe fn load_aligned(data: *const u8) -> Self {
        Self(_mm_load_si128(data as *const __m128i))
    }

    #[inline(always)]
    unsafe fn load_unaligned(data: *const u8) -> Self {
        Self(_mm_loadu_si128(data as *const __m128i))
    }
    #[inline(always)]
    unsafe fn eq_u8(self, other: Self) -> Self {
        Self(_mm_cmpeq_epi8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn gt_u8(self, other: Self) -> Self {
        Self(_mm_cmpgt_epi8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn lt_u8(self, other: Self) -> Self {
        Self(_mm_cmplt_epi8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn max_u16(self, other: Self) -> Self {
        Self(_mm_max_epu16(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn smax_u16(self) -> u16 {
        // PHMINPOSUW finds minimum, so we invert to find maximum
        let all_ones = _mm_set1_epi16(-1); // 0xFFFF
        let inverted = _mm_xor_si128(self.0, all_ones); // ~v

        // Find minimum of inverted values (= maximum of original)
        let min_pos = _mm_minpos_epu16(inverted);

        // Extract and invert back
        let min_val = _mm_extract_epi16(min_pos, 0) as u16;
        !min_val // Invert to get original max
    }

    #[inline(always)]
    unsafe fn add_u16(self, other: Self) -> Self {
        Self(_mm_add_epi16(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn subs_u16(self, other: Self) -> Self {
        Self(_mm_subs_epu16(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(_mm_and_si128(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        Self(_mm_or_si128(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        Self(_mm_xor_si128(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1)))
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u16<const L: i32>(self, other: Self) -> Self {
        Self(_mm_alignr_epi8::<L>(self.0, other.0))
    }
}

impl super::Vector128 for SSEVector {
    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        Self(match len {
            0 => _mm_setzero_si128(),
            8 => _mm_loadl_epi64(data as *const __m128i),
            16 => _mm_loadu_si128(data as *const __m128i),

            1..=7 if Self::can_overread_8(data) => {
                let low = _mm_loadl_epi64(data as *const __m128i);
                let mask = _mm_set_epi64x(0, (1i64 << (len * 8)) - 1);
                _mm_and_si128(low, mask)
            }
            1..=7 => Self::load_partial_safe(data, len),
            9..=15 => {
                let lo = _mm_loadl_epi64(data as *const __m128i);

                let high_start = len - 8;
                let high = _mm_loadl_epi64(data.add(high_start) as *const __m128i);
                let mask = _mm_set_epi64x(0, (1i64 << ((len - 8) * 8)) - 1);
                let high = _mm_and_si128(high, mask);

                _mm_unpacklo_epi64(lo, high)
            }

            _ if start + 16 <= len => _mm_loadu_si128(data.add(start) as *const __m128i),
            _ => {
                let overlap = start + 16 - len; // bytes we need to shift out
                let data = _mm_loadu_si128(data.add(len - 16) as *const __m128i);

                // Shift left by 'overlap' bytes to align data to start position
                // This zeros out the rightmost 'overlap' bytes and shifts content left
                match overlap {
                    1 => _mm_slli_si128(data, 1),
                    2 => _mm_slli_si128(data, 2),
                    3 => _mm_slli_si128(data, 3),
                    4 => _mm_slli_si128(data, 4),
                    5 => _mm_slli_si128(data, 5),
                    6 => _mm_slli_si128(data, 6),
                    7 => _mm_slli_si128(data, 7),
                    8 => _mm_slli_si128(data, 8),
                    9 => _mm_slli_si128(data, 9),
                    10 => _mm_slli_si128(data, 10),
                    11 => _mm_slli_si128(data, 11),
                    12 => _mm_slli_si128(data, 12),
                    13 => _mm_slli_si128(data, 13),
                    14 => _mm_slli_si128(data, 14),
                    15 => _mm_slli_si128(data, 15),
                    _ => _mm_setzero_si128(),
                }
            }
        })
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u8<const L: i32>(self, other: Self) -> Self {
        Self(_mm_alignr_epi8::<L>(self.0, other.0))
    }
}

impl super::Vector128Expansion<AVXVector> for SSEVector {
    #[inline(always)]
    unsafe fn cast_i8_to_i16(self) -> AVXVector {
        AVXVector(_mm256_cvtepi8_epi16(self.0))
    }
}

impl super::Vector128Expansion<SSE256Vector> for SSEVector {
    #[inline(always)]
    unsafe fn cast_i8_to_i16(self) -> SSE256Vector {
        // Lower 8 bytes → 8 x i16
        let lo = _mm_cvtepi8_epi16(self.0);

        // Shift upper 8 bytes to lower position, then expand
        let hi = _mm_cvtepi8_epi16(_mm_srli_si128(self.0, 8));

        SSE256Vector((lo, hi))
    }
}
