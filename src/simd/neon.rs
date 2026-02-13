use std::arch::aarch64::*;

use super::NEON256Vector;

#[derive(Debug, Clone, Copy)]
pub struct NEONVector(uint8x16_t);

impl NEONVector {
    #[inline(always)]
    unsafe fn load_partial_safe(ptr: *const u8, len: usize) -> uint8x16_t {
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
                lo | ((hi & 0xFFFFFF) << 32)
            }
            _ => std::hint::unreachable_unchecked(),
        };

        vcombine_u8(vreinterpret_u8_u64(vdup_n_u64(val)), vdup_n_u8(0))
    }

    #[inline(always)]
    fn can_overread_8(ptr: *const u8) -> bool {
        (ptr as usize & 0xFFF) <= (4096 - 8)
    }
}

impl super::Vector for NEONVector {
    #[inline]
    fn is_available() -> bool {
        // NEON is mandatory on aarch64
        cfg!(target_arch = "aarch64")
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(vdupq_n_u8(0))
    }

    #[inline(always)]
    unsafe fn splat_u8(value: u8) -> Self {
        Self(vdupq_n_u8(value))
    }

    #[inline(always)]
    unsafe fn splat_u16(value: u16) -> Self {
        Self(vreinterpretq_u8_u16(vdupq_n_u16(value)))
    }

    #[inline(always)]
    unsafe fn eq_u8(self, other: Self) -> Self {
        Self(vceqq_u8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn gt_u8(self, other: Self) -> Self {
        Self(vcgtq_u8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn lt_u8(self, other: Self) -> Self {
        Self(vcltq_u8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn max_u16(self, other: Self) -> Self {
        Self(vreinterpretq_u8_u16(vmaxq_u16(
            vreinterpretq_u16_u8(self.0),
            vreinterpretq_u16_u8(other.0),
        )))
    }

    #[inline(always)]
    unsafe fn smax_u16(self) -> u16 {
        vmaxvq_u16(vreinterpretq_u16_u8(self.0))
    }

    #[inline(always)]
    unsafe fn add_u16(self, other: Self) -> Self {
        Self(vreinterpretq_u8_u16(vaddq_u16(
            vreinterpretq_u16_u8(self.0),
            vreinterpretq_u16_u8(other.0),
        )))
    }

    #[inline(always)]
    unsafe fn subs_u16(self, other: Self) -> Self {
        Self(vreinterpretq_u8_u16(vqsubq_u16(
            vreinterpretq_u16_u8(self.0),
            vreinterpretq_u16_u8(other.0),
        )))
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(vandq_u8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        Self(vorrq_u8(self.0, other.0))
    }

    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self(vmvnq_u8(self.0))
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u16<const L: i32>(self, other: Self) -> Self {
        assert!(L >= 0 && L <= 8);
        match L {
            0 => self,
            1 => Self(vextq_u8(other.0, self.0, 14)),
            2 => Self(vextq_u8(other.0, self.0, 12)),
            3 => Self(vextq_u8(other.0, self.0, 10)),
            4 => Self(vextq_u8(other.0, self.0, 8)),
            5 => Self(vextq_u8(other.0, self.0, 6)),
            6 => Self(vextq_u8(other.0, self.0, 4)),
            7 => Self(vextq_u8(other.0, self.0, 2)),
            8 => Self(other.0),
            _ => unreachable!(),
        }
    }

    #[cfg(test)]
    fn from_array(arr: [u8; 16]) -> Self {
        Self(unsafe { vld1q_u8(arr.as_ptr()) })
    }
    #[cfg(test)]
    fn to_array(self) -> [u8; 16] {
        let mut arr = [0u8; 16];
        unsafe { vst1q_u8(arr.as_mut_ptr(), self.0) };
        arr
    }
    #[cfg(test)]
    fn from_array_u16(arr: [u16; 8]) -> Self {
        Self(unsafe { vld1q_u8(arr.as_ptr() as *const u8) })
    }
    #[cfg(test)]
    fn to_array_u16(self) -> [u16; 8] {
        let mut arr = [0u16; 8];
        unsafe { vst1q_u16(arr.as_mut_ptr(), vreinterpretq_u16_u8(self.0)) };
        arr
    }
}

impl super::Vector128 for NEONVector {
    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        Self(match len {
            0 => vdupq_n_u8(0),
            8 => {
                let lo = vld1_u8(data);
                vcombine_u8(lo, vdup_n_u8(0))
            }
            16 => vld1q_u8(data),

            1..=7 if Self::can_overread_8(data) => {
                let lo = vld1_u8(data);
                // Create mask: bytes 0..len are 0xFF, rest are 0x00
                let mask_vals: [u8; 8] = [
                    if 0 < len { 0xFF } else { 0 },
                    if 1 < len { 0xFF } else { 0 },
                    if 2 < len { 0xFF } else { 0 },
                    if 3 < len { 0xFF } else { 0 },
                    if 4 < len { 0xFF } else { 0 },
                    if 5 < len { 0xFF } else { 0 },
                    if 6 < len { 0xFF } else { 0 },
                    if 7 < len { 0xFF } else { 0 },
                ];
                let mask = vld1_u8(mask_vals.as_ptr());
                let masked = vand_u8(lo, mask);
                vcombine_u8(masked, vdup_n_u8(0))
            }
            1..=7 => Self::load_partial_safe(data, len),
            9..=15 => {
                let lo = vld1_u8(data);

                let hi_start = len - 8;
                let hi = vld1_u8(data.add(hi_start));
                let hi = match 16 - len {
                    1 => vext_u8(hi, vdup_n_u8(0), 1),
                    2 => vext_u8(hi, vdup_n_u8(0), 2),
                    3 => vext_u8(hi, vdup_n_u8(0), 3),
                    4 => vext_u8(hi, vdup_n_u8(0), 4),
                    5 => vext_u8(hi, vdup_n_u8(0), 5),
                    6 => vext_u8(hi, vdup_n_u8(0), 6),
                    7 => vext_u8(hi, vdup_n_u8(0), 7),
                    _ => unreachable!(),
                };

                vcombine_u8(lo, hi)
            }

            _ if start + 16 <= len => vld1q_u8(data.add(start)),
            _ => {
                let overlap = start + 16 - len;
                let loaded = vld1q_u8(data.add(len - 16));

                // Shift left by 'overlap' bytes (zeros enter from the right)
                match overlap {
                    1 => vextq_u8(loaded, vdupq_n_u8(0), 1),
                    2 => vextq_u8(loaded, vdupq_n_u8(0), 2),
                    3 => vextq_u8(loaded, vdupq_n_u8(0), 3),
                    4 => vextq_u8(loaded, vdupq_n_u8(0), 4),
                    5 => vextq_u8(loaded, vdupq_n_u8(0), 5),
                    6 => vextq_u8(loaded, vdupq_n_u8(0), 6),
                    7 => vextq_u8(loaded, vdupq_n_u8(0), 7),
                    8 => vextq_u8(loaded, vdupq_n_u8(0), 8),
                    9 => vextq_u8(loaded, vdupq_n_u8(0), 9),
                    10 => vextq_u8(loaded, vdupq_n_u8(0), 10),
                    11 => vextq_u8(loaded, vdupq_n_u8(0), 11),
                    12 => vextq_u8(loaded, vdupq_n_u8(0), 12),
                    13 => vextq_u8(loaded, vdupq_n_u8(0), 13),
                    14 => vextq_u8(loaded, vdupq_n_u8(0), 14),
                    15 => vextq_u8(loaded, vdupq_n_u8(0), 15),
                    _ => vdupq_n_u8(0),
                }
            }
        })
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u8<const L: i32>(self, other: Self) -> Self {
        assert!(L >= 0 && L <= 15);
        match L {
            0 => self,
            1 => Self(vextq_u8(other.0, self.0, 15)),
            2 => Self(vextq_u8(other.0, self.0, 14)),
            3 => Self(vextq_u8(other.0, self.0, 13)),
            4 => Self(vextq_u8(other.0, self.0, 12)),
            5 => Self(vextq_u8(other.0, self.0, 11)),
            6 => Self(vextq_u8(other.0, self.0, 10)),
            7 => Self(vextq_u8(other.0, self.0, 9)),
            8 => Self(vextq_u8(other.0, self.0, 8)),
            9 => Self(vextq_u8(other.0, self.0, 7)),
            10 => Self(vextq_u8(other.0, self.0, 6)),
            11 => Self(vextq_u8(other.0, self.0, 5)),
            12 => Self(vextq_u8(other.0, self.0, 4)),
            13 => Self(vextq_u8(other.0, self.0, 3)),
            14 => Self(vextq_u8(other.0, self.0, 2)),
            15 => Self(vextq_u8(other.0, self.0, 1)),
            _ => unreachable!(),
        }
    }
}

impl super::Vector128Expansion<NEON256Vector> for NEONVector {
    #[inline(always)]
    unsafe fn cast_i8_to_i16(self) -> NEON256Vector {
        NEON256Vector((
            vreinterpretq_u8_s16(vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(self.0)))),
            vreinterpretq_u8_s16(vmovl_high_s8(vreinterpretq_s8_u8(self.0))),
        ))
    }
}
