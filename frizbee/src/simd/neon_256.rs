use std::arch::aarch64::*;

#[derive(Debug, Clone, Copy)]
pub struct NEON256Vector(pub(crate) (uint8x16_t, uint8x16_t));

impl super::Vector for NEON256Vector {
    #[inline]
    fn is_available() -> bool {
        // NEON is mandatory on aarch64
        cfg!(target_arch = "aarch64")
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self((vdupq_n_u8(0), vdupq_n_u8(0))) }
    }

    #[inline(always)]
    unsafe fn splat_u8(value: u8) -> Self {
        unsafe { Self((vdupq_n_u8(value), vdupq_n_u8(value))) }
    }

    #[inline(always)]
    unsafe fn splat_u16(value: u16) -> Self {
        unsafe {
            Self((
                vreinterpretq_u8_u16(vdupq_n_u16(value)),
                vreinterpretq_u8_u16(vdupq_n_u16(value)),
            ))
        }
    }

    #[inline(always)]
    unsafe fn eq_u8(self, other: Self) -> Self {
        unsafe { Self((vceqq_u8(self.0.0, other.0.0), vceqq_u8(self.0.1, other.0.1))) }
    }

    #[inline(always)]
    unsafe fn gt_u8(self, other: Self) -> Self {
        unsafe { Self((vcgtq_u8(self.0.0, other.0.0), vcgtq_u8(self.0.1, other.0.1))) }
    }

    #[inline(always)]
    unsafe fn lt_u8(self, other: Self) -> Self {
        unsafe { Self((vcltq_u8(self.0.0, other.0.0), vcltq_u8(self.0.1, other.0.1))) }
    }

    #[inline(always)]
    unsafe fn max_u16(self, other: Self) -> Self {
        unsafe {
            Self((
                vreinterpretq_u8_u16(vmaxq_u16(
                    vreinterpretq_u16_u8(self.0.0),
                    vreinterpretq_u16_u8(other.0.0),
                )),
                vreinterpretq_u8_u16(vmaxq_u16(
                    vreinterpretq_u16_u8(self.0.1),
                    vreinterpretq_u16_u8(other.0.1),
                )),
            ))
        }
    }

    #[inline(always)]
    unsafe fn smax_u16(self) -> u16 {
        unsafe {
            vmaxvq_u16(vreinterpretq_u16_u8(self.0.0))
                .max(vmaxvq_u16(vreinterpretq_u16_u8(self.0.1)))
        }
    }

    #[inline(always)]
    unsafe fn add_u16(self, other: Self) -> Self {
        unsafe {
            Self((
                vreinterpretq_u8_u16(vaddq_u16(
                    vreinterpretq_u16_u8(self.0.0),
                    vreinterpretq_u16_u8(other.0.0),
                )),
                vreinterpretq_u8_u16(vaddq_u16(
                    vreinterpretq_u16_u8(self.0.1),
                    vreinterpretq_u16_u8(other.0.1),
                )),
            ))
        }
    }

    #[inline(always)]
    unsafe fn subs_u16(self, other: Self) -> Self {
        unsafe {
            Self((
                vreinterpretq_u8_u16(vqsubq_u16(
                    vreinterpretq_u16_u8(self.0.0),
                    vreinterpretq_u16_u8(other.0.0),
                )),
                vreinterpretq_u8_u16(vqsubq_u16(
                    vreinterpretq_u16_u8(self.0.1),
                    vreinterpretq_u16_u8(other.0.1),
                )),
            ))
        }
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self((vandq_u8(self.0.0, other.0.0), vandq_u8(self.0.1, other.0.1))) }
    }

    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        unsafe { Self((vorrq_u8(self.0.0, other.0.0), vorrq_u8(self.0.1, other.0.1))) }
    }

    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self((vmvnq_u8(self.0.0), vmvnq_u8(self.0.1))) }
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u16<const L: i32>(self, other: Self) -> Self {
        unsafe {
            const { assert!(L >= 0 && L <= 15) };
            match L {
                0 => self,
                1 => Self((
                    vextq_u8(other.0.1, self.0.0, 14),
                    vextq_u8(self.0.0, self.0.1, 14),
                )),
                2 => Self((
                    vextq_u8(other.0.1, self.0.0, 12),
                    vextq_u8(self.0.0, self.0.1, 12),
                )),
                3 => Self((
                    vextq_u8(other.0.1, self.0.0, 10),
                    vextq_u8(self.0.0, self.0.1, 10),
                )),
                4 => Self((
                    vextq_u8(other.0.1, self.0.0, 8),
                    vextq_u8(self.0.0, self.0.1, 8),
                )),
                5 => Self((
                    vextq_u8(other.0.1, self.0.0, 6),
                    vextq_u8(self.0.0, self.0.1, 6),
                )),
                6 => Self((
                    vextq_u8(other.0.1, self.0.0, 4),
                    vextq_u8(self.0.0, self.0.1, 4),
                )),
                7 => Self((
                    vextq_u8(other.0.1, self.0.0, 2),
                    vextq_u8(self.0.0, self.0.1, 2),
                )),
                8 => Self((other.0.1, self.0.0)),
                9 => Self((
                    vextq_u8(other.0.0, other.0.1, 14),
                    vextq_u8(other.0.1, self.0.0, 14),
                )),
                10 => Self((
                    vextq_u8(other.0.0, other.0.1, 12),
                    vextq_u8(other.0.1, self.0.0, 12),
                )),
                11 => Self((
                    vextq_u8(other.0.0, other.0.1, 10),
                    vextq_u8(other.0.1, self.0.0, 10),
                )),
                12 => Self((
                    vextq_u8(other.0.0, other.0.1, 8),
                    vextq_u8(other.0.1, self.0.0, 8),
                )),
                13 => Self((
                    vextq_u8(other.0.0, other.0.1, 6),
                    vextq_u8(other.0.1, self.0.0, 6),
                )),
                14 => Self((
                    vextq_u8(other.0.0, other.0.1, 4),
                    vextq_u8(other.0.1, self.0.0, 4),
                )),
                15 => Self((
                    vextq_u8(other.0.0, other.0.1, 2),
                    vextq_u8(other.0.1, self.0.0, 2),
                )),
                _ => unreachable!(),
            }
        }
    }

    #[cfg(test)]
    fn from_array(arr: [u8; 16]) -> Self {
        Self((unsafe { vld1q_u8(arr.as_ptr()) }, unsafe {
            vld1q_u8(arr.as_ptr())
        }))
    }
    #[cfg(test)]
    fn to_array(self) -> [u8; 16] {
        let mut arr = [0u8; 16];
        unsafe { vst1q_u8(arr.as_mut_ptr(), self.0.0) };
        arr
    }
    #[cfg(test)]
    fn from_array_u16(arr: [u16; 8]) -> Self {
        Self((unsafe { vld1q_u8(arr.as_ptr() as *const u8) }, unsafe {
            vld1q_u8(arr.as_ptr() as *const u8)
        }))
    }
    #[cfg(test)]
    fn to_array_u16(self) -> [u16; 8] {
        let mut arr = [0u16; 8];
        unsafe { vst1q_u16(arr.as_mut_ptr(), vreinterpretq_u16_u8(self.0.0)) };
        arr
    }
}

impl super::Vector256 for NEON256Vector {
    #[cfg(test)]
    fn from_array_256_u16(arr: [u16; 16]) -> Self {
        Self((unsafe { vld1q_u8(arr.as_ptr() as *const u8) }, unsafe {
            vld1q_u8(arr.as_ptr().add(8) as *const u8)
        }))
    }
    #[cfg(test)]
    fn to_array_256_u16(self) -> [u16; 16] {
        let mut arr = [0u16; 16];
        unsafe { vst1q_u16(arr.as_mut_ptr(), vreinterpretq_u16_u8(self.0.0)) };
        unsafe { vst1q_u16(arr.as_mut_ptr().add(8), vreinterpretq_u16_u8(self.0.1)) };
        arr
    }

    #[inline(always)]
    unsafe fn load_unaligned(data: [u8; 32]) -> Self {
        Self((unsafe { vld1q_u8(data.as_ptr()) }, unsafe {
            vld1q_u8(data.as_ptr().add(16))
        }))
    }

    #[inline(always)]
    unsafe fn idx_u16(self, search: u16) -> usize {
        unsafe {
            // Compare all elements with search value (0xFFFF where equal, 0 otherwise)
            let cmp_a = vceqq_u16(vreinterpretq_u16_u8(self.0.0), vdupq_n_u16(search));
            let cmp_b = vceqq_u16(vreinterpretq_u16_u8(self.0.1), vdupq_n_u16(search));

            // Narrow to 8-bit (0xFF or 0x00 per original 16-bit lane)
            let narrowed_a = vmovn_u16(cmp_a);
            let narrowed_b = vmovn_u16(cmp_b);

            // Reinterpret as u64 for efficient bit manipulation
            let bits_a = vget_lane_u64(vreinterpret_u64_u8(narrowed_a), 0);
            let bits_b = vget_lane_u64(vreinterpret_u64_u8(narrowed_b), 0);

            if bits_a == 0 {
                if bits_b == 0 {
                    return 16; // Not found
                }
                bits_b.trailing_zeros() as usize / 8 + 8
            } else {
                // Each byte represents one u16 lane, find first 0xFF byte
                bits_a.trailing_zeros() as usize / 8
            }
        }
    }
}
