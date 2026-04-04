/// 256-bit scalar vector using u16 arrays, for platforms without SIMD support.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Scalar256Vector(pub(crate) [u16; 16]);

impl Scalar256Vector {
    #[inline(always)]
    fn to_bytes(self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for i in 0..16 {
            let b = self.0[i].to_ne_bytes();
            bytes[i * 2] = b[0];
            bytes[i * 2 + 1] = b[1];
        }
        bytes
    }

    #[inline(always)]
    fn from_bytes(bytes: [u8; 32]) -> Self {
        let mut arr = [0u16; 16];
        for i in 0..16 {
            arr[i] = u16::from_ne_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        }
        Self(arr)
    }
}

impl super::Vector for Scalar256Vector {
    #[inline]
    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0u16; 16])
    }

    #[inline(always)]
    unsafe fn splat_u8(value: u8) -> Self {
        let u16_val = u16::from_ne_bytes([value, value]);
        Self([u16_val; 16])
    }

    #[inline(always)]
    unsafe fn splat_u16(value: u16) -> Self {
        Self([value; 16])
    }

    #[inline(always)]
    unsafe fn eq_u8(self, other: Self) -> Self {
        let a = self.to_bytes();
        let b = other.to_bytes();
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = if a[i] == b[i] { 0xFF } else { 0x00 };
        }
        Self::from_bytes(result)
    }

    #[inline(always)]
    unsafe fn gt_u8(self, other: Self) -> Self {
        let a = self.to_bytes();
        let b = other.to_bytes();
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = if a[i] > b[i] { 0xFF } else { 0x00 };
        }
        Self::from_bytes(result)
    }

    #[inline(always)]
    unsafe fn lt_u8(self, other: Self) -> Self {
        let a = self.to_bytes();
        let b = other.to_bytes();
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = if a[i] < b[i] { 0xFF } else { 0x00 };
        }
        Self::from_bytes(result)
    }

    #[inline(always)]
    unsafe fn max_u16(self, other: Self) -> Self {
        let mut result = [0u16; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = a.max(b);
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn smax_u16(self) -> u16 {
        let mut max = self.0[0];
        for &v in &self.0[1..] {
            if v > max {
                max = v;
            }
        }
        max
    }

    #[inline(always)]
    unsafe fn add_u16(self, other: Self) -> Self {
        let mut result = [0u16; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = a.wrapping_add(b);
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn subs_u16(self, other: Self) -> Self {
        let mut result = [0u16; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = a.saturating_sub(b);
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        let mut result = [0u16; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = a & b;
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        let mut result = [0u16; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = a | b;
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn not(self) -> Self {
        let mut result = [0u16; 16];
        for (r, &a) in result.iter_mut().zip(self.0.iter()) {
            *r = !a;
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u16<const N: i32>(self, other: Self) -> Self {
        assert!(N >= 0 && N <= 16);
        let n = N as usize;
        let mut result = [0u16; 16];
        for (i, r) in result.iter_mut().enumerate().take(n.min(16)) {
            *r = other.0[16 - n + i];
        }
        result[n..16].copy_from_slice(&self.0[..(16 - n)]);
        Self(result)
    }

    #[cfg(test)]
    fn from_array(arr: [u8; 16]) -> Self {
        // Load same 16 bytes into both halves (matching SSE256/NEON256 behavior)
        let mut u16s = [0u16; 16];
        for i in 0..8 {
            u16s[i] = u16::from_ne_bytes([arr[i * 2], arr[i * 2 + 1]]);
        }
        for i in 0..8 {
            u16s[i + 8] = u16s[i];
        }
        Self(u16s)
    }
    #[cfg(test)]
    fn to_array(self) -> [u8; 16] {
        let mut arr = [0u8; 16];
        for i in 0..8 {
            let bytes = self.0[i].to_ne_bytes();
            arr[i * 2] = bytes[0];
            arr[i * 2 + 1] = bytes[1];
        }
        arr
    }
    #[cfg(test)]
    fn from_array_u16(arr: [u16; 8]) -> Self {
        let mut u16s = [0u16; 16];
        u16s[0..8].copy_from_slice(&arr);
        u16s[8..16].copy_from_slice(&arr);
        Self(u16s)
    }
    #[cfg(test)]
    fn to_array_u16(self) -> [u16; 8] {
        let mut arr = [0u16; 8];
        arr.copy_from_slice(&self.0[0..8]);
        arr
    }
}

impl super::Vector256 for Scalar256Vector {
    #[cfg(test)]
    fn from_array_256_u16(arr: [u16; 16]) -> Self {
        Self(arr)
    }
    #[cfg(test)]
    fn to_array_256_u16(self) -> [u16; 16] {
        self.0
    }

    #[inline(always)]
    unsafe fn load_unaligned(data: [u8; 32]) -> Self {
        let mut arr = [0u16; 16];
        for i in 0..16 {
            arr[i] = u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]);
        }
        Self(arr)
    }

    #[inline(always)]
    unsafe fn idx_u16(self, search: u16) -> usize {
        for i in 0..16 {
            if self.0[i] == search {
                return i;
            }
        }
        16
    }
}
