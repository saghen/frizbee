/// 128-bit scalar vector using byte arrays, for platforms without SIMD support.
#[derive(Debug, Clone, Copy)]
pub struct ScalarVector(pub(crate) [u8; 16]);

/// 256-bit scalar vector using u16 arrays, for platforms without SIMD support.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Scalar256Vector(pub(crate) [u16; 16]);

impl ScalarVector {
    #[inline(always)]
    fn as_u16s(&self) -> [u16; 8] {
        let mut result = [0u16; 8];
        for (i, r) in result.iter_mut().enumerate() {
            *r = u16::from_ne_bytes([self.0[i * 2], self.0[i * 2 + 1]]);
        }
        result
    }

    #[inline(always)]
    fn from_u16s(arr: [u16; 8]) -> Self {
        let mut bytes = [0u8; 16];
        for i in 0..8 {
            let b = arr[i].to_ne_bytes();
            bytes[i * 2] = b[0];
            bytes[i * 2 + 1] = b[1];
        }
        Self(bytes)
    }
}

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

// --- Vector trait for ScalarVector (128-bit) ---

impl super::Vector for ScalarVector {
    #[inline]
    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0u8; 16])
    }

    #[inline(always)]
    unsafe fn splat_u8(value: u8) -> Self {
        Self([value; 16])
    }

    #[inline(always)]
    unsafe fn splat_u16(value: u16) -> Self {
        Self::from_u16s([value; 8])
    }

    #[inline(always)]
    unsafe fn eq_u8(self, other: Self) -> Self {
        let mut result = [0u8; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = if a == b { 0xFF } else { 0x00 };
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn gt_u8(self, other: Self) -> Self {
        let mut result = [0u8; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = if a > b { 0xFF } else { 0x00 };
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn lt_u8(self, other: Self) -> Self {
        let mut result = [0u8; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = if a < b { 0xFF } else { 0x00 };
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn max_u16(self, other: Self) -> Self {
        let a = self.as_u16s();
        let b = other.as_u16s();
        let mut result = [0u16; 8];
        for ((r, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
            *r = a_val.max(b_val);
        }
        Self::from_u16s(result)
    }

    #[inline(always)]
    unsafe fn smax_u16(self) -> u16 {
        let vals = self.as_u16s();
        let mut max = vals[0];
        for &v in &vals[1..] {
            if v > max {
                max = v;
            }
        }
        max
    }

    #[inline(always)]
    unsafe fn add_u16(self, other: Self) -> Self {
        let a = self.as_u16s();
        let b = other.as_u16s();
        let mut result = [0u16; 8];
        for ((r, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
            *r = a_val.wrapping_add(b_val);
        }
        Self::from_u16s(result)
    }

    #[inline(always)]
    unsafe fn subs_u16(self, other: Self) -> Self {
        let a = self.as_u16s();
        let b = other.as_u16s();
        let mut result = [0u16; 8];
        for ((r, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
            *r = a_val.saturating_sub(b_val);
        }
        Self::from_u16s(result)
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        let mut result = [0u8; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = a & b;
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        let mut result = [0u8; 16];
        for ((r, &a), &b) in result.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = a | b;
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn not(self) -> Self {
        let mut result = [0u8; 16];
        for (r, &a) in result.iter_mut().zip(self.0.iter()) {
            *r = !a;
        }
        Self(result)
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u16<const N: i32>(self, other: Self) -> Self {
        let a = self.as_u16s();
        let b = other.as_u16s();
        let n = N as usize;
        let mut result = [0u16; 8];
        for (i, r) in result.iter_mut().enumerate().take(n.min(8)) {
            *r = b[8 - n + i];
        }
        result[n..8].copy_from_slice(&a[..(8 - n)]);
        Self::from_u16s(result)
    }

    #[cfg(test)]
    fn from_array(arr: [u8; 16]) -> Self {
        Self(arr)
    }
    #[cfg(test)]
    fn to_array(self) -> [u8; 16] {
        self.0
    }
    #[cfg(test)]
    fn from_array_u16(arr: [u16; 8]) -> Self {
        Self::from_u16s(arr)
    }
    #[cfg(test)]
    fn to_array_u16(self) -> [u16; 8] {
        self.as_u16s()
    }
}

// --- Vector128 trait for ScalarVector ---

impl super::Vector128 for ScalarVector {
    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        let mut arr = [0u8; 16];
        let available = len.saturating_sub(start);
        let to_copy = available.min(16);
        if to_copy > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(data.add(start), arr.as_mut_ptr(), to_copy);
            }
        }
        Self(arr)
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u8<const L: i32>(self, other: Self) -> Self {
        let l = L as usize;
        let mut result = [0u8; 16];
        for (i, r) in result.iter_mut().enumerate().take(l.min(16)) {
            *r = other.0[16 - l + i];
        }
        result[l..16].copy_from_slice(&self.0[..(16 - l)]);
        Self(result)
    }
}

// --- Vector128Expansion for ScalarVector -> Scalar256Vector ---

impl super::Vector128Expansion<Scalar256Vector> for ScalarVector {
    #[inline(always)]
    unsafe fn cast_i8_to_i16(self) -> Scalar256Vector {
        let mut result = [0u16; 16];
        for (r, &b) in result.iter_mut().zip(self.0.iter()) {
            *r = (b as i8 as i16) as u16;
        }
        Scalar256Vector(result)
    }
}

// --- Vector trait for Scalar256Vector (256-bit) ---

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

// --- Vector256 trait for Scalar256Vector ---

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
