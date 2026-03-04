#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
mod neon_256;
#[cfg(target_arch = "x86_64")]
mod sse;
#[cfg(target_arch = "x86_64")]
mod sse_256;

#[cfg(target_arch = "x86_64")]
pub use avx::AVXVector;
#[cfg(target_arch = "aarch64")]
pub use neon::NEONVector;
#[cfg(target_arch = "aarch64")]
pub use neon_256::NEON256Vector;
#[cfg(target_arch = "x86_64")]
pub use sse::SSEVector;
#[cfg(target_arch = "x86_64")]
pub use sse_256::SSE256Vector;

pub trait Vector: Copy + core::fmt::Debug {
    /// Checks available vector extensions at runtime and returns whether the vector implementation
    /// may be safely used.
    /// We use `raw_cpuid` instead of the `is_x86_feature_detected` macro because the latter
    /// compiles to a constant when compiled with `RUSTFLAGS="-C target-cpu=x86-64-v3"`
    fn is_available() -> bool;

    /// Create a vector with zeros in all lanes.
    unsafe fn zero() -> Self;
    /// Create a vector with 8-bit lanes with the given byte repeated into each
    /// lane.
    unsafe fn splat_u8(value: u8) -> Self;
    /// Create a vector with 16-bit lanes with the given byte repeated into each
    /// lane.
    unsafe fn splat_u16(value: u16) -> Self;

    unsafe fn eq_u8(self, other: Self) -> Self;
    unsafe fn gt_u8(self, other: Self) -> Self;
    unsafe fn lt_u8(self, other: Self) -> Self;

    unsafe fn max_u16(self, other: Self) -> Self;
    /// Get the maximum value in the vector as a scalar
    unsafe fn smax_u16(self) -> u16;

    unsafe fn add_u16(self, other: Self) -> Self;
    unsafe fn subs_u16(self, other: Self) -> Self;

    unsafe fn and(self, other: Self) -> Self;
    unsafe fn or(self, other: Self) -> Self;
    unsafe fn not(self) -> Self;

    unsafe fn shift_right_padded_u16<const N: i32>(self, other: Self) -> Self;

    #[cfg(test)]
    fn from_array(arr: [u8; 16]) -> Self;
    #[cfg(test)]
    fn to_array(self) -> [u8; 16];
    #[cfg(test)]
    fn from_array_u16(arr: [u16; 8]) -> Self;
    #[cfg(test)]
    fn to_array_u16(self) -> [u16; 8];
}

pub trait Vector128: Vector {
    /// Loads from the given pointer, where the number of remaining bytes may be less than
    /// the vector size. The pointer does not need to be aligned.
    ///
    /// # Safety
    ///
    /// Callers must guarantee that the pointer contains `len` bytes and `start < len`.
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self;

    /// Shift `self` right by `L` bytes, filling in the low bytes with the right most values in
    /// `other`
    unsafe fn shift_right_padded_u8<const L: i32>(self, other: Self) -> Self;
}

pub trait Vector128Expansion<Expanded: Vector256>: Vector128 {
    /// Expands the vector from 128-bit to 256-bit by expanding each byte
    unsafe fn cast_i8_to_i16(self) -> Expanded;
}

pub trait Vector256: Vector {
    #[cfg(test)]
    fn from_array_256_u16(arr: [u16; 16]) -> Self;
    #[cfg(test)]
    fn to_array_256_u16(self) -> [u16; 16];

    unsafe fn load_unaligned(data: [u8; 32]) -> Self;

    /// Extract the value at the given index from the vector
    unsafe fn idx_u16(self, search: u16) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    trait VectorTests: Vector {
        unsafe fn test_zero() {
            unsafe {
                assert_eq!(Self::zero().to_array(), [0u8; 16]);
            }
        }

        unsafe fn test_splat_u8() {
            unsafe {
                assert_eq!(Self::splat_u8(0x42).to_array(), [0x42u8; 16]);
            }
        }

        unsafe fn test_eq_u8() {
            unsafe {
                let a = Self::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255]);
                let b = Self::from_array([
                    0, 254, 2, 253, 4, 252, 6, 251, 8, 250, 10, 249, 12, 248, 14, 255,
                ]);
                let expected = [
                    0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
                    0x00, 0xFF, 0xFF,
                ];
                assert_eq!(a.eq_u8(b).to_array(), expected);
            }
        }

        unsafe fn test_gt_u8() {
            unsafe {
                let a = Self::from_array([
                    1, 254, 2, 253, 4, 252, 6, 251, 8, 250, 10, 249, 12, 248, 14, 255,
                ]);
                let b = Self::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 254]);
                let expected = [
                    0xFF, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
                    0xFF, 0x00, 0xFF,
                ];
                assert_eq!(a.gt_u8(b).to_array(), expected);
            }
        }

        unsafe fn test_max_u16() {
            unsafe {
                let a = Self::from_array_u16([0, 2, 4, 6, 8, 10, 12, 14]);
                let b = Self::from_array_u16([0, 252, 2, 253, 4, 252, 6, 255]);
                let expected = [0, 252, 4, 253, 8, 252, 12, 255];
                assert_eq!(a.max_u16(b).to_array_u16(), expected);
            }
        }

        unsafe fn test_smax_u16() {
            unsafe {
                let a = Self::from_array_u16([u16::MAX, 2, 4, 6, 8, 10, 12, 14]);
                assert_eq!(a.smax_u16(), u16::MAX);
                let b = Self::from_array_u16([0, 252, 2, 253, 4, 252, 6, u16::MAX - 1]);
                assert_eq!(b.smax_u16(), u16::MAX - 1);
                let c = Self::from_array_u16([0, 1, 2, 3, 80, 3, 2, 1]);
                assert_eq!(c.smax_u16(), 80);
            }
        }

        unsafe fn test_add_u16() {
            unsafe {
                let a = Self::from_array_u16([0, 3, 4, 6, 8, 10, 12, 14]);
                let b = Self::from_array_u16([0, u16::MAX - 3, 2, 100, 4, 50, 6, 40]);
                let expected = [0, u16::MAX, 6, 106, 12, 60, 18, 54];
                assert_eq!(a.add_u16(b).to_array_u16(), expected);
            }
        }

        unsafe fn test_subs_u16() {
            unsafe {
                let a = Self::from_array_u16([u16::MAX, 0, 4, 99, 8, 100, 12, 40]);
                let b = Self::from_array_u16([0, u16::MAX, 2, 100, 4, 50, 6, 40]);
                let expected = [u16::MAX, 0, 2, 0, 4, 50, 6, 0];
                assert_eq!(a.subs_u16(b).to_array_u16(), expected);
            }
        }

        unsafe fn test_and() {
            unsafe {
                let a = Self::from_array([
                    0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
                    0x00, 0xFF, 0x00,
                ]);
                let b = Self::from_array([
                    0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF,
                    0xFF, 0xFF, 0x00,
                ]);
                let expected = [
                    0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
                    0x00, 0xFF, 0x00,
                ];
                assert_eq!(a.and(b).to_array(), expected);
            }
        }

        unsafe fn test_or() {
            unsafe {
                let a = Self::from_array([
                    0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
                    0x00, 0xFF, 0x00,
                ]);
                let b = Self::from_array([
                    0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF,
                    0xFF, 0xFF, 0x00,
                ]);
                let expected = [
                    0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF,
                    0xFF, 0xFF, 0x00,
                ];
                assert_eq!(a.or(b).to_array(), expected);
            }
        }

        unsafe fn test_not() {
            unsafe {
                let a = Self::from_array([
                    0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF,
                    0xFF, 0xFF, 0x00,
                ]);
                let expected = [
                    0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00,
                    0x00, 0x00, 0xFF,
                ];
                assert_eq!(a.not().to_array(), expected);
            }
        }

        unsafe fn test_shift_right_padded_u16() {
            unsafe {
                let a = Self::from_array_u16([0, 1, 2, 3, 4, 5, 6, 7]);
                let b = Self::from_array_u16([50, 51, 52, 53, 54, 55, 56, 57]);

                fn get_expected(i: usize) -> [u16; 8] {
                    let a = [0, 1, 2, 3, 4, 5, 6, 7];
                    let b = [50, 51, 52, 53, 54, 55, 56, 57];

                    let mut expected = [0; 8];
                    expected[i..8].copy_from_slice(&a[0..(8 - i)]);
                    expected[0..i].copy_from_slice(&b[(8 - i)..8]);
                    expected
                }

                assert_eq!(
                    a.shift_right_padded_u16::<1>(b).to_array_u16(),
                    get_expected(1)
                );
                assert_eq!(
                    a.shift_right_padded_u16::<2>(b).to_array_u16(),
                    get_expected(2)
                );
                assert_eq!(
                    a.shift_right_padded_u16::<3>(b).to_array_u16(),
                    get_expected(3)
                );
                assert_eq!(
                    a.shift_right_padded_u16::<4>(b).to_array_u16(),
                    get_expected(4)
                );
                assert_eq!(
                    a.shift_right_padded_u16::<5>(b).to_array_u16(),
                    get_expected(5)
                );
                assert_eq!(
                    a.shift_right_padded_u16::<6>(b).to_array_u16(),
                    get_expected(6)
                );
                assert_eq!(
                    a.shift_right_padded_u16::<7>(b).to_array_u16(),
                    get_expected(7)
                );
            }
        }
    }

    impl<T: Vector> VectorTests for T {}

    pub trait Vector128Tests: Vector128 {
        unsafe fn test_load_partial() {
            unsafe {
                let data = (1..=32).collect::<Vec<_>>();

                for start in (0..32).step_by(16) {
                    for len in (start + 1)..32 {
                        let a = Self::load_partial(data.as_ptr(), start, len);
                        let mut expected = [0; 16];
                        if len - start > 16 {
                            expected[0..16].copy_from_slice(&data[start..start + 16]);
                        } else {
                            expected[0..len - start].copy_from_slice(&data[start..len]);
                        }
                        assert_eq!(a.to_array(), expected);
                    }
                }
            }
        }

        unsafe fn test_shift_right_padded_u8() {
            unsafe {
                let a = Self::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
                let b = Self::from_array([
                    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                ]);

                fn get_expected(i: usize) -> [u8; 16] {
                    let a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
                    let b = [
                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                    ];

                    let mut expected = [0; 16];
                    expected[i..16].copy_from_slice(&a[0..(16 - i)]);
                    expected[0..i].copy_from_slice(&b[(16 - i)..16]);
                    expected
                }

                assert_eq!(a.shift_right_padded_u8::<1>(b).to_array(), get_expected(1));
                assert_eq!(a.shift_right_padded_u8::<2>(b).to_array(), get_expected(2));
                assert_eq!(a.shift_right_padded_u8::<3>(b).to_array(), get_expected(3));
                assert_eq!(a.shift_right_padded_u8::<4>(b).to_array(), get_expected(4));
                assert_eq!(a.shift_right_padded_u8::<5>(b).to_array(), get_expected(5));
                assert_eq!(a.shift_right_padded_u8::<6>(b).to_array(), get_expected(6));
                assert_eq!(a.shift_right_padded_u8::<7>(b).to_array(), get_expected(7));
                assert_eq!(a.shift_right_padded_u8::<8>(b).to_array(), get_expected(8));
                assert_eq!(a.shift_right_padded_u8::<9>(b).to_array(), get_expected(9));
                assert_eq!(
                    a.shift_right_padded_u8::<10>(b).to_array(),
                    get_expected(10)
                );
                assert_eq!(
                    a.shift_right_padded_u8::<11>(b).to_array(),
                    get_expected(11)
                );
                assert_eq!(
                    a.shift_right_padded_u8::<12>(b).to_array(),
                    get_expected(12)
                );
                assert_eq!(
                    a.shift_right_padded_u8::<13>(b).to_array(),
                    get_expected(13)
                );
                assert_eq!(
                    a.shift_right_padded_u8::<14>(b).to_array(),
                    get_expected(14)
                );
                assert_eq!(
                    a.shift_right_padded_u8::<15>(b).to_array(),
                    get_expected(15)
                );
            }
        }
    }

    impl<T: Vector128> Vector128Tests for T {}

    pub trait Vector128ExpansionTests<Expanded: Vector256>: Vector128Expansion<Expanded> {
        #[cfg(test)]
        unsafe fn test_cast_i8_to_i16() {
            unsafe {
                let a = Self::splat_u8(0x00);
                assert_eq!(a.cast_i8_to_i16().to_array_256_u16(), [0x0000; 16]);

                let b = Self::splat_u8(0xFF);
                assert_eq!(b.cast_i8_to_i16().to_array_256_u16(), [0xFFFF; 16]);
            }
        }
    }

    impl<T: Vector128Expansion<Expanded>, Expanded: Vector256> Vector128ExpansionTests<Expanded> for T {}

    pub trait Vector256Tests: Vector256 {
        unsafe fn test_idx_u16() {
            unsafe {
                let a = Self::from_array_256_u16([
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                ]);
                assert_eq!(a.idx_u16(0), 0);
                assert_eq!(a.idx_u16(1), 1);
                assert_eq!(a.idx_u16(2), 2);
                assert_eq!(a.idx_u16(3), 3);
                assert_eq!(a.idx_u16(4), 4);
                assert_eq!(a.idx_u16(5), 5);
                assert_eq!(a.idx_u16(6), 6);
                assert_eq!(a.idx_u16(7), 7);
                assert_eq!(a.idx_u16(8), 8);
                assert_eq!(a.idx_u16(9), 9);
                assert_eq!(a.idx_u16(10), 10);
                assert_eq!(a.idx_u16(11), 11);
                assert_eq!(a.idx_u16(12), 12);
                assert_eq!(a.idx_u16(13), 13);
                assert_eq!(a.idx_u16(14), 14);
                assert_eq!(a.idx_u16(15), 15);

                let b = Self::from_array_256_u16([
                    200, 150, 2, 3, 4, 5, 6, 7, 150, 9, 2, 11, 12, 13, 14, 200,
                ]);
                assert_eq!(b.idx_u16(200), 0);
                assert_eq!(b.idx_u16(150), 1);
                assert_eq!(b.idx_u16(2), 2);
            }
        }
    }

    impl<T: Vector256> Vector256Tests for T {}

    macro_rules! simd_test {
        ($name:ident) => {
            #[test]
            fn $name() {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    SSEVector::$name();
                    AVXVector::$name();
                    SSE256Vector::$name();
                };
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    NEONVector::$name();
                    NEON256Vector::$name();
                };
            }
        };
    }

    macro_rules! simd128_test {
        ($name:ident) => {
            #[test]
            fn $name() {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    SSEVector::$name();
                };
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    NEONVector::$name();
                };
            }
        };
    }

    macro_rules! simd256_test {
        ($name:ident) => {
            #[test]
            fn $name() {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    SSE256Vector::$name();
                    AVXVector::$name();
                };
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    NEON256Vector::$name();
                };
            }
        };
    }

    simd_test!(test_zero);
    simd_test!(test_splat_u8);
    simd_test!(test_eq_u8);
    simd_test!(test_gt_u8);
    simd_test!(test_max_u16);
    simd_test!(test_smax_u16);
    simd_test!(test_add_u16);
    simd_test!(test_subs_u16);
    simd_test!(test_and);
    simd_test!(test_or);
    simd_test!(test_not);
    simd_test!(test_shift_right_padded_u16);
    simd128_test!(test_load_partial);
    simd128_test!(test_shift_right_padded_u8);
    simd256_test!(test_idx_u16);

    #[test]
    fn test_cast_i8_to_i16() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            <SSEVector as Vector128ExpansionTests<SSE256Vector>>::test_cast_i8_to_i16();
            <SSEVector as Vector128ExpansionTests<AVXVector>>::test_cast_i8_to_i16();
        };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            <NEONVector as Vector128ExpansionTests<NEON256Vector>>::test_cast_i8_to_i16()
        };
    }
}
