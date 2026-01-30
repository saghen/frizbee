use std::simd::cmp::*;
use std::simd::{Mask, Select, Simd};

const CAPITAL_START: u16 = 65; // A
const CAPITAL_END: u16 = 90; // Z
const LOWER_START: u16 = 97; // a
const LOWER_END: u16 = 122; // z
const TO_LOWERCASE_MASK: u16 = 0x20;

#[inline(always)]
pub(crate) fn simd_to_lowercase_with_mask<const L: usize>(
    data: Simd<u16, L>,
) -> (Mask<i16, L>, Mask<i16, L>, Simd<u16, L>) {
    let is_capital_mask: Mask<i16, L> =
        data.simd_ge(Simd::splat(CAPITAL_START)) & data.simd_le(Simd::splat(CAPITAL_END));
    let is_lower_mask: Mask<i16, L> =
        data.simd_ge(Simd::splat(LOWER_START)) & data.simd_le(Simd::splat(LOWER_END));
    let lowercase = data | is_capital_mask.select(Simd::splat(TO_LOWERCASE_MASK), Simd::splat(0));
    (is_capital_mask, is_lower_mask, lowercase)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NeedleChar<const L: usize> {
    pub(crate) lowercase: Simd<u16, L>,
    pub(crate) is_capital_mask: Mask<i16, L>,
}
impl<const L: usize> NeedleChar<L> {
    #[inline(always)]
    pub(crate) fn new(char: u16) -> Self {
        let (is_capital_mask, _, lowercase) = simd_to_lowercase_with_mask::<L>(Simd::splat(char));
        Self {
            lowercase,
            is_capital_mask,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HaystackChar<const L: usize> {
    pub(crate) lowercase: Simd<u16, L>,
    pub(crate) is_lower_mask: Mask<i16, L>,
    pub(crate) is_capital_mask: Mask<i16, L>,
    pub(crate) is_delimiter_mask: Mask<i16, L>,
}
impl<const L: usize> HaystackChar<L> {
    #[inline(always)]
    pub(crate) fn new(chars: Simd<u16, L>) -> Self {
        let (is_capital_mask, is_lower_mask, lowercase) = simd_to_lowercase_with_mask::<L>(chars);
        let is_delimiter_mask: Mask<i16, L> = Simd::splat(b' ' as u16).simd_eq(lowercase)
            | Simd::splat(b'/' as u16).simd_eq(lowercase)
            | Simd::splat(b'.' as u16).simd_eq(lowercase)
            | Simd::splat(b',' as u16).simd_eq(lowercase)
            | Simd::splat(b'_' as u16).simd_eq(lowercase)
            | Simd::splat(b'-' as u16).simd_eq(lowercase);
        Self {
            lowercase,
            is_lower_mask,
            is_capital_mask,
            is_delimiter_mask,
        }
    }

    #[inline(always)]
    pub(crate) fn from_haystack(haystacks: &[&str; L], i: usize) -> Self {
        // Convert haystacks to a static array of bytes chunked for SIMD
        let chars = std::array::from_fn(|j| {
            *haystacks[j].as_bytes().get(i).to_owned().unwrap_or(&0) as u16
        });
        // pre-compute haystack case mask, delimiter mask, and lowercase
        HaystackChar::new(Simd::from_array(chars))
    }
}

impl<const L: usize> Default for HaystackChar<L> {
    fn default() -> Self {
        Self {
            lowercase: Simd::splat(0),
            is_lower_mask: Mask::splat(false),
            is_capital_mask: Mask::splat(false),
            is_delimiter_mask: Mask::splat(false),
        }
    }
}
