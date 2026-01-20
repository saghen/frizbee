#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

pub enum Alignment {
    None,
    Diagonal,
    Left,
    Up,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AlignmentChunk {
    bit0: u16,
    bit1: u16,
}

impl AlignmentChunk {
    pub unsafe fn new(diag_mask: __m256i, left_mask: __m256i, up_mask: __m256i) -> AlignmentChunk {
        let not_mask1 = _mm256_xor_si256(diag_mask, _mm256_set1_epi16(-1));
        let not_mask2 = _mm256_xor_si256(left_mask, _mm256_set1_epi16(-1));

        // Encoding: bit1:bit0 = 00(none), 01(mask1), 10(mask2), 11(mask3)
        // bit0 = mask1 | (mask3 & ~mask2)
        let bit0_vec = _mm256_or_si256(diag_mask, _mm256_and_si256(up_mask, not_mask2));
        // bit1 = ~mask1 & (mask2 | mask3)
        let bit1_vec = _mm256_and_si256(not_mask1, _mm256_or_si256(left_mask, up_mask));

        // Pack bits together
        let packed = _mm256_packs_epi16(bit0_vec, bit1_vec);

        // Fix lane interleaving: vpermq with immediate
        // Before: [bit1_lane1, bit0_lane1 | bit1_lane0, bit0_lane0]
        //              d           c            b           a      (64-bit chunks)
        // Want:   [bit1_lane1, bit1_lane0 | bit0_lane1, bit0_lane0]
        let fixed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00); // d,b,c,a

        let mask = _mm256_movemask_epi8(fixed) as u32;

        AlignmentChunk {
            bit0: (mask & 0xFFFF) as u16,
            bit1: (mask >> 16) as u16,
        }
    }

    pub fn alignment(&self, index: usize) -> Alignment {
        match ((self.bit1 >> index) & 1u16, ((self.bit0 >> index) & 1u16)) {
            (0, 0) => Alignment::None,
            (1, 0) => Alignment::Diagonal,
            (0, 1) => Alignment::Left,
            (1, 1) => Alignment::Up,
            _ => unreachable!(),
        }
    }
}
