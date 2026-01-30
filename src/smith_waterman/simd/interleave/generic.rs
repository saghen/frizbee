use std::simd::{Simd, num::SimdUint};

#[inline(never)]
pub fn interleave<const W: usize, const L: usize>(strs: [&str; L]) -> [Simd<u16, L>; W] {
    // Ensure the strings are all the length of W
    let strs = std::array::from_fn(|i| {
        let mut tmp = [0u8; W];
        tmp[0..strs[i].len()].copy_from_slice(strs[i].as_bytes());
        tmp
    });

    let chunk_count = W.div_ceil(L);
    let mut interleaved = [Simd::splat(0); W];

    for chunk_idx in 0..chunk_count {
        let offset = chunk_idx * L;

        let simds = to_simd::<W, L>(strs, offset);
        let interleaved_chunk = interleave_chunk::<L>(simds);

        if offset + L > W {
            interleaved[offset..W].copy_from_slice(&interleaved_chunk[0..(W - offset)]);
        } else {
            interleaved[offset..(offset + L)].copy_from_slice(&interleaved_chunk);
        }
    }

    interleaved
}

#[inline(always)]
fn to_simd<const W: usize, const L: usize>(strs: [[u8; W]; L], offset: usize) -> [Simd<u16, L>; L] {
    std::array::from_fn(|i| {
        Simd::load_or_default(&strs[i][offset..(offset + L).min(W)]).cast::<u16>()
    })
}

#[inline(never)]
pub fn interleave_chunk<const L: usize>(mut simds: [Simd<u16, L>; L]) -> [Simd<u16, L>; L] {
    // Assert that L is a power of 2
    debug_assert!(L.is_power_of_two());

    // Perform the interleave operations in stages
    // Starting with the largest distance and halving each time
    let mut distance = L / 2;

    while distance > 0 {
        // Process pairs at the current distance
        for base in 0..L {
            // Only process if this is the first element of a pair at this distance
            if base & distance == 0 {
                let pair_idx = base + distance;
                if pair_idx < L {
                    // Perform the interleave operation on this pair
                    let (new_base, new_pair) = simds[base].interleave(simds[pair_idx]);
                    simds[base] = new_base;
                    simds[pair_idx] = new_pair;
                }
            }
        }

        distance /= 2;
    }

    // Cast all results to u16
    simds
}

#[cfg(test)]
mod tests {
    use std::simd::Simd;

    use super::interleave;

    fn assert_matrix_eq<const L: usize, const W: usize>(a: [Simd<u16, L>; W], b: [[u8; L]; W]) {
        let a = a.map(|a| {
            a.to_array()
                .into_iter()
                .map(|x| x as u8)
                .collect::<Vec<_>>()
        });
        assert_eq!(a, b);
    }

    #[test]
    fn test_interleave_simd_2() {
        let strs = ["ab", "cd"];
        let interleaved = interleave::<2, 2>(strs);
        assert_matrix_eq(interleaved, [[b'a', b'c'], [b'b', b'd']]);
    }

    #[test]
    fn test_interleave_simd_chunks_2() {
        let strs = ["abcd", "efgh"];
        let interleaved = interleave::<4, 2>(strs);
        assert_matrix_eq(
            interleaved,
            [[b'a', b'e'], [b'b', b'f'], [b'c', b'g'], [b'd', b'h']],
        );
    }

    #[test]
    fn test_interleave_simd_4() {
        let strs = ["abcd", "efgh", "ijkl", "mnop"];
        let interleaved = interleave::<4, 4>(strs);
        assert_matrix_eq(
            interleaved,
            [
                [b'a', b'e', b'i', b'm'],
                [b'b', b'f', b'j', b'n'],
                [b'c', b'g', b'k', b'o'],
                [b'd', b'h', b'l', b'p'],
            ],
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_interleave_simd_8() {
        let strs = ["abcdefgh", "ijklmnop", "qrstuvwx", "yzABCDEF", "GHIJKLMN", "OPQRSTUV", "WXYZ1234", "56789012"];
        let interleaved = interleave::<8, 8>(strs);

        assert_matrix_eq(
            interleaved,
            [
                [b'a', b'i', b'q', b'y', b'G', b'O', b'W', b'5'],
                [b'b', b'j', b'r', b'z', b'H', b'P', b'X', b'6'],
                [b'c', b'k', b's', b'A', b'I', b'Q', b'Y', b'7'],
                [b'd', b'l', b't', b'B', b'J', b'R', b'Z', b'8'],
                [b'e', b'm', b'u', b'C', b'K', b'S', b'1', b'9'],
                [b'f', b'n', b'v', b'D', b'L', b'T', b'2', b'0'],
                [b'g', b'o', b'w', b'E', b'M', b'U', b'3', b'1'],
                [b'h', b'p', b'x', b'F', b'N', b'V', b'4', b'2'],
            ],
        );
    }
}
