//! Per-backend tests, using the scalar implementation as the reference

use super::super::{Backend, BytesVec, MaskVec, ScoreVec};

// -------------------
// BytesVec / MaskVec
// -------------------

fn check_bytes_splat<B: Backend>() {
    unsafe {
        let v = B::Bytes::splat(0x42);
        assert_eq!(BytesVec::to_lanes(v), vec![0x42; B::LANES]);
    }
}

fn check_bytes_eq<B: Backend>() {
    unsafe {
        let a_in = (0u8..(B::LANES as u8)).collect::<Vec<_>>();
        let b_in = a_in
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { v.wrapping_add(1) })
            .collect::<Vec<_>>();
        let a = B::Bytes::from_lanes(&a_in);
        let b = B::Bytes::from_lanes(&b_in);
        let result = MaskVec::to_lanes(a.eq(b));
        for (i, r) in result.iter().enumerate() {
            assert_eq!(*r, i % 2 == 0, "lane {i}");
        }
    }
}

fn check_bytes_gt_lt<B: Backend>() {
    unsafe {
        let a_in = vec![5u8; B::LANES];
        let mut b_in = vec![5u8; B::LANES];
        if B::LANES >= 2 {
            b_in[0] = 4;
            b_in[1] = 6;
        }
        let a = B::Bytes::from_lanes(&a_in);
        let b = B::Bytes::from_lanes(&b_in);
        let gt = MaskVec::to_lanes(a.gt(b));
        let lt = MaskVec::to_lanes(a.lt(b));
        if B::LANES >= 2 {
            assert!(gt[0]);
            assert!(!gt[1]);
            assert!(!lt[0]);
            assert!(lt[1]);
        }
    }
}

fn check_mask_and_or_not<B: Backend>() {
    unsafe {
        let pattern_a: Vec<bool> = (0..B::LANES).map(|i| i % 2 == 0).collect();
        let pattern_b: Vec<bool> = (0..B::LANES).map(|i| i % 3 == 0).collect();
        let a = B::Mask::from_lanes(&pattern_a);
        let b = B::Mask::from_lanes(&pattern_b);
        let and = a.and(b).to_lanes();
        let or = a.or(b).to_lanes();
        let not_a = a.not().to_lanes();
        for i in 0..B::LANES {
            assert_eq!(and[i], pattern_a[i] && pattern_b[i], "and lane {i}");
            assert_eq!(or[i], pattern_a[i] || pattern_b[i], "or lane {i}");
            assert_eq!(not_a[i], !pattern_a[i], "not lane {i}");
        }
    }
}

fn check_mask_zero<B: Backend>() {
    unsafe {
        let z = B::Mask::zero();
        assert_eq!(MaskVec::to_lanes(z), vec![false; B::LANES]);
    }
}

fn check_bytes_load_partial<B: Backend>() {
    unsafe {
        let data: Vec<u8> = (1..=64).collect();
        let lanes = B::LANES;
        for start in (0..32).step_by(lanes) {
            for len in (start + 1)..=32 {
                let v = B::Bytes::load_partial(data.as_ptr(), start, len);
                let got = BytesVec::to_lanes(v);
                let mut expected = vec![0u8; lanes];
                let take = (len - start).min(lanes);
                expected[..take].copy_from_slice(&data[start..start + take]);
                assert_eq!(got, expected, "start={start} len={len}");
            }
        }
    }
}

fn check_mask_shift_right_padded_1<B: Backend>() {
    unsafe {
        let a_in: Vec<bool> = (0..B::LANES).map(|i| i % 2 == 0).collect();
        let p_in: Vec<bool> = (0..B::LANES).map(|i| i % 3 == 0).collect();
        let a = B::Mask::from_lanes(&a_in);
        let p = B::Mask::from_lanes(&p_in);
        let got = MaskVec::to_lanes(a.shift_right_padded_1(p));

        let mut expected = vec![false; B::LANES];
        expected[0] = p_in[B::LANES - 1];
        expected[1..B::LANES].copy_from_slice(&a_in[0..(B::LANES - 1)]);
        assert_eq!(got, expected);
    }
}

// -------------------
// ScoreVec
// -------------------

fn check_score_zero<B: Backend>() {
    unsafe {
        assert_eq!(B::Score::zero().to_lanes(), vec![0u16; B::LANES]);
    }
}

fn check_score_splat<B: Backend>() {
    unsafe {
        assert_eq!(B::Score::splat(0xAB).to_lanes(), vec![0xABu16; B::LANES]);
    }
}

fn check_score_first_lane<B: Backend>() {
    unsafe {
        let v = B::Score::first_lane(0xCD);
        let lanes = v.to_lanes();
        assert_eq!(lanes[0], 0xCD);
        for &l in &lanes[1..] {
            assert_eq!(l, 0);
        }
    }
}

fn check_score_max_add_subs<B: Backend>() {
    unsafe {
        let a_in: Vec<u16> = (0..B::LANES).map(|i| (i % 100) as u16).collect();
        let b_in: Vec<u16> = (0..B::LANES).map(|i| (B::LANES - i) as u16 % 50).collect();
        let a = B::Score::from_lanes(&a_in);
        let b = B::Score::from_lanes(&b_in);
        let max = a.max(b).to_lanes();
        for i in 0..B::LANES {
            assert_eq!(max[i], a_in[i].max(b_in[i]));
        }
        let added = a.add(b).to_lanes();
        for i in 0..B::LANES {
            assert_eq!(added[i], a_in[i].wrapping_add(b_in[i]));
        }
        let subbed = a.subs(b).to_lanes();
        for i in 0..B::LANES {
            assert_eq!(subbed[i], a_in[i].saturating_sub(b_in[i]));
        }
    }
}

fn check_score_horizontal_max<B: Backend>() {
    unsafe {
        let mut a_in = vec![10u16; B::LANES];
        a_in[B::LANES / 2] = 222;
        let a = B::Score::from_lanes(&a_in);
        assert_eq!(a.horizontal_max(), 222);

        let zero = B::Score::zero();
        assert_eq!(zero.horizontal_max(), 0);
    }
}

fn check_score_find_lane<B: Backend>() {
    unsafe {
        // for 32 lanes, step of 7 stays under 224, fits in u8
        let a_in: Vec<u16> = (0..B::LANES).map(|i| (i as u16) * 7).collect();
        let a = B::Score::from_lanes(&a_in);
        for (i, &v) in a_in.iter().enumerate() {
            assert_eq!(a.find_lane(v), i);
        }
        // search for a value that can't appear in the data
        // works for both u8 and u16 backends
        assert_eq!(a.find_lane(251), B::LANES);
    }
}

fn check_score_shift_right_padded_at<B: Backend, const L: i32>() {
    unsafe {
        // Distinct values for `a` and `p` that all fit in u8
        let a_in: Vec<u16> = (0..B::LANES).map(|i| (i as u16) + 1).collect();
        let p_in: Vec<u16> = (0..B::LANES).map(|i| (i as u16) + 100).collect();
        let a = B::Score::from_lanes(&a_in);
        let p = B::Score::from_lanes(&p_in);
        let got = a.shift_right_padded::<L>(p).to_lanes();

        let n = L as usize;
        let mut expected = vec![0u16; B::LANES];
        for i in 0..n {
            expected[i] = p_in[B::LANES - n + i];
        }
        expected[n..B::LANES].copy_from_slice(&a_in[0..(B::LANES - n)]);
        assert_eq!(got, expected, "L = {}", L);
    }
}

fn check_score_shift_right_padded<B: Backend>() {
    check_score_shift_right_padded_at::<B, 0>();
    check_score_shift_right_padded_at::<B, 1>();
    check_score_shift_right_padded_at::<B, 2>();
    check_score_shift_right_padded_at::<B, 3>();
    check_score_shift_right_padded_at::<B, 4>();
    if B::LANES >= 8 {
        check_score_shift_right_padded_at::<B, 5>();
        check_score_shift_right_padded_at::<B, 6>();
        check_score_shift_right_padded_at::<B, 7>();
        check_score_shift_right_padded_at::<B, 8>();
    }
}

pub(super) fn check_score_shift_right_padded_16<B: Backend>() {
    check_score_shift_right_padded_at::<B, 16>();
}

pub(super) fn check_score_shift_right_padded_32<B: Backend>() {
    check_score_shift_right_padded_at::<B, 32>();
}

// -------------------
// Dispatch
// -------------------

fn check_backend_contract<B: Backend>() {
    check_bytes_splat::<B>();
    check_bytes_eq::<B>();
    check_bytes_gt_lt::<B>();
    check_mask_and_or_not::<B>();
    check_mask_zero::<B>();
    check_bytes_load_partial::<B>();
    check_mask_shift_right_padded_1::<B>();
    check_score_zero::<B>();
    check_score_splat::<B>();
    check_score_first_lane::<B>();
    check_score_max_add_subs::<B>();
    check_score_horizontal_max::<B>();
    check_score_find_lane::<B>();
    check_score_shift_right_padded::<B>();
}

/// Generate a `#[test]` that runs against `$backend` (skipping it when the CPU lacks
/// the required features), plus any backend-specific extra checks
macro_rules! backend_contract_tests {
    ($name:ident, $backend:ty $(, $extra:ident)*) => {
        #[test]
        fn $name() {
            if !<$backend>::is_available() {
                return;
            }
            check_backend_contract::<$backend>();
            $(
                $extra::<$backend>();
            )*
        }
    };
}

backend_contract_tests!(scalar8, super::super::BackendScalar8);
backend_contract_tests!(scalar16_u8, super::super::BackendScalar16U8);
#[cfg(target_arch = "x86_64")]
backend_contract_tests!(sse, super::super::BackendSSE);
#[cfg(target_arch = "x86_64")]
backend_contract_tests!(sse_u8, super::super::BackendSSEU8);
#[cfg(target_arch = "x86_64")]
backend_contract_tests!(avx, super::super::BackendAVX);
#[cfg(target_arch = "x86_64")]
backend_contract_tests!(
    avx_u8,
    super::super::BackendAVXU8,
    check_score_shift_right_padded_16
);
#[cfg(target_arch = "x86_64")]
backend_contract_tests!(
    avx512,
    super::super::BackendAVX512,
    check_score_shift_right_padded_16
);
#[cfg(target_arch = "x86_64")]
backend_contract_tests!(
    avx512_u8,
    super::super::BackendAVX512U8,
    check_score_shift_right_padded_16,
    check_score_shift_right_padded_32
);
#[cfg(target_arch = "aarch64")]
backend_contract_tests!(neon, super::super::BackendNEON);
#[cfg(target_arch = "aarch64")]
backend_contract_tests!(neon_u8, super::super::BackendNEONU8);
