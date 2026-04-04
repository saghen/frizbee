use crate::simd::Vector256;

#[inline(always)]
pub unsafe fn propagate_horizontal_gaps_u8<Simd256: Vector256>(
    row: Simd256,
    adjacent_row: Simd256,
    match_mask: Simd256,
    adjacent_match_mask: Simd256,
    gap_open_penalty: Simd256,
    gap_extend_penalty: Simd256,
) -> Simd256 {
    unsafe {
        // shift by 1 element (1 bytes), decay by 1
        // row: [0, 4, 0, 0, 0, ...]
        let shifted_row = row.shift_right_padded_u8::<1>(adjacent_row);
        let shifted_match_mask = match_mask.shift_right_padded_u8::<1>(adjacent_match_mask);
        let gap_penalty = gap_extend_penalty.add_u8(gap_open_penalty.and(shifted_match_mask));
        let decayed = shifted_row.subs_u8(gap_penalty);
        let row = row.max_u8(decayed); // [0, 4, 3, 0, 0, ...]

        // shift by 2 elements (2 bytes), decay by 2
        let shifted_row = row.shift_right_padded_u8::<2>(adjacent_row);
        let shifted_match_mask = match_mask.shift_right_padded_u8::<2>(adjacent_match_mask);
        let gap_extend_penalty = gap_extend_penalty.add_u8(gap_extend_penalty);
        let gap_penalty = gap_extend_penalty.add_u8(gap_open_penalty.and(shifted_match_mask));
        let decayed = shifted_row.subs_u8(gap_penalty);
        let row = row.max_u8(decayed);

        // shift by 4 elements (4 bytes), decay by 4
        let shifted_row = row.shift_right_padded_u8::<4>(adjacent_row);
        let shifted_match_mask = match_mask.shift_right_padded_u8::<4>(adjacent_match_mask);
        let gap_extend_penalty = gap_extend_penalty.add_u8(gap_extend_penalty);
        let gap_penalty = gap_extend_penalty.add_u8(gap_open_penalty.and(shifted_match_mask));
        let decayed = shifted_row.subs_u8(gap_penalty);
        let row = row.max_u8(decayed);

        // shift by 8 elements (16 bytes), decay by 8
        let shifted_row = row.shift_right_padded_u8::<8>(adjacent_row);
        let shifted_match_mask = match_mask.shift_right_padded_u8::<8>(adjacent_match_mask);
        let gap_extend_penalty = gap_extend_penalty.add_u8(gap_extend_penalty);
        let gap_penalty = gap_extend_penalty.add_u8(gap_open_penalty.and(shifted_match_mask));
        let decayed = shifted_row.subs_u8(gap_penalty);
        let row = row.max_u8(decayed);

        // shift by 16 elements (32 bytes), decay by 16
        let shifted_row = row.shift_right_padded_u8::<16>(adjacent_row);
        let shifted_match_mask = match_mask.shift_right_padded_u8::<16>(adjacent_match_mask);
        let gap_extend_penalty = gap_extend_penalty.add_u8(gap_extend_penalty);
        let gap_penalty = gap_extend_penalty.add_u8(gap_open_penalty.and(shifted_match_mask));
        let decayed = shifted_row.subs_u8(gap_penalty);
        row.max_u8(decayed)
    }
}

#[inline(always)]
pub unsafe fn propagate_horizontal_gaps_u16<Simd256: Vector256>(
    row: Simd256,
    adjacent_row: Simd256,
    match_mask: Simd256,
    adjacent_match_mask: Simd256,
    gap_open_penalty: Simd256,
    gap_extend_penalty: Simd256,
) -> Simd256 {
    unsafe {
        // shift by 1 element (2 bytes), decay by 1
        // row: [0, 4, 0, 0, 0, ...]
        let shifted_row = row.shift_right_padded_u16::<1>(adjacent_row);
        let shifted_match_mask = match_mask.shift_right_padded_u16::<1>(adjacent_match_mask);
        let gap_penalty = gap_extend_penalty.add_u16(gap_open_penalty.and(shifted_match_mask));
        let decayed = shifted_row.subs_u16(gap_penalty);
        let row = row.max_u16(decayed); // [0, 4, 3, 0, 0, ...]

        // shift by 2 elements (4 bytes), decay by 2
        let shifted_row = row.shift_right_padded_u16::<2>(adjacent_row);
        let shifted_match_mask = match_mask.shift_right_padded_u16::<2>(adjacent_match_mask);
        let gap_extend_penalty = gap_extend_penalty.add_u16(gap_extend_penalty);
        let gap_penalty = gap_extend_penalty.add_u16(gap_open_penalty.and(shifted_match_mask));
        let decayed = shifted_row.subs_u16(gap_penalty);
        let row = row.max_u16(decayed);

        // shift by 4 elements (8 bytes), decay by 4
        let shifted_row = row.shift_right_padded_u16::<4>(adjacent_row);
        let shifted_match_mask = match_mask.shift_right_padded_u16::<4>(adjacent_match_mask);
        let gap_extend_penalty = gap_extend_penalty.add_u16(gap_extend_penalty);
        let gap_penalty = gap_extend_penalty.add_u16(gap_open_penalty.and(shifted_match_mask));
        let decayed = shifted_row.subs_u16(gap_penalty);
        let row = row.max_u16(decayed);

        // shift by 8 elements (16 bytes), decay by 8
        let shifted_row = row.shift_right_padded_u16::<8>(adjacent_row);
        let shifted_match_mask = match_mask.shift_right_padded_u16::<8>(adjacent_match_mask);
        let gap_extend_penalty = gap_extend_penalty.add_u16(gap_extend_penalty);
        let gap_penalty = gap_extend_penalty.add_u16(gap_open_penalty.and(shifted_match_mask));
        let decayed = shifted_row.subs_u16(gap_penalty);
        row.max_u16(decayed)
    }
}
