use std::arch::x86_64::*;

use crate::prefilter::algo::can_overread;

use super::Backend;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Utf8ToUtf32AVX512;

impl Backend for Utf8ToUtf32AVX512 {
    #[inline(always)]
    fn is_available() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("popcnt")
    }

    #[inline(always)]
    unsafe fn convert_into(input: &str, out: &mut Vec<u32>) {
        unsafe { convert_into_avx512(input, out) }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,popcnt")]
unsafe fn convert_into_avx512(input: &str, out: &mut Vec<u32>) {
    unsafe { convert_avx512(input, out) }
}

#[inline(always)]
unsafe fn convert_avx512(input: &str, out: &mut Vec<u32>) {
    unsafe {
        out.clear();

        let bytes = input.as_bytes();
        let len = bytes.len();
        out.reserve(len);

        let input_ptr = bytes.as_ptr();
        let output_ptr = out.as_mut_ptr();
        let high_bit = _mm512_set1_epi8(0x80u8 as i8);
        let mut input_pos = 0usize;
        let mut output_len = 0usize;

        // Mixed UTF-8 looks four bytes ahead so lane-end codepoints stay vectorized.
        while input_pos + 68 <= len {
            let input = input_ptr.add(input_pos);
            let chunk = _mm512_loadu_si512(input as *const __m512i);
            if _mm512_test_epi8_mask(chunk, high_bit) == 0 {
                store_ascii_64(chunk, output_ptr.add(output_len));
                output_len += 64;
            } else {
                output_len += convert_mixed_64(chunk, input, output_ptr.add(output_len));
            }
            input_pos += 64;
        }

        // With 52+ tail bytes, process three lanes and leave the final lane scalar.
        let remaining = len - input_pos;
        if remaining >= 64 {
            let input = input_ptr.add(input_pos);
            let chunk = _mm512_loadu_si512(input as *const __m512i);
            if _mm512_test_epi8_mask(chunk, high_bit) == 0 {
                store_ascii_64(chunk, output_ptr.add(output_len));
                output_len += 64;
                input_pos += 64;
            } else if can_overread(input.add(64), 4) {
                // Page-safe overread: extra bytes only complete starts inside this block.
                output_len += convert_mixed_64(chunk, input, output_ptr.add(output_len));
                input_pos += 64;
            } else {
                output_len += convert_mixed_prefix_48(chunk, output_ptr.add(output_len));
                input_pos += 48;
            }
        } else if remaining >= 52 {
            let input = input_ptr.add(input_pos);
            let mask = 1u64.wrapping_shl(remaining as u32).wrapping_sub(1) as __mmask64;
            let chunk = _mm512_maskz_loadu_epi8(mask, input as *const i8);
            output_len += convert_mixed_prefix_48(chunk, output_ptr.add(output_len));
            input_pos += 48;
        }

        out.set_len(output_len);

        // Skip continuations already consumed by the vector block before scalar tail.
        while input_pos < len && (*input_ptr.add(input_pos) & 0xc0) == 0x80 {
            input_pos += 1;
        }

        let tail = std::str::from_utf8_unchecked(&bytes[input_pos..]);
        out.extend(tail.chars().map(|c| c as u32));
    }
}

#[inline(always)]
unsafe fn store_ascii_64(input: __m512i, output: *mut u32) {
    unsafe {
        // Four 16-byte zero-extends turn a 64-byte ASCII block into UTF-32.
        let first = _mm512_castsi512_si128(input);
        let second = _mm512_extracti32x4_epi32::<1>(input);
        let third = _mm512_extracti32x4_epi32::<2>(input);
        let fourth = _mm512_extracti32x4_epi32::<3>(input);

        _mm512_storeu_si512(output as *mut __m512i, _mm512_cvtepu8_epi32(first));
        _mm512_storeu_si512(output.add(16) as *mut __m512i, _mm512_cvtepu8_epi32(second));
        _mm512_storeu_si512(output.add(32) as *mut __m512i, _mm512_cvtepu8_epi32(third));
        _mm512_storeu_si512(output.add(48) as *mut __m512i, _mm512_cvtepu8_epi32(fourth));
    }
}

#[inline(always)]
unsafe fn convert_mixed_64(input_block: __m512i, input: *const u8, output: *mut u32) -> usize {
    unsafe {
        let first_expanded = expand_window::<0>(input_block);
        let second_expanded = expand_window::<16>(input_block);
        let third_expanded = expand_window::<32>(input_block);

        let lookahead = _mm512_castsi128_si512(_mm_cvtsi32_si128(
            input.add(64).cast::<u32>().read_unaligned() as i32,
        ));
        let fourth_expanded = expand_final_window(input_block, lookahead);

        let (first, first_count) = identify_expanded(first_expanded);
        let (second, second_count) = identify_expanded(second_expanded);
        let (third, third_count) = identify_expanded(third_expanded);
        let (fourth, fourth_count) = identify_expanded(fourth_expanded);

        let first_pair_count = first_count + second_count;
        let second_pair_count = third_count + fourth_count;
        let total = first_pair_count + second_pair_count;
        if total <= 16 {
            // Four-byte-heavy text fits one output vector, so decode it once.
            let first_pair = combine_pair(first, first_count, second, second_count);
            let second_pair = combine_pair(third, third_count, fourth, fourth_count);
            let combined =
                combine_pair(first_pair, first_pair_count, second_pair, second_pair_count);
            let utf32 = expand_utf8_to_utf32(combined);
            if total == 16 {
                store_utf32_full(output, utf32);
            } else {
                store_utf32(output, utf32, total);
            }
            total
        } else {
            let mut output_len = write_pair(first, first_count, second, second_count, output);
            let output = output.add(output_len);
            output_len += write_pair(third, third_count, fourth, fourth_count, output);
            output_len
        }
    }
}

#[inline(always)]
unsafe fn convert_mixed_prefix_48(input_block: __m512i, output: *mut u32) -> usize {
    unsafe {
        let first_expanded = expand_window::<0>(input_block);
        let second_expanded = expand_window::<16>(input_block);
        let third_expanded = expand_window::<32>(input_block);

        let (first, first_count) = identify_expanded(first_expanded);
        let (second, second_count) = identify_expanded(second_expanded);
        let output_len = write_pair(first, first_count, second, second_count, output);
        let (third, third_count) = identify_expanded(third_expanded);
        store_utf32(
            output.add(output_len),
            expand_utf8_to_utf32(third),
            third_count,
        );
        output_len + third_count
    }
}

#[inline(always)]
unsafe fn expand_window<const START: u8>(input: __m512i) -> __m512i {
    unsafe {
        // VBMI gathers every 4-byte candidate substring for starts in each 16-byte window.
        _mm512_permutexvar_epi8(window_index::<START>(), input)
    }
}

#[inline(always)]
unsafe fn expand_final_window(input: __m512i, lookahead: __m512i) -> __m512i {
    unsafe { _mm512_permutex2var_epi8(input, window_index::<48>(), lookahead) }
}

#[inline(always)]
unsafe fn window_index<const START: u8>() -> __m512i {
    unsafe {
        _mm512_setr_epi64(
            byte_index_pair(START),
            byte_index_pair(START + 2),
            byte_index_pair(START + 4),
            byte_index_pair(START + 6),
            byte_index_pair(START + 8),
            byte_index_pair(START + 10),
            byte_index_pair(START + 12),
            byte_index_pair(START + 14),
        )
    }
}

#[inline(always)]
fn byte_index_pair(start: u8) -> i64 {
    (start as i64)
        | ((start as i64 + 1) << 8)
        | ((start as i64 + 2) << 16)
        | ((start as i64 + 3) << 24)
        | ((start as i64 + 1) << 32)
        | ((start as i64 + 2) << 40)
        | ((start as i64 + 3) << 48)
        | ((start as i64 + 4) << 56)
}

#[inline(always)]
unsafe fn identify_expanded(expanded: __m512i) -> (__m512i, usize) {
    unsafe {
        let byte_class = _mm512_and_si512(expanded, _mm512_set1_epi32(0xc0));
        let leading_bytes = _mm512_cmpneq_epu32_mask(byte_class, _mm512_set1_epi32(0x80));
        let count = leading_bytes.count_ones() as usize;
        (
            _mm512_mask_compress_epi32(_mm512_setzero_si512(), leading_bytes, expanded),
            count,
        )
    }
}

#[inline(always)]
unsafe fn write_pair(
    first: __m512i,
    first_count: usize,
    second: __m512i,
    second_count: usize,
    output: *mut u32,
) -> usize {
    unsafe {
        let total = first_count + second_count;
        if total <= 16 {
            let combined = combine_pair(first, first_count, second, second_count);
            store_utf32(output, expand_utf8_to_utf32(combined), total);
        } else {
            store_utf32(output, expand_utf8_to_utf32(first), first_count);
            store_utf32(
                output.add(first_count),
                expand_utf8_to_utf32(second),
                second_count,
            );
        }
        total
    }
}

#[inline(always)]
unsafe fn combine_pair(
    first: __m512i,
    first_count: usize,
    second: __m512i,
    second_count: usize,
) -> __m512i {
    unsafe {
        let second_mask = (((1u32 << second_count) - 1) << first_count) as __mmask16;
        _mm512_mask_expand_epi32(first, second_mask, second)
    }
}

#[inline(always)]
unsafe fn store_utf32_full(output: *mut u32, utf32: __m512i) {
    unsafe { _mm512_storeu_si512(output as *mut __m512i, utf32) }
}

#[inline(always)]
unsafe fn store_utf32(output: *mut u32, utf32: __m512i, count: usize) {
    unsafe { _mm512_mask_storeu_epi32(output as *mut i32, valid_lane_mask(count), utf32) }
}

#[inline(always)]
unsafe fn expand_utf8_to_utf32(input: __m512i) -> __m512i {
    unsafe {
        let mut char_class = _mm512_srli_epi32::<4>(input);
        char_class = _mm512_ternarylogic_epi32::<0xea>(
            char_class,
            _mm512_set1_epi32(0x0f),
            _mm512_set1_epi32(0x80808000u32 as i32),
        );

        // Each u32 lane contains b0|b1|b2|b3 for a possible UTF-8 sequence.
        // Rust's &str invariant gives us valid UTF-8, so this only removes
        // prefix bits and joins payload fields into UTF-32 codepoints.
        let mut values = _mm512_and_si512(input, _mm512_set1_epi32(0x3f3f3f7f));
        values = _mm512_maddubs_epi16(values, _mm512_set1_epi32(0x01400140));
        values = _mm512_madd_epi16(values, _mm512_set1_epi32(0x00011000));

        let shift_left = _mm512_setr_epi64(
            0x0707070707070707,
            0x0b0a090900000000,
            0x0707070707070707,
            0x0b0a090900000000,
            0x0707070707070707,
            0x0b0a090900000000,
            0x0707070707070707,
            0x0b0a090900000000,
        );
        values = _mm512_sllv_epi32(values, _mm512_shuffle_epi8(shift_left, char_class));

        let shift_right = _mm512_setr_epi64(
            0x1919191919191919,
            0x0b10151500000000,
            0x1919191919191919,
            0x0b10151500000000,
            0x1919191919191919,
            0x0b10151500000000,
            0x1919191919191919,
            0x0b10151500000000,
        );
        _mm512_srlv_epi32(values, _mm512_shuffle_epi8(shift_right, char_class))
    }
}

#[inline(always)]
fn valid_lane_mask(count: usize) -> __mmask16 {
    debug_assert!(count <= 16);
    1u32.wrapping_shl(count as u32).wrapping_sub(1) as __mmask16
}
