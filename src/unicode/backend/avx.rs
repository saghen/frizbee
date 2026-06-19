use std::arch::x86_64::*;

use super::Backend;

const ONE_OR_TWO_BYTE_SHUFFLES: [[u8; 16]; 256] = build_one_or_two_byte_shuffles::<256, 8>();
const SIX_ONE_OR_TWO_BYTE_SHUFFLES: [[u8; 16]; 64] = build_one_or_two_byte_shuffles::<64, 6>();
const ONE_TO_THREE_BYTE_SHUFFLES: [[u8; 16]; 81] = build_one_to_three_byte_shuffles();
const SIX_ONE_OR_TWO_BYTE_MASKS: [u16; 4096] = build_six_one_or_two_byte_masks();
const INVALID_ONE_OR_TWO_BYTE_MASK: u16 = u16::MAX;
const TWO_BYTE_BOUNDARY_MASK_8: u32 = 0x0001_ffff;
const TWO_BYTE_CONTINUATION_MASK_8: u32 = 0x0000_aaaa;
const THREE_BYTE_BOUNDARY_MASK_8: u32 = 0x00ff_ffff;
const THREE_BYTE_CONTINUATION_MASK_8: u32 = 0x00db_6db6;
const FOUR_BYTE_CONTINUATION_MASK_8: u32 = 0xeeee_eeee;

// UTF-8 to UTF-32 conversion optimized for the case Frizbee expects most:
// mostly-ASCII fuzzy-match inputs with occasional Unicode.
//
// The hot path is a pure ASCII block converter. It scans 32 bytes at a time,
// checks the high bit with movemask, and widens bytes to u32 lanes with AVX2
// zero-extension instructions. That path is why the Latin benchmark is close
// to simdutf.
//
// A block containing any non-ASCII byte switches to mixed-width paths inspired
// by simdutf's Haswell converter. The 64-byte loop computes byte-class masks
// once, then shifts those masks as each small decoder consumes input. That
// keeps the table-driven one/two-byte path from paying a fresh vector load and
// movemask for every 6 decoded codepoints.
//
// Continuation masks alone do not validate UTF-8; they are only layout hints.
// That is fine here because the public input is already a valid `str`. The
// fast paths still have to preserve codepoint boundaries: the all-2-byte check
// looks at one extra continuation bit so a 4-byte sequence cannot masquerade as
// two 2-byte codepoints at the edge of the 16-byte decoder.
//
// Dense fixed-width cases are decoded in larger groups: six codepoints when
// every sequence is one or two bytes, eight codepoints for all 3-byte
// sequences, and eight codepoints for all 4-byte sequences. The fallback still
// handles arbitrary four-codepoint groups with scalar length checks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Utf8ToUtf32AVX2;

impl Backend for Utf8ToUtf32AVX2 {
    #[inline(always)]
    fn is_available() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[inline(always)]
    unsafe fn convert_into(input: &str, out: &mut Vec<u32>) {
        unsafe { convert_into_avx2(input, out) }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convert_into_avx2(input: &str, out: &mut Vec<u32>) {
    unsafe { convert_avx2(input, out) }
}

#[inline(always)]
unsafe fn convert_avx2(input: &str, out: &mut Vec<u32>) {
    unsafe {
        out.clear();

        let bytes = input.as_bytes();
        let len = bytes.len();
        // A valid UTF-8 string never has more codepoints than bytes, so byte
        // length is a safe upper bound for the reusable UTF-32 buffer.
        out.reserve(len);

        let input_ptr = bytes.as_ptr();
        let output_ptr = out.as_mut_ptr();
        let mut input_pos = 0usize;
        let mut output_len = 0usize;

        // ASCII dominates common fuzzy-match inputs. When the movemask is
        // zero, every byte is already a complete codepoint and conversion is
        // just zero-extension into u32 lanes.
        while input_pos + 64 <= len {
            let first = _mm256_loadu_si256(input_ptr.add(input_pos) as *const __m256i);
            let second = _mm256_loadu_si256(input_ptr.add(input_pos + 32) as *const __m256i);
            let non_ascii = _mm256_movemask_epi8(first) | _mm256_movemask_epi8(second);

            if non_ascii == 0 {
                store_ascii_32(first, output_ptr.add(output_len));
                store_ascii_32(second, output_ptr.add(output_len + 32));
                input_pos += 64;
                output_len += 64;
            } else {
                let block_start = input_pos;
                let mut continuation_mask = u64::from(continuation_mask_32(first))
                    | (u64::from(continuation_mask_32(second)) << 32);

                // Dense 3-byte text otherwise wastes the last bytes of each
                // 64-byte mask window because 64 is not divisible by 3. Load a
                // third mask only for that shape and decode through byte 80;
                // the generic path stays at 48 so every fast decoder has the
                // boundary bits it needs inside the original 64-byte mask.
                if block_start + 96 <= len
                    && (continuation_mask as u32 & THREE_BYTE_BOUNDARY_MASK_8)
                        == THREE_BYTE_CONTINUATION_MASK_8
                {
                    let block_limit = block_start + 80;
                    let mut continuation_mask_hi = continuation_mask_32(_mm256_loadu_si256(
                        input_ptr.add(input_pos + 64) as *const __m256i,
                    ));

                    while input_pos < block_limit {
                        let (consumed, written) = convert_masked_or_fallback(
                            input_ptr.add(input_pos),
                            output_ptr.add(output_len),
                            len - input_pos,
                            continuation_mask as u32,
                        );
                        input_pos += consumed;
                        output_len += written;
                        shift_continuation_mask(
                            &mut continuation_mask,
                            &mut continuation_mask_hi,
                            consumed,
                        );
                    }
                } else {
                    let block_limit = block_start + 48;

                    while input_pos < block_limit {
                        let (consumed, written) = convert_masked_or_fallback(
                            input_ptr.add(input_pos),
                            output_ptr.add(output_len),
                            len - input_pos,
                            continuation_mask as u32,
                        );
                        input_pos += consumed;
                        output_len += written;
                        continuation_mask >>= consumed;
                    }
                }
            }
        }

        while input_pos + 32 <= len {
            let chunk = _mm256_loadu_si256(input_ptr.add(input_pos) as *const __m256i);
            if _mm256_movemask_epi8(chunk) == 0 {
                store_ascii_32(chunk, output_ptr.add(output_len));
                input_pos += 32;
                output_len += 32;
            } else {
                let continuation_mask = continuation_mask_32(chunk);
                let (consumed, written) = convert_masked_or_fallback(
                    input_ptr.add(input_pos),
                    output_ptr.add(output_len),
                    len - input_pos,
                    continuation_mask,
                );
                input_pos += consumed;
                output_len += written;
            }
        }

        // Finish any remaining half-register ASCII block before handing the
        // short tail to scalar chars().
        while input_pos + 16 <= len {
            let chunk = _mm_loadu_si128(input_ptr.add(input_pos) as *const __m128i);
            if _mm_movemask_epi8(chunk) == 0 {
                store_ascii_16(chunk, output_ptr.add(output_len));
                input_pos += 16;
                output_len += 16;
            } else {
                let (consumed, written) = convert_mixed_codepoints_fallback(
                    input_ptr.add(input_pos),
                    output_ptr.add(output_len),
                    len - input_pos,
                );
                input_pos += consumed;
                output_len += written;
            }
        }

        // All SIMD stores above wrote initialized u32 lanes directly into the
        // spare capacity. Publish only those lanes before using safe Vec
        // extension for the short suffix.
        out.set_len(output_len);
        let tail = std::str::from_utf8_unchecked(&bytes[input_pos..]);
        out.extend(tail.chars().map(|c| c as u32));
    }
}

#[inline(always)]
unsafe fn store_ascii_32(chunk: __m256i, output: *mut u32) {
    unsafe {
        let lo = _mm256_castsi256_si128(chunk);
        let hi = _mm256_extracti128_si256::<1>(chunk);

        _mm256_storeu_si256(output as *mut __m256i, _mm256_cvtepu8_epi32(lo));
        _mm256_storeu_si256(
            output.add(8) as *mut __m256i,
            _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(lo)),
        );
        _mm256_storeu_si256(output.add(16) as *mut __m256i, _mm256_cvtepu8_epi32(hi));
        _mm256_storeu_si256(
            output.add(24) as *mut __m256i,
            _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(hi)),
        );
    }
}

#[inline(always)]
unsafe fn store_ascii_16(chunk: __m128i, output: *mut u32) {
    unsafe {
        _mm256_storeu_si256(output as *mut __m256i, _mm256_cvtepu8_epi32(chunk));
        _mm256_storeu_si256(
            output.add(8) as *mut __m256i,
            _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(chunk)),
        );
    }
}

#[inline(always)]
unsafe fn convert_masked_codepoints(
    input: *const u8,
    output: *mut u32,
    continuation_mask: u32,
) -> Option<(usize, usize)> {
    unsafe {
        // Check bit 16 as a boundary guard. Without it, a 4-byte sequence that
        // starts at byte 14 has the same low 16 continuation bits as eight
        // 2-byte sequences.
        if (continuation_mask & TWO_BYTE_BOUNDARY_MASK_8) == TWO_BYTE_CONTINUATION_MASK_8 {
            convert_eight_one_or_two_byte_codepoints(input, output, 0xff);
            return Some((16, 8));
        }

        if let Some((consumed, shuffle_index)) =
            six_one_or_two_byte_mask_from_continuation_mask(continuation_mask)
        {
            convert_six_one_or_two_byte_codepoints(input, output, shuffle_index);
            return Some((consumed, 6));
        }

        if (continuation_mask & THREE_BYTE_BOUNDARY_MASK_8) == THREE_BYTE_CONTINUATION_MASK_8 {
            convert_eight_three_byte_codepoints(input, output);
            return Some((24, 8));
        }

        if continuation_mask == FOUR_BYTE_CONTINUATION_MASK_8 {
            convert_eight_four_byte_codepoints(input, output);
            return Some((32, 8));
        }

        None
    }
}

#[inline(always)]
unsafe fn convert_masked_or_fallback(
    input: *const u8,
    output: *mut u32,
    remaining: usize,
    continuation_mask: u32,
) -> (usize, usize) {
    unsafe {
        if let Some(decoded) = convert_masked_codepoints(input, output, continuation_mask) {
            decoded
        } else {
            convert_mixed_codepoints_fallback(input, output, remaining)
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convert_mixed_codepoints_fallback(
    input: *const u8,
    output: *mut u32,
    remaining: usize,
) -> (usize, usize) {
    unsafe {
        let first_len = utf8_codepoint_len(*input);
        let second_start = first_len;
        let second_len = utf8_codepoint_len(*input.add(second_start));
        let third_start = second_start + second_len;
        let third_len = utf8_codepoint_len(*input.add(third_start));
        let fourth_start = third_start + third_len;
        let fourth_len = utf8_codepoint_len(*input.add(fourth_start));

        if first_len <= 2 && second_len <= 2 && third_len <= 2 && fourth_len <= 2 {
            let fifth_start = fourth_start + fourth_len;
            let fifth_len = utf8_codepoint_len(*input.add(fifth_start));
            let sixth_start = fifth_start + fifth_len;
            let sixth_len = utf8_codepoint_len(*input.add(sixth_start));
            let seventh_start = sixth_start + sixth_len;
            let seventh_len = utf8_codepoint_len(*input.add(seventh_start));
            let eighth_start = seventh_start + seventh_len;
            let eighth_len = utf8_codepoint_len(*input.add(eighth_start));

            if fifth_len <= 2 && sixth_len <= 2 && seventh_len <= 2 && eighth_len <= 2 {
                let consumed = eighth_start + eighth_len;
                debug_assert!(consumed <= 16);
                convert_eight_one_or_two_byte_codepoints(
                    input,
                    output,
                    one_or_two_byte_shuffle_index([
                        first_len,
                        second_len,
                        third_len,
                        fourth_len,
                        fifth_len,
                        sixth_len,
                        seventh_len,
                        eighth_len,
                    ]),
                );
                return (consumed, 8);
            }
        }

        if first_len == 3 && second_len == 3 && third_len == 3 && fourth_len == 3 && remaining >= 24
        {
            let fifth_len = utf8_codepoint_len(*input.add(12));
            let sixth_len = utf8_codepoint_len(*input.add(15));
            let seventh_len = utf8_codepoint_len(*input.add(18));
            let eighth_len = utf8_codepoint_len(*input.add(21));

            if fifth_len == 3 && sixth_len == 3 && seventh_len == 3 && eighth_len == 3 {
                convert_eight_three_byte_codepoints(input, output);
                return (24, 8);
            }
        }

        if first_len <= 3 && second_len <= 3 && third_len <= 3 && fourth_len <= 3 {
            let consumed = fourth_start + fourth_len;
            debug_assert!(consumed <= 12);
            convert_four_one_to_three_byte_codepoints(
                input,
                output,
                one_to_three_byte_shuffle_index([first_len, second_len, third_len, fourth_len]),
            );
            return (consumed, 4);
        }

        if first_len == 4 && second_len == 4 && third_len == 4 && fourth_len == 4 {
            if remaining >= 32
                && utf8_codepoint_len(*input.add(16)) == 4
                && utf8_codepoint_len(*input.add(20)) == 4
                && utf8_codepoint_len(*input.add(24)) == 4
                && utf8_codepoint_len(*input.add(28)) == 4
            {
                convert_eight_four_byte_codepoints(input, output);
                return (32, 8);
            }

            convert_four_four_byte_codepoints(input, output);
            return (16, 4);
        }

        convert_four_codepoints(
            input,
            output,
            [0, second_start, third_start, fourth_start],
            [first_len, second_len, third_len, fourth_len],
        )
    }
}

#[inline(always)]
unsafe fn continuation_mask_32(chunk: __m256i) -> u32 {
    unsafe {
        let continuation = _mm256_cmpgt_epi8(_mm256_set1_epi8(-64), chunk);
        _mm256_movemask_epi8(continuation) as u32
    }
}

#[inline(always)]
fn shift_continuation_mask(mask: &mut u64, mask_hi: &mut u32, consumed: usize) {
    debug_assert!((1..=32).contains(&consumed));

    if consumed == 32 {
        *mask = (*mask >> 32) | (u64::from(*mask_hi) << 32);
        *mask_hi = 0;
    } else {
        *mask = (*mask >> consumed) | (u64::from(*mask_hi) << (64 - consumed));
        *mask_hi >>= consumed;
    }
}

#[inline(always)]
unsafe fn convert_six_one_or_two_byte_codepoints(
    input: *const u8,
    output: *mut u32,
    shuffle_index: usize,
) {
    unsafe {
        let shuffle =
            _mm_loadu_si128(SIX_ONE_OR_TWO_BYTE_SHUFFLES[shuffle_index].as_ptr() as *const __m128i);
        convert_one_or_two_byte_codepoints(input, output, shuffle);
    }
}

#[inline(always)]
unsafe fn convert_eight_one_or_two_byte_codepoints(
    input: *const u8,
    output: *mut u32,
    shuffle_index: usize,
) {
    unsafe {
        let shuffle =
            _mm_loadu_si128(ONE_OR_TWO_BYTE_SHUFFLES[shuffle_index].as_ptr() as *const __m128i);
        convert_one_or_two_byte_codepoints(input, output, shuffle);
    }
}

#[inline(always)]
unsafe fn convert_one_or_two_byte_codepoints(input: *const u8, output: *mut u32, shuffle: __m128i) {
    unsafe {
        let input = _mm_loadu_si128(input as *const __m128i);
        let packed = _mm_shuffle_epi8(input, shuffle);
        let low_payload = _mm_and_si128(packed, _mm_set1_epi16(0x7f));
        let high_payload = _mm_and_si128(packed, _mm_set1_epi16(0x1f00));
        let codepoints = _mm_or_si128(low_payload, _mm_srli_epi16::<2>(high_payload));
        _mm256_storeu_si256(output as *mut __m256i, _mm256_cvtepu16_epi32(codepoints));
    }
}

#[inline(always)]
unsafe fn convert_eight_three_byte_codepoints(input: *const u8, output: *mut u32) {
    unsafe {
        convert_four_three_byte_codepoints(input, output);
        convert_four_three_byte_codepoints(input.add(12), output.add(4));
    }
}

#[inline(always)]
unsafe fn convert_four_three_byte_codepoints(input: *const u8, output: *mut u32) {
    unsafe {
        let input = _mm_loadu_si128(input as *const __m128i);
        let shuffle = _mm_setr_epi8(2, 1, 0, -1, 5, 4, 3, -1, 8, 7, 6, -1, 11, 10, 9, -1);
        let packed = _mm_shuffle_epi8(input, shuffle);
        let low_payload = _mm_and_si128(packed, _mm_set1_epi32(0x7f));
        let middle_payload = _mm_and_si128(packed, _mm_set1_epi32(0x3f00));
        let high_payload = _mm_and_si128(packed, _mm_set1_epi32(0x0f0000));
        let codepoints = _mm_or_si128(
            _mm_or_si128(low_payload, _mm_srli_epi32::<2>(middle_payload)),
            _mm_srli_epi32::<4>(high_payload),
        );
        _mm_storeu_si128(output as *mut __m128i, codepoints);
    }
}

#[inline(always)]
unsafe fn convert_four_one_to_three_byte_codepoints(
    input: *const u8,
    output: *mut u32,
    shuffle_index: usize,
) {
    unsafe {
        let input = _mm_loadu_si128(input as *const __m128i);
        let shuffle =
            _mm_loadu_si128(ONE_TO_THREE_BYTE_SHUFFLES[shuffle_index].as_ptr() as *const __m128i);
        let packed = _mm_shuffle_epi8(input, shuffle);
        let low_payload = _mm_and_si128(packed, _mm_set1_epi32(0x7f));
        let middle_payload = _mm_and_si128(packed, _mm_set1_epi32(0x3f00));
        let high_payload = _mm_and_si128(packed, _mm_set1_epi32(0x0f0000));
        let codepoints = _mm_or_si128(
            _mm_or_si128(low_payload, _mm_srli_epi32::<2>(middle_payload)),
            _mm_srli_epi32::<4>(high_payload),
        );
        _mm_storeu_si128(output as *mut __m128i, codepoints);
    }
}

#[inline(always)]
unsafe fn convert_eight_four_byte_codepoints(input: *const u8, output: *mut u32) {
    unsafe {
        convert_four_four_byte_codepoints(input, output);
        convert_four_four_byte_codepoints(input.add(16), output.add(4));
    }
}

#[inline(always)]
unsafe fn convert_four_four_byte_codepoints(input: *const u8, output: *mut u32) {
    unsafe {
        let input = _mm_loadu_si128(input as *const __m128i);
        let shuffle = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        let packed = _mm_shuffle_epi8(input, shuffle);
        let low_payload = _mm_and_si128(packed, _mm_set1_epi32(0x7f));
        let middle_payload = _mm_and_si128(packed, _mm_set1_epi32(0x3f00));
        let upper_middle_payload = _mm_and_si128(packed, _mm_set1_epi32(0x3f0000));
        let high_payload = _mm_and_si128(packed, _mm_set1_epi32(0x07000000));
        let correction = _mm_srli_epi32::<1>(_mm_and_si128(packed, _mm_set1_epi32(0x400000)));
        let upper_middle_payload = _mm_xor_si128(upper_middle_payload, correction);
        let codepoints = _mm_or_si128(
            _mm_or_si128(low_payload, _mm_srli_epi32::<2>(middle_payload)),
            _mm_or_si128(
                _mm_srli_epi32::<4>(upper_middle_payload),
                _mm_srli_epi32::<6>(high_payload),
            ),
        );
        _mm_storeu_si128(output as *mut __m128i, codepoints);
    }
}

#[inline(always)]
unsafe fn convert_four_codepoints(
    input: *const u8,
    output: *mut u32,
    starts: [usize; 4],
    lens: [usize; 4],
) -> (usize, usize) {
    unsafe {
        // The caller only enters here while at least 16 bytes remain. Four
        // valid UTF-8 codepoints consume at most 16 bytes, and input_pos is
        // always advanced by whole codepoints, so all leading-byte reads and
        // the 16-byte vector load below stay within the input.
        let consumed = starts[3] + lens[3];
        debug_assert!(consumed <= 16);

        // Build a pshufb control vector that places each UTF-8 sequence into a
        // u32 lane as b0|b1|b2|b3. Unused bytes in shorter sequences are
        // zeroed. This per-call stack mask is simple, but it is one of the
        // main costs on dense Unicode text.
        let input = _mm_loadu_si128(input as *const __m128i);
        let shuffle = shuffle_for_codepoints(starts, lens);
        let packed = _mm_shuffle_epi8(input, shuffle);

        // Decode all possible sequence lengths in parallel and select the
        // correct result per lane using masks derived from the scalar lengths.
        let byte_mask = _mm_set1_epi32(0xff);
        let continuation_mask = _mm_set1_epi32(0x3f);
        let first_payload_mask = _mm_set1_epi32(0x1f);
        let second_payload_mask = _mm_set1_epi32(0x0f);
        let third_payload_mask = _mm_set1_epi32(0x07);

        let b0 = _mm_and_si128(packed, byte_mask);
        let b1 = _mm_and_si128(_mm_srli_epi32::<8>(packed), byte_mask);
        let b2 = _mm_and_si128(_mm_srli_epi32::<16>(packed), byte_mask);
        let b3 = _mm_and_si128(_mm_srli_epi32::<24>(packed), byte_mask);

        let cp1 = b0;
        let cp2 = _mm_or_si128(
            _mm_slli_epi32::<6>(_mm_and_si128(b0, first_payload_mask)),
            _mm_and_si128(b1, continuation_mask),
        );
        let cp3 = _mm_or_si128(
            _mm_or_si128(
                _mm_slli_epi32::<12>(_mm_and_si128(b0, second_payload_mask)),
                _mm_slli_epi32::<6>(_mm_and_si128(b1, continuation_mask)),
            ),
            _mm_and_si128(b2, continuation_mask),
        );
        let cp4 = _mm_or_si128(
            _mm_or_si128(
                _mm_slli_epi32::<18>(_mm_and_si128(b0, third_payload_mask)),
                _mm_slli_epi32::<12>(_mm_and_si128(b1, continuation_mask)),
            ),
            _mm_or_si128(
                _mm_slli_epi32::<6>(_mm_and_si128(b2, continuation_mask)),
                _mm_and_si128(b3, continuation_mask),
            ),
        );

        let codepoints = _mm_or_si128(
            _mm_or_si128(
                _mm_and_si128(cp1, lane_mask(lens, 1)),
                _mm_and_si128(cp2, lane_mask(lens, 2)),
            ),
            _mm_or_si128(
                _mm_and_si128(cp3, lane_mask(lens, 3)),
                _mm_and_si128(cp4, lane_mask(lens, 4)),
            ),
        );

        _mm_storeu_si128(output as *mut __m128i, codepoints);
        (consumed, 4)
    }
}

#[inline(always)]
fn utf8_codepoint_len(first_byte: u8) -> usize {
    if first_byte < 0x80 {
        1
    } else if first_byte < 0xe0 {
        2
    } else if first_byte < 0xf0 {
        3
    } else {
        4
    }
}

#[inline(always)]
fn one_or_two_byte_shuffle_index(lens: [usize; 8]) -> usize {
    let mut index = 0usize;
    for (lane, len) in lens.into_iter().enumerate() {
        debug_assert!((1..=2).contains(&len));
        index |= (len - 1) << lane;
    }
    index
}

#[inline(always)]
fn six_one_or_two_byte_mask_from_continuation_mask(
    continuation_mask: u32,
) -> Option<(usize, usize)> {
    let entry = SIX_ONE_OR_TWO_BYTE_MASKS[(continuation_mask & 0x0fff) as usize];
    if entry == INVALID_ONE_OR_TWO_BYTE_MASK {
        return None;
    }

    let consumed = (entry >> 8) as usize;
    if consumed == 12 && (continuation_mask & (1 << 12)) != 0 {
        return None;
    }

    Some((consumed, (entry & 0x3f) as usize))
}

#[inline(always)]
fn one_to_three_byte_shuffle_index(lens: [usize; 4]) -> usize {
    let mut index = 0usize;
    let mut multiplier = 1usize;
    for len in lens {
        debug_assert!((1..=3).contains(&len));
        index += (len - 1) * multiplier;
        multiplier *= 3;
    }
    index
}

#[inline(always)]
unsafe fn shuffle_for_codepoints(starts: [usize; 4], lens: [usize; 4]) -> __m128i {
    unsafe {
        let mut mask = [0x80u8; 16];
        for lane in 0..4 {
            for byte in 0..lens[lane] {
                mask[lane * 4 + byte] = (starts[lane] + byte) as u8;
            }
        }
        _mm_loadu_si128(mask.as_ptr() as *const __m128i)
    }
}

#[inline(always)]
unsafe fn lane_mask(lens: [usize; 4], target: usize) -> __m128i {
    unsafe {
        _mm_setr_epi32(
            full_mask(lens[0] == target),
            full_mask(lens[1] == target),
            full_mask(lens[2] == target),
            full_mask(lens[3] == target),
        )
    }
}

#[inline(always)]
fn full_mask(enabled: bool) -> i32 {
    if enabled { -1 } else { 0 }
}

const fn build_one_or_two_byte_shuffles<const ENTRIES: usize, const LANES: usize>()
-> [[u8; 16]; ENTRIES] {
    let mut table = [[0x80u8; 16]; ENTRIES];
    let mut index = 0usize;
    while index < table.len() {
        let mut input_pos = 0usize;
        let mut lane = 0usize;
        while lane < LANES {
            let output_pos = lane * 2;
            let len = 1 + ((index >> lane) & 1);
            if len == 1 {
                table[index][output_pos] = input_pos as u8;
            } else {
                table[index][output_pos] = (input_pos + 1) as u8;
                table[index][output_pos + 1] = input_pos as u8;
            }
            input_pos += len;
            lane += 1;
        }
        index += 1;
    }
    table
}

const fn build_six_one_or_two_byte_masks() -> [u16; 4096] {
    let mut table = [INVALID_ONE_OR_TWO_BYTE_MASK; 4096];
    let mut mask = 0usize;
    while mask < table.len() {
        table[mask] = six_one_or_two_byte_mask_entry(mask);
        mask += 1;
    }
    table
}

const fn six_one_or_two_byte_mask_entry(mask: usize) -> u16 {
    let mut input_pos = 0usize;
    let mut shuffle_index = 0usize;
    let mut lane = 0usize;

    while lane < 6 {
        if ((mask >> input_pos) & 1) != 0 {
            return INVALID_ONE_OR_TWO_BYTE_MASK;
        }

        if ((mask >> (input_pos + 1)) & 1) != 0 {
            if input_pos + 2 < 12 && ((mask >> (input_pos + 2)) & 1) != 0 {
                return INVALID_ONE_OR_TWO_BYTE_MASK;
            }
            shuffle_index |= 1 << lane;
            input_pos += 2;
        } else {
            input_pos += 1;
        }

        lane += 1;
    }

    ((input_pos as u16) << 8) | shuffle_index as u16
}

const fn build_one_to_three_byte_shuffles() -> [[u8; 16]; 81] {
    let mut table = [[0x80u8; 16]; 81];
    let mut index = 0usize;
    while index < table.len() {
        let mut pattern = index;
        let mut input_pos = 0usize;
        let mut lane = 0usize;
        while lane < 4 {
            let output_pos = lane * 4;
            let len = 1 + (pattern % 3);
            pattern /= 3;

            let mut byte = 0usize;
            while byte < len {
                table[index][output_pos + byte] = (input_pos + len - 1 - byte) as u8;
                byte += 1;
            }
            input_pos += len;
            lane += 1;
        }
        index += 1;
    }
    table
}
