# Frizbee

Frizbee is a SIMD typo-resistant fuzzy string matcher written in Rust. The core of the algorithm uses Smith-Waterman with affine gaps, similar to FZF. In the included benchmark, with typo resistance disabled, it outperforms [Nucleo](https://github.com/helix-editor/nucleo) by ~4x and [FZF](https://github.com/junegunn/fzf) by ~5x and supports multithreading, see [benchmarks](./BENCHMARKS.md). When matching against unicode, it outperforms nucleo and fzf by 20x.

Used by [blink.cmp](https://github.com/saghen/blink.cmp), [skim](https://github.com/skim-rs/skim), and [fff](https://github.com/dmtrKovalenko/fff). Special thank you to [stefanboca](https://github.com/stefanboca) and [ii14](https://github.com/ii14)!

For commercial support, please [contact me](mailto:frizbee@liam.super.fish). I'd be happy to work with you directly! Also, please consider [sponsoring me](https://github.com/sponsors/saghen).

## Usage

See [the docs](https://docs.rs/frizbee) for more usage examples.

```rust
use frizbee::{match_list, match_list_parallel, Config};

let needle = "fBr";
let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];

let matches = match_list(needle, &haystacks, &Config::default());
// or in parallel (8 threads)
let matches = match_list_parallel(needle, &haystacks, &Config::default(), 8);
```

## Benchmarks

See [BENCHMARKS.md](./BENCHMARKS.md)

## Algorithm

The core of the algorithm is Smith-Waterman with affine gaps and row-wise parallelism via SIMD. Besides the parallelism, this is the basis of other popular fuzzy matching algorithms like [FZF](https://github.com/junegunn/fzf) and [Nucleo](https://github.com/helix-editor/nucleo). The main properties of Smith-Waterman are:

- Always finds the best alignment
- Supports insertion (unmatched char in haystack, basis of fuzzy matching)
- Supports deletion (unmatched char in needle, basis of typo-resistance)
- Supports substitution (haystack and needle char mismatch, basis of typo-resistance)

### Prefiltering

Nucleo and FZF use a prefiltering step that removes any haystacks that do not include all of the characters in the needle. Frizbee does this by default but supports alternative algorithms to allow for typos. You may control the maximum number of typos with the `max_typos` property.

Nucleo uses [`memchr`](https://docs.rs/memchr/2.7.6/memchr/) to ensure that the needle is wholly contained in the haystack, in the correct order. For case insensitive matching, it checks the lowercase and uppercase needle chars separately. For each character, it loads the haystack from the previously matched position and performs a sequential search via SIMD. This results in many unnecessary loads (for a needle of 6 chars with a haystack of length 32 with 16 byte wide SIMD, this would lead to 6 + 2 - 1 = 7 loads).

Frizbee improves upon this by loading the haystack chunk by chunk and masking out the already-matched prefix for each needle char. This results in ahaystack of length 32 with 16 byte wide SIMD only performing 2 loads. Frizbee also uses prefiltering to find the prefix/suffix of the haystack of characters that are impossible to match (not in the needle), similar to FZF.

```
needle: "foo"
haystack: "oFoo"

// assuming 4 byte SIMD width for simplicity
// in reality, the widths are 16 (SSE/NEON), 32 (AVX2), or 64 (AVX512) bytes

haystack: [_, F, o, o]

// first iter
needle: ([f, f, f, f], [F, F, F, F])
mask: [00, FF, 00, 00] // needle.0 == haystack | needle.1 == haystack
bitmask: 0b0010 & haystack_mask // movemask(mask)
bitmask > 0 // needle found in haystack, check next needle char
min_bit: 1 // bitmask.trailing_zeros()
haystack_mask: 0b1100 // !1u32 << min_bit

haystack_start_pos: min_bit

// second iter
needle: ([o, o, o, o], [O, O, O, O])
mask: [FF, 00, FF, FF] // needle.0 == haystack | needle.1 == haystack
bitmask: 0b1101 & haystack_mask // movemask(mask)
bitmask: 0b1100
min_bit: 2 // bitmask.trailing_zeros()
haystack_mask: 0b0111 // !1u32 << min_bit

// third iter
needle: ([o, o, o, o], [O, O, O, O])
mask: [FF, 00, FF, FF] // needle.0 == haystack | needle.1 == haystack
bitmask: 0b1101 & haystack_mask // movemask(mask)
bitmask: 0b1000
min_bit: 3 // bitmask.trailing_zeros()
haystack_mask: 0b1111 // !1u32 << min_bit

// in reality, if we're not on the last chunk, we search for
// the last occurrence of the last needle char separately
haystack_end_pos: min_bit

// ran out of needle chars, matched!
return (true, haystack_start_pos, haystack_end_pos)
```

See the full implementation as well as 1-typo, 2-typo and N-typo implementations in [`src/prefilter/algo`](src/prefilter/algo/ascii.rs).

### Smith Waterman

The [Smith Waterman algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) performs local sequence alignment ([explanation](https://kaell.se/bibook/pairwise/waterman.html)), originally designed to find similar sequences between two DNA strings. The algorithm's time and space complexity of O(nm) led to plenty of research on parallelization. Each cell in the matrix has a data dependency on the cell to the left, up, and left-up diagonal. For biology, DNA sequences are typically quite large (m > 1000), so most of the parallelization approaches focused on large matrices ([see this paper for common parallelization techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419822)).

As a fuzzy matcher, the matrices in Frizbee are typically much smaller than those in DNA alignment (m < 128). Frizbee uses an approach similar to [sequential layout](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419822/#Sec11), except the horizontal (vertical in the paper, but flipped in Frizbee) data dependency [is applied immediately](src/smith_waterman/algo/ascii_gap.rs). This approach supports [affine gaps](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Affine). When the maximum score is < 256 (based on needle length), Frizbee internally uses u8 scoring, effectively doubling SIMD width over the default u16 scoring.

```
needle: "foo"
haystack: "some/long/foo/path"

// assuming 4 lane SIMD for simplicity
// in reality, we use 16 SIMD lanes (16 bits per lane, 256 bit)

// the final matrix would look like this
score_matrix:
   [s   o   m   e]   [/   l   o   n]   [g   /   f   o]   [o   /   p   a]   [t   h   _   _]
f  [0   0   0   0]   [0   0   0   0]   [0   0   16  11]  [10  9   8   7]   [6   5   4   3]
o  [0   16  11  10]  [9   8   16  11]  [10  9   8   32]  [27  26  25  24]  [23  22  21  20]
o  [0   16  11  10]  [9   8   24  19]  [18  17  16  24]  [48  43  42  41]  [40  39  38  37]

// for the SIMD register at row 3, col 2, before applying gap propgation, we would start with

needle:      [o   o   o   o]
haystack:    [/   l   o   n]
match mask:  [N   N   Y   N]

// left-up diagonal data dependency
diagonal:     [10  9   8   16]
match:        [-6  -6  16  -6]

// up data dependency
up:          [9   8   16  11]
gap_penalty: [-1  -1  -1  -1]

current:     [8   7   24  10] <- max(diagonal + match, up + gap_penalty)

// now we propagate the left data dependency

left:        [0   16  11  10]
// shift `current` right by 1 element, filling in with right most element from left
shifted:     [10  8   7   24]
// decay by gap extend penalty (1)
// last element decayed by 5 (gap open penalty) instead of 1 (gap extend penalty)
// because the previous element matched (affine gaps)
decayed:     [9   7   6   19]
// max with current
current:     [9   7   24  19]
// repeat for shifting by 2 elements
shifted:     [11  10  9   7]
decayed:     [9   8   7   5] <- gap extend penalty * 2 or gap open penalty + extend penalty
current:     [9   8   24  19]

final:       [8   7   24  19]
```

Frizbee previously used inter-sequence parallelism (one needle, $LANES haystacks) but this performed only slightly better than sequential layout due to requiring interleaving the haystacks and bucketing based on haystack length, while performing worse in parallel due to the required bucketing. This may be revisited in the future (PRs welcome, search git history for "bucket").

### Multithreading

The parallel implementation uses work-stealing to distribute the work across threads. Each thread sorts the matches individually and the final result uses k-way merge for concatenation. In the chromium benchmark, this gets reasonably close to perfect scaling: 3.6ms vs 2.8ms (theoretical perfect)

### Scoring

- `MATCH_SCORE`: Score for a match
- `MISMATCH_PENALTY`: Penalty for a mismatch (substitution)
- `GAP_OPEN_PENALTY`: Penalty for opening a gap (deletion/insertion)
- `GAP_EXTEND_PENALTY`: Penalty for extending a gap (deletion/insertion)
- `PREFIX_BONUS`: Bonus for matching the first character of the haystack (e.g. "h" on "hello_world")
- `DELIMITER_BONUS`: Bonus for matching after a non-alphanumeric character (e.g. "hw" on "hello_world", will give a bonus on "w")
- `CAPITALIZATION_BONUS`: Bonus for matching a capital letter after a lowercase letter (e.g. "b" on "fooBar" will receive a bonus on "B")
- `MATCHING_CASE_BONUS`: Bonus for matching the case of the needle (e.g. "WorLd" on "WoRld" will receive a bonus on "W", "o", "d")
- `EXACT_MATCH_BONUS`: Bonus for matching the exact needle (e.g. "foo" on "foo" will receive the bonus)

### Unicode

Frizbee matches against UTF-8 bytes directly rather than converting to UTF-32 codepoints like Nucleo. This results in near-native performance without preprocessing or any extra memory usage. By default, with `config.unicode = UnicodeMatching::Smart`, the unicode path will only be taken when the needle contains non-ASCII characters.

#### Prefiltering

The prefilter path is largely the same as the ASCII path, except we match on the final byte of the UTF-8 codepoint, since it's by far the most discriminating. The prefix bytes of the UTF-8 codepoint identify the script/block, and nearby characters are likely to have a similar prefix. If `!mask.is_zero()`, we then perform offset loads to match the rest of the bytes in the unicode char.

```rust
let mask = eq(needle_char.last(), load(haystack, start + needle_char_len - 1)
if mask.is_zero() { /* continue to next chunk */ }

match needle_char_len {
  1 => {}
  2 => {
    mask = mask.and(eq(needle_char[0], load(haystack, start + needle_char_len - 2)))
  }
  3 => {
    mask = mask.and(eq(needle_char[0], load(haystack, start + needle_char_len - 3)))
    mask = mask.and(eq(needle_char[1], load(haystack, start + needle_char_len - 2)))
  }
  4 => { ... }
}
```

#### Smith Waterman

The matrix receives a row per needle UTF-8 codepoint, rather than per needle byte. This significantly reduces the size of the matrix versus the naive path (i.e. 5 chars of 3 bytes/char would be 15 rows vs 5 rows). The match mask uses a similar approach to the prefilter with the offset loads. The tricky part was applying the gap propagation, since we only want to apply a gap penalty once per multi-byte haystack char. The solution was to keep track of the number of lead/continuation bytes we've crossed, see the [docs on the implementation](src/smith_waterman/algo/unicode_gap.rs).

#### Limitations

Frizbee will not perform any [unicode normalization](https://docs.rs/unicode-normalization/latest/unicode_normalization/) before matching. You should apply this yourself if you need it.

With the default `UnicodeMatching::Smart`, an ASCII needle matching against a haystack with multi-byte UTF-8 codepoints will have a slightly lower score than a pure ASCII haystack. For example, `hw` matched against `h😀w` will receive a penalty to the score of `gap_open_penality + gap_extend_penalty * 4` due to the emoji taking up 4 bytes. If the haystack was instead `hew`, the penalty would be `gap_open_penality + gap_extend_penalty`. This should typically be a non-issue, but you can force the slower unicode path with `UnicodeMatching::Respect` on ASCII needles, if necessary.

For case-insenstive matching, the case flipped version will be skipped if it's a different byte length or has multiple codepoints (such as the German `ß` -> `SS`).

Unlike FZF, Frizbee will not match `a` against `á`.

## Safety

On stable Rust, it's only possible to use SIMD via intrinsics ([portable-simd](https://github.com/rust-lang/portable-simd) is nightly-only). Many existing crates for safe SIMD abstractions do not currently support AVX512, or left performance on the table. The codebase isolates the vast majority of the unsafe code to SIMD "Backend"s ([prefilter](src/prefilter/backend) and [smith waterman](src/smith_waterman/backend)) which contain many unit/property tests, checked through Miri.

Without the `safe_read` feature, Frizbee will over-read haystacks when safe to do so (within page-boundary) which will trigger the `AddressSanitizer`. Without AVX512, performance regresses by ~40% with `safe_read` enabled. Over-reads are automatically disabled when running inside of miri (`cfg!(miri)`).
