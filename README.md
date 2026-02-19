# Frizbee

Frizbee is a SIMD typo-resistant fuzzy string matcher written in Rust. The core of the algorithm uses Smith-Waterman with affine gaps, similar to FZF, but with many of the scoring bonuses from FZY. In the included benchmark, with typo resistance disabled, it outperforms [nucleo](https://github.com/helix-editor/nucleo) by ~1.7x and supports multithreading, see [benchmarks](./BENCHMARKS.md). It matches against bytes directly, ignoring unicode. Used by [blink.cmp](https://github.com/saghen/blink.cmp), [skim](https://github.com/skim-rs/skim), [fff.nvim](https://github.com/dmtrKovalenko/fff.nvim). Special thank you to [stefanboca](https://github.com/stefanboca) and [ii14](https://github.com/ii14)!

## Usage

```rust
use frizbee::{match_list, Options};

let needle = "fBr";
let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];

let matches = match_list(needle, &haystacks, Options::default());
```

## Benchmarks

See [BENCHMARKS.md](./BENCHMARKS.md)

## Algorithm

The core of the algorithm is Smith-Waterman with affine gaps and row-wise parallelism via SIMD. Besides the parallelism, this is the basis of other popular fuzzy matching algorithms like [FZF](https://github.com/junegunn/fzf) and [Nucleo](https://github.com/helix-editor/nucleo). The main properties of Smith-Waterman are:

- Always finds the best alignment
- Supports insertion, deletion and substitution

### Prefiltering

Nucleo and FZF use a prefiltering step that removes any haystacks that do not include all of the characters in the needle. Frizbee does this by default but supports disabling it to allow for typos. You may control the maximum number of typos with the `max_typos` property.

Nucleo uses [`memchr`](https://docs.rs/memchr/2.7.6/memchr/) to ensure that the needle is wholly contained in the haystack, in the correct order. For case insensitive matching, it checks the lowercase and uppercase needle chars separately.

Frizbee improves upon this by checking both lowercase and uppercase needle chars in the same comparison (when AVX2 is available) and by ignoring the ordering of the needle characters in the haystack. Ignoring the ordering allows for false-positives, which must be filtered out during the alignment phase of the Smith Waterman. However, the speed-up from re-using the haystack loads drastically outweigh the slow down from the alignment phase.

```
needle: "foo"
haystack: "Fo_o"

// assuming 4 and 8 byte SIMD widths for simplicity
// in reality, the widths are 16 and 32 bytes

needle: [f, f, f, f, F, F, F, F]

haystack: [F, o, _, o]
// expand to 8 bytes by broadcasting lo to hi
haystack: [F, o, _, o, F, o, _, o]

needle == haystack
mask: [00, 00, 00, 00, FF, 00, 00, 00]
movemask(mask) > 0 // needle found in haystack, check next needle char
```

See the full implementation in [`src/prefilter/x86_64/avx2.rs`](src/prefilter/x86_64/avx2.rs). When 256-bit SIMD is not available (no AVX2 or ARM), we simply check the uppercase and lowercase separately.

### Smith Waterman

The [Smith Waterman algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) performs local sequence alignment ([explanation](https://kaell.se/bibook/pairwise/waterman.html)), originally designed to find similar sequences between two DNA strings. The algorithm's time and space complexity of O(nm) led to plenty of research on parallelization. Each cell in the matrix has a data dependency on the cell to the left, up, and left-up diagonal. For biology, DNA sequences are typically quite large (m > 1000), so most of the parallelization approaches focused on large matrices ([see this paper for common parallelization techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419822)).

As a fuzzy matcher, the matrices in Frizbee are typically much smaller than those in DNA alignment (m < 128). Frizbee uses an approach similar to [sequential layout](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419822/#Sec11), except the horizontal (vertical in the paper, but flipped in frizbee) data dependency [is applied immediately](src/smith_waterman/simd/gaps.rs). This approach supports [affine gaps](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Affine).

```
needle: "foo"
haystack: "some/long/foo/path"

// assuming 4 lane SIMD for simplicity
// in reality, we use 16 SIMD lanes (16 bytes each, 256 bit)

score_matrix:
   [s   o   m   e]   [/   l   o   n]   [g   /   f   o]   [o   /   p   a]   [t   h   _   _]
f  [0   0   0   0]   [0   0   0   0]   [0   0   16  11]  [10  9   8   7]   [6   5   4   3]
o  [0   16  11  10]  [9   8   16  11]  [10  9   8   32]  [27  26  25  24]  [23  22  21  20]
o  [0   16  11  10]  [9   8   24  19]  [18  17  16  24]  [48  43  42  41]  [40  39  38  37]

// for the SIMD register at row 2, col 1, we would start with

needle:      [o   o   o   o]
haystack:    [/   l   o   n]
match mask:  [f   f   t   f]

diagonal:    [10  9   8   16]
up:          [9   8   16  11]
current:     [8   7   24  9]

// now we propagate the left data dependency

left:        [0   16  11  10]
// shift current right by 1 element, filling in with right most element from left
shifted:     [10  8   7   24]
// decay by gap extend penalty (1)
// last element decayed by 5 (gap open penalty) instead of 1 because the previous element matched (affine gaps)
decayed:     [9   7   6   19]
// max with current
current:     [9   7   24  19]
// repeat for shifting by 2 elements
shifted:     [11  10  9   7]
decayed:     [9   8   7   5] // gap extend penalty * 2 or gap open penalty + extend penalty
current:     [9   8   24  19]

final:       [8   7   24  19]
```

Frizbee previously used inter-sequence parallelism (one needle, $LANES haystacks) but this performed about the same as sequential layout due to requiring interleaving the haystacks and bucketing based on haystack length, while performing worse in parallel due to the required bucketing.

### Multithreading

The parallel implementation uses work-stealing to distribute the work across threads. Each thread sorts the matches individually and the final result concatenated via k-way merge from itertools. In the chromium benchmark, this gets reasonably close to perfect scaling (7.6ms vs 6.8ms).

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
