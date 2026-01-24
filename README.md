# Frizbee

Frizbee is a SIMD fuzzy string matcher written in Rust. The core of the algorithm uses Smith-Waterman with affine gaps, similar to FZF, but with many of the scoring bonuses from FZY. In the included benchmark, with typo resistance disabled, it outperforms [nucleo](https://github.com/helix-editor/nucleo) by ~2x and supports multithreading (WIP), see [benchmarks](./BENCHMARKS.md). It matches against bytes directly, ignoring unicode. Used by [blink.cmp](https://github.com/saghen/blink.cmp), [fff.nvim](https://github.com/dmtrKovalenko/fff.nvim), [skim](https://github.com/helix-editor/nucleo). Special thank you to [stefanboca](https://github.com/stefanboca) and [ii14](https://github.com/ii14)!

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
- Does not support transposition (swapping two adjacent characters)

### Prefiltering

### Smith Waterman

The SIMD width will be chosen at runtime based on available instruction set extensions. Currently, only x86_64's AVX2 (256-bit) and AVX512 (512-bit) will be detected at runtime, falling back to 128-bit SIMD if neither is available.

Nucleo and FZF use a prefiltering step that removes any haystacks that do not include all of the characters in the needle. Frizbee does this by default but supports disabling it to allow for typos. You may control the maximum number of typos with the `max_typos` property.

### Scoring

- `MATCH_SCORE`: Score for a match
- `MISMATCH_PENALTY`: Penalty for a mismatch (substitution)
- `GAP_OPEN_PENALTY`: Penalty for opening a gap (deletion/insertion)
- `GAP_EXTEND_PENALTY`: Penalty for extending a gap (deletion/insertion)
- `PREFIX_BONUS`: Bonus for matching the first character of the haystack (e.g. "h" on "hello_world")
- `DELIMITER_BONUS`: Bonus for matching _after_ a delimiter character (e.g. "hw" on "hello_world", will give a bonus on "w")
- `CAPITALIZATION_BONUS`: Bonus for matching a capital letter after a lowercase letter (e.g. "b" on "fooBar" will receive a bonus on "B")
- `MATCHING_CASE_BONUS`: Bonus for matching the case of the needle (e.g. "WorLd" on "WoRld" will receive a bonus on "W", "o", "d")
- `EXACT_MATCH_BONUS`: Bonus for matching the exact needle (e.g. "foo" on "foo" will receive the bonus)

### Implementation

1. **Prefiltering**: When `max_typos = Some(x)`, perform a fast prefiltering step on the haystacks to exclude any items that can't match the needle
    - Loads the haystack in 16 byte chunks (128-bit SIMD) and checks for the presence of each character in the needle sequentially.
    - Notably, while iterating over the needle chars, the implementation does not check that the previous needle character comes before the current one, within a 16 byte chunk. This results in a ~40% speed-up but requires that we perform a backward-pass to check the number of typos.
    - This approach is ~3x faster than the `memchr` implementation used in Nucleo
2. **Bucketing**: Group the haystacks by length into buckets of various haystack lengths (`4`, `8`, `12`, ...) until the bucket reaches `$LANES` items, where `$LANES` is the number of available SIMD lanes
    - If the item would cause excessive memory usage, or we don't have a bucket big enough for the haystack (currently max bucket size is `512`), fallback to a greedy matcher
3. **Smith Waterman Forward Pass**: When a bucket is full, perform SIMD smith waterman on `$LANES` items at a time
4. **Smith Waterman Backward Pass**: If `max_typos != None`, perform a backward (alignment) pass to find the number of typos in the haystack
5. **Finalize:** Optionally sort (`opts.sort`) and return the matches

## Ideas

- [x] Runtime instruction selection (512-bit and 256-bit SIMD)
- [x] Multithreading
- [ ] Calculate alignment directions during matrix building
  - Might speed up typo calculation
- [ ] Experiment with u8 for math, converting to u16 for score
- [x] Drop u8 based scoring and double scoring to support longer fuzzy matches
  - Currently, alignment can be lost on longer matches causing us to mark them as having typos
- [x] Incremental matching
  - [ ] Runtime instruction selection (512-bit and 256-bit SIMD)
  - [ ] Prefiltering
  - [ ] Exact match bonus
