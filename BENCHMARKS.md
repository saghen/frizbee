# Benchmarks

## Table of Contents

- [Environment](#environment)
- [Overview](#overview)
- [Benchmark Results](#benchmark-results)
    - [Chromium](#chromium)
    - [Arabic](#arabic)
    - [Korean](#korean)
    - [Partial Match](#partial-match)
    - [All Match](#all-match)
    - [No Match with Partial](#no-match-with-partial)
    - [No Match](#no-match)
    - [Copy](#copy)
    - [Sort](#sort)

## Environment

You may test these cases yourself via the included benchmarks. Benchmarks were run on a Ryzen 9950x3D and the following environment:

```bash
$ cargo version -v
cargo 1.98.0-nightly (a595d0da2 2026-06-20)
release: 1.98.0-nightly
commit-hash: a595d0da21f228b7fdae64d3d5c0e527ea66bb59
commit-date: 2026-06-20
host: x86_64-unknown-linux-gnu
libgit2: 1.9.4 (sys:0.21.0 vendored)
libcurl: 8.20.0-DEV (sys:0.4.89+curl-8.20.0 vendored ssl:OpenSSL/3.6.2)
ssl: OpenSSL 3.6.2 7 Apr 2026
os: NixOS 26.11.0 (zokor) [64-bit]
```

## Overview

In each of the benchmarks, the median length of the haystacks is varied from 8 to 128.

- **Frizbee**: Uses the `Config::default()`, where we perform the fastest prefilter since no typos are allowed
- **Iter**: Same as **Frizbee**, but uses the `match_iter` API to lazily match haystacks one at a time
- **Parallel (x8)**: Same as $BENCH, but uses 8 threads to perform the matching in parallel
- **All Scores**: Set via `max_typos: None`, gets the scores for all of the items without any filtering
- **1/2 Typos**: Set via `max_typos: Some(1 || 2)`, performs a slower, but still effective prefilter since a small number of typos are allowed
- **3 Typos**: Set via `max_typos: Some(3)`, skips prefiltering since in non-syntheic data (Chromium), the prefilter has to pass over the data up to 4 times and most items will not be filtered out
- **Nucleo**: Runs with normalization disabled, case insensitivity enabled and fuzzy matching enabled
- **FZF**: Times only the matching part via `fzf --filter linux --tiebreak index --bench 10s --threads 1 < benches/match_list/data.txt`
- **$BENCH/Parallel (x8)**: Same as $BENCH, but uses 8 threads to perform the matching in parallel

## Benchmark Results

### Chromium

List of all file paths in the chromium repository, with a median length of 67 characters.
```rust
needle: "linux"
match_percentage: 0.08
partial_match_percentage: unknown
median_length: 67
std_dev_length: unknown
haystack_size: 1406941
```

|                     | `Nucleo`                 | `FZF`                            | `Frizbee`                       | `Iter`                          | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                         |
|:--------------------|:-------------------------|:---------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------- |
| **`Sequential`**    | `90.53 ms` (**1.00x**) | `120.84 ms` (*1.33x slower*)   | `22.36 ms` (**4.05x faster**) | `24.68 ms` (**3.67x faster**) | `84.64 ms` (**1.07x faster**) | `60.76 ms` (**1.49x faster**) | `99.15 ms` (**1.10x slower**) | `142.39 ms` (*1.57x slower*)    |
| **`Parallel (x8)`** | `36.78 ms` (**1.00x**) | `16.36 ms` (**2.25x faster**)  | `3.48 ms` (**10.58x faster**) | `N/A`                           | `13.81 ms` (**2.66x faster**) | `9.50 ms` (**3.87x faster**)  | `15.58 ms` (**2.36x faster**) | `20.29 ms` (**1.81x faster**)   |

### Arabic

Arabic sentence dataset to test UTF-8 throughput for 2-bytes per character

```rust
needle: "إن"
match_percentage: 0.07934
partial_match_percentage: 0.59514
median_length: 37 bytes / 21 chars
mean_length: 43.18 bytes / 23.80 chars
std_dev_length: 33.27 bytes / 18.40 chars
haystack_size: 285587
total_bytes: 12331053
```

|                     | `Nucleo`                  | `FZF`                            | `Frizbee`                         | `Iter`                          | `All Scores`                    | `1 Typo`                         | `2 Typos`                       | `3 Typos`                        |
|:--------------------|:--------------------------|:---------------------------------|:----------------------------------|:--------------------------------|:--------------------------------|:---------------------------------|:--------------------------------|:-------------------------------- |
| **`Sequential`**    | `119.61 ms` (**1.00x**) | `165.94 ms` (*1.39x slower*)   | `2.60 ms` (**46.07x faster**)   | `2.98 ms` (**40.10x faster**) | `15.13 ms` (**7.90x faster**) | `11.46 ms` (**10.43x faster**) | `14.92 ms` (**8.02x faster**) | `15.39 ms` (**7.77x faster**)  |
| **`Parallel (x8)`** | `9.49 ms` (**1.00x**)   | `22.15 ms` (*2.33x slower*)    | `480.86 us` (**19.74x faster**) | `N/A`                           | `2.65 ms` (**3.58x faster**)  | `1.97 ms` (**4.81x faster**)   | `2.61 ms` (**3.64x faster**)  | `2.70 ms` (**3.51x faster**)   |

### Korean

Korean sentence dataset to test UTF-8 throughput for 3-bytes per character

```rust
needle: "니다"
match_percentage: 0.08419
partial_match_percentage: 0.40674
median_length: 36 bytes / 15 chars
mean_length: 39.95 bytes / 16.38 chars
std_dev_length: 24.76 bytes / 10.20 chars
haystack_size: 281471
total_bytes: 11244535
```

|                     | `Nucleo`                  | `FZF`                            | `Frizbee`                         | `Iter`                          | `All Scores`                    | `1 Typo`                         | `2 Typos`                       | `3 Typos`                        |
|:--------------------|:--------------------------|:---------------------------------|:----------------------------------|:--------------------------------|:--------------------------------|:---------------------------------|:--------------------------------|:-------------------------------- |
| **`Sequential`**    | `104.00 ms` (**1.00x**) | `114.52 ms` (**1.10x slower**) | `2.53 ms` (**41.11x faster**)   | `3.46 ms` (**30.07x faster**) | `15.51 ms` (**6.70x faster**) | `10.07 ms` (**10.32x faster**) | `15.10 ms` (**6.89x faster**) | `15.65 ms` (**6.64x faster**)  |
| **`Parallel (x8)`** | `8.48 ms` (**1.00x**)   | `15.59 ms` (*1.84x slower*)    | `477.78 us` (**17.74x faster**) | `N/A`                           | `2.61 ms` (**3.25x faster**)  | `1.66 ms` (**5.09x faster**)   | `2.57 ms` (**3.29x faster**)  | `2.68 ms` (**3.16x faster**)   |

### Partial Match

What I would consider the typical case, where 5% of the haystack matches the needle and 20% of the haystack includes characters from the needle, but doesn't fully match.

```rust
needle: "deadbeef"
partial_match_percentage: 0.20
match_percentage: 0.05
median_length: varies
std_dev_length: median_length / 4
haystack_size: 100000
```

|           | `Nucleo`                | `Frizbee`                        | `Iter`                         | `All Scores`                    | `1 Typo`                       | `2 Typos`                      | `3 Typos`                       |
|:----------|:------------------------|:---------------------------------|:-------------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:------------------------------- |
| **`16`**  | `3.23 ms` (**1.00x**) | `879.17 us` (**3.67x faster**) | `1.05 ms` (**3.08x faster**) | `7.78 ms` (*2.41x slower*)    | `1.39 ms` (**2.33x faster**) | `1.82 ms` (**1.77x faster**) | `3.07 ms` (**1.05x faster**)  |
| **`32`**  | `3.96 ms` (**1.00x**) | `867.80 us` (**4.57x faster**) | `1.02 ms` (**3.88x faster**) | `7.79 ms` (*1.96x slower*)    | `1.37 ms` (**2.89x faster**) | `1.82 ms` (**2.18x faster**) | `3.16 ms` (**1.25x faster**)  |
| **`64`**  | `5.42 ms` (**1.00x**) | `1.24 ms` (**4.36x faster**)   | `1.43 ms` (**3.79x faster**) | `11.35 ms` (*2.10x slower*)   | `1.85 ms` (**2.93x faster**) | `2.36 ms` (**2.29x faster**) | `3.80 ms` (**1.43x faster**)  |
| **`128`** | `8.89 ms` (**1.00x**) | `1.78 ms` (**4.99x faster**)   | `1.95 ms` (**4.55x faster**) | `18.50 ms` (*2.08x slower*)   | `2.66 ms` (**3.34x faster**) | `3.37 ms` (**2.64x faster**) | `5.07 ms` (**1.75x faster**)  |

### All Match

All of the haystacks match the needle. The "All Scores" case will always be the fastest since it skips the prefiltering step, which no longer filters any of the items out.

```rust
needle: "deadbeef"
match_percentage: 1.0
partial_match_percentage: 0.0
median_length: varies
std_dev_length: median_length / 4
haystack_size: 100000
```

|           | `Nucleo`                  | `Frizbee`                       | `Iter`                          | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                        |
|:----------|:--------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `22.91 ms` (**1.00x**)  | `8.57 ms` (**2.67x faster**)  | `9.24 ms` (**2.48x faster**)  | `7.64 ms` (**3.00x faster**)  | `11.90 ms` (**1.92x faster**) | `12.43 ms` (**1.84x faster**) | `14.06 ms` (**1.63x faster**)  |
| **`32`**  | `38.46 ms` (**1.00x**)  | `8.54 ms` (**4.50x faster**)  | `9.16 ms` (**4.20x faster**)  | `7.64 ms` (**5.03x faster**)  | `11.92 ms` (**3.23x faster**) | `12.36 ms` (**3.11x faster**) | `14.74 ms` (**2.61x faster**)  |
| **`64`**  | `62.85 ms` (**1.00x**)  | `10.65 ms` (**5.90x faster**) | `11.03 ms` (**5.70x faster**) | `11.13 ms` (**5.65x faster**) | `14.16 ms` (**4.44x faster**) | `14.66 ms` (**4.29x faster**) | `18.52 ms` (**3.39x faster**)  |
| **`128`** | `118.71 ms` (**1.00x**) | `18.15 ms` (**6.54x faster**) | `18.40 ms` (**6.45x faster**) | `18.27 ms` (**6.50x faster**) | `21.05 ms` (**5.64x faster**) | `22.13 ms` (**5.36x faster**) | `26.43 ms` (**4.49x faster**)  |

### No Match with Partial

None of the haystacks fully match the needle while 15% of the haystack includes characters from the needle, but doesn't fully match.

```rust
needle: "deadbeef"
match_percentage: 0.0
partial_match_percentage: 0.15
median_length: varies
std_dev_length: median_length / 4
haystack_size: 100000
```

|           | `Nucleo`                | `Frizbee`                        | `Iter`                           | `All Scores`                    | `1 Typo`                         | `2 Typos`                        | `3 Typos`                       |
|:----------|:------------------------|:---------------------------------|:---------------------------------|:--------------------------------|:---------------------------------|:---------------------------------|:------------------------------- |
| **`16`**  | `1.87 ms` (**1.00x**) | `254.39 us` (**7.33x faster**) | `415.28 us` (**4.49x faster**) | `7.81 ms` (*4.19x slower*)    | `647.76 us` (**2.88x faster**) | `991.05 us` (**1.88x faster**) | `1.97 ms` (**1.05x slower**)  |
| **`32`**  | `1.87 ms` (**1.00x**) | `249.07 us` (**7.51x faster**) | `406.41 us` (**4.60x faster**) | `7.80 ms` (*4.17x slower*)    | `646.65 us` (**2.89x faster**) | `981.22 us` (**1.91x faster**) | `1.89 ms` (**1.01x slower**)  |
| **`64`**  | `2.16 ms` (**1.00x**) | `563.52 us` (**3.83x faster**) | `716.38 us` (**3.01x faster**) | `11.29 ms` (*5.24x slower*)   | `1.04 ms` (**2.07x faster**)   | `1.43 ms` (**1.51x faster**)   | `2.45 ms` (*1.13x slower*)    |
| **`128`** | `2.74 ms` (**1.00x**) | `692.58 us` (**3.96x faster**) | `845.07 us` (**3.24x faster**) | `18.42 ms` (*6.72x slower*)   | `1.34 ms` (**2.04x faster**)   | `1.90 ms` (**1.44x faster**)   | `3.13 ms` (*1.14x slower*)    |

### No Match

None of the haystacks partially or fully match the needle, meaning none of the characters in the needle are present in the haystack.

```rust
needle: "deadbeef"
match_percentage: 0.0
partial_match_percentage: 0.0
median_length: varies
std_dev_length: median_length / 4
haystack_size: 100000
```

|           | `Nucleo`                | `Frizbee`                         | `Iter`                           | `All Scores`                    | `1 Typo`                         | `2 Typos`                        | `3 Typos`                         |
|:----------|:------------------------|:----------------------------------|:---------------------------------|:--------------------------------|:---------------------------------|:---------------------------------|:--------------------------------- |
| **`16`**  | `1.55 ms` (**1.00x**) | `140.28 us` (**11.02x faster**) | `262.80 us` (**5.88x faster**) | `7.57 ms` (*4.89x slower*)    | `223.39 us` (**6.92x faster**) | `300.68 us` (**5.14x faster**) | `923.12 us` (**1.67x faster**)  |
| **`32`**  | `1.59 ms` (**1.00x**) | `137.86 us` (**11.50x faster**) | `262.32 us` (**6.04x faster**) | `7.61 ms` (*4.80x slower*)    | `221.39 us` (**7.16x faster**) | `297.90 us` (**5.32x faster**) | `961.37 us` (**1.65x faster**)  |
| **`64`**  | `1.85 ms` (**1.00x**) | `416.57 us` (**4.43x faster**)  | `584.90 us` (**3.16x faster**) | `11.02 ms` (*5.97x slower*)   | `576.75 us` (**3.20x faster**) | `682.34 us` (**2.70x faster**) | `1.26 ms` (**1.47x faster**)    |
| **`128`** | `2.39 ms` (**1.00x**) | `462.27 us` (**5.17x faster**)  | `668.34 us` (**3.57x faster**) | `18.14 ms` (*7.60x slower*)   | `716.41 us` (**3.33x faster**) | `897.33 us` (**2.66x faster**) | `1.58 ms` (**1.51x faster**)    |

### Copy

Zero-sized needle, all haystacks match the needle.

```rust
needle: ""
match_percentage: 1.0
partial_match_percentage: 0.0
median_length: varies
std_dev_length: median_length / 4
haystack_size: 100000
```

|           | `Nucleo`                 | `Frizbee`                       | `Iter`                            | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                        |
|:----------|:-------------------------|:--------------------------------|:----------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `20.49 us` (**1.00x**) | `15.98 us` (**1.28x faster**) | `228.33 us` (*11.14x slower*)   | `15.94 us` (**1.29x faster**) | `15.84 us` (**1.29x faster**) | `15.96 us` (**1.28x faster**) | `15.84 us` (**1.29x faster**)  |
| **`32`**  | `20.32 us` (**1.00x**) | `15.85 us` (**1.28x faster**) | `227.89 us` (*11.22x slower*)   | `15.92 us` (**1.28x faster**) | `15.92 us` (**1.28x faster**) | `16.03 us` (**1.27x faster**) | `15.84 us` (**1.28x faster**)  |
| **`64`**  | `20.39 us` (**1.00x**) | `15.97 us` (**1.28x faster**) | `224.55 us` (*11.01x slower*)   | `15.87 us` (**1.28x faster**) | `15.88 us` (**1.28x faster**) | `15.85 us` (**1.29x faster**) | `15.88 us` (**1.28x faster**)  |
| **`128`** | `20.42 us` (**1.00x**) | `15.89 us` (**1.28x faster**) | `220.34 us` (*10.79x slower*)   | `15.90 us` (**1.28x faster**) | `15.94 us` (**1.28x faster**) | `15.82 us` (**1.29x faster**) | `15.97 us` (**1.28x faster**)  |

### Sort

|              | `std`                     | `radix`                            |
|:-------------|:--------------------------|:---------------------------------- |
| **`10`**     | `10.40 ns` (**1.00x**)  | `149.20 ns` (*14.34x slower*)    |
| **`100`**    | `307.41 ns` (**1.00x**) | `253.71 ns` (**1.21x faster**)   |
| **`1000`**   | `4.23 us` (**1.00x**)   | `1.40 us` (**3.02x faster**)     |
| **`10000`**  | `53.31 us` (**1.00x**)  | `12.29 us` (**4.34x faster**)    |
| **`100000`** | `1.01 ms` (**1.00x**)   | `131.88 us` (**7.66x faster**)   |
