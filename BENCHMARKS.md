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
cargo 1.98.0-nightly (0b1123a48 2026-06-01)
release: 1.98.0-nightly
commit-hash: 0b1123a48825309b697312b44fdb64b3df00c958
commit-date: 2026-06-01
host: x86_64-unknown-linux-gnu
libgit2: 1.9.4 (sys:0.21.0 vendored)
libcurl: 8.20.0-DEV (sys:0.4.88+curl-8.20.0 vendored ssl:OpenSSL/3.6.2)
ssl: OpenSSL 3.6.2 7 Apr 2026
os: NixOS 26.11.0 (zokor) [64-bit]
```

## Overview

In each of the benchmarks, the median length of the haystacks is varied from 8 to 128.

- **Frizbee**: Uses the `Options::default()`, where we perform the fastest prefilter since no typos are allowed
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

|                     | `Nucleo`                 | `FZF`                            | `Frizbee`                       | `All Scores`                    | `1 Typo`                        | `2 Typos`                        | `3 Typos`                         |
|:--------------------|:-------------------------|:---------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:---------------------------------|:--------------------------------- |
| **`Sequential`**    | `90.61 ms` (**1.00x**) | `120.66 ms` (*1.33x slower*)   | `23.27 ms` (**3.89x faster**) | `90.44 ms` (**1.00x faster**) | `65.03 ms` (**1.39x faster**) | `105.46 ms` (*1.16x slower*)   | `136.86 ms` (*1.51x slower*)    |
| **`Parallel (x8)`** | `36.42 ms` (**1.00x**) | `16.23 ms` (**2.24x faster**)  | `3.57 ms` (**10.21x faster**) | `14.88 ms` (**2.45x faster**) | `9.94 ms` (**3.67x faster**)  | `16.44 ms` (**2.21x faster**)  | `20.91 ms` (**1.74x faster**)   |

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

|                     | `Nucleo`                  | `FZF`                            | `Frizbee`                         | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                        |
|:--------------------|:--------------------------|:---------------------------------|:----------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`Sequential`**    | `122.34 ms` (**1.00x**) | `165.77 ms` (*1.36x slower*)   | `2.62 ms` (**46.72x faster**)   | `15.60 ms` (**7.84x faster**) | `13.07 ms` (**9.36x faster**) | `16.28 ms` (**7.51x faster**) | `16.36 ms` (**7.48x faster**)  |
| **`Parallel (x8)`** | `9.49 ms` (**1.00x**)   | `22.01 ms` (*2.32x slower*)    | `495.83 us` (**19.13x faster**) | `2.81 ms` (**3.38x faster**)  | `2.18 ms` (**4.36x faster**)  | `2.92 ms` (**3.25x faster**)  | `2.90 ms` (**3.28x faster**)   |

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

|                     | `Nucleo`                  | `FZF`                            | `Frizbee`                         | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                        |
|:--------------------|:--------------------------|:---------------------------------|:----------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`Sequential`**    | `108.82 ms` (**1.00x**) | `114.37 ms` (**1.05x slower**) | `2.66 ms` (**40.93x faster**)   | `16.18 ms` (**6.73x faster**) | `10.92 ms` (**9.96x faster**) | `16.84 ms` (**6.46x faster**) | `16.58 ms` (**6.56x faster**)  |
| **`Parallel (x8)`** | `8.58 ms` (**1.00x**)   | `15.45 ms` (*1.80x slower*)    | `497.30 us` (**17.25x faster**) | `2.73 ms` (**3.15x faster**)  | `1.82 ms` (**4.71x faster**)  | `2.87 ms` (**2.99x faster**)  | `2.87 ms` (**2.99x faster**)   |

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

|           | `Nucleo`                | `Frizbee`                        | `All Scores`                    | `1 Typo`                       | `2 Typos`                      | `3 Typos`                       |
|:----------|:------------------------|:---------------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:------------------------------- |
| **`16`**  | `3.29 ms` (**1.00x**) | `896.09 us` (**3.67x faster**) | `8.24 ms` (*2.51x slower*)    | `1.43 ms` (**2.29x faster**) | `1.85 ms` (**1.78x faster**) | `3.12 ms` (**1.05x faster**)  |
| **`32`**  | `3.94 ms` (**1.00x**) | `897.13 us` (**4.40x faster**) | `8.16 ms` (*2.07x slower*)    | `1.42 ms` (**2.78x faster**) | `1.83 ms` (**2.15x faster**) | `3.27 ms` (**1.21x faster**)  |
| **`64`**  | `5.44 ms` (**1.00x**) | `1.26 ms` (**4.33x faster**)   | `11.62 ms` (*2.13x slower*)   | `1.90 ms` (**2.86x faster**) | `2.38 ms` (**2.29x faster**) | `3.84 ms` (**1.42x faster**)  |
| **`128`** | `8.82 ms` (**1.00x**) | `1.83 ms` (**4.83x faster**)   | `19.32 ms` (*2.19x slower*)   | `2.75 ms` (**3.21x faster**) | `3.49 ms` (**2.53x faster**) | `5.32 ms` (**1.66x faster**)  |

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

|           | `Nucleo`                  | `Frizbee`                       | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                        |
|:----------|:--------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `22.49 ms` (**1.00x**)  | `8.69 ms` (**2.59x faster**)  | `8.02 ms` (**2.80x faster**)  | `11.98 ms` (**1.88x faster**) | `12.54 ms` (**1.79x faster**) | `14.72 ms` (**1.53x faster**)  |
| **`32`**  | `38.21 ms` (**1.00x**)  | `8.62 ms` (**4.43x faster**)  | `7.97 ms` (**4.80x faster**)  | `12.04 ms` (**3.17x faster**) | `12.36 ms` (**3.09x faster**) | `15.36 ms` (**2.49x faster**)  |
| **`64`**  | `62.42 ms` (**1.00x**)  | `10.80 ms` (**5.78x faster**) | `11.61 ms` (**5.38x faster**) | `14.27 ms` (**4.37x faster**) | `14.84 ms` (**4.21x faster**) | `18.87 ms` (**3.31x faster**)  |
| **`128`** | `117.84 ms` (**1.00x**) | `18.58 ms` (**6.34x faster**) | `19.59 ms` (**6.01x faster**) | `21.94 ms` (**5.37x faster**) | `22.43 ms` (**5.25x faster**) | `27.08 ms` (**4.35x faster**)  |

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

|           | `Nucleo`                | `Frizbee`                        | `All Scores`                    | `1 Typo`                         | `2 Typos`                      | `3 Typos`                       |
|:----------|:------------------------|:---------------------------------|:--------------------------------|:---------------------------------|:-------------------------------|:------------------------------- |
| **`16`**  | `1.98 ms` (**1.00x**) | `282.59 us` (**7.00x faster**) | `8.16 ms` (*4.12x slower*)    | `694.69 us` (**2.85x faster**) | `1.02 ms` (**1.93x faster**) | `2.05 ms` (**1.04x slower**)  |
| **`32`**  | `1.96 ms` (**1.00x**) | `281.63 us` (**6.94x faster**) | `8.21 ms` (*4.20x slower*)    | `689.85 us` (**2.83x faster**) | `1.02 ms` (**1.92x faster**) | `2.02 ms` (**1.03x slower**)  |
| **`64`**  | `2.26 ms` (**1.00x**) | `598.24 us` (**3.78x faster**) | `11.68 ms` (*5.16x slower*)   | `1.06 ms` (**2.13x faster**)   | `1.41 ms` (**1.60x faster**) | `2.53 ms` (*1.12x slower*)    |
| **`128`** | `2.79 ms` (**1.00x**) | `738.58 us` (**3.77x faster**) | `18.97 ms` (*6.81x slower*)   | `1.44 ms` (**1.93x faster**)   | `2.07 ms` (**1.35x faster**) | `3.23 ms` (*1.16x slower*)    |

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

|           | `Nucleo`                | `Frizbee`                         | `All Scores`                    | `1 Typo`                         | `2 Typos`                        | `3 Typos`                         |
|:----------|:------------------------|:----------------------------------|:--------------------------------|:---------------------------------|:---------------------------------|:--------------------------------- |
| **`16`**  | `1.58 ms` (**1.00x**) | `156.37 us` (**10.13x faster**) | `7.98 ms` (*5.04x slower*)    | `245.24 us` (**6.46x faster**) | `313.66 us` (**5.05x faster**) | `916.74 us` (**1.73x faster**)  |
| **`32`**  | `1.64 ms` (**1.00x**) | `151.64 us` (**10.81x faster**) | `7.97 ms` (*4.87x slower*)    | `245.55 us` (**6.67x faster**) | `313.27 us` (**5.23x faster**) | `916.65 us` (**1.79x faster**)  |
| **`64`**  | `1.94 ms` (**1.00x**) | `435.20 us` (**4.45x faster**)  | `11.51 ms` (*5.94x slower*)   | `589.21 us` (**3.29x faster**) | `683.72 us` (**2.83x faster**) | `1.32 ms` (**1.47x faster**)    |
| **`128`** | `2.37 ms` (**1.00x**) | `517.15 us` (**4.59x faster**)  | `18.52 ms` (*7.81x slower*)   | `746.81 us` (**3.18x faster**) | `1.03 ms` (**2.30x faster**)   | `1.66 ms` (**1.43x faster**)    |

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

|           | `Nucleo`                 | `Frizbee`                       | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                        |
|:----------|:-------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `20.76 us` (**1.00x**) | `41.94 us` (*2.02x slower*)   | `43.13 us` (*2.08x slower*)   | `33.30 us` (*1.60x slower*)   | `43.87 us` (*2.11x slower*)   | `46.86 us` (*2.26x slower*)    |
| **`32`**  | `20.72 us` (**1.00x**) | `46.56 us` (*2.25x slower*)   | `47.14 us` (*2.27x slower*)   | `46.75 us` (*2.26x slower*)   | `42.49 us` (*2.05x slower*)   | `46.22 us` (*2.23x slower*)    |
| **`64`**  | `20.67 us` (**1.00x**) | `46.43 us` (*2.25x slower*)   | `38.82 us` (*1.88x slower*)   | `42.21 us` (*2.04x slower*)   | `44.76 us` (*2.17x slower*)   | `46.09 us` (*2.23x slower*)    |
| **`128`** | `20.75 us` (**1.00x**) | `39.59 us` (*1.91x slower*)   | `42.70 us` (*2.06x slower*)   | `40.85 us` (*1.97x slower*)   | `42.50 us` (*2.05x slower*)   | `42.27 us` (*2.04x slower*)    |

### Sort

|              | `std`                     | `radix`                            |
|:-------------|:--------------------------|:---------------------------------- |
| **`10`**     | `9.14 ns` (**1.00x**)   | `161.87 ns` (*17.72x slower*)    |
| **`100`**    | `316.03 ns` (**1.00x**) | `252.43 ns` (**1.25x faster**)   |
| **`1000`**   | `4.18 us` (**1.00x**)   | `1.40 us` (**2.98x faster**)     |
| **`10000`**  | `53.13 us` (**1.00x**)  | `12.51 us` (**4.25x faster**)    |
| **`100000`** | `984.60 us` (**1.00x**) | `131.40 us` (**7.49x faster**)   |

