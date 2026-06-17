# Benchmarks

## Table of Contents

- [Environment](#environment)
- [Overview](#overview)
- [Benchmark Results](#benchmark-results)
    - [Chromium](#chromium)
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
| **`Sequential`**    | `91.64 ms` (**1.00x**) | `120.67 ms` (*1.32x slower*)   | `22.40 ms` (**4.09x faster**) | `88.39 ms` (**1.04x faster**) | `63.50 ms` (**1.44x faster**) | `102.32 ms` (*1.12x slower*)   | `133.37 ms` (*1.46x slower*)    |
| **`Parallel (x8)`** | `36.41 ms` (**1.00x**) | `16.23 ms` (**2.24x faster**)  | `3.51 ms` (**10.38x faster**) | `14.24 ms` (**2.56x faster**) | `9.65 ms` (**3.77x faster**)  | `16.02 ms` (**2.27x faster**)  | `20.47 ms` (**1.78x faster**)   |

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
| **`16`**  | `3.29 ms` (**1.00x**) | `842.31 us` (**3.90x faster**) | `8.20 ms` (*2.49x slower*)    | `1.36 ms` (**2.42x faster**) | `1.80 ms` (**1.83x faster**) | `2.97 ms` (**1.11x faster**)  |
| **`32`**  | `3.95 ms` (**1.00x**) | `842.15 us` (**4.69x faster**) | `8.18 ms` (*2.07x slower*)    | `1.36 ms` (**2.91x faster**) | `1.81 ms` (**2.18x faster**) | `3.00 ms` (**1.32x faster**)  |
| **`64`**  | `5.56 ms` (**1.00x**) | `1.23 ms` (**4.52x faster**)   | `11.74 ms` (*2.11x slower*)   | `1.81 ms` (**3.06x faster**) | `2.35 ms` (**2.37x faster**) | `3.76 ms` (**1.48x faster**)  |
| **`128`** | `9.10 ms` (**1.00x**) | `1.79 ms` (**5.09x faster**)   | `19.01 ms` (*2.09x slower*)   | `2.72 ms` (**3.35x faster**) | `3.42 ms` (**2.66x faster**) | `5.13 ms` (**1.77x faster**)  |

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
| **`16`**  | `22.67 ms` (**1.00x**)  | `8.53 ms` (**2.66x faster**)  | `8.00 ms` (**2.83x faster**)  | `11.99 ms` (**1.89x faster**) | `12.57 ms` (**1.80x faster**) | `14.61 ms` (**1.55x faster**)  |
| **`32`**  | `38.52 ms` (**1.00x**)  | `8.47 ms` (**4.55x faster**)  | `8.04 ms` (**4.79x faster**)  | `11.94 ms` (**3.22x faster**) | `12.54 ms` (**3.07x faster**) | `15.00 ms` (**2.57x faster**)  |
| **`64`**  | `63.92 ms` (**1.00x**)  | `10.86 ms` (**5.89x faster**) | `11.70 ms` (**5.46x faster**) | `14.20 ms` (**4.50x faster**) | `14.88 ms` (**4.29x faster**) | `19.13 ms` (**3.34x faster**)  |
| **`128`** | `122.37 ms` (**1.00x**) | `18.65 ms` (**6.56x faster**) | `19.05 ms` (**6.42x faster**) | `21.62 ms` (**5.66x faster**) | `23.00 ms` (**5.32x faster**) | `27.60 ms` (**4.43x faster**)  |

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
| **`16`**  | `1.90 ms` (**1.00x**) | `277.38 us` (**6.85x faster**) | `8.19 ms` (*4.31x slower*)    | `659.50 us` (**2.88x faster**) | `1.01 ms` (**1.88x faster**) | `1.89 ms` (**1.00x faster**)  |
| **`32`**  | `1.91 ms` (**1.00x**) | `271.66 us` (**7.02x faster**) | `8.27 ms` (*4.34x slower*)    | `658.10 us` (**2.90x faster**) | `1.01 ms` (**1.89x faster**) | `2.02 ms` (**1.06x slower**)  |
| **`64`**  | `2.23 ms` (**1.00x**) | `573.95 us` (**3.89x faster**) | `11.65 ms` (*5.22x slower*)   | `999.69 us` (**2.23x faster**) | `1.43 ms` (**1.56x faster**) | `2.53 ms` (*1.13x slower*)    |
| **`128`** | `2.74 ms` (**1.00x**) | `750.50 us` (**3.64x faster**) | `19.18 ms` (*7.01x slower*)   | `1.42 ms` (**1.92x faster**)   | `2.01 ms` (**1.36x faster**) | `3.22 ms` (*1.18x slower*)    |

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
| **`16`**  | `1.55 ms` (**1.00x**) | `149.83 us` (**10.35x faster**) | `7.95 ms` (*5.12x slower*)    | `226.60 us` (**6.85x faster**) | `314.94 us` (**4.93x faster**) | `979.11 us` (**1.58x faster**)  |
| **`32`**  | `1.61 ms` (**1.00x**) | `148.67 us` (**10.83x faster**) | `7.98 ms` (*4.95x slower*)    | `221.72 us` (**7.26x faster**) | `316.16 us` (**5.09x faster**) | `977.19 us` (**1.65x faster**)  |
| **`64`**  | `1.89 ms` (**1.00x**) | `415.95 us` (**4.55x faster**)  | `11.47 ms` (*6.06x slower*)   | `538.43 us` (**3.51x faster**) | `694.66 us` (**2.72x faster**) | `1.30 ms` (**1.45x faster**)    |
| **`128`** | `2.32 ms` (**1.00x**) | `521.52 us` (**4.44x faster**)  | `18.55 ms` (*8.01x slower*)   | `706.80 us` (**3.28x faster**) | `902.33 us` (**2.57x faster**) | `1.59 ms` (**1.46x faster**)    |

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

|           | `Nucleo`                 | `Frizbee`                        | `All Scores`                     | `1 Typo`                         | `2 Typos`                        | `3 Typos`                         |
|:----------|:-------------------------|:---------------------------------|:---------------------------------|:---------------------------------|:---------------------------------|:--------------------------------- |
| **`16`**  | `20.47 us` (**1.00x**) | `144.95 us` (*7.08x slower*)   | `145.66 us` (*7.11x slower*)   | `145.39 us` (*7.10x slower*)   | `145.10 us` (*7.09x slower*)   | `145.77 us` (*7.12x slower*)    |
| **`32`**  | `20.44 us` (**1.00x**) | `145.63 us` (*7.13x slower*)   | `145.01 us` (*7.10x slower*)   | `145.66 us` (*7.13x slower*)   | `145.92 us` (*7.14x slower*)   | `145.29 us` (*7.11x slower*)    |
| **`64`**  | `20.53 us` (**1.00x**) | `142.25 us` (*6.93x slower*)   | `141.90 us` (*6.91x slower*)   | `141.92 us` (*6.91x slower*)   | `142.90 us` (*6.96x slower*)   | `142.47 us` (*6.94x slower*)    |
| **`128`** | `20.51 us` (**1.00x**) | `143.34 us` (*6.99x slower*)   | `143.49 us` (*7.00x slower*)   | `143.24 us` (*6.98x slower*)   | `144.12 us` (*7.03x slower*)   | `137.04 us` (*6.68x slower*)    |

### Sort

|              | `std`                     | `radix`                            |
|:-------------|:--------------------------|:---------------------------------- |
| **`10`**     | `11.07 ns` (**1.00x**)  | `138.50 ns` (*12.51x slower*)    |
| **`100`**    | `303.45 ns` (**1.00x**) | `258.31 ns` (**1.17x faster**)   |
| **`1000`**   | `4.44 us` (**1.00x**)   | `1.39 us` (**3.19x faster**)     |
| **`10000`**  | `54.46 us` (**1.00x**)  | `12.59 us` (**4.32x faster**)    |
| **`100000`** | `1.00 ms` (**1.00x**)   | `134.12 us` (**7.48x faster**)   |

