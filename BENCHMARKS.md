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
| **`Sequential`**    | `90.16 ms` (**1.00x**) | `120.67 ms` (*1.34x slower*)   | `22.82 ms` (**3.95x faster**) | `90.83 ms` (**1.01x slower**) | `64.08 ms` (**1.41x faster**) | `103.94 ms` (*1.15x slower*)   | `135.70 ms` (*1.51x slower*)    |
| **`Parallel (x8)`** | `36.05 ms` (**1.00x**) | `16.23 ms` (**2.22x faster**)  | `3.45 ms` (**10.44x faster**) | `14.37 ms` (**2.51x faster**) | `9.67 ms` (**3.73x faster**)  | `15.98 ms` (**2.26x faster**)  | `20.45 ms` (**1.76x faster**)   |

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

|                     | `Nucleo`                  | `FZF`                            | `Frizbee`                         | `All Scores`                     | `1 Typo`                         | `2 Typos`                       | `3 Typos`                        |
|:--------------------|:--------------------------|:---------------------------------|:----------------------------------|:---------------------------------|:---------------------------------|:--------------------------------|:-------------------------------- |
| **`Sequential`**    | `123.27 ms` (**1.00x**) | `165.79 ms` (*1.34x slower*)   | `2.07 ms` (**59.59x faster**)   | `11.47 ms` (**10.75x faster**) | `10.76 ms` (**11.46x faster**) | `17.68 ms` (**6.97x faster**) | `19.92 ms` (**6.19x faster**)  |
| **`Parallel (x8)`** | `9.48 ms` (**1.00x**)   | `22.02 ms` (*2.32x slower*)    | `409.48 us` (**23.15x faster**) | `2.09 ms` (**4.55x faster**)   | `1.85 ms` (**5.11x faster**)   | `2.96 ms` (**3.21x faster**)  | `3.24 ms` (**2.92x faster**)   |

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
| **`Sequential`**    | `105.13 ms` (**1.00x**) | `114.38 ms` (**1.09x slower**) | `2.84 ms` (**37.00x faster**)   | `17.30 ms` (**6.08x faster**) | `10.97 ms` (**9.58x faster**) | `16.44 ms` (**6.39x faster**) | `25.26 ms` (**4.16x faster**)  |
| **`Parallel (x8)`** | `8.31 ms` (**1.00x**)   | `15.45 ms` (*1.86x slower*)    | `518.40 us` (**16.02x faster**) | `2.87 ms` (**2.89x faster**)  | `1.72 ms` (**4.82x faster**)  | `2.62 ms` (**3.17x faster**)  | `3.89 ms` (**2.14x faster**)   |

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
| **`16`**  | `3.16 ms` (**1.00x**) | `879.79 us` (**3.60x faster**) | `8.19 ms` (*2.59x slower*)    | `1.42 ms` (**2.23x faster**) | `1.84 ms` (**1.72x faster**) | `3.12 ms` (**1.01x faster**)  |
| **`32`**  | `3.92 ms` (**1.00x**) | `862.79 us` (**4.55x faster**) | `8.25 ms` (*2.10x slower*)    | `1.43 ms` (**2.75x faster**) | `1.82 ms` (**2.16x faster**) | `3.17 ms` (**1.24x faster**)  |
| **`64`**  | `5.48 ms` (**1.00x**) | `1.25 ms` (**4.39x faster**)   | `11.75 ms` (*2.15x slower*)   | `1.86 ms` (**2.94x faster**) | `2.34 ms` (**2.35x faster**) | `3.91 ms` (**1.40x faster**)  |
| **`128`** | `8.88 ms` (**1.00x**) | `1.83 ms` (**4.86x faster**)   | `19.93 ms` (*2.24x slower*)   | `2.79 ms` (**3.18x faster**) | `3.43 ms` (**2.59x faster**) | `5.31 ms` (**1.67x faster**)  |

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
| **`16`**  | `22.16 ms` (**1.00x**)  | `8.71 ms` (**2.54x faster**)  | `8.00 ms` (**2.77x faster**)  | `11.95 ms` (**1.86x faster**) | `12.34 ms` (**1.80x faster**) | `14.62 ms` (**1.52x faster**)  |
| **`32`**  | `37.43 ms` (**1.00x**)  | `8.58 ms` (**4.36x faster**)  | `8.13 ms` (**4.61x faster**)  | `11.92 ms` (**3.14x faster**) | `12.49 ms` (**3.00x faster**) | `15.27 ms` (**2.45x faster**)  |
| **`64`**  | `61.96 ms` (**1.00x**)  | `10.81 ms` (**5.73x faster**) | `11.73 ms` (**5.28x faster**) | `14.45 ms` (**4.29x faster**) | `15.14 ms` (**4.09x faster**) | `18.98 ms` (**3.26x faster**)  |
| **`128`** | `116.85 ms` (**1.00x**) | `18.93 ms` (**6.17x faster**) | `19.03 ms` (**6.14x faster**) | `21.88 ms` (**5.34x faster**) | `23.36 ms` (**5.00x faster**) | `27.42 ms` (**4.26x faster**)  |

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

|           | `Nucleo`                | `Frizbee`                        | `All Scores`                    | `1 Typo`                         | `2 Typos`                        | `3 Typos`                       |
|:----------|:------------------------|:---------------------------------|:--------------------------------|:---------------------------------|:---------------------------------|:------------------------------- |
| **`16`**  | `1.87 ms` (**1.00x**) | `272.62 us` (**6.84x faster**) | `8.23 ms` (*4.42x slower*)    | `667.21 us` (**2.80x faster**) | `997.97 us` (**1.87x faster**) | `1.98 ms` (**1.06x slower**)  |
| **`32`**  | `1.93 ms` (**1.00x**) | `273.80 us` (**7.06x faster**) | `8.28 ms` (*4.28x slower*)    | `675.26 us` (**2.86x faster**) | `996.38 us` (**1.94x faster**) | `2.00 ms` (**1.03x slower**)  |
| **`64`**  | `2.25 ms` (**1.00x**) | `594.07 us` (**3.78x faster**) | `11.68 ms` (*5.20x slower*)   | `1.05 ms` (**2.14x faster**)   | `1.42 ms` (**1.58x faster**)   | `2.46 ms` (**1.10x slower**)  |
| **`128`** | `2.82 ms` (**1.00x**) | `765.15 us` (**3.68x faster**) | `19.25 ms` (*6.83x slower*)   | `1.47 ms` (**1.91x faster**)   | `1.98 ms` (**1.43x faster**)   | `3.38 ms` (*1.20x slower*)    |

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
| **`16`**  | `1.57 ms` (**1.00x**) | `147.96 us` (**10.61x faster**) | `8.04 ms` (*5.12x slower*)    | `228.65 us` (**6.87x faster**) | `307.04 us` (**5.11x faster**) | `886.89 us` (**1.77x faster**)  |
| **`32`**  | `1.65 ms` (**1.00x**) | `148.01 us` (**11.12x faster**) | `8.01 ms` (*4.87x slower*)    | `230.45 us` (**7.14x faster**) | `311.29 us` (**5.29x faster**) | `1.71 ms` (**1.04x slower**)    |
| **`64`**  | `1.92 ms` (**1.00x**) | `433.26 us` (**4.43x faster**)  | `11.42 ms` (*5.95x slower*)   | `568.62 us` (**3.38x faster**) | `676.29 us` (**2.84x faster**) | `1.28 ms` (**1.50x faster**)    |
| **`128`** | `2.37 ms` (**1.00x**) | `527.64 us` (**4.49x faster**)  | `18.56 ms` (*7.83x slower*)   | `750.99 us` (**3.16x faster**) | `896.98 us` (**2.64x faster**) | `1.78 ms` (**1.33x faster**)    |

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
| **`16`**  | `20.92 us` (**1.00x**) | `176.87 us` (*8.45x slower*)   | `177.27 us` (*8.47x slower*)   | `179.78 us` (*8.59x slower*)   | `183.62 us` (*8.78x slower*)   | `181.70 us` (*8.69x slower*)    |
| **`32`**  | `20.59 us` (**1.00x**) | `178.97 us` (*8.69x slower*)   | `183.46 us` (*8.91x slower*)   | `171.15 us` (*8.31x slower*)   | `180.61 us` (*8.77x slower*)   | `182.07 us` (*8.84x slower*)    |
| **`64`**  | `20.64 us` (**1.00x**) | `179.76 us` (*8.71x slower*)   | `179.97 us` (*8.72x slower*)   | `180.30 us` (*8.74x slower*)   | `177.93 us` (*8.62x slower*)   | `174.56 us` (*8.46x slower*)    |
| **`128`** | `20.50 us` (**1.00x**) | `163.15 us` (*7.96x slower*)   | `173.30 us` (*8.45x slower*)   | `171.92 us` (*8.38x slower*)   | `172.03 us` (*8.39x slower*)   | `172.41 us` (*8.41x slower*)    |

### Sort

|              | `std`                     | `radix`                            |
|:-------------|:--------------------------|:---------------------------------- |
| **`10`**     | `11.82 ns` (**1.00x**)  | `159.88 ns` (*13.52x slower*)    |
| **`100`**    | `305.49 ns` (**1.00x**) | `252.61 ns` (**1.21x faster**)   |
| **`1000`**   | `4.26 us` (**1.00x**)   | `1.42 us` (**3.00x faster**)     |
| **`10000`**  | `54.23 us` (**1.00x**)  | `12.70 us` (**4.27x faster**)    |
| **`100000`** | `998.50 us` (**1.00x**) | `130.41 us` (**7.66x faster**)   |

---
Made with [criterion-table](https://github.com/nu11ptr/criterion-table)

