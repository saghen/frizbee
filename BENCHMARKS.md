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
cargo 1.95.0-nightly (fe2f314ae 2026-01-30)
release: 1.95.0-nightly
commit-hash: fe2f314aef06e688a9517da1ac0577bb1854d01f
commit-date: 2026-01-30
host: x86_64-unknown-linux-gnu
libgit2: 1.9.2 (sys:0.20.3 vendored)
libcurl: 8.15.0-DEV (sys:0.4.83+curl-8.15.0 vendored ssl:OpenSSL/3.5.4)
ssl: OpenSSL 3.5.4 30 Sep 2025
os: NixOS 26.5.0 [64-bit]
```

## Overview

In each of the benchmarks, the median length of the haystacks is varied from 8 to 128.

- **Frizbee**: Uses the `Options::default()`, where we perform the fastest prefilter since no typos are allowed
- **Parallel (x8)**: Same as $BENCH, but uses 8 threads to perform the matching in parallel
- **All Scores**: Set via `max_typos: None`, gets the scores for all of the items without any filtering
- **1/2 Typos**: Set via `max_typos: Some(1 || 2)`, performs a slower, but still effective prefilter since a small number of typos are allowed
- **3 Typos**: Set via `max_typos: Some(3)`, skips prefiltering since in non-syntheic data (Chromium), the prefilter has to pass over the data up to 4 times and most items will not be filtered out
- **Nucleo**: Runs with normalization disabled, case insentivity enabled and fuzzy matching enabled

NOTE: The nucleo parallel benchmark is not included since I haven't discovered a way to ensure the matcher has finished running.

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

|          | `Nucleo`                 | `Frizbee`                       | `Parallel (x8)`                 | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                         |
|:---------|:-------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------- |
| **`67`** | `88.99 ms` (**1.00x**) | `22.26 ms` (**4.00x faster**) | `3.67 ms` (**24.25x faster**) | `88.52 ms` (**1.01x faster**) | `56.55 ms` (**1.57x faster**) | `92.16 ms` (**1.04x slower**) | `142.75 ms` (*1.60x slower*)    |

#### FZF

Version: 0.73.1

Single threaded: `119.01ms` (*5.3x slower*)

Multi threaded (8 threads): `15.95ms` (*4.3x slower* vs parallel)

`fzf --filter linux --tiebreak index --bench 10s --threads 1 < benches/match_list/data.txt`

We added `--tiebreak index` to only sort the results by their matching scores.

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

|           | `Nucleo`                | `Frizbee`                        | `Parallel (x8)`                   | `All Scores`                    | `1 Typo`                       | `2 Typos`                      | `3 Typos`                       |
|:----------|:------------------------|:---------------------------------|:----------------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:------------------------------- |
| **`16`**  | `3.09 ms` (**1.00x**) | `850.82 us` (**3.63x faster**) | `246.40 us` (**12.53x faster**) | `8.14 ms` (*2.64x slower*)    | `1.29 ms` (**2.40x faster**) | `1.66 ms` (**1.86x faster**) | `3.32 ms` (**1.08x slower**)  |
| **`32`**  | `3.87 ms` (**1.00x**) | `854.17 us` (**4.53x faster**) | `249.96 us` (**15.49x faster**) | `8.21 ms` (*2.12x slower*)    | `1.31 ms` (**2.95x faster**) | `1.66 ms` (**2.33x faster**) | `3.37 ms` (**1.15x faster**)  |
| **`64`**  | `5.37 ms` (**1.00x**) | `1.25 ms` (**4.30x faster**)   | `307.59 us` (**17.44x faster**) | `12.34 ms` (*2.30x slower*)   | `1.82 ms` (**2.95x faster**) | `2.19 ms` (**2.45x faster**) | `4.30 ms` (**1.25x faster**)  |
| **`128`** | `9.17 ms` (**1.00x**) | `1.77 ms` (**5.18x faster**)   | `376.44 us` (**24.36x faster**) | `20.47 ms` (*2.23x slower*)   | `2.63 ms` (**3.49x faster**) | `3.26 ms` (**2.82x faster**) | `5.89 ms` (**1.56x faster**)  |

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

|           | `Nucleo`                  | `Frizbee`                       | `Parallel (x8)`                 | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                        |
|:----------|:--------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `23.16 ms` (**1.00x**)  | `8.85 ms` (**2.62x faster**)  | `1.53 ms` (**15.09x faster**) | `8.23 ms` (**2.81x faster**)  | `11.52 ms` (**2.01x faster**) | `12.15 ms` (**1.91x faster**) | `15.30 ms` (**1.51x faster**)  |
| **`32`**  | `38.77 ms` (**1.00x**)  | `8.67 ms` (**4.47x faster**)  | `1.59 ms` (**24.31x faster**) | `8.26 ms` (**4.69x faster**)  | `11.45 ms` (**3.39x faster**) | `12.20 ms` (**3.18x faster**) | `15.81 ms` (**2.45x faster**)  |
| **`64`**  | `63.40 ms` (**1.00x**)  | `11.60 ms` (**5.47x faster**) | `1.93 ms` (**32.91x faster**) | `12.39 ms` (**5.12x faster**) | `14.25 ms` (**4.45x faster**) | `15.24 ms` (**4.16x faster**) | `20.51 ms` (**3.09x faster**)  |
| **`128`** | `118.67 ms` (**1.00x**) | `20.17 ms` (**5.88x faster**) | `2.96 ms` (**40.05x faster**) | `21.56 ms` (**5.50x faster**) | `22.90 ms` (**5.18x faster**) | `23.70 ms` (**5.01x faster**) | `29.26 ms` (**4.06x faster**)  |

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

|           | `Nucleo`                | `Frizbee`                        | `Parallel (x8)`                   | `All Scores`                    | `1 Typo`                         | `2 Typos`                        | `3 Typos`                       |
|:----------|:------------------------|:---------------------------------|:----------------------------------|:--------------------------------|:---------------------------------|:---------------------------------|:------------------------------- |
| **`16`**  | `1.81 ms` (**1.00x**) | `265.77 us` (**6.81x faster**) | `111.00 us` (**16.29x faster**) | `8.34 ms` (*4.61x slower*)    | `616.14 us` (**2.94x faster**) | `864.91 us` (**2.09x faster**) | `2.30 ms` (*1.27x slower*)    |
| **`32`**  | `1.82 ms` (**1.00x**) | `259.96 us` (**7.01x faster**) | `109.27 us` (**16.67x faster**) | `8.48 ms` (*4.66x slower*)    | `609.62 us` (**2.99x faster**) | `868.56 us` (**2.10x faster**) | `2.28 ms` (*1.25x slower*)    |
| **`64`**  | `2.17 ms` (**1.00x**) | `544.84 us` (**3.99x faster**) | `144.70 us` (**15.01x faster**) | `11.95 ms` (*5.50x slower*)   | `997.14 us` (**2.18x faster**) | `1.27 ms` (**1.72x faster**)   | `2.92 ms` (*1.34x slower*)    |
| **`128`** | `2.81 ms` (**1.00x**) | `711.33 us` (**3.96x faster**) | `167.11 us` (**16.84x faster**) | `20.11 ms` (*7.15x slower*)   | `1.34 ms` (**2.09x faster**)   | `1.79 ms` (**1.57x faster**)   | `4.04 ms` (*1.44x slower*)    |

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

|           | `Nucleo`                | `Frizbee`                         | `Parallel (x8)`                   | `All Scores`                    | `1 Typo`                         | `2 Typos`                        | `3 Typos`                       |
|:----------|:------------------------|:----------------------------------|:----------------------------------|:--------------------------------|:---------------------------------|:---------------------------------|:------------------------------- |
| **`16`**  | `1.51 ms` (**1.00x**) | `140.23 us` (**10.75x faster**) | `91.93 us` (**16.39x faster**)  | `8.12 ms` (*5.39x slower*)    | `186.38 us` (**8.09x faster**) | `208.82 us` (**7.22x faster**) | `1.15 ms` (**1.31x faster**)  |
| **`32`**  | `1.57 ms` (**1.00x**) | `138.81 us` (**11.28x faster**) | `92.51 us` (**16.92x faster**)  | `8.20 ms` (*5.24x slower*)    | `181.83 us` (**8.61x faster**) | `211.93 us` (**7.38x faster**) | `1.16 ms` (**1.35x faster**)  |
| **`64`**  | `1.82 ms` (**1.00x**) | `396.95 us` (**4.58x faster**)  | `123.66 us` (**14.72x faster**) | `11.67 ms` (*6.41x slower*)   | `533.36 us` (**3.41x faster**) | `586.63 us` (**3.10x faster**) | `1.70 ms` (**1.07x faster**)  |
| **`128`** | `2.33 ms` (**1.00x**) | `463.80 us` (**5.03x faster**)  | `136.77 us` (**17.07x faster**) | `19.00 ms` (*8.14x slower*)   | `615.34 us` (**3.79x faster**) | `663.13 us` (**3.52x faster**) | `2.30 ms` (**1.01x faster**)  |

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

|           | `Nucleo`                 | `Frizbee`                      | `Parallel (x8)`                 | `All Scores`                   | `1 Typo`                       | `2 Typos`                      | `3 Typos`                       |
|:----------|:-------------------------|:-------------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:-------------------------------|:------------------------------- |
| **`16`**  | `20.75 us` (**1.00x**) | `2.92 us` (**7.11x faster**) | `14.93 us` (**1.39x faster**) | `2.89 us` (**7.18x faster**) | `2.88 us` (**7.20x faster**) | `2.94 us` (**7.07x faster**) | `2.89 us` (**7.19x faster**)  |
| **`32`**  | `20.67 us` (**1.00x**) | `2.95 us` (**7.01x faster**) | `14.97 us` (**1.38x faster**) | `2.97 us` (**6.95x faster**) | `2.92 us` (**7.07x faster**) | `2.93 us` (**7.06x faster**) | `2.94 us` (**7.04x faster**)  |
| **`64`**  | `20.96 us` (**1.00x**) | `2.95 us` (**7.12x faster**) | `14.94 us` (**1.40x faster**) | `2.91 us` (**7.20x faster**) | `2.97 us` (**7.07x faster**) | `2.92 us` (**7.18x faster**) | `2.93 us` (**7.15x faster**)  |
| **`128`** | `20.93 us` (**1.00x**) | `3.49 us` (**6.00x faster**) | `14.92 us` (**1.40x faster**) | `3.51 us` (**5.97x faster**) | `3.48 us` (**6.01x faster**) | `3.49 us` (**5.99x faster**) | `3.51 us` (**5.97x faster**)  |

### Sort

|              | `std`                     | `radix`                            |
|:-------------|:--------------------------|:---------------------------------- |
| **`10`**     | `12.34 ns` (**1.00x**)  | `173.93 ns` (*14.09x slower*)    |
| **`100`**    | `293.93 ns` (**1.00x**) | `259.59 ns` (**1.13x faster**)   |
| **`1000`**   | `4.28 us` (**1.00x**)   | `1.43 us` (**3.00x faster**)     |
| **`10000`**  | `53.27 us` (**1.00x**)  | `12.50 us` (**4.26x faster**)    |
| **`100000`** | `998.22 us` (**1.00x**) | `132.68 us` (**7.52x faster**)   |

