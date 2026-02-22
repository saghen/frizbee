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

|          | `Nucleo`                 | `Frizbee`                       | `Parallel (x8)`                 | `All Scores`                     | `1 Typo`                         | `2 Typos`                        | `3 Typos`                         |
|:---------|:-------------------------|:--------------------------------|:--------------------------------|:---------------------------------|:---------------------------------|:---------------------------------|:--------------------------------- |
| **`67`** | `94.12 ms` (**1.00x**) | `55.09 ms` (**1.71x faster**) | `7.95 ms` (**11.84x faster**) | `217.86 ms` (*2.31x slower*)   | `159.66 ms` (*1.70x slower*)   | `245.31 ms` (*2.61x slower*)   | `270.86 ms` (*2.88x slower*)    |


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

|           | `Nucleo`                 | `Frizbee`                      | `Parallel (x8)`                   | `All Scores`                    | `1 Typo`                       | `2 Typos`                       | `3 Typos`                        |
|:----------|:-------------------------|:-------------------------------|:----------------------------------|:--------------------------------|:-------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `3.59 ms` (**1.00x**)  | `2.17 ms` (**1.66x faster**) | `437.92 us` (**8.20x faster**)  | `10.79 ms` (*3.00x slower*)   | `3.04 ms` (**1.18x faster**) | `3.68 ms` (**1.03x slower**)  | `11.48 ms` (*3.19x slower*)    |
| **`32`**  | `4.21 ms` (**1.00x**)  | `2.66 ms` (**1.58x faster**) | `489.24 us` (**8.60x faster**)  | `15.89 ms` (*3.78x slower*)   | `3.87 ms` (**1.09x faster**) | `4.72 ms` (*1.12x slower*)    | `17.33 ms` (*4.12x slower*)    |
| **`64`**  | `5.72 ms` (**1.00x**)  | `3.59 ms` (**1.59x faster**) | `545.21 us` (**10.49x faster**) | `26.18 ms` (*4.58x slower*)   | `5.26 ms` (**1.09x faster**) | `6.80 ms` (*1.19x slower*)    | `28.16 ms` (*4.93x slower*)    |
| **`128`** | `10.23 ms` (**1.00x**) | `5.75 ms` (**1.78x faster**) | `719.20 us` (**14.22x faster**) | `46.42 ms` (*4.54x slower*)   | `8.60 ms` (**1.19x faster**) | `10.61 ms` (**1.04x slower**) | `47.51 ms` (*4.64x slower*)    |

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
| **`16`**  | `26.70 ms` (**1.00x**)  | `23.80 ms` (**1.12x faster**) | `3.18 ms` (**8.39x faster**)  | `13.46 ms` (**1.98x faster**) | `23.36 ms` (**1.14x faster**) | `23.30 ms` (**1.15x faster**) | `24.33 ms` (**1.10x faster**)  |
| **`32`**  | `41.55 ms` (**1.00x**)  | `29.31 ms` (**1.42x faster**) | `4.17 ms` (**9.96x faster**)  | `18.02 ms` (**2.31x faster**) | `29.51 ms` (**1.41x faster**) | `29.40 ms` (**1.41x faster**) | `29.70 ms` (**1.40x faster**)  |
| **`64`**  | `67.91 ms` (**1.00x**)  | `37.59 ms` (**1.81x faster**) | `5.15 ms` (**13.18x faster**) | `28.36 ms` (**2.39x faster**) | `38.78 ms` (**1.75x faster**) | `38.94 ms` (**1.74x faster**) | `37.55 ms` (**1.81x faster**)  |
| **`128`** | `123.71 ms` (**1.00x**) | `56.16 ms` (**2.20x faster**) | `7.45 ms` (**16.61x faster**) | `49.12 ms` (**2.52x faster**) | `58.27 ms` (**2.12x faster**) | `59.76 ms` (**2.07x faster**) | `58.50 ms` (**2.11x faster**)  |

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

|           | `Nucleo`                | `Frizbee`                      | `Parallel (x8)`                   | `All Scores`                     | `1 Typo`                       | `2 Typos`                      | `3 Typos`                         |
|:----------|:------------------------|:-------------------------------|:----------------------------------|:---------------------------------|:-------------------------------|:-------------------------------|:--------------------------------- |
| **`16`**  | `2.16 ms` (**1.00x**) | `1.03 ms` (**2.11x faster**) | `218.95 us` (**9.88x faster**)  | `10.54 ms` (*4.87x slower*)    | `1.75 ms` (**1.23x faster**) | `2.32 ms` (**1.07x slower**) | `10.82 ms` (*5.00x slower*)     |
| **`32`**  | `2.03 ms` (**1.00x**) | `1.05 ms` (**1.94x faster**) | `222.01 us` (**9.13x faster**)  | `15.36 ms` (*7.58x slower*)    | `2.04 ms` (**1.01x slower**) | `2.81 ms` (*1.39x slower*)   | `15.81 ms` (*7.80x slower*)     |
| **`64`**  | `2.38 ms` (**1.00x**) | `1.28 ms` (**1.85x faster**) | `243.99 us` (**9.75x faster**)  | `25.21 ms` (*10.60x slower*)   | `2.64 ms` (**1.11x slower**) | `3.69 ms` (*1.55x slower*)   | `25.77 ms` (*10.84x slower*)    |
| **`128`** | `3.11 ms` (**1.00x**) | `2.58 ms` (**1.21x faster**) | `308.13 us` (**10.09x faster**) | `45.33 ms` (*14.58x slower*)   | `4.49 ms` (*1.44x slower*)   | `5.96 ms` (*1.92x slower*)   | `45.04 ms` (*14.48x slower*)    |

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

|           | `Nucleo`                | `Frizbee`                        | `Parallel (x8)`                   | `All Scores`                     | `1 Typo`                         | `2 Typos`                        | `3 Typos`                         |
|:----------|:------------------------|:---------------------------------|:----------------------------------|:---------------------------------|:---------------------------------|:---------------------------------|:--------------------------------- |
| **`16`**  | `1.67 ms` (**1.00x**) | `558.38 us` (**2.99x faster**) | `155.06 us` (**10.76x faster**) | `7.72 ms` (*4.63x slower*)     | `645.02 us` (**2.59x faster**) | `723.88 us` (**2.30x faster**) | `7.88 ms` (*4.72x slower*)      |
| **`32`**  | `1.64 ms` (**1.00x**) | `612.91 us` (**2.67x faster**) | `161.65 us` (**10.14x faster**) | `12.53 ms` (*7.64x slower*)    | `716.19 us` (**2.29x faster**) | `873.70 us` (**1.88x faster**) | `13.04 ms` (*7.96x slower*)     |
| **`64`**  | `1.91 ms` (**1.00x**) | `849.58 us` (**2.25x faster**) | `193.63 us` (**9.88x faster**)  | `22.87 ms` (*11.95x slower*)   | `975.41 us` (**1.96x faster**) | `1.24 ms` (**1.54x faster**)   | `22.93 ms` (*11.98x slower*)    |
| **`128`** | `2.37 ms` (**1.00x**) | `1.21 ms` (**1.96x faster**)   | `231.35 us` (**10.23x faster**) | `42.56 ms` (*17.97x slower*)   | `1.46 ms` (**1.63x faster**)   | `1.98 ms` (**1.20x faster**)   | `43.01 ms` (*18.17x slower*)    |

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

|           | `Nucleo`                 | `Frizbee`                       | `Parallel (x8)`                 | `All Scores`                    | `1 Typo`                        | `2 Typos`                       | `3 Typos`                        |
|:----------|:-------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `20.98 us` (**1.00x**) | `15.53 us` (**1.35x faster**) | `13.99 us` (**1.50x faster**) | `15.52 us` (**1.35x faster**) | `15.46 us` (**1.36x faster**) | `15.57 us` (**1.35x faster**) | `15.47 us` (**1.36x faster**)  |
| **`32`**  | `20.95 us` (**1.00x**) | `15.46 us` (**1.36x faster**) | `13.32 us` (**1.57x faster**) | `15.49 us` (**1.35x faster**) | `15.67 us` (**1.34x faster**) | `15.66 us` (**1.34x faster**) | `15.55 us` (**1.35x faster**)  |
| **`64`**  | `20.90 us` (**1.00x**) | `15.52 us` (**1.35x faster**) | `14.68 us` (**1.42x faster**) | `15.47 us` (**1.35x faster**) | `15.52 us` (**1.35x faster**) | `15.45 us` (**1.35x faster**) | `15.44 us` (**1.35x faster**)  |
| **`128`** | `20.80 us` (**1.00x**) | `15.46 us` (**1.34x faster**) | `14.81 us` (**1.40x faster**) | `15.61 us` (**1.33x faster**) | `15.50 us` (**1.34x faster**) | `15.57 us` (**1.34x faster**) | `15.55 us` (**1.34x faster**)  |
