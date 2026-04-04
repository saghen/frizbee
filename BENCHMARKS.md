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
| **`67`** | `90.68 ms` (**1.00x**) | `50.55 ms` (**1.79x faster**) | `7.21 ms` (**12.58x faster**) | `175.00 ms` (*1.93x slower*)   | `158.12 ms` (*1.74x slower*)   | `231.77 ms` (*2.56x slower*)   | `232.87 ms` (*2.57x slower*)    |

#### FZF

Version: 0.70.0 (d57ed157)

Single threaded: `118.7ms` (*2.34x slower*)

Multi threaded (8 threads): `31.15ms` (*4.4x slower* vs parallel)

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

|           | `Nucleo`                | `Frizbee`                      | `Parallel (x8)`                   | `All Scores`                    | `1 Typo`                       | `2 Typos`                       | `3 Typos`                        |
|:----------|:------------------------|:-------------------------------|:----------------------------------|:--------------------------------|:-------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `3.06 ms` (**1.00x**) | `1.84 ms` (**1.66x faster**) | `382.12 us` (**8.01x faster**)  | `7.88 ms` (*2.57x slower*)    | `2.66 ms` (**1.15x faster**) | `3.26 ms` (**1.07x slower**)  | `9.80 ms` (*3.20x slower*)     |
| **`32`**  | `3.88 ms` (**1.00x**) | `2.32 ms` (**1.67x faster**) | `440.14 us` (**8.81x faster**)  | `12.93 ms` (*3.33x slower*)   | `3.52 ms` (**1.10x faster**) | `4.39 ms` (*1.13x slower*)    | `15.01 ms` (*3.87x slower*)    |
| **`64`**  | `5.43 ms` (**1.00x**) | `3.02 ms` (**1.80x faster**) | `507.21 us` (**10.70x faster**) | `22.91 ms` (*4.22x slower*)   | `4.85 ms` (**1.12x faster**) | `6.35 ms` (*1.17x slower*)    | `25.38 ms` (*4.68x slower*)    |
| **`128`** | `9.07 ms` (**1.00x**) | `4.46 ms` (**2.04x faster**) | `649.45 us` (**13.97x faster**) | `44.69 ms` (*4.92x slower*)   | `7.64 ms` (**1.19x faster**) | `10.40 ms` (*1.15x slower*)   | `45.25 ms` (*4.99x slower*)    |

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
| **`16`**  | `22.74 ms` (**1.00x**)  | `16.39 ms` (**1.39x faster**) | `2.55 ms` (**8.90x faster**)  | `7.89 ms` (**2.88x faster**)  | `16.32 ms` (**1.39x faster**) | `16.32 ms` (**1.39x faster**) | `15.85 ms` (**1.43x faster**)  |
| **`32`**  | `38.82 ms` (**1.00x**)  | `24.12 ms` (**1.61x faster**) | `3.63 ms` (**10.70x faster**) | `12.81 ms` (**3.03x faster**) | `24.33 ms` (**1.60x faster**) | `24.40 ms` (**1.59x faster**) | `23.33 ms` (**1.66x faster**)  |
| **`64`**  | `62.65 ms` (**1.00x**)  | `35.19 ms` (**1.78x faster**) | `4.99 ms` (**12.55x faster**) | `23.98 ms` (**2.61x faster**) | `37.02 ms` (**1.69x faster**) | `37.30 ms` (**1.68x faster**) | `35.55 ms` (**1.76x faster**)  |
| **`128`** | `118.12 ms` (**1.00x**) | `53.54 ms` (**2.21x faster**) | `7.13 ms` (**16.58x faster**) | `45.70 ms` (**2.58x faster**) | `56.77 ms` (**2.08x faster**) | `57.79 ms` (**2.04x faster**) | `55.16 ms` (**2.14x faster**)  |

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

|           | `Nucleo`                | `Frizbee`                        | `Parallel (x8)`                   | `All Scores`                     | `1 Typo`                       | `2 Typos`                      | `3 Typos`                         |
|:----------|:------------------------|:---------------------------------|:----------------------------------|:---------------------------------|:-------------------------------|:-------------------------------|:--------------------------------- |
| **`16`**  | `1.81 ms` (**1.00x**) | `883.82 us` (**2.05x faster**) | `190.32 us` (**9.53x faster**)  | `7.86 ms` (*4.34x slower*)     | `1.55 ms` (**1.17x faster**) | `2.01 ms` (**1.11x slower**) | `8.90 ms` (*4.91x slower*)      |
| **`32`**  | `1.83 ms` (**1.00x**) | `959.90 us` (**1.91x faster**) | `199.70 us` (**9.16x faster**)  | `13.18 ms` (*7.20x slower*)    | `1.89 ms` (**1.04x slower**) | `2.61 ms` (*1.42x slower*)   | `14.07 ms` (*7.69x slower*)     |
| **`64`**  | `2.22 ms` (**1.00x**) | `1.15 ms` (**1.93x faster**)   | `222.92 us` (**9.94x faster**)  | `23.09 ms` (*10.42x slower*)   | `2.55 ms` (*1.15x slower*)   | `3.68 ms` (*1.66x slower*)   | `24.33 ms` (*10.97x slower*)    |
| **`128`** | `2.97 ms` (**1.00x**) | `1.66 ms` (**1.79x faster**)   | `276.87 us` (**10.73x faster**) | `44.88 ms` (*15.11x slower*)   | `4.04 ms` (*1.36x slower*)   | `5.99 ms` (*2.02x slower*)   | `44.08 ms` (*14.84x slower*)    |

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
| **`16`**  | `1.48 ms` (**1.00x**) | `502.86 us` (**2.95x faster**) | `137.23 us` (**10.82x faster**) | `7.71 ms` (*5.19x slower*)     | `600.05 us` (**2.47x faster**) | `658.09 us` (**2.26x faster**) | `7.77 ms` (*5.23x slower*)      |
| **`32`**  | `1.56 ms` (**1.00x**) | `563.73 us` (**2.76x faster**) | `143.23 us` (**10.88x faster**) | `12.73 ms` (*8.17x slower*)    | `689.17 us` (**2.26x faster**) | `822.64 us` (**1.89x faster**) | `12.83 ms` (*8.23x slower*)     |
| **`64`**  | `1.81 ms` (**1.00x**) | `756.68 us` (**2.39x faster**) | `168.28 us` (**10.77x faster**) | `22.80 ms` (*12.58x slower*)   | `1.01 ms` (**1.80x faster**)   | `1.22 ms` (**1.48x faster**)   | `22.84 ms` (*12.61x slower*)    |
| **`128`** | `2.54 ms` (**1.00x**) | `1.09 ms` (**2.32x faster**)   | `210.31 us` (**12.08x faster**) | `44.07 ms` (*17.34x slower*)   | `1.55 ms` (**1.64x faster**)   | `2.00 ms` (**1.27x faster**)   | `42.49 ms` (*16.72x slower*)    |

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
| **`16`**  | `20.75 us` (**1.00x**) | `15.36 us` (**1.35x faster**) | `15.02 us` (**1.38x faster**) | `15.37 us` (**1.35x faster**) | `15.46 us` (**1.34x faster**) | `15.31 us` (**1.35x faster**) | `15.34 us` (**1.35x faster**)  |
| **`32`**  | `20.67 us` (**1.00x**) | `15.27 us` (**1.35x faster**) | `14.92 us` (**1.39x faster**) | `15.38 us` (**1.34x faster**) | `15.37 us` (**1.34x faster**) | `15.41 us` (**1.34x faster**) | `15.27 us` (**1.35x faster**)  |
| **`64`**  | `20.86 us` (**1.00x**) | `15.48 us` (**1.35x faster**) | `14.95 us` (**1.40x faster**) | `15.33 us` (**1.36x faster**) | `15.40 us` (**1.35x faster**) | `15.37 us` (**1.36x faster**) | `15.34 us` (**1.36x faster**)  |
| **`128`** | `20.80 us` (**1.00x**) | `15.39 us` (**1.35x faster**) | `15.01 us` (**1.39x faster**) | `15.37 us` (**1.35x faster**) | `15.41 us` (**1.35x faster**) | `15.32 us` (**1.36x faster**) | `15.45 us` (**1.35x faster**)  |
