# Benchmarks

## Table of Contents

- [Environment](#environment)
- [Explanation](#explanation)
- [Benchmark Results](#benchmark-results)
    - [Chromium](#chromium)
    - [Partial Match](#partial-match)
    - [All Match](#all-match)
    - [No Match with Partial](#no-match-with-partial)
    - [No Match](#no-match)

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

## Explanation

In each of the benchmarks, the median length of the haystacks is varied from 8 to 128.

- **One Shot**: Uses the `Options::default()`, where we perform the fastest prefilter since no typos are allowed
- **All Scores**: Set via `max_typos: None`, gets the scores for all of the items without any filtering
- **1 Typo**: Set via `max_typos: Some(1)`, performs a slower, but still effective prefilter since a set number of typos are allowed
- **Nucleo**: Runs with normalization disabled, case insentivity enabled and fuzzy matching enabled
- **\$BENCH (Parallel)**: Same as $BENCH, but uses 8 threads to perform the matching in parallel

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
num_samples: 1406941
```

|          | `Nucleo`                 | `One Shot`                      | `Parallel`                      | `All Scores`                     | `1 Typo`                          |
|:---------|:-------------------------|:--------------------------------|:--------------------------------|:---------------------------------|:--------------------------------- |
| **`67`** | `90.90 ms` (**1.00x**) | `54.76 ms` (**1.66x faster**) | `7.68 ms` (**11.84x faster**) | `212.67 ms` (*2.34x slower*)   | `161.24 ms` (*1.77x slower*)    |

### Partial Match

What I would consider the typical case, where 5% of the haystack matches the needle and 20% of the haystack includes characters from the needle, but doesn't fully match.

```rust
needle: "deadbeef"
partial_match_percentage: 0.20
match_percentage: 0.05
median_length: varies
std_dev_length: median_length / 4
num_samples: 100000
```

|           | `Nucleo`                | `One Shot`                     | `Parallel`                        | `All Scores`                    | `1 Typo`                        |
|:----------|:------------------------|:-------------------------------|:----------------------------------|:--------------------------------|:------------------------------- |
| **`16`**  | `3.28 ms` (**1.00x**) | `1.84 ms` (**1.79x faster**) | `361.35 us` (**9.08x faster**)  | `10.34 ms` (*3.15x slower*)   | `2.88 ms` (**1.14x faster**)  |
| **`32`**  | `3.91 ms` (**1.00x**) | `2.22 ms` (**1.76x faster**) | `396.84 us` (**9.85x faster**)  | `15.38 ms` (*3.93x slower*)   | `3.74 ms` (**1.05x faster**)  |
| **`64`**  | `5.50 ms` (**1.00x**) | `3.04 ms` (**1.81x faster**) | `478.54 us` (**11.49x faster**) | `25.25 ms` (*4.59x slower*)   | `4.90 ms` (**1.12x faster**)  |
| **`128`** | `9.99 ms` (**1.00x**) | `5.12 ms` (**1.95x faster**) | `662.66 us` (**15.08x faster**) | `45.40 ms` (*4.54x slower*)   | `7.86 ms` (**1.27x faster**)  |

### All Match

All of the haystacks match the needle. The "All Scores" case will always be the fastest since it skips the prefiltering step, which no longer filters any of the items out.

```rust
needle: "deadbeef"
match_percentage: 1.0
partial_match_percentage: 0.0
median_length: varies
std_dev_length: median_length / 4
num_samples: 100000
```

|           | `Nucleo`                  | `One Shot`                      | `Parallel`                      | `All Scores`                    | `1 Typo`                         |
|:----------|:--------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `23.92 ms` (**1.00x**)  | `16.88 ms` (**1.42x faster**) | `2.40 ms` (**9.95x faster**)  | `11.29 ms` (**2.12x faster**) | `17.81 ms` (**1.34x faster**)  |
| **`32`**  | `39.59 ms` (**1.00x**)  | `22.30 ms` (**1.78x faster**) | `3.15 ms` (**12.59x faster**) | `16.14 ms` (**2.45x faster**) | `25.86 ms` (**1.53x faster**)  |
| **`64`**  | `64.33 ms` (**1.00x**)  | `32.08 ms` (**2.00x faster**) | `4.34 ms` (**14.81x faster**) | `26.71 ms` (**2.41x faster**) | `35.63 ms` (**1.81x faster**)  |
| **`128`** | `120.17 ms` (**1.00x**) | `51.84 ms` (**2.32x faster**) | `6.96 ms` (**17.26x faster**) | `47.17 ms` (**2.55x faster**) | `55.55 ms` (**2.16x faster**)  |

### No Match with Partial

None of the haystacks fully match the needle while 15% of the haystack includes characters from the needle, but doesn't fully match.

```rust
needle: "deadbeef"
match_percentage: 0.0
partial_match_percentage: 0.15
median_length: varies
std_dev_length: median_length / 4
num_samples: 100000
```

|           | `Nucleo`                | `One Shot`                       | `Parallel`                        | `All Scores`                     | `1 Typo`                        |
|:----------|:------------------------|:---------------------------------|:----------------------------------|:---------------------------------|:------------------------------- |
| **`16`**  | `1.91 ms` (**1.00x**) | `917.26 us` (**2.08x faster**) | `203.21 us` (**9.39x faster**)  | `9.86 ms` (*5.16x slower*)     | `1.70 ms` (**1.12x faster**)  |
| **`32`**  | `1.90 ms` (**1.00x**) | `1.01 ms` (**1.88x faster**)   | `206.99 us` (**9.18x faster**)  | `15.13 ms` (*7.97x slower*)    | `2.01 ms` (**1.06x slower**)  |
| **`64`**  | `2.22 ms` (**1.00x**) | `1.21 ms` (**1.83x faster**)   | `233.30 us` (**9.53x faster**)  | `24.67 ms` (*11.09x slower*)   | `2.56 ms` (*1.15x slower*)    |
| **`128`** | `3.30 ms` (**1.00x**) | `2.09 ms` (**1.58x faster**)   | `295.17 us` (**11.18x faster**) | `44.80 ms` (*13.58x slower*)   | `4.24 ms` (*1.28x slower*)    |

### No Match

None of the haystacks partially or fully match the needle, meaning none of the characters in the needle are present in the haystack.

```rust
needle: "deadbeef"
match_percentage: 0.0
partial_match_percentage: 0.0
median_length: varies
std_dev_length: median_length / 4
num_samples: 100000
```

|           | `Nucleo`                | `One Shot`                       | `Parallel`                        | `All Scores`                     | `1 Typo`                          |
|:----------|:------------------------|:---------------------------------|:----------------------------------|:---------------------------------|:--------------------------------- |
| **`16`**  | `1.53 ms` (**1.00x**) | `552.28 us` (**2.77x faster**) | `147.06 us` (**10.42x faster**) | `7.55 ms` (*4.93x slower*)     | `707.38 us` (**2.17x faster**)  |
| **`32`**  | `1.58 ms` (**1.00x**) | `607.76 us` (**2.60x faster**) | `155.18 us` (**10.17x faster**) | `12.61 ms` (*7.99x slower*)    | `833.93 us` (**1.89x faster**)  |
| **`64`**  | `1.88 ms` (**1.00x**) | `817.39 us` (**2.29x faster**) | `180.49 us` (**10.39x faster**) | `22.53 ms` (*12.01x slower*)   | `1.13 ms` (**1.66x faster**)    |
| **`128`** | `2.43 ms` (**1.00x**) | `1.18 ms` (**2.06x faster**)   | `224.60 us` (**10.80x faster**) | `42.21 ms` (*17.40x slower*)   | `1.62 ms` (**1.50x faster**)    |
