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
haystack_size: 1406941
```

|          | `Nucleo`                 | `One Shot`                      | `Parallel`                      | `All Scores`                     | `1 Typo`                          |
|:---------|:-------------------------|:--------------------------------|:--------------------------------|:---------------------------------|:--------------------------------- |
| **`67`** | `95.06 ms` (**1.00x**) | `53.97 ms` (**1.76x faster**) | `7.60 ms` (**12.51x faster**) | `212.92 ms` (*2.24x slower*)   | `158.76 ms` (*1.67x slower*)    |

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

|           | `Nucleo`                | `One Shot`                     | `Parallel`                        | `All Scores`                    | `1 Typo`                        |
|:----------|:------------------------|:-------------------------------|:----------------------------------|:--------------------------------|:------------------------------- |
| **`16`**  | `3.52 ms` (**1.00x**) | `1.91 ms` (**1.84x faster**) | `368.71 us` (**9.54x faster**)  | `10.52 ms` (*2.99x slower*)   | `2.79 ms` (**1.26x faster**)  |
| **`32`**  | `4.48 ms` (**1.00x**) | `2.24 ms` (**2.00x faster**) | `393.12 us` (**11.39x faster**) | `15.44 ms` (*3.45x slower*)   | `3.59 ms` (**1.25x faster**)  |
| **`64`**  | `6.08 ms` (**1.00x**) | `3.10 ms` (**1.96x faster**) | `489.86 us` (**12.42x faster**) | `25.81 ms` (*4.24x slower*)   | `4.94 ms` (**1.23x faster**)  |
| **`128`** | `9.76 ms` (**1.00x**) | `5.19 ms` (**1.88x faster**) | `678.08 us` (**14.39x faster**) | `46.29 ms` (*4.74x slower*)   | `8.12 ms` (**1.20x faster**)  |

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

|           | `Nucleo`                  | `One Shot`                      | `Parallel`                      | `All Scores`                    | `1 Typo`                         |
|:----------|:--------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `23.63 ms` (**1.00x**)  | `17.16 ms` (**1.38x faster**) | `2.43 ms` (**9.74x faster**)  | `11.98 ms` (**1.97x faster**) | `18.52 ms` (**1.28x faster**)  |
| **`32`**  | `40.41 ms` (**1.00x**)  | `21.87 ms` (**1.85x faster**) | `2.99 ms` (**13.51x faster**) | `16.55 ms` (**2.44x faster**) | `25.15 ms` (**1.61x faster**)  |
| **`64`**  | `65.89 ms` (**1.00x**)  | `31.78 ms` (**2.07x faster**) | `4.37 ms` (**15.08x faster**) | `27.35 ms` (**2.41x faster**) | `35.68 ms` (**1.85x faster**)  |
| **`128`** | `120.53 ms` (**1.00x**) | `52.63 ms` (**2.29x faster**) | `7.07 ms` (**17.04x faster**) | `47.70 ms` (**2.53x faster**) | `56.32 ms` (**2.14x faster**)  |

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

|           | `Nucleo`                | `One Shot`                       | `Parallel`                        | `All Scores`                     | `1 Typo`                        |
|:----------|:------------------------|:---------------------------------|:----------------------------------|:---------------------------------|:------------------------------- |
| **`16`**  | `2.13 ms` (**1.00x**) | `982.93 us` (**2.16x faster**) | `208.02 us` (**10.22x faster**) | `10.22 ms` (*4.81x slower*)    | `1.66 ms` (**1.28x faster**)  |
| **`32`**  | `2.37 ms` (**1.00x**) | `1.03 ms` (**2.30x faster**)   | `213.57 us` (**11.08x faster**) | `15.19 ms` (*6.42x slower*)    | `1.97 ms` (**1.20x faster**)  |
| **`64`**  | `2.66 ms` (**1.00x**) | `1.26 ms` (**2.11x faster**)   | `245.55 us` (**10.85x faster**) | `25.08 ms` (*9.42x slower*)    | `2.66 ms` (**1.00x faster**)  |
| **`128`** | `3.45 ms` (**1.00x**) | `2.21 ms` (**1.56x faster**)   | `308.48 us` (**11.19x faster**) | `45.61 ms` (*13.21x slower*)   | `4.45 ms` (*1.29x slower*)    |

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

|           | `Nucleo`                | `One Shot`                       | `Parallel`                        | `All Scores`                     | `1 Typo`                          |
|:----------|:------------------------|:---------------------------------|:----------------------------------|:---------------------------------|:--------------------------------- |
| **`16`**  | `1.60 ms` (**1.00x**) | `580.95 us` (**2.75x faster**) | `154.37 us` (**10.34x faster**) | `7.82 ms` (*4.90x slower*)     | `678.05 us` (**2.35x faster**)  |
| **`32`**  | `1.85 ms` (**1.00x**) | `645.27 us` (**2.86x faster**) | `160.67 us` (**11.49x faster**) | `12.83 ms` (*6.95x slower*)    | `817.53 us` (**2.26x faster**)  |
| **`64`**  | `2.03 ms` (**1.00x**) | `855.61 us` (**2.38x faster**) | `188.01 us` (**10.82x faster**) | `22.97 ms` (*11.29x slower*)   | `1.15 ms` (**1.77x faster**)    |
| **`128`** | `2.51 ms` (**1.00x**) | `1.22 ms` (**2.05x faster**)   | `232.28 us` (**10.79x faster**) | `42.95 ms` (*17.13x slower*)   | `1.71 ms` (**1.47x faster**)    |

### Copy

|           | `Nucleo`                 | `One Shot`                      | `Parallel`                      | `All Scores`                    | `1 Typo`                         |
|:----------|:-------------------------|:--------------------------------|:--------------------------------|:--------------------------------|:-------------------------------- |
| **`16`**  | `20.74 us` (**1.00x**) | `15.38 us` (**1.35x faster**) | `14.95 us` (**1.39x faster**) | `15.41 us` (**1.35x faster**) | `15.34 us` (**1.35x faster**)  |
| **`32`**  | `20.72 us` (**1.00x**) | `15.43 us` (**1.34x faster**) | `14.95 us` (**1.39x faster**) | `15.34 us` (**1.35x faster**) | `15.35 us` (**1.35x faster**)  |
| **`64`**  | `20.74 us` (**1.00x**) | `15.37 us` (**1.35x faster**) | `14.92 us` (**1.39x faster**) | `15.35 us` (**1.35x faster**) | `15.34 us` (**1.35x faster**)  |
| **`128`** | `20.68 us` (**1.00x**) | `15.28 us` (**1.35x faster**) | `14.89 us` (**1.39x faster**) | `15.23 us` (**1.36x faster**) | `15.32 us` (**1.35x faster**)  |
