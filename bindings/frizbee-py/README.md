# Frizbee (Python bindings)

Python bindings for [frizbee](https://github.com/saghen/frizbee), SIMD fuzzy string matching. Supports CPython 3.10+ on linux x64/arm64 (glibc + musl), macOS x64/arm64 and windows x64.

```sh
pip install frizbee
```

## Usage

```python
import frizbee

matcher = frizbee.Matcher("fBr", max_typos=1, casing="smart")
matches = matcher.match_list(["fooBar", "foo_bar", "barfoo", "prelude"])
# [Match(score=53, index=0, exact=False), ...]

# Owned haystacks (recommended): copy to Rust once for much faster matching
haystacks = frizbee.Haystacks(["fooBar", "foo_bar", "barfoo", "prelude"])
matches = matcher.match_list(haystacks)

# Perform multi-pattern matching (whitespace separated) with syntax for controlling
# the matching mode:
# fuzzy  substring  prefix    suffix    exact    negated (combines with others)
# foo    'foo       ^foo      foo$      ^foo$    !foo
matcher = frizbee.Matcher.from_query("foo !^bar")

# Per-pattern config overrides
patterns = frizbee.Pattern.parse_query("foo !^bar")
for pattern in patterns:
    pattern.max_typos = len(pattern.needle) // 4
matcher = frizbee.Matcher.from_patterns(patterns)
```

## Performance

These bindings achieve native performance to the rust crate, after a one-time copy of strings to Rust. On the Chromium file list (1.4M haystacks, needle "linux", maxItems 1000):

```
new Haystacks(string[]):               33.99 ms
match_list (string[]):                 58.47 ms/iter
match_list (Haystacks):                23.60 ms/iter
match_list_parallel(8) (string[]):     40.14 ms/iter
match_list_parallel(8) (Haystacks):     4.52 ms/iter
```

## Development

```sh
# use nix dev shell to get all clis/pkgs
nix develop

just build-py
just test-py
just bench-py
```
