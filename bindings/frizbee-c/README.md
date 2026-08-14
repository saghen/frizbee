# Frizbee C and C++ bindings

C bindings and a header-only C++17 wrapper for [frizbee](https://github.com/saghen/frizbee), SIMD fuzzy string matching.

## Install

Download the SDK for your target from [GitHub Releases](https://github.com/saghen/frizbee/releases) (x86_64-linux, x86_64-windows, x86_64-macos, aarch64-linux, aarch64-macos).

Use pkg-config for the shared library (the pkg-config file is relocatable via `${pcfiledir}`):

```sh
cc main.c $(PKG_CONFIG_PATH=frizbee/lib/pkgconfig pkg-config --cflags --libs frizbee) -o main
```

CMake consumers can use `frizbee::frizbee` (shared) or `frizbee::static`:

```cmake
find_package(frizbee CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE frizbee::frizbee)
```

## Usage (C)

```c
#include <stdio.h>
#include <frizbee/frizbee.h>

// Passing NULL for config uses default configuration
frizbee_matcher_t *matcher = frizbee_matcher_new((frizbee_str_t){"fBr", 3}, NULL);

frizbee_str_t haystacks[] = {
    {"fooBar", 6},
    {"foo_bar", 7},
    {"barfoo", 6},
    {"prelude", 7},
};

frizbee_matches_t matches = frizbee_matcher_match_list(matcher, haystacks, 4);
for (size_t i = 0; i < matches.len; i++)
    printf("#%u scored %u\n", matches.items[i].index, matches.items[i].score);

frizbee_matches_free(&matches);
frizbee_matcher_free(matcher);
```

See [examples/smoke.c](examples/smoke.c)

## Usage (C++)

[include/frizbee/frizbee.hpp](include/frizbee/frizbee.hpp) is a header-only RAII wrapper over the C API, requiring C++17.

```cpp
#include <frizbee/frizbee.hpp>

frizbee::Matcher matcher("fBr");
auto matches = matcher.match_list({"fooBar", "foo_bar", "barfoo", "prelude"});
```

The wrapper also provides `match_list_parallel`, `match_list_indices`,
`match_one`, and `Matcher::from_query`. See [examples/smoke.cpp](examples/smoke.cpp).

## Building from source

```sh
nix develop
just build-c
```

The static and shared libraries are written to `target/release`. To assemble the same relocatable SDK used for releases:

```sh
bindings/frizbee-c/package.sh "$(rustc -vV | sed -n 's/^host: //p')" dist
```

This uses `cargo-c` to build and install the libraries, version the shared library and generate pkg-config metadata.
