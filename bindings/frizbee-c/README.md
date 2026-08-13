# Frizbee (C/C++ bindings)

C bindings plus a header-only C++17 wrapper for [frizbee](https://github.com/saghen/frizbee), SIMD fuzzy string matching.

## Install

Download the tarball for your target from [GitHub Releases](https://github.com/saghen/frizbee/releases) (Linux gnu/musl, macOS and Windows, x86_64 and aarch64). Each contains `include/` with the C and C++ headers, static and shared libraries in `lib/`, and `lib/pkgconfig/frizbee.pc`.

```sh
cc main.c -Ifrizbee/include frizbee/lib/libfrizbee.a -lpthread -ldl -lm
```

Or use pkg-config after adjusting the `prefix` in `frizbee.pc` to where you extracted the tarball:

```sh
cc main.c $(pkg-config --cflags --libs --static frizbee)
```

## Usage (C)

```c
#include <stdio.h>
#include "frizbee.h"

frizbee_config_t config = frizbee_config_default();
frizbee_matcher_t *matcher = frizbee_matcher_new((frizbee_str_t){"fBr", 3}, &config);

frizbee_str_t haystacks[] = {
    {"fooBar", 6}, {"foo_bar", 7}, {"barfoo", 6}, {"prelude", 7},
};

frizbee_matches_t matches = frizbee_match_list(matcher, haystacks, 4);
for (size_t i = 0; i < matches.len; i++)
    printf("#%u scored %u\n", matches.items[i].index, matches.items[i].score);

frizbee_matches_free(&matches);
frizbee_matcher_free(matcher);
```

See [examples/smoke.c](examples/smoke.c)

## Usage (C++)

[include/frizbee.hpp](include/frizbee.hpp) is a header-only RAII wrapper over the C API, requiring C++17.

```cpp
#include "frizbee.hpp"

frizbee::Matcher matcher("fBr");
auto matches = matcher.match_list({"fooBar", "foo_bar", "barfoo", "prelude"});
// or: matcher.match_list_parallel(haystacks, 8), matcher.match_list_indices(...),
//     matcher.match_one("fooBar", 0), frizbee::Matcher::from_query("foo !^bar")
```

See [examples/smoke.cpp](examples/smoke.cpp)

## Building from source

```sh
nix develop
just build-c
```

The static and shared libraries can be found in `target/release` (`libfrizbee.a` plus `libfrizbee.so`/`libfrizbee.dylib`/`frizbee.dll` depending on your platform).
