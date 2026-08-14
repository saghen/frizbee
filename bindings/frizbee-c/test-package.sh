#!/usr/bin/env bash

# test that the pkg-config shared library, CMake shared library,
# and CMake static library work correctly and are relocatable

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

BINDING_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TEST_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEST_DIR"' EXIT

BUILT_PREFIX=$(
  CARGO_TARGET_DIR="$TEST_DIR/cargo-target" FRIZBEE_C_DEBUG=1 \
    "$BINDING_DIR/package.sh" \
    "$(rustc -vV | sed -n 's/^host: //p')" "$TEST_DIR"
)

# Consume the SDK after moving it away from cargo-c's installation destination
mkdir -p "$TEST_DIR/relocated"
PREFIX="$TEST_DIR/relocated/$(basename "$BUILT_PREFIX")"
mv "$BUILT_PREFIX" "$PREFIX"

UNAME_S=$(uname -s)
case $UNAME_S in
  MINGW* | MSYS* | CYGWIN*)
    export PATH="$PREFIX/bin:$PATH"
    CMAKE_EXE_DIR="$TEST_DIR/cmake-build/Debug"
    EXE_SUFFIX=.exe
    ;;
  Darwin)
    export DYLD_LIBRARY_PATH="$PREFIX/lib"
    CMAKE_EXE_DIR="$TEST_DIR/cmake-build"
    EXE_SUFFIX=
    ;;
  *)
    export LD_LIBRARY_PATH="$PREFIX/lib"
    CMAKE_EXE_DIR="$TEST_DIR/cmake-build"
    EXE_SUFFIX=
    ;;
esac

if [[ $UNAME_S != MINGW* && $UNAME_S != MSYS* && $UNAME_S != CYGWIN* ]]; then
  cc -std=c11 -Wall -Wextra -Werror \
    "$BINDING_DIR/examples/smoke.c" \
    $(PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" pkg-config --cflags --libs frizbee) \
    -o "$TEST_DIR/smoke-c-pkg-config"
  "$TEST_DIR/smoke-c-pkg-config"
fi

cmake -S "$BINDING_DIR/examples/cmake" -B "$TEST_DIR/cmake-build" \
  -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build "$TEST_DIR/cmake-build" --config Debug
"$CMAKE_EXE_DIR/frizbee-smoke-shared$EXE_SUFFIX"
"$CMAKE_EXE_DIR/frizbee-smoke-static$EXE_SUFFIX"
"$CMAKE_EXE_DIR/frizbee-smoke-cpp$EXE_SUFFIX"
