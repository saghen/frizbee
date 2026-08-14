#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <target> <destination>" >&2
  exit 2
fi

TARGET=$1
DESTINATION=$2

BINDING_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$BINDING_DIR/../.." && pwd)
PACKAGE_ID=$(cargo pkgid --manifest-path "$BINDING_DIR/Cargo.toml" frizbee-c)
VERSION=${PACKAGE_ID##*#}
PACKAGE_NAME="frizbee-$VERSION-$TARGET"
PREFIX="$DESTINATION/$PACKAGE_NAME"

CARGO_C_PROFILE=--release
if [[ ${FRIZBEE_C_DEBUG:-0} == 1 ]]; then
  CARGO_C_PROFILE=--debug
fi

# build static library, versioned shared library, C header, pkg-config metadata
cargo cinstall --locked --manifest-path "$BINDING_DIR/Cargo.toml" \
  --target "$TARGET" "$CARGO_C_PROFILE" \
  --destdir "$PREFIX" --prefix / --libdir lib --includedir include \
  --pkgconfigdir lib/pkgconfig

# make the pkg-config file relocatable
PC_FILE="$PREFIX/lib/pkgconfig/frizbee.pc"
sed 's|^prefix=.*|prefix=${pcfiledir}/../..|' "$PC_FILE" >"$PC_FILE.tmp"
mv "$PC_FILE.tmp" "$PC_FILE"

# add CMake package config, C++ header, LICENSE and README
mkdir -p "$PREFIX/lib/cmake/frizbee"
cp "$BINDING_DIR/include/frizbee/frizbee.hpp" \
  "$PREFIX/include/frizbee/frizbee.hpp"
cp "$REPO_DIR/LICENSE" "$PREFIX/LICENSE"
cp "$BINDING_DIR/README.md" "$PREFIX/README.md"
cp "$BINDING_DIR/cmake/frizbee-config.cmake.in" \
  "$PREFIX/lib/cmake/frizbee/frizbee-config.cmake"

echo "$PREFIX"
