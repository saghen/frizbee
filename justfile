# Run from inside `nix develop`

_default:
    @just --list

# --------------

[group('test')]
test: test-rust test-c test-py test-wasm

[group('test')]
test-rust:
    cargo test

[group('test')]
test-c:
    cd bindings/frizbee-c && cbindgen --crate frizbee-c --output include/frizbee.h && git diff --exit-code include/frizbee.h
    cargo build -p frizbee-c
    cd bindings/frizbee-c && cc -std=c11 -Wall -Wextra -Werror examples/smoke.c -Iinclude ../../target/debug/libfrizbee.a -lpthread -ldl -lm -o ../../target/smoke-c && ../../target/smoke-c
    cd bindings/frizbee-c && c++ -std=c++17 -Wall -Wextra -Werror examples/smoke.cpp -Iinclude ../../target/debug/libfrizbee.a -lpthread -ldl -lm -o ../../target/smoke-cpp && ../../target/smoke-cpp

[group('test')]
test-py:
    cd bindings/frizbee-py && [ -d .venv ] || python3 -m venv .venv
    cd bindings/frizbee-py && . .venv/bin/activate && maturin develop && python -m unittest discover tests -v

[group('test')]
test-wasm:
    cd bindings/frizbee-wasm && wasm-pack test --node
    cd bindings/frizbee-wasm && wasm-pack build --target web --release && node --test tests/wrapper.test.mjs tests/wrapper-node.test.mjs

# --------------

[group('build')]
build: build-rust build-c build-py build-wasm

[group('build')]
build-rust:
    cargo build --release

[group('build')]
build-c:
    cd bindings/frizbee-c && cbindgen --crate frizbee-c --output include/frizbee.h
    cargo build --release -p frizbee-c

[group('build')]
build-py:
    cd bindings/frizbee-py && [ -d .venv ] || python3 -m venv .venv
    cd bindings/frizbee-py && . .venv/bin/activate && maturin develop --release

[group('build')]
build-wasm:
    cd bindings/frizbee-wasm && wasm-pack build --target web --release

# --------------

[group('bench')]
bench: bench-rust bench-c bench-py bench-wasm

[group('bench')]
bench-rust:
    cargo bench

[group('bench')]
bench-c: build-c
    cd bindings/frizbee-c && cc -O2 -std=c11 bench/bench.c -Iinclude ../../target/release/libfrizbee.a -lpthread -ldl -lm -o ../../target/bench-c && ../../target/bench-c

[group('bench')]
bench-py: build-py
    cd bindings/frizbee-py && . .venv/bin/activate && python bench.py

[group('bench')]
bench-wasm: build-wasm
    cd bindings/frizbee-wasm && node bench.mjs

[group('bench')]
download-bench-data:
    echo "Downloading chromium.txt benchmark data..."
    curl -L -o benches/data/chromium.txt https://gist.github.com/ii14/637689ef8d071824e881a78044670310/raw/dc1dbc859daa38b62f4b9a69dec1fc599e4735e7/data.txt
    echo
    echo "Downloading arabic_unicode.txt benchmark data..."
    curl -L -o benches/data/arabic_unicode.txt https://gist.github.com/saghen/d6d582b681bebecc2d2ad4a6ec9534e5/raw/3ffad717bbee2f77262c5d2d9f2e92b032f18760/arabic_unicode.txt
    echo
    echo "Downloading korean_unicode.txt benchmark data..."
    curl -L -o benches/data/korean_unicode.txt https://gist.github.com/saghen/f9b4c6e9870ee08913275520c178bafa/raw/8fd9355ee26ddc99a493eccd2475da382e6b7f8a/korean_unicode.txt
    echo
    echo "All benchmarks downloaded"

# --------------

bump version: test
    sed -i 's/^version = ".*"/version = "{{ version }}"/' Cargo.toml
    sed -i 's/"version": ".*"/"version": "{{ version }}"/' bindings/frizbee-wasm/package.json
    cargo update --workspace --offline

    git add Cargo.toml bindings/frizbee-wasm/package.json Cargo.lock
    git commit -m "chore: bump version to {{ version }}"
    git tag v{{ version }} -s -m v{{ version }}

    git-cliff --github-token $(gh auth token) -o CHANGELOG.md
    git add CHANGELOG.md
    git reset --soft HEAD~1
    git commit -m "chore: bump version to {{ version }}"
    git tag -d v{{ version }}
    git tag v{{ version }} -s -m v{{ version }}
