# Fuzzing

The fuzz targets use Bolero and can run either as bounded random smoke checks
or under AFL.

```bash
# discover targets
cargo bolero list --features fuzzing

# bounded local check
cargo test --features fuzzing --test fuzz_api_equivalence
cargo test --features fuzzing --test fuzz_prefilter
cargo test --features fuzzing --test fuzz_smith_waterman
cargo test --features fuzzing --test fuzz_long_inputs
cargo test --features fuzzing --test fuzz_unicode

# AFL
cargo bolero test --features fuzzing --engine afl --corpus-dir fuzz/corpus/api_equivalence fuzz_api_equivalence
cargo bolero test --features fuzzing --engine afl --corpus-dir fuzz/corpus/prefilter fuzz_prefilter
cargo bolero test --features fuzzing --engine afl --corpus-dir fuzz/corpus/smith_waterman fuzz_smith_waterman
cargo bolero test --features fuzzing --engine afl --corpus-dir fuzz/corpus/long_inputs fuzz_long_inputs
cargo bolero test --features fuzzing --engine afl --corpus-dir fuzz/corpus/unicode fuzz_unicode
```

The `fuzzing` feature only exposes `frizbee::fuzz_support` to the harnesses.
It is not part of the normal library surface.
