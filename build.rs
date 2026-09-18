//! Compiles the scalar fallback out when a SIMD tier is guaranteed at compile
//! time, since the runtime backend selection could never reach it. The
//! `force_backend` feature keeps it so benchmarks can still select it.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(scalar_fallback)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |feature: &str| features.split(',').any(|f| f == feature);
    let simd_guaranteed = match arch.as_str() {
        "aarch64" => has("neon"),
        "wasm32" => has("simd128"),
        "x86_64" => has("ssse3") && has("sse4.1"),
        _ => false,
    };
    if !simd_guaranteed || std::env::var("CARGO_FEATURE_FORCE_BACKEND").is_ok() {
        println!("cargo:rustc-cfg=scalar_fallback");
    }
}
