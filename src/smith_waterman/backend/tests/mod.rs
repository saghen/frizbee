//! Tests for the SIMD backend abstraction.
//!
//! - [`contract`] checks each backend's primitives (eq, gt, shift, ...) against the scalar impl
//! - [`parity`] checks that every backend produces identical scores and indices to the scalar impl
//! - [`generator`] randomized-input generator used by `parity` and `/tests/api_properties.rs`

mod contract;
mod generator;
mod parity;
