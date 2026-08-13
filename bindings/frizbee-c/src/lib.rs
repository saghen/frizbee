//! C ABI for the frizbee crate, exported as `libfrizbee`.
//!
//! Strings cross the boundary as [`frizbee_str_t`] (pointer + length) and are
//! **not** validated as UTF-8.
//!
//! Result buffers are allocated by Rust and must be released with the matching
//! `frizbee_*_free` function, never with `free()`.

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

mod config;
mod matcher;

pub use config::*;
pub use matcher::*;
