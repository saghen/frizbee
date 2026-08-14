//! C ABI for the frizbee crate, exported as `libfrizbee`

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

mod config;
mod matcher;

pub use config::*;
pub use matcher::*;
