//! # kittentts-rs
//!
//! Ultra-lightweight text-to-speech inference in Rust.
//!
//! A Rust port of the [KittenTTS](https://github.com/kittenml/kittentts) Python
//! package, using `ort` for ONNX Runtime inference and `espeakng` for phonemisation.

pub mod config;
pub mod espeak;
pub mod model;
pub mod preprocess;
pub mod text_cleaner;

// Re-export the main model type for convenience.
pub use model::KittenTTSModel;
