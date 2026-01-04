//! FerrousForge - A unified inference platform for multiple model types
//!
//! FerrousForge is a Rust implementation inspired by Ollama, supporting
//! text, image, audio, video, and multimodal models through a unified interface.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod api;
pub mod cli;
pub mod config;
pub mod core;
pub mod gateway;
pub mod inference;
pub mod management;
pub mod models;
pub mod monitoring;
pub mod plugins;
pub mod server;
pub mod storage;
pub mod utils;

// Re-export commonly used types
pub use crate::api::error::{FerrousForgeError, Result};
pub use crate::config::Config;

/// FerrousForge version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

