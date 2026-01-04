//! 具体后端实现
//! 
//! 包含各种推理后端的具体实现。

pub mod llama_cpp;

#[cfg(feature = "llama-cpp")]
pub(crate) mod llama_cpp_ffi;

// Re-export commonly used types
pub use llama_cpp::{LlamaCppBackend, LlamaCppModelHandle, LlamaCppContextParams};

