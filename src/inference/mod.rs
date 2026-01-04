pub mod backend;
pub mod executor;
pub mod hardware;

pub mod backends;

// Re-export commonly used types
pub use backend::{
    InferenceBackend, BackendConfig, BackendInput, BackendOutput,
    TensorData, TensorDtype, MultimodalInput, MultimodalOutput,
    AcceleratorInfo, BackendRegistry, BackendInfo,
};

