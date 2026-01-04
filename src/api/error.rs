use thiserror::Error;

/// FerrousForge 错误类型
#[derive(Debug, Error)]
pub enum FerrousForgeError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("API error: {0}")]
    Api(#[from] ApiError),

    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
}

/// 模型错误
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model not found: {0}")]
    NotFound(String),

    #[error("Model already loaded: {0}")]
    AlreadyLoaded(String),

    #[error("Model not loaded: {0}")]
    NotLoaded(String),

    #[error("Unsupported model format: {0}")]
    UnsupportedFormat(String),

    #[error("Insufficient memory: required {required}, available {available}")]
    InsufficientMemory { required: u64, available: u64 },

    #[error("GPU required but not available")]
    GpuRequired,
}

/// 推理错误
#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("Inference failed: {0}")]
    Failed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Timeout")]
    Timeout,

    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),
}

/// 存储错误
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Storage path not found: {0}")]
    PathNotFound(String),

    #[error("Failed to download model: {0}")]
    DownloadFailed(String),

    #[error("Failed to read model file: {0}")]
    ReadFailed(String),

    #[error("Invalid model file: {0}")]
    InvalidFile(String),
}

/// 配置错误
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to load config: {0}")]
    LoadFailed(String),

    #[error("Invalid config: {0}")]
    Invalid(String),

    #[error("Config file not found: {0}")]
    NotFound(String),
}

/// API 错误
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Unauthorized")]
    Unauthorized,

    #[error("Internal server error: {0}")]
    Internal(String),
}

/// 后端错误
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Backend not available: {0}")]
    NotAvailable(String),

    #[error("Backend initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Unsupported backend: {0}")]
    Unsupported(String),
}

/// Result 类型别名
pub type Result<T> = std::result::Result<T, FerrousForgeError>;

