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

// ========== 错误转换 ==========

impl From<std::io::Error> for FerrousForgeError {
    fn from(err: std::io::Error) -> Self {
        FerrousForgeError::Storage(StorageError::ReadFailed(err.to_string()))
    }
}

impl From<serde_json::Error> for FerrousForgeError {
    fn from(err: serde_json::Error) -> Self {
        FerrousForgeError::Api(ApiError::InvalidRequest(format!("JSON error: {}", err)))
    }
}

impl From<reqwest::Error> for FerrousForgeError {
    fn from(err: reqwest::Error) -> Self {
        FerrousForgeError::Storage(StorageError::DownloadFailed(err.to_string()))
    }
}

// ========== HTTP 错误响应 ==========

use axum::{
    response::{IntoResponse, Response},
    http::StatusCode,
    Json,
};
use serde::Serialize;

/// HTTP 错误响应
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl IntoResponse for FerrousForgeError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            FerrousForgeError::Model(e) => match e {
                ModelError::NotFound(_) => (StatusCode::NOT_FOUND, "ModelNotFound"),
                ModelError::AlreadyLoaded(_) => (StatusCode::CONFLICT, "ModelAlreadyLoaded"),
                ModelError::NotLoaded(_) => (StatusCode::BAD_REQUEST, "ModelNotLoaded"),
                ModelError::UnsupportedFormat(_) => (StatusCode::BAD_REQUEST, "UnsupportedFormat"),
                ModelError::InsufficientMemory { .. } => (StatusCode::INSUFFICIENT_STORAGE, "InsufficientMemory"),
                ModelError::GpuRequired => (StatusCode::SERVICE_UNAVAILABLE, "GpuRequired"),
            },
            FerrousForgeError::Inference(e) => match e {
                InferenceError::Failed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "InferenceFailed"),
                InferenceError::InvalidInput(_) => (StatusCode::BAD_REQUEST, "InvalidInput"),
                InferenceError::Timeout => (StatusCode::REQUEST_TIMEOUT, "Timeout"),
                InferenceError::BackendNotAvailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "BackendNotAvailable"),
            },
            FerrousForgeError::Storage(e) => match e {
                StorageError::PathNotFound(_) => (StatusCode::NOT_FOUND, "PathNotFound"),
                StorageError::DownloadFailed(_) => (StatusCode::BAD_GATEWAY, "DownloadFailed"),
                StorageError::ReadFailed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "ReadFailed"),
                StorageError::InvalidFile(_) => (StatusCode::BAD_REQUEST, "InvalidFile"),
            },
            FerrousForgeError::Config(e) => match e {
                ConfigError::LoadFailed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "ConfigLoadFailed"),
                ConfigError::Invalid(_) => (StatusCode::BAD_REQUEST, "ConfigInvalid"),
                ConfigError::NotFound(_) => (StatusCode::NOT_FOUND, "ConfigNotFound"),
            },
            FerrousForgeError::Api(e) => match e {
                ApiError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "InvalidRequest"),
                ApiError::RateLimitExceeded => (StatusCode::TOO_MANY_REQUESTS, "RateLimitExceeded"),
                ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized"),
                ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "InternalError"),
            },
            FerrousForgeError::Backend(e) => match e {
                BackendError::NotAvailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "BackendNotAvailable"),
                BackendError::InitializationFailed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "BackendInitFailed"),
                BackendError::Unsupported(_) => (StatusCode::BAD_REQUEST, "BackendUnsupported"),
            },
        };

        let error_message = self.to_string();
        let details = if error_message.len() > 100 {
            Some(error_message.clone())
        } else {
            None::<String>
        };

        let body = Json(ErrorResponse {
            error: error_message,
            error_type: error_type.to_string(),
            details,
        });

        (status, body).into_response()
    }
}

// ========== 错误代码 ==========

impl FerrousForgeError {
    /// 获取 HTTP 状态码
    pub fn status_code(&self) -> StatusCode {
        match self {
            FerrousForgeError::Model(e) => match e {
                ModelError::NotFound(_) => StatusCode::NOT_FOUND,
                ModelError::AlreadyLoaded(_) => StatusCode::CONFLICT,
                ModelError::NotLoaded(_) => StatusCode::BAD_REQUEST,
                ModelError::UnsupportedFormat(_) => StatusCode::BAD_REQUEST,
                ModelError::InsufficientMemory { .. } => StatusCode::INSUFFICIENT_STORAGE,
                ModelError::GpuRequired => StatusCode::SERVICE_UNAVAILABLE,
            },
            FerrousForgeError::Inference(e) => match e {
                InferenceError::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
                InferenceError::InvalidInput(_) => StatusCode::BAD_REQUEST,
                InferenceError::Timeout => StatusCode::REQUEST_TIMEOUT,
                InferenceError::BackendNotAvailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            },
            FerrousForgeError::Storage(e) => match e {
                StorageError::PathNotFound(_) => StatusCode::NOT_FOUND,
                StorageError::DownloadFailed(_) => StatusCode::BAD_GATEWAY,
                StorageError::ReadFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
                StorageError::InvalidFile(_) => StatusCode::BAD_REQUEST,
            },
            FerrousForgeError::Config(e) => match e {
                ConfigError::LoadFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
                ConfigError::Invalid(_) => StatusCode::BAD_REQUEST,
                ConfigError::NotFound(_) => StatusCode::NOT_FOUND,
            },
            FerrousForgeError::Api(e) => match e {
                ApiError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
                ApiError::RateLimitExceeded => StatusCode::TOO_MANY_REQUESTS,
                ApiError::Unauthorized => StatusCode::UNAUTHORIZED,
                ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            },
            FerrousForgeError::Backend(e) => match e {
                BackendError::NotAvailable(_) => StatusCode::SERVICE_UNAVAILABLE,
                BackendError::InitializationFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
                BackendError::Unsupported(_) => StatusCode::BAD_REQUEST,
            },
        }
    }

    /// 获取错误类型字符串
    pub fn error_type(&self) -> &'static str {
        match self {
            FerrousForgeError::Model(e) => match e {
                ModelError::NotFound(_) => "ModelNotFound",
                ModelError::AlreadyLoaded(_) => "ModelAlreadyLoaded",
                ModelError::NotLoaded(_) => "ModelNotLoaded",
                ModelError::UnsupportedFormat(_) => "UnsupportedFormat",
                ModelError::InsufficientMemory { .. } => "InsufficientMemory",
                ModelError::GpuRequired => "GpuRequired",
            },
            FerrousForgeError::Inference(e) => match e {
                InferenceError::Failed(_) => "InferenceFailed",
                InferenceError::InvalidInput(_) => "InvalidInput",
                InferenceError::Timeout => "Timeout",
                InferenceError::BackendNotAvailable(_) => "BackendNotAvailable",
            },
            FerrousForgeError::Storage(e) => match e {
                StorageError::PathNotFound(_) => "PathNotFound",
                StorageError::DownloadFailed(_) => "DownloadFailed",
                StorageError::ReadFailed(_) => "ReadFailed",
                StorageError::InvalidFile(_) => "InvalidFile",
            },
            FerrousForgeError::Config(e) => match e {
                ConfigError::LoadFailed(_) => "ConfigLoadFailed",
                ConfigError::Invalid(_) => "ConfigInvalid",
                ConfigError::NotFound(_) => "ConfigNotFound",
            },
            FerrousForgeError::Api(e) => match e {
                ApiError::InvalidRequest(_) => "InvalidRequest",
                ApiError::RateLimitExceeded => "RateLimitExceeded",
                ApiError::Unauthorized => "Unauthorized",
                ApiError::Internal(_) => "InternalError",
            },
            FerrousForgeError::Backend(e) => match e {
                BackendError::NotAvailable(_) => "BackendNotAvailable",
                BackendError::InitializationFailed(_) => "BackendInitFailed",
                BackendError::Unsupported(_) => "BackendUnsupported",
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_code() {
        let err = FerrousForgeError::Model(ModelError::NotFound("test".to_string()));
        assert_eq!(err.status_code(), StatusCode::NOT_FOUND);
        assert_eq!(err.error_type(), "ModelNotFound");
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let ff_err: FerrousForgeError = io_err.into();
        assert!(matches!(ff_err, FerrousForgeError::Storage(_)));
    }

    #[test]
    fn test_error_response() {
        let err = FerrousForgeError::Api(ApiError::InvalidRequest("test".to_string()));
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}

