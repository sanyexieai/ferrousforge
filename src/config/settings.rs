use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// 主配置结构
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub inference: InferenceConfig,
    pub logging: LoggingConfig,
    pub metrics: Option<MetricsConfig>,
}

/// 服务器配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
    pub max_connections: Option<usize>,
    pub timeout: Option<Duration>,
    pub cors: Option<CorsConfig>,
}

/// CORS 配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CorsConfig {
    pub allowed_origins: Vec<String>,
    pub allowed_methods: Vec<String>,
    pub allowed_headers: Vec<String>,
}

/// 存储配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StorageConfig {
    pub base_path: PathBuf,
    pub cache_size: Option<u64>,
    pub auto_download: bool,
    pub download_timeout: Option<Duration>,
}

/// 推理配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceConfig {
    pub default_backend: String,
    pub max_concurrent_requests: usize,
    pub max_memory_per_model: Option<u64>,
    pub auto_unload_timeout: Option<Duration>,
    pub backends: HashMap<String, BackendConfig>,
}

/// 后端配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BackendConfig {
    pub enabled: bool,
    pub device: Option<String>,
    pub threads: Option<usize>,
}

/// 日志配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub output: Vec<String>,
}

/// 指标配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub port: Option<u16>,
    pub path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 11434,
                workers: None,
                max_connections: Some(100),
                timeout: Some(Duration::from_secs(300)),
                cors: None,
            },
            storage: StorageConfig {
                base_path: PathBuf::from("./models"),
                cache_size: Some(10 * 1024 * 1024 * 1024), // 10GB
                auto_download: false,
                download_timeout: Some(Duration::from_secs(3600)),
            },
            inference: InferenceConfig {
                default_backend: "candle".to_string(),
                max_concurrent_requests: 10,
                max_memory_per_model: None,
                auto_unload_timeout: Some(Duration::from_secs(300)),
                backends: HashMap::new(),
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "json".to_string(),
                output: vec!["stdout".to_string()],
            },
            metrics: None,
        }
    }
}

impl Config {
    /// 从文件加载配置
    pub fn from_file(path: &str) -> crate::Result<Self> {
        crate::config::loader::load_from_file(path)
    }

    /// 从环境变量加载配置
    pub fn from_env() -> crate::Result<Self> {
        crate::config::loader::load_from_env()
    }
}

