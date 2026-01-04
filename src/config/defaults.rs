// 默认配置常量

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 11434;
pub const DEFAULT_MAX_CONNECTIONS: usize = 100;
pub const DEFAULT_TIMEOUT_SECS: u64 = 300;

pub const DEFAULT_STORAGE_PATH: &str = "./models";
pub const DEFAULT_CACHE_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10GB
pub const DEFAULT_DOWNLOAD_TIMEOUT_SECS: u64 = 3600;

pub const DEFAULT_BACKEND: &str = "candle";
pub const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 10;
pub const DEFAULT_AUTO_UNLOAD_TIMEOUT_SECS: u64 = 300;

pub const DEFAULT_LOG_LEVEL: &str = "info";
pub const DEFAULT_LOG_FORMAT: &str = "json";

