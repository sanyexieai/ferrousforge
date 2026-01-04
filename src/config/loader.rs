use crate::Result;
use crate::config::settings::Config;
use config::{Config as ConfigBuilder, Environment, File};

/// 从文件加载配置
pub fn load_from_file(path: &str) -> Result<Config> {
    let config = ConfigBuilder::builder()
        .add_source(File::with_name(path))
        .add_source(Environment::with_prefix("FERROUSFORGE"))
        .build()
        .map_err(|e| crate::api::error::ConfigError::LoadFailed(e.to_string()))?;

    config
        .try_deserialize()
        .map_err(|e| crate::api::error::ConfigError::Invalid(e.to_string()).into())
}

/// 从环境变量加载配置
pub fn load_from_env() -> Result<Config> {
    let config = ConfigBuilder::builder()
        .add_source(Environment::with_prefix("FERROUSFORGE"))
        .build()
        .map_err(|e| crate::api::error::ConfigError::LoadFailed(e.to_string()))?;

    config
        .try_deserialize()
        .map_err(|e| crate::api::error::ConfigError::Invalid(e.to_string()).into())
}

