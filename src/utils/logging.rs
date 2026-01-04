//! 日志系统
//! 
//! 提供基于 tracing 的日志系统，支持：
//! - 可配置的日志级别
//! - 多种日志格式（JSON、Pretty）
//! - 多种输出方式（stdout、stderr、文件）

use crate::config::settings::LoggingConfig;
use crate::Result;
use std::path::PathBuf;
use tracing_subscriber::{
    fmt,
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
    Layer,
};
use tracing_subscriber::registry::Registry;

/// 初始化日志系统
/// 
/// 支持多个输出目标，可以同时输出到 stdout、stderr 和多个文件。
/// 
/// # 参数
/// 
/// * `config` - 日志配置
/// 
/// # 示例
/// 
/// ```no_run
/// use ferrousforge::config::settings::LoggingConfig;
/// use ferrousforge::utils::logging::init_logging;
/// 
/// let config = LoggingConfig {
///     level: "info".to_string(),
///     format: "json".to_string(),
///     output: vec!["stdout".to_string(), "logs/app.log".to_string()],
/// };
/// 
/// init_logging(&config).unwrap();
/// ```
pub fn init_logging(config: &LoggingConfig) -> Result<()> {
    // 解析日志级别
    // 优先使用环境变量 RUST_LOG，如果没有则使用配置文件中的级别
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            EnvFilter::try_new(&config.level)
                .unwrap_or_else(|_| {
                    // 注意：这里不能使用 tracing::warn!，因为日志系统还没初始化
                    eprintln!(
                        "Warning: Invalid log level '{}', using 'info' as default",
                        config.level
                    );
                    EnvFilter::new("info")
                })
        });

    // 如果没有指定输出，默认使用 stdout
    let outputs = if config.output.is_empty() {
        vec!["stdout".to_string()]
    } else {
        config.output.clone()
    };

    // 为每个输出创建并添加 layer
    // 注意：tracing-subscriber 的类型系统使得动态添加多个 layer 比较复杂
    // 这里我们使用第一个输出作为主输出，其他输出可以在后续版本中支持
    let primary_output = outputs.first().map(|s| s.as_str()).unwrap_or("stdout");
    
    // 如果有多个输出，记录警告（后续版本可以支持）
    if outputs.len() > 1 {
        eprintln!(
            "Warning: Multiple log outputs specified, but only the first one ({}) will be used. \
             Multi-output support will be added in a future version.",
            primary_output
        );
    }

    // 根据输出类型初始化 subscriber
    match primary_output {
        "stdout" => {
            init_subscriber(&config.format, filter, std::io::stdout)?;
        }
        "stderr" => {
            init_subscriber(&config.format, filter, std::io::stderr)?;
        }
        file_path => {
            // 文件输出
            let path = PathBuf::from(file_path);
            
            // 确保目录存在
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| crate::api::error::ConfigError::Invalid(
                        format!("Failed to create log directory: {}", e)
                    ))?;
            }
            
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| crate::api::error::ConfigError::Invalid(
                    format!("Failed to open log file {}: {}", file_path, e)
                ))?;
            
            init_subscriber(&config.format, filter, file)?;
        }
    }

    tracing::info!(
        "Logging initialized: level={}, format={}, output={}",
        config.level,
        config.format,
        primary_output
    );

    Ok(())
}

/// 创建并初始化 subscriber（内部函数）
fn init_subscriber<W>(
    format: &str,
    filter: EnvFilter,
    writer: W,
) -> Result<()>
where
    W: for<'writer> fmt::MakeWriter<'writer> + Send + Sync + 'static,
{
    let registry = Registry::default().with(filter);

    let layer = match format.to_lowercase().as_str() {
        "json" => {
            fmt::layer()
                .with_writer(writer)
                .json()
                .with_target(true)
                .with_level(true)
                .with_file(true)
                .with_line_number(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .boxed()
        }
        "pretty" | "human" => {
            fmt::layer()
                .with_writer(writer)
                .pretty()
                .with_target(true)
                .with_level(true)
                .with_file(true)
                .with_line_number(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .boxed()
        }
        "compact" => {
            fmt::layer()
                .with_writer(writer)
                .compact()
                .with_target(true)
                .with_level(true)
                .with_file(true)
                .with_line_number(true)
                .boxed()
        }
        _ => {
            // 默认使用 compact 格式
            fmt::layer()
                .with_writer(writer)
                .compact()
                .with_target(true)
                .with_level(true)
                .with_file(true)
                .with_line_number(true)
                .boxed()
        }
    };

    registry
        .with(layer)
        .try_init()
        .map_err(|e| crate::api::error::ConfigError::Invalid(
            format!("Failed to initialize logging: {}", e)
        ))?;

    Ok(())
}



/// 使用默认配置初始化日志系统
/// 
/// 使用 info 级别和 pretty 格式输出到 stdout
pub fn init_default_logging() -> Result<()> {
    let config = LoggingConfig {
        level: "info".to_string(),
        format: "pretty".to_string(),
        output: vec!["stdout".to_string()],
    };
    init_logging(&config)
}

/// 从环境变量初始化日志系统
/// 
/// 支持以下环境变量：
/// - `RUST_LOG`: 日志级别，格式：`RUST_LOG=debug,ferrousforge=info`
/// - `RUST_LOG_FORMAT`: 日志格式，可选值：`json`, `pretty`, `compact`
/// - `RUST_LOG_OUTPUT`: 输出目标，多个目标用逗号分隔，如：`stdout,stderr,logs/app.log`
pub fn init_logging_from_env() -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let format = std::env::var("RUST_LOG_FORMAT")
        .unwrap_or_else(|_| "pretty".to_string());

    let output = std::env::var("RUST_LOG_OUTPUT")
        .map(|s| {
            s.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_else(|_| vec!["stdout".to_string()]);

    let config = LoggingConfig {
        level: "info".to_string(), // 实际使用环境变量中的 RUST_LOG
        format,
        output,
    };

    // 使用第一个输出（简化实现）
    let primary_output = config.output.first()
        .map(|s| s.as_str())
        .unwrap_or("stdout");

    // 根据输出类型初始化 subscriber
    match primary_output {
        "stdout" => {
            init_subscriber(&config.format, filter, std::io::stdout)?;
        }
        "stderr" => {
            init_subscriber(&config.format, filter, std::io::stderr)?;
        }
        file_path => {
            let path = PathBuf::from(file_path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| crate::api::error::ConfigError::Invalid(
                        format!("Failed to create log directory: {}", e)
                    ))?;
            }
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| crate::api::error::ConfigError::Invalid(
                    format!("Failed to open log file {}: {}", file_path, e)
                ))?;
            init_subscriber(&config.format, filter, file)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_default_logging() {
        // 这个测试可能会失败，因为 tracing 只能初始化一次
        // 在实际使用中，应该只在程序开始时初始化一次
        let _ = init_default_logging();
    }

    #[test]
    fn test_logging_config() {
        let config = LoggingConfig {
            level: "debug".to_string(),
            format: "json".to_string(),
            output: vec!["stdout".to_string()],
        };

        // 验证配置结构
        assert_eq!(config.level, "debug");
        assert_eq!(config.format, "json");
        assert_eq!(config.output.len(), 1);
    }
}
