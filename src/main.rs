use clap::Parser;
use ferrousforge::cli::Cli;
use ferrousforge::config::Config;
use ferrousforge::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // 解析命令行参数（需要在初始化日志前解析，以便使用配置文件中的日志设置）
    let cli = Cli::parse();
    
    // 加载配置（如果提供了配置文件）
    let config = if let Some(path) = &cli.config {
        Some(Config::from_file(path)?)
    } else {
        None
    };
    
    // 初始化日志系统
    if let Some(ref cfg) = config {
        // 使用配置文件中的日志设置
        ferrousforge::utils::logging::init_logging(&cfg.logging)?;
    } else {
        // 使用默认或环境变量配置
        ferrousforge::utils::logging::init_logging_from_env()
            .or_else(|_| ferrousforge::utils::logging::init_default_logging())?;
    }

    // 执行命令，如果没有指定命令则默认启动服务器
    match cli.command {
        None => {
            // 默认行为：启动 web 服务器
            let config = config.unwrap_or_else(|| Config::default());
            tracing::info!("Starting FerrousForge server (default mode)");
            ferrousforge::server::serve(config).await?;
        }
        Some(ferrousforge::cli::Command::Serve { config: serve_config }) => {
            let config = if let Some(path) = serve_config {
                Config::from_file(&path)?
            } else {
                config.unwrap_or_else(|| Config::default())
            };
            tracing::info!("Starting FerrousForge server");
            ferrousforge::server::serve(config).await?;
        }
        Some(ferrousforge::cli::Command::Pull { model }) => {
            ferrousforge::cli::commands::pull(&model).await?;
        }
        Some(ferrousforge::cli::Command::List) => {
            ferrousforge::cli::commands::list().await?;
        }
        Some(ferrousforge::cli::Command::Run { model, prompt }) => {
            ferrousforge::cli::commands::run(&model, &prompt).await?;
        }
        Some(ferrousforge::cli::Command::Remove { model }) => {
            ferrousforge::cli::commands::remove(&model).await?;
        }
    }

    Ok(())
}

