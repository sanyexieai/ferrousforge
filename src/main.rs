use clap::Parser;
use ferrousforge::cli::Cli;
use ferrousforge::config::Config;
use ferrousforge::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // 解析命令行参数
    let cli = Cli::parse();

    // 执行命令，如果没有指定命令则默认启动服务器
    match cli.command {
        None => {
            // 默认行为：启动 web 服务器
            let config = if let Some(path) = cli.config {
                Config::from_file(&path)?
            } else {
                Config::default()
            };
            tracing::info!("Starting FerrousForge server (default mode)");
            ferrousforge::server::serve(config).await?;
        }
        Some(ferrousforge::cli::Command::Serve { config }) => {
            let config = if let Some(path) = config {
                Config::from_file(&path)?
            } else if let Some(path) = cli.config {
                Config::from_file(&path)?
            } else {
                Config::default()
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

