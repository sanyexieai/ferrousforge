pub mod commands;

use clap::{Parser, Subcommand};

/// FerrousForge CLI
#[derive(Parser)]
#[command(name = "ferrousforge")]
#[command(about = "A unified inference platform for multiple model types")]
#[command(version)]
pub struct Cli {
    /// 配置文件路径（仅用于默认启动服务器时）
    #[arg(short, long)]
    pub config: Option<String>,
    
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// CLI 命令
#[derive(Subcommand)]
pub enum Command {
    /// 启动服务器
    Serve {
        /// 配置文件路径
        #[arg(short, long)]
        config: Option<String>,
    },
    /// 下载模型
    Pull {
        /// 模型名称
        model: String,
    },
    /// 列出已安装的模型
    List,
    /// 运行模型（本地）
    Run {
        /// 模型名称
        model: String,
        /// 提示词
        prompt: String,
    },
    /// 删除模型
    Remove {
        /// 模型名称
        model: String,
    },
}

