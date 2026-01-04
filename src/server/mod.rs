pub mod http;

use crate::Result;
use crate::config::Config;

/// 启动服务器
pub async fn serve(config: Config) -> Result<()> {
    tracing::info!("Starting server...");
    
    // TODO: 实现服务器启动逻辑
    // 目前先启动 HTTP 服务器
    http::serve(config).await
}

