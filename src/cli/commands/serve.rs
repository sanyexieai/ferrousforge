use crate::Result;
use crate::config::Config;

/// 启动服务器
pub async fn serve(config: Config) -> Result<()> {
    tracing::info!("Starting FerrousForge server on {}:{}", config.server.host, config.server.port);
    
    // TODO: 实现服务器启动逻辑
    crate::server::serve(config).await
}

