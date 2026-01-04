use crate::Result;

/// 运行模型（本地）
pub async fn run(model: &str, prompt: &str) -> Result<()> {
    tracing::info!("Running model: {} with prompt: {}", model, prompt);
    
    // TODO: 实现本地运行逻辑
    Ok(())
}

