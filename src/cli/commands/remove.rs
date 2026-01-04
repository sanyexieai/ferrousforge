use crate::Result;

/// 删除模型
pub async fn remove(model: &str) -> Result<()> {
    tracing::info!("Removing model: {}", model);
    
    // TODO: 实现模型删除逻辑
    Ok(())
}

