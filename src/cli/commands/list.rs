use crate::Result;

/// 列出已安装的模型
pub async fn list() -> Result<()> {
    tracing::info!("Listing installed models");
    
    // TODO: 实现模型列表逻辑
    println!("No models installed yet.");
    Ok(())
}

