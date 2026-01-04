//! 存储管理器
//! 
//! 提供统一的存储接口，管理模型文件的存储、检索和删除。

use crate::Result;
use crate::config::Config;
use std::path::{Path, PathBuf};
use async_trait::async_trait;

/// 存储 trait - 定义存储操作的统一接口
#[async_trait]
pub trait Storage: Send + Sync {
    /// 获取模型文件路径
    async fn get_model_path(&self, name: &str) -> Result<PathBuf>;
    
    /// 检查模型是否存在
    async fn model_exists(&self, name: &str) -> bool;
    
    /// 列出所有已安装的模型
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    
    /// 删除模型
    async fn remove_model(&self, name: &str) -> Result<()>;
    
    /// 获取存储基础路径
    fn base_path(&self) -> &Path;
}

/// 模型信息
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    pub format: Option<String>,
}

/// 文件系统存储实现
pub struct FileSystemStorage {
    base_path: PathBuf,
}

impl FileSystemStorage {
    /// 创建新的文件系统存储
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// 从配置创建
    pub fn from_config(config: &Config) -> Self {
        Self::new(&config.storage.base_path)
    }

    /// 确保基础目录存在
    pub async fn ensure_base_path(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.base_path)
            .await
            .map_err(|e| crate::api::error::StorageError::PathNotFound(
                format!("Failed to create storage directory: {}", e)
            ).into())
    }

    /// 获取模型目录路径
    fn model_dir(&self, name: &str) -> PathBuf {
        self.base_path.join(name)
    }
}

#[async_trait]
impl Storage for FileSystemStorage {
    async fn get_model_path(&self, name: &str) -> Result<PathBuf> {
        let model_dir = self.model_dir(name);
        
        if !model_dir.exists() {
            return Err(crate::api::error::ModelError::NotFound(name.to_string()).into());
        }

        // 查找模型文件（支持多种格式）
        let model_extensions = ["gguf", "safetensors", "onnx", "bin", "pt", "pth"];
        
        for ext in &model_extensions {
            let model_file = model_dir.join(format!("model.{}", ext));
            if model_file.exists() {
                return Ok(model_file);
            }
        }

        // 如果没有找到标准格式，返回目录（可能包含多个文件）
        Ok(model_dir)
    }

    async fn model_exists(&self, name: &str) -> bool {
        self.model_dir(name).exists()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.ensure_base_path().await?;

        let mut models = Vec::new();

        let mut entries = tokio::fs::read_dir(&self.base_path)
            .await
            .map_err(|e| crate::api::error::StorageError::ReadFailed(
                format!("Failed to read storage directory: {}", e)
            ))?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| crate::api::error::StorageError::ReadFailed(
                format!("Failed to read directory entry: {}", e)
            ))? {
            
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                // 计算目录大小
                let size = calculate_dir_size(&path).await.unwrap_or(0);

                // 检测模型格式
                let format = detect_model_format(&path).await;

                models.push(ModelInfo {
                    name,
                    path,
                    size,
                    format,
                });
            }
        }

        Ok(models)
    }

    async fn remove_model(&self, name: &str) -> Result<()> {
        let model_dir = self.model_dir(name);
        
        if !model_dir.exists() {
            return Err(crate::api::error::ModelError::NotFound(name.to_string()).into());
        }

        tokio::fs::remove_dir_all(&model_dir)
            .await
            .map_err(|e| crate::api::error::StorageError::ReadFailed(
                format!("Failed to remove model directory: {}", e)
            ))?;

        Ok(())
    }

    fn base_path(&self) -> &Path {
        &self.base_path
    }
}

/// 计算目录大小
async fn calculate_dir_size(path: &Path) -> Result<u64> {
    let mut total_size = 0u64;
    
    let mut stack = vec![path.to_path_buf()];
    
    while let Some(current_path) = stack.pop() {
        let mut entries = tokio::fs::read_dir(&current_path)
            .await
            .map_err(|e| crate::api::error::StorageError::ReadFailed(
                format!("Failed to read directory: {}", e)
            ))?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| crate::api::error::StorageError::ReadFailed(
                format!("Failed to read entry: {}", e)
            ))? {
            
            let path = entry.path();
            let metadata = entry.metadata().await
                .map_err(|e| crate::api::error::StorageError::ReadFailed(
                    format!("Failed to read metadata: {}", e)
                ))?;

            if path.is_dir() {
                stack.push(path);
            } else {
                total_size += metadata.len();
            }
        }
    }

    Ok(total_size)
}

/// 检测模型格式
async fn detect_model_format(path: &Path) -> Option<String> {
    let model_extensions = ["gguf", "safetensors", "onnx", "bin", "pt", "pth"];
    
    for ext in &model_extensions {
        let model_file = path.join(format!("model.{}", ext));
        if model_file.exists() {
            return Some(ext.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_filesystem_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileSystemStorage::new(temp_dir.path());
        
        // 测试确保基础路径存在
        storage.ensure_base_path().await.unwrap();
        assert!(storage.base_path().exists());
        
        // 测试模型不存在
        assert!(!storage.model_exists("nonexistent").await);
    }

    #[tokio::test]
    async fn test_model_info() {
        let info = ModelInfo {
            name: "test-model".to_string(),
            path: PathBuf::from("/test/path"),
            size: 1024,
            format: Some("gguf".to_string()),
        };
        
        assert_eq!(info.name, "test-model");
        assert_eq!(info.size, 1024);
    }
}
