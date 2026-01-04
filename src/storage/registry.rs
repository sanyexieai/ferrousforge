//! 模型注册表
//! 
//! 管理模型清单（类似 Ollama 的 Modelfile），存储模型的元数据和配置。

use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;

/// 模型注册表
pub struct ModelRegistry {
    registry_path: PathBuf,
    models: HashMap<String, ModelManifest>,
}

/// 模型清单（类似 Modelfile）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// 模型名称
    pub name: String,
    
    /// 模型版本
    pub version: Option<String>,
    
    /// 模型类型
    pub model_type: String,
    
    /// 模型格式
    pub format: String,
    
    /// 模型文件路径（相对于存储根目录）
    pub file_path: PathBuf,
    
    /// 模型大小（字节）
    pub size: u64,
    
    /// 参数量
    pub parameters: Option<u64>,
    
    /// 量化类型
    pub quantization: Option<String>,
    
    /// 架构
    pub architecture: Option<String>,
    
    /// 上下文大小
    pub context_size: Option<usize>,
    
    /// 模型描述
    pub description: Option<String>,
    
    /// 标签
    pub tags: Vec<String>,
    
    /// 作者
    pub author: Option<String>,
    
    /// 许可证
    pub license: Option<String>,
    
    /// 创建时间
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    
    /// 更新时间
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    
    /// 自定义配置
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

impl ModelRegistry {
    /// 创建新的模型注册表
    pub fn new(registry_path: impl AsRef<Path>) -> Self {
        Self {
            registry_path: registry_path.as_ref().to_path_buf(),
            models: HashMap::new(),
        }
    }

    /// 从文件加载注册表
    pub async fn load(&mut self) -> Result<()> {
        // 确保注册表目录存在
        if let Some(parent) = self.registry_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| crate::api::error::StorageError::PathNotFound(
                    format!("Failed to create registry directory: {}", e)
                ))?;
        }

        // 如果注册表文件不存在，创建空注册表
        if !self.registry_path.exists() {
            self.save().await?;
            return Ok(());
        }

        // 读取注册表文件
        let content = tokio::fs::read_to_string(&self.registry_path)
            .await
            .map_err(|e| crate::api::error::StorageError::ReadFailed(
                format!("Failed to read registry file: {}", e)
            ))?;

        let registry_data: RegistryData = toml::from_str(&content)
            .map_err(|e| crate::api::error::StorageError::InvalidFile(
                format!("Failed to parse registry file: {}", e)
            ))?;

        self.models = registry_data.models;

        Ok(())
    }

    /// 保存注册表到文件
    pub async fn save(&self) -> Result<()> {
        let registry_data = RegistryData {
            version: "1.0".to_string(),
            models: self.models.clone(),
        };

        let content = toml::to_string_pretty(&registry_data)
            .map_err(|e| crate::api::error::StorageError::InvalidFile(
                format!("Failed to serialize registry: {}", e)
            ))?;

        tokio::fs::write(&self.registry_path, content)
            .await
            .map_err(|e| crate::api::error::StorageError::ReadFailed(
                format!("Failed to write registry file: {}", e)
            ))?;

        Ok(())
    }

    /// 注册模型
    pub async fn register(&mut self, manifest: ModelManifest) -> Result<()> {
        let name = manifest.name.clone();
        self.models.insert(name.clone(), manifest);
        self.save().await?;
        Ok(())
    }

    /// 获取模型清单
    pub fn get(&self, name: &str) -> Option<&ModelManifest> {
        self.models.get(name)
    }

    /// 列出所有模型
    pub fn list(&self) -> Vec<&ModelManifest> {
        self.models.values().collect()
    }

    /// 删除模型注册
    pub async fn remove(&mut self, name: &str) -> Result<()> {
        if self.models.remove(name).is_some() {
            self.save().await?;
        }
        Ok(())
    }

    /// 检查模型是否已注册
    pub fn is_registered(&self, name: &str) -> bool {
        self.models.contains_key(name)
    }

    /// 更新模型清单
    pub async fn update(&mut self, name: &str, mut manifest: ModelManifest) -> Result<()> {
        manifest.name = name.to_string();
        manifest.updated_at = Some(chrono::Utc::now());
        self.models.insert(name.to_string(), manifest);
        self.save().await?;
        Ok(())
    }
}

/// 注册表数据（用于序列化）
#[derive(Debug, Serialize, Deserialize)]
struct RegistryData {
    version: String,
    models: HashMap<String, ModelManifest>,
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: None,
            model_type: "text".to_string(),
            format: "gguf".to_string(),
            file_path: PathBuf::new(),
            size: 0,
            parameters: None,
            quantization: None,
            architecture: None,
            context_size: None,
            description: None,
            tags: Vec::new(),
            author: None,
            license: None,
            created_at: Some(chrono::Utc::now()),
            updated_at: Some(chrono::Utc::now()),
            custom: HashMap::new(),
        }
    }
}
