//! 模型注册表
//! 
//! 提供统一的模型注册、查找和元数据管理功能。
//! 整合存储注册表和生命周期管理，提供高级查询接口。

use crate::Result;
use crate::models::{
    BaseModel, BaseModelConfig, ModelMetadata, ModelState, ModelType,
};
use crate::storage::registry::{ModelManifest, ModelRegistry as StorageRegistry};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 模型注册表
/// 
/// 统一管理模型的注册、查找和元数据。
/// 整合了存储注册表和运行时模型实例管理。
pub struct ModelRegistry {
    /// 存储注册表（持久化模型清单）
    storage_registry: Arc<RwLock<StorageRegistry>>,
    /// 运行时模型实例（已加载的模型）
    runtime_models: Arc<RwLock<HashMap<String, Arc<RwLock<BaseModel>>>>>,
    /// 模型元数据缓存
    metadata_cache: Arc<RwLock<HashMap<String, ModelMetadata>>>,
}

impl ModelRegistry {
    /// 创建新的模型注册表
    pub fn new(registry_path: impl AsRef<std::path::Path>) -> Self {
        Self {
            storage_registry: Arc::new(RwLock::new(StorageRegistry::new(registry_path))),
            runtime_models: Arc::new(RwLock::new(HashMap::new())),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 从文件加载注册表
    pub async fn load(&self) -> Result<()> {
        let mut storage = self.storage_registry.write().await;
        storage.load().await?;
        
        // 加载元数据缓存
        let mut cache = self.metadata_cache.write().await;
        for manifest in storage.list() {
            if let Some(metadata) = Self::manifest_to_metadata(manifest) {
                cache.insert(manifest.name.clone(), metadata);
            }
        }
        
        Ok(())
    }

    /// 注册模型
    /// 
    /// 将模型注册到注册表中，包括：
    /// - 保存模型清单到存储注册表
    /// - 缓存模型元数据
    pub async fn register(&self, manifest: ModelManifest) -> Result<()> {
        let name = manifest.name.clone();
        
        // 保存到存储注册表
        {
            let mut storage = self.storage_registry.write().await;
            storage.register(manifest.clone()).await?;
        }
        
        // 更新元数据缓存
        if let Some(metadata) = Self::manifest_to_metadata(&manifest) {
            let mut cache = self.metadata_cache.write().await;
            cache.insert(name.clone(), metadata);
        }
        
        tracing::info!("Registered model: {}", name);
        Ok(())
    }

    /// 注销模型
    /// 
    /// 从注册表中移除模型，包括：
    /// - 从存储注册表删除
    /// - 如果模型已加载，先卸载
    /// - 清除元数据缓存
    pub async fn unregister(&self, name: &str) -> Result<()> {
        // 如果模型已加载，先卸载
        {
            let runtime = self.runtime_models.read().await;
            if let Some(model) = runtime.get(name) {
                let mut model_guard = model.write().await;
                if model_guard.is_loaded().await {
                    model_guard.begin_unload().await?;
                    model_guard.finish_unload().await;
                }
            }
        }
        
        // 从运行时模型移除
        self.runtime_models.write().await.remove(name);
        
        // 从存储注册表删除
        {
            let mut storage = self.storage_registry.write().await;
            storage.remove(name).await?;
        }
        
        // 清除元数据缓存
        self.metadata_cache.write().await.remove(name);
        
        tracing::info!("Unregistered model: {}", name);
        Ok(())
    }

    /// 查找模型（按名称）
    /// 
    /// 返回模型的元数据和运行时状态。
    pub async fn find(&self, name: &str) -> Option<ModelInfo> {
        // 获取元数据
        let metadata = {
            let cache = self.metadata_cache.read().await;
            cache.get(name).cloned()
        };
        
        if metadata.is_none() {
            return None;
        }
        let metadata = metadata.unwrap();
        
        // 获取运行时状态
        let state = {
            let runtime = self.runtime_models.read().await;
            if let Some(model) = runtime.get(name) {
                let model_guard = model.read().await;
                Some(model_guard.state().await)
            } else {
                Some(ModelState::Uninitialized)
            }
        };
        
        Some(ModelInfo {
            metadata,
            state: state.unwrap_or(ModelState::Uninitialized),
        })
    }

    /// 查找模型（按名称，返回运行时实例）
    pub async fn find_runtime(&self, name: &str) -> Option<Arc<RwLock<BaseModel>>> {
        self.runtime_models.read().await.get(name).cloned()
    }

    /// 列出所有已注册的模型
    pub async fn list(&self) -> Vec<ModelInfo> {
        let cache = self.metadata_cache.read().await;
        let runtime = self.runtime_models.read().await;
        let mut models = Vec::new();
        
        for (name, metadata) in cache.iter() {
            let state = if let Some(model) = runtime.get(name) {
                let model_guard = model.read().await;
                model_guard.state().await
            } else {
                ModelState::Uninitialized
            };
            
            models.push(ModelInfo {
                metadata: metadata.clone(),
                state,
            });
        }
        
        models
    }

    /// 按类型查找模型
    pub async fn find_by_type(&self, model_type: ModelType) -> Vec<ModelInfo> {
        let all_models = self.list().await;
        all_models
            .into_iter()
            .filter(|info| info.metadata.model_type == model_type)
            .collect()
    }

    /// 按标签查找模型
    pub async fn find_by_tag(&self, tag: &str) -> Vec<ModelInfo> {
        let all_models = self.list().await;
        all_models
            .into_iter()
            .filter(|info| info.metadata.tags.contains(&tag.to_string()))
            .collect()
    }

    /// 搜索模型（按名称、标签、描述等）
    pub async fn search(&self, query: &str) -> Vec<ModelInfo> {
        let query_lower = query.to_lowercase();
        let all_models = self.list().await;
        
        all_models
            .into_iter()
            .filter(|info| {
                // 搜索名称
                info.metadata.name.to_lowercase().contains(&query_lower) ||
                // 搜索标签
                info.metadata.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lower)) ||
                // 搜索描述
                info.metadata.description.as_ref()
                    .map(|d| d.to_lowercase().contains(&query_lower))
                    .unwrap_or(false) ||
                // 搜索架构
                info.metadata.architecture.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// 注册运行时模型实例
    /// 
    /// 将已加载的模型实例注册到运行时注册表。
    pub async fn register_runtime(&self, name: String, model: BaseModel) {
        self.runtime_models.write().await.insert(name.clone(), Arc::new(RwLock::new(model)));
        tracing::debug!("Registered runtime model: {}", name);
    }

    /// 移除运行时模型实例
    pub async fn unregister_runtime(&self, name: &str) {
        self.runtime_models.write().await.remove(name);
        tracing::debug!("Unregistered runtime model: {}", name);
    }

    /// 获取模型元数据
    pub async fn get_metadata(&self, name: &str) -> Option<ModelMetadata> {
        let cache = self.metadata_cache.read().await;
        cache.get(name).cloned()
    }

    /// 更新模型元数据
    pub async fn update_metadata(&self, name: &str, metadata: ModelMetadata) -> Result<()> {
        // 更新缓存
        {
            let mut cache = self.metadata_cache.write().await;
            cache.insert(name.to_string(), metadata.clone());
        }
        
        // 更新存储注册表
        if let Some(manifest) = Self::metadata_to_manifest(&metadata) {
            let mut storage = self.storage_registry.write().await;
            storage.update(name, manifest).await?;
        }
        
        Ok(())
    }

    /// 检查模型是否已注册
    pub async fn is_registered(&self, name: &str) -> bool {
        let cache = self.metadata_cache.read().await;
        cache.contains_key(name)
    }

    /// 检查模型是否已加载
    pub async fn is_loaded(&self, name: &str) -> bool {
        let runtime = self.runtime_models.read().await;
        if let Some(model) = runtime.get(name) {
            let model_guard = model.read().await;
            model_guard.is_loaded().await
        } else {
            false
        }
    }

    /// 获取所有已加载的模型
    pub async fn get_loaded_models(&self) -> Vec<String> {
        let runtime = self.runtime_models.read().await;
        let mut loaded = Vec::new();
        
        for (name, model) in runtime.iter() {
            let model_guard = model.read().await;
            if model_guard.is_loaded().await {
                loaded.push(name.clone());
            }
        }
        
        loaded
    }

    /// 获取模型统计信息
    pub async fn get_stats(&self) -> RegistryStats {
        let cache = self.metadata_cache.read().await;
        let runtime = self.runtime_models.read().await;
        
        let total = cache.len();
        let loaded = runtime.values()
            .filter(|model| {
                let guard = model.try_read();
                guard.map(|m| m.is_loaded_sync()).unwrap_or(false)
            })
            .count();
        
        RegistryStats {
            total_models: total,
            loaded_models: loaded,
            registered_models: total,
        }
    }

    /// 将 ModelManifest 转换为 ModelMetadata
    fn manifest_to_metadata(manifest: &ModelManifest) -> Option<ModelMetadata> {
        // 解析模型类型
        let model_type = match manifest.model_type.as_str() {
            "text" => ModelType::Text(crate::models::types::TextModelType::LanguageModel),
            "image" => ModelType::Image(crate::models::types::ImageModelType::Generation),
            "audio" => ModelType::Audio(crate::models::types::AudioModelType::SpeechRecognition),
            "video" => ModelType::Video(crate::models::types::VideoModelType::Generation),
            "multimodal" => ModelType::Multimodal(crate::models::types::MultimodalModelType::VisionLanguage),
            _ => return None,
        };
        
        // 解析模型格式
        let format = match manifest.format.as_str() {
            "gguf" => crate::models::types::ModelFormat::Gguf,
            "safetensors" => crate::models::types::ModelFormat::Safetensors,
            "onnx" => crate::models::types::ModelFormat::Onnx,
            "pytorch" => crate::models::types::ModelFormat::PyTorch,
            "tensorflow" => crate::models::types::ModelFormat::TensorFlow,
            "tvm" => crate::models::types::ModelFormat::Tvm,
            "binary" => crate::models::types::ModelFormat::Binary,
            _ => crate::models::types::ModelFormat::Gguf,
        };
        
        // 解析量化类型
        let quantization = manifest.quantization.as_ref()
            .and_then(|q| match q.as_str() {
                "q4_0" => Some(crate::models::types::QuantizationType::Q4_0),
                "q4_1" => Some(crate::models::types::QuantizationType::Q4_1),
                "q5_0" => Some(crate::models::types::QuantizationType::Q5_0),
                "q5_1" => Some(crate::models::types::QuantizationType::Q5_1),
                "q8_0" => Some(crate::models::types::QuantizationType::Q8_0),
                "fp16" | "f16" => Some(crate::models::types::QuantizationType::Fp16),
                "fp32" | "f32" => Some(crate::models::types::QuantizationType::Fp32),
                "bf16" => Some(crate::models::types::QuantizationType::Bf16),
                "int8" => Some(crate::models::types::QuantizationType::Int8),
                "int4" => Some(crate::models::types::QuantizationType::Int4),
                _ => None,
            });
        
        Some(ModelMetadata {
            name: manifest.name.clone(),
            version: manifest.version.clone().unwrap_or_else(|| "1.0.0".to_string()),
            model_type,
            format,
            size: manifest.size,
            parameters: manifest.parameters,
            quantization,
            architecture: manifest.architecture.clone().unwrap_or_else(|| "unknown".to_string()),
            context_size: manifest.context_size,
            input_shapes: vec![],
            output_shapes: vec![],
            requirements: crate::models::metadata::ModelRequirements::default(),
            tags: manifest.tags.clone(),
            description: manifest.description.clone(),
            license: manifest.license.clone(),
            author: manifest.author.clone(),
            created_at: manifest.created_at,
            updated_at: manifest.updated_at,
        })
    }

    /// 将 ModelMetadata 转换为 ModelManifest
    fn metadata_to_manifest(metadata: &ModelMetadata) -> Option<ModelManifest> {
        let model_type = match metadata.model_type {
            ModelType::Text(_) => "text",
            ModelType::Image(_) => "image",
            ModelType::Audio(_) => "audio",
            ModelType::Video(_) => "video",
            ModelType::Multimodal(_) => "multimodal",
        };
        
        let format = match metadata.format {
            crate::models::types::ModelFormat::Gguf => "gguf",
            crate::models::types::ModelFormat::Safetensors => "safetensors",
            crate::models::types::ModelFormat::Onnx => "onnx",
            crate::models::types::ModelFormat::PyTorch => "pytorch",
            crate::models::types::ModelFormat::TensorFlow => "tensorflow",
            crate::models::types::ModelFormat::Tvm => "tvm",
            crate::models::types::ModelFormat::Binary => "binary",
            crate::models::types::ModelFormat::Unknown => "unknown",
        };
        
        let quantization = metadata.quantization.map(|q| match q {
            crate::models::types::QuantizationType::Q4_0 => "q4_0",
            crate::models::types::QuantizationType::Q4_1 => "q4_1",
            crate::models::types::QuantizationType::Q5_0 => "q5_0",
            crate::models::types::QuantizationType::Q5_1 => "q5_1",
            crate::models::types::QuantizationType::Q8_0 => "q8_0",
            crate::models::types::QuantizationType::Fp16 => "fp16",
            crate::models::types::QuantizationType::Fp32 => "fp32",
            crate::models::types::QuantizationType::Bf16 => "bf16",
            crate::models::types::QuantizationType::Int8 => "int8",
            crate::models::types::QuantizationType::Int4 => "int4",
        }.to_string());
        
        Some(ModelManifest {
            name: metadata.name.clone(),
            version: Some(metadata.version.clone()),
            model_type: model_type.to_string(),
            format: format.to_string(),
            file_path: PathBuf::new(), // 需要从其他地方获取
            size: metadata.size,
            parameters: metadata.parameters,
            quantization,
            architecture: Some(metadata.architecture.clone()),
            context_size: metadata.context_size,
            description: metadata.description.clone(),
            tags: metadata.tags.clone(),
            author: metadata.author.clone(),
            license: metadata.license.clone(),
            created_at: metadata.created_at,
            updated_at: metadata.updated_at,
            custom: std::collections::HashMap::new(),
        })
    }
}

/// 模型信息
/// 
/// 包含模型的元数据和运行时状态。
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// 模型元数据
    pub metadata: ModelMetadata,
    /// 运行时状态
    pub state: ModelState,
}

/// 注册表统计信息
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// 总模型数
    pub total_models: usize,
    /// 已加载模型数
    pub loaded_models: usize,
    /// 已注册模型数
    pub registered_models: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::types::*;
    use crate::models::metadata::ModelRequirements;
    use tempfile::TempDir;

    fn create_test_metadata() -> ModelMetadata {
        ModelMetadata {
            name: "test-model".to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::Text(TextModelType::LanguageModel),
            format: ModelFormat::Gguf,
            size: 1024 * 1024 * 1024,
            parameters: Some(7_000_000_000),
            quantization: Some(QuantizationType::Q4_0),
            architecture: "llama".to_string(),
            context_size: Some(4096),
            input_shapes: vec![],
            output_shapes: vec![],
            requirements: ModelRequirements::default(),
            tags: vec!["test".to_string(), "llm".to_string()],
            description: Some("Test model".to_string()),
            license: None,
            author: None,
            created_at: Some(chrono::Utc::now()),
            updated_at: Some(chrono::Utc::now()),
        }
    }

    #[tokio::test]
    async fn test_register_and_find() {
        let temp_dir = TempDir::new().unwrap();
        let registry_path = temp_dir.path().join("registry.toml");
        let registry = ModelRegistry::new(registry_path);
        
        // 创建模型清单
        let manifest = ModelManifest {
            name: "test-model".to_string(),
            version: Some("1.0.0".to_string()),
            model_type: "text".to_string(),
            format: "gguf".to_string(),
            file_path: PathBuf::from("test.gguf"),
            size: 1024 * 1024 * 1024,
            parameters: Some(7_000_000_000),
            quantization: Some("q4_0".to_string()),
            architecture: Some("llama".to_string()),
            context_size: Some(4096),
            description: Some("Test model".to_string()),
            tags: vec!["test".to_string()],
            author: None,
            license: None,
            created_at: Some(chrono::Utc::now()),
            updated_at: Some(chrono::Utc::now()),
            custom: std::collections::HashMap::new(),
        };
        
        // 注册模型
        registry.register(manifest).await.unwrap();
        
        // 查找模型
        let info = registry.find("test-model").await;
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.metadata.name, "test-model");
        assert_eq!(info.state, ModelState::Uninitialized);
    }

    #[tokio::test]
    async fn test_list_models() {
        let temp_dir = TempDir::new().unwrap();
        let registry_path = temp_dir.path().join("registry.toml");
        let registry = ModelRegistry::new(registry_path);
        
        // 注册多个模型
        let manifest1 = ModelManifest {
            name: "model1".to_string(),
            version: Some("1.0.0".to_string()),
            model_type: "text".to_string(),
            format: "gguf".to_string(),
            file_path: PathBuf::from("model1.gguf"),
            size: 1024,
            parameters: None,
            quantization: None,
            architecture: None,
            context_size: None,
            description: None,
            tags: vec![],
            author: None,
            license: None,
            created_at: Some(chrono::Utc::now()),
            updated_at: Some(chrono::Utc::now()),
            custom: std::collections::HashMap::new(),
        };
        
        let manifest2 = ModelManifest {
            name: "model2".to_string(),
            version: Some("1.0.0".to_string()),
            model_type: "image".to_string(),
            format: "safetensors".to_string(),
            file_path: PathBuf::from("model2.safetensors"),
            size: 2048,
            parameters: None,
            quantization: None,
            architecture: None,
            context_size: None,
            description: None,
            tags: vec![],
            author: None,
            license: None,
            created_at: Some(chrono::Utc::now()),
            updated_at: Some(chrono::Utc::now()),
            custom: std::collections::HashMap::new(),
        };
        
        registry.register(manifest1).await.unwrap();
        registry.register(manifest2).await.unwrap();
        
        // 列出所有模型
        let models = registry.list().await;
        assert_eq!(models.len(), 2);
    }

    #[tokio::test]
    async fn test_search_models() {
        let temp_dir = TempDir::new().unwrap();
        let registry_path = temp_dir.path().join("registry.toml");
        let registry = ModelRegistry::new(registry_path);
        
        let manifest = ModelManifest {
            name: "llama-7b".to_string(),
            version: Some("1.0.0".to_string()),
            model_type: "text".to_string(),
            format: "gguf".to_string(),
            file_path: PathBuf::from("llama.gguf"),
            size: 1024,
            parameters: Some(7_000_000_000),
            quantization: Some("q4_0".to_string()),
            architecture: Some("llama".to_string()),
            context_size: Some(4096),
            description: Some("Large language model".to_string()),
            tags: vec!["llm".to_string(), "text".to_string()],
            author: None,
            license: None,
            created_at: Some(chrono::Utc::now()),
            updated_at: Some(chrono::Utc::now()),
            custom: std::collections::HashMap::new(),
        };
        
        registry.register(manifest).await.unwrap();
        
        // 搜索模型
        let results = registry.search("llama").await;
        assert_eq!(results.len(), 1);
        
        let results = registry.search("language").await;
        assert_eq!(results.len(), 1);
        
        let results = registry.search("nonexistent").await;
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_unregister() {
        let temp_dir = TempDir::new().unwrap();
        let registry_path = temp_dir.path().join("registry.toml");
        let registry = ModelRegistry::new(registry_path);
        
        let manifest = ModelManifest {
            name: "test-model".to_string(),
            version: Some("1.0.0".to_string()),
            model_type: "text".to_string(),
            format: "gguf".to_string(),
            file_path: PathBuf::from("test.gguf"),
            size: 1024,
            parameters: None,
            quantization: None,
            architecture: None,
            context_size: None,
            description: None,
            tags: vec![],
            author: None,
            license: None,
            created_at: Some(chrono::Utc::now()),
            updated_at: Some(chrono::Utc::now()),
            custom: std::collections::HashMap::new(),
        };
        
        registry.register(manifest).await.unwrap();
        assert!(registry.is_registered("test-model").await);
        
        registry.unregister("test-model").await.unwrap();
        assert!(!registry.is_registered("test-model").await);
    }
}
