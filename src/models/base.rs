//! 基础模型实现
//! 
//! 提供通用的模型基础实现，包括：
//! - 通用模型元数据管理
//! - 模型生命周期管理
//! - 模型状态管理

use crate::Result;
use crate::models::metadata::{ModelMetadata, MemoryUsage};
use crate::models::types::ModelType;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

/// 模型状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    /// 未初始化
    Uninitialized,
    /// 加载中
    Loading,
    /// 已加载
    Loaded,
    /// 卸载中
    Unloading,
    /// 错误状态
    Error(String),
}

/// 基础模型配置
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct BaseModelConfig {
    /// 模型名称
    pub name: String,
    /// 模型路径
    pub path: Option<String>,
    /// 自动卸载超时（秒），None 表示不自动卸载
    pub auto_unload_timeout: Option<u64>,
}

/// 基础模型
/// 
/// 提供所有模型的通用功能，包括：
/// - 元数据管理
/// - 生命周期管理
/// - 状态跟踪
/// - 内存使用跟踪
pub struct BaseModel {
    /// 模型元数据
    metadata: ModelMetadata,
    /// 模型状态
    state: Arc<RwLock<ModelState>>,
    /// 加载时间
    loaded_at: Arc<RwLock<Option<Instant>>>,
    /// 最后使用时间
    last_used: Arc<RwLock<Option<Instant>>>,
    /// 内存使用
    memory_usage: Arc<RwLock<MemoryUsage>>,
    /// 配置
    config: BaseModelConfig,
    /// 自动卸载任务句柄（用于取消自动卸载）
    #[allow(dead_code)]
    auto_unload_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl BaseModel {
    /// 创建新的基础模型
    pub fn new(metadata: ModelMetadata, config: BaseModelConfig) -> Self {
        Self {
            metadata,
            state: Arc::new(RwLock::new(ModelState::Uninitialized)),
            loaded_at: Arc::new(RwLock::new(None)),
            last_used: Arc::new(RwLock::new(None)),
            memory_usage: Arc::new(RwLock::new(MemoryUsage {
                used: 0,
                peak: 0,
                vram_used: None,
                vram_peak: None,
            })),
            config,
            auto_unload_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// 从元数据创建
    pub fn from_metadata(metadata: ModelMetadata) -> Self {
        let config = BaseModelConfig {
            name: metadata.name.clone(),
            path: None,
            auto_unload_timeout: None,
        };
        Self::new(metadata, config)
    }

    /// 获取模型状态
    pub async fn state(&self) -> ModelState {
        self.state.read().await.clone()
    }

    /// 设置模型状态
    pub async fn set_state(&self, state: ModelState) {
        *self.state.write().await = state;
    }

    /// 更新最后使用时间
    pub async fn update_last_used(&self) {
        *self.last_used.write().await = Some(Instant::now());
    }

    /// 获取最后使用时间
    pub async fn last_used(&self) -> Option<Instant> {
        *self.last_used.read().await
    }

    /// 获取加载时间
    pub async fn loaded_at(&self) -> Option<Instant> {
        *self.loaded_at.read().await
    }

    /// 更新内存使用
    pub async fn update_memory_usage(&self, usage: MemoryUsage) {
        let mut current = self.memory_usage.write().await;
        current.used = usage.used;
        current.peak = current.peak.max(usage.used);
        if let Some(vram) = usage.vram_used {
            current.vram_used = Some(vram);
            if let Some(peak) = current.vram_peak {
                current.vram_peak = Some(peak.max(vram));
            } else {
                current.vram_peak = Some(vram);
            }
        }
    }

    /// 获取配置
    pub fn config(&self) -> &BaseModelConfig {
        &self.config
    }

    /// 设置自动卸载
    pub async fn set_auto_unload(&self, timeout: Duration) {
        // 取消之前的自动卸载任务
        if let Some(handle) = self.auto_unload_handle.write().await.take() {
            handle.abort();
        }

        let state = Arc::clone(&self.state);
        let auto_unload_handle = Arc::clone(&self.auto_unload_handle);
        
        let handle = tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            
            // 检查状态，如果仍然是 Loaded 且没有使用，则卸载
            let current_state = state.read().await.clone();
            if current_state == ModelState::Loaded {
                tracing::info!("Auto-unloading model after timeout");
                // 这里应该调用实际的卸载逻辑
                // 目前只是设置状态
                *state.write().await = ModelState::Unloading;
            }
        });

        *auto_unload_handle.write().await = Some(handle);
    }

    /// 取消自动卸载
    pub async fn cancel_auto_unload(&self) {
        if let Some(handle) = self.auto_unload_handle.write().await.take() {
            handle.abort();
        }
    }

    /// 检查是否应该自动卸载
    pub async fn should_auto_unload(&self) -> bool {
        if let Some(timeout) = self.config.auto_unload_timeout {
            if let Some(last_used) = *self.last_used.read().await {
                let elapsed = last_used.elapsed();
                return elapsed.as_secs() >= timeout;
            }
        }
        false
    }

    /// 标记为已加载
    pub async fn mark_loaded(&self) {
        *self.loaded_at.write().await = Some(Instant::now());
        *self.last_used.write().await = Some(Instant::now());
        self.set_state(ModelState::Loaded).await;
        
        // 如果配置了自动卸载，启动定时器
        if let Some(timeout_secs) = self.config.auto_unload_timeout {
            self.set_auto_unload(Duration::from_secs(timeout_secs)).await;
        }
    }

    /// 标记为已卸载
    pub async fn mark_unloaded(&self) {
        self.cancel_auto_unload().await;
        *self.loaded_at.write().await = None;
        *self.last_used.write().await = None;
        self.set_state(ModelState::Uninitialized).await;
        
        // 重置内存使用
        *self.memory_usage.write().await = MemoryUsage {
            used: 0,
            peak: 0,
            vram_used: None,
            vram_peak: None,
        };
    }

    /// 标记为错误状态
    pub async fn mark_error(&self, error: String) {
        self.cancel_auto_unload().await;
        self.set_state(ModelState::Error(error)).await;
    }

    /// 获取模型名称
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// 获取模型类型
    pub fn model_type(&self) -> ModelType {
        self.metadata.model_type
    }

    /// 获取模型元数据
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// 检查模型是否已加载
    pub async fn is_loaded(&self) -> bool {
        matches!(self.state().await, ModelState::Loaded)
    }

    /// 同步检查模型是否已加载（使用 try_read）
    pub fn is_loaded_sync(&self) -> bool {
        matches!(
            self.state.try_read().map(|s| s.clone()).unwrap_or(ModelState::Uninitialized),
            ModelState::Loaded
        )
    }

    /// 获取内存使用（同步版本）
    pub fn memory_usage_sync(&self) -> MemoryUsage {
        self.memory_usage.try_read()
            .map(|u| u.clone())
            .unwrap_or_else(|_| MemoryUsage {
                used: 0,
                peak: 0,
                vram_used: None,
                vram_peak: None,
            })
    }

    /// 获取内存使用（异步版本）
    pub async fn memory_usage_async(&self) -> MemoryUsage {
        self.memory_usage.read().await.clone()
    }

    /// 开始加载流程（由具体实现调用）
    pub async fn begin_load(&self) -> Result<()> {
        let current_state = self.state().await;
        if current_state == ModelState::Loaded {
            return Err(crate::api::error::ModelError::AlreadyLoaded(
                self.metadata.name.clone()
            ).into());
        }

        if current_state == ModelState::Loading {
            return Err(crate::api::error::ModelError::AlreadyLoaded(
                "Model is already loading".to_string()
            ).into());
        }

        self.set_state(ModelState::Loading).await;
        Ok(())
    }

    /// 完成加载流程（由具体实现调用）
    pub async fn finish_load(&self) {
        self.mark_loaded().await;
    }

    /// 开始卸载流程（由具体实现调用）
    pub async fn begin_unload(&self) -> Result<()> {
        let current_state = self.state().await;
        if current_state != ModelState::Loaded {
            return Err(crate::api::error::ModelError::NotLoaded(
                self.metadata.name.clone()
            ).into());
        }

        self.set_state(ModelState::Unloading).await;
        Ok(())
    }

    /// 完成卸载流程（由具体实现调用）
    pub async fn finish_unload(&self) {
        self.mark_unloaded().await;
    }
}

// BaseModel 不直接实现 Model trait
// 它提供通用的状态管理和生命周期功能
// 具体的模型实现可以包含 BaseModel 并使用其功能

/// 模型生命周期管理器
/// 
/// 管理多个模型的生命周期，提供统一的加载、卸载和状态查询接口。
pub struct ModelLifecycleManager {
    models: Arc<RwLock<std::collections::HashMap<String, Arc<RwLock<BaseModel>>>>>,
}

impl ModelLifecycleManager {
    /// 创建新的生命周期管理器
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// 注册模型
    pub async fn register(&self, name: String, model: BaseModel) {
        self.models.write().await.insert(name, Arc::new(RwLock::new(model)));
    }

    /// 获取模型
    pub async fn get(&self, name: &str) -> Option<Arc<RwLock<BaseModel>>> {
        self.models.read().await.get(name).cloned()
    }

    /// 列出所有模型
    pub async fn list(&self) -> Vec<String> {
        self.models.read().await.keys().cloned().collect()
    }

    /// 移除模型
    pub async fn remove(&self, name: &str) -> Option<Arc<RwLock<BaseModel>>> {
        self.models.write().await.remove(name)
    }

    /// 获取所有模型的状态
    pub async fn get_all_states(&self) -> std::collections::HashMap<String, ModelState> {
        let mut states = std::collections::HashMap::new();
        let models = self.models.read().await;
        
        for (name, model) in models.iter() {
            let model_guard = model.read().await;
            let state = model_guard.state().await;
            states.insert(name.clone(), state);
        }
        
        states
    }

    /// 清理未使用的模型
    pub async fn cleanup_unused(&self, _timeout: Duration) {
        let models = self.models.read().await;
        let mut to_unload = Vec::new();

        for (name, model) in models.iter() {
            let model_guard = model.read().await;
            if model_guard.should_auto_unload().await {
                to_unload.push(name.clone());
            }
        }

        drop(models);

        for name in to_unload {
            if let Some(model) = self.get(&name).await {
                {
                    let model_guard = model.write().await;
                    if let Err(e) = model_guard.begin_unload().await {
                        tracing::warn!("Failed to begin unloading model {}: {}", name, e);
                        continue;
                    }
                }
                // 释放锁后调用 finish_unload
                {
                    let model_guard = model.write().await;
                    model_guard.finish_unload().await;
                }
            }
        }
    }
}

impl Default for ModelLifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::types::*;
    use crate::models::metadata::ModelRequirements;

    fn create_test_metadata() -> ModelMetadata {
        ModelMetadata {
            name: "test-model".to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::Text(TextModelType::LanguageModel),
            format: ModelFormat::Gguf,
            size: 1024 * 1024 * 1024, // 1GB
            parameters: Some(7_000_000_000),
            quantization: Some(QuantizationType::Q4_0),
            architecture: "llama".to_string(),
            context_size: Some(4096),
            input_shapes: vec![],
            output_shapes: vec![],
            requirements: ModelRequirements::default(),
            tags: vec!["test".to_string()],
            description: Some("Test model".to_string()),
            license: None,
            author: None,
            created_at: Some(chrono::Utc::now()),
            updated_at: Some(chrono::Utc::now()),
        }
    }

    #[tokio::test]
    async fn test_base_model_creation() {
        let metadata = create_test_metadata();
        let config = BaseModelConfig {
            name: "test-model".to_string(),
            path: None,
            auto_unload_timeout: None,
        };
        let model = BaseModel::new(metadata, config);
        
        assert_eq!(model.name(), "test-model");
        assert!(!model.is_loaded().await);
    }

    #[tokio::test]
    async fn test_model_state_management() {
        let metadata = create_test_metadata();
        let config = BaseModelConfig {
            name: "test-model".to_string(),
            path: None,
            auto_unload_timeout: None,
        };
        let model = BaseModel::new(metadata, config);
        
        // 初始状态应该是 Uninitialized
        assert_eq!(model.state().await, ModelState::Uninitialized);
        
        // 开始加载模型
        model.begin_load().await.unwrap();
        
        // 状态应该是 Loading
        assert_eq!(model.state().await, ModelState::Loading);
        
        // 完成加载
        model.finish_load().await;
        
        // 状态应该是 Loaded
        assert_eq!(model.state().await, ModelState::Loaded);
        assert!(model.is_loaded().await);
        
        // 开始卸载模型
        model.begin_unload().await.unwrap();
        
        // 状态应该是 Unloading
        assert_eq!(model.state().await, ModelState::Unloading);
        
        // 完成卸载
        model.finish_unload().await;
        
        // 状态应该是 Uninitialized
        assert_eq!(model.state().await, ModelState::Uninitialized);
        assert!(!model.is_loaded().await);
    }

    #[tokio::test]
    async fn test_model_lifecycle_manager() {
        let manager = ModelLifecycleManager::new();
        
        let metadata = create_test_metadata();
        let config = BaseModelConfig {
            name: "test-model".to_string(),
            path: None,
            auto_unload_timeout: None,
        };
        let model = BaseModel::new(metadata, config);
        
        manager.register("test-model".to_string(), model).await;
        
        let model = manager.get("test-model").await;
        assert!(model.is_some());
        
        let list = manager.list().await;
        assert!(list.contains(&"test-model".to_string()));
    }

    #[tokio::test]
    async fn test_memory_usage_tracking() {
        let metadata = create_test_metadata();
        let config = BaseModelConfig {
            name: "test-model".to_string(),
            path: None,
            auto_unload_timeout: None,
        };
        let model = BaseModel::new(metadata, config);
        
        let usage = MemoryUsage {
            used: 1024 * 1024 * 1024, // 1GB
            peak: 1024 * 1024 * 1024,
            vram_used: Some(512 * 1024 * 1024), // 512MB
            vram_peak: Some(512 * 1024 * 1024),
        };
        
        model.update_memory_usage(usage).await;
        
        let current_usage = model.memory_usage_async().await;
        assert_eq!(current_usage.used, 1024 * 1024 * 1024);
        assert_eq!(current_usage.peak, 1024 * 1024 * 1024);
        assert_eq!(current_usage.vram_used, Some(512 * 1024 * 1024));
    }
}
