//! 推理后端抽象
//! 
//! 定义推理后端的统一接口，支持多种后端实现（Candle、llama.cpp、ONNX 等）。

use crate::Result;
use crate::api::request::InferenceOptions;
use crate::models::types::*;
use crate::models::metadata::ModelMetadata;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use futures::Stream;

/// 推理后端 trait
/// 
/// 所有推理后端都必须实现这个 trait，提供统一的模型加载和推理接口。
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// 模型句柄类型
    /// 
    /// 每个后端可以定义自己的模型句柄类型，用于标识已加载的模型。
    type ModelHandle: Send + Sync;
    
    /// 后端名称
    fn name(&self) -> &str;
    
    /// 后端类型
    fn backend_type(&self) -> BackendType;
    
    /// 支持的模型格式
    fn supported_formats(&self) -> &[ModelFormat];
    
    /// 支持的硬件加速器
    fn supported_accelerators(&self) -> &[AcceleratorType];
    
    /// 检查是否支持指定的模型格式
    fn supports_format(&self, format: ModelFormat) -> bool {
        self.supported_formats().contains(&format)
    }
    
    /// 检查是否支持指定的硬件加速器
    fn supports_accelerator(&self, accelerator: AcceleratorType) -> bool {
        self.supported_accelerators().contains(&accelerator)
    }
    
    /// 加载模型
    /// 
    /// # 参数
    /// - `path`: 模型文件路径
    /// - `config`: 后端配置
    /// - `accelerator`: 可选的硬件加速器
    /// 
    /// # 返回
    /// 返回模型句柄，用于后续的推理操作
    async fn load_model(
        &self,
        path: &Path,
        config: BackendConfig,
        accelerator: Option<AcceleratorType>,
    ) -> Result<Self::ModelHandle>;
    
    /// 卸载模型
    /// 
    /// # 参数
    /// - `handle`: 模型句柄
    async fn unload_model(&self, handle: Self::ModelHandle) -> Result<()>;
    
    /// 单次推理
    /// 
    /// # 参数
    /// - `handle`: 模型句柄
    /// - `input`: 输入数据
    /// - `options`: 推理选项
    /// 
    /// # 返回
    /// 推理结果
    async fn infer(
        &self,
        handle: &Self::ModelHandle,
        input: BackendInput,
        options: InferenceOptions,
    ) -> Result<BackendOutput>;
    
    /// 批处理推理
    /// 
    /// # 参数
    /// - `handle`: 模型句柄
    /// - `inputs`: 输入数据列表
    /// - `options`: 推理选项
    /// 
    /// # 返回
    /// 推理结果列表
    async fn infer_batch(
        &self,
        handle: &Self::ModelHandle,
        inputs: Vec<BackendInput>,
        options: InferenceOptions,
    ) -> Result<Vec<BackendOutput>>;
    
    /// 流式推理（可选）
    /// 
    /// 如果后端不支持流式推理，可以返回 `None`。
    /// 
    /// # 参数
    /// - `handle`: 模型句柄
    /// - `input`: 输入数据
    /// - `options`: 推理选项
    /// 
    /// # 返回
    /// 流式输出，如果后端不支持则返回 `None`
    async fn infer_stream(
        &self,
        handle: &Self::ModelHandle,
        input: BackendInput,
        options: InferenceOptions,
    ) -> Result<Option<Pin<Box<dyn Stream<Item = Result<BackendOutput>> + Send>>>> {
        // 默认实现：不支持流式推理
        let _ = (handle, input, options);
        Ok(None)
    }
    
    /// 获取硬件加速器信息
    /// 
    /// # 参数
    /// - `accelerator`: 硬件加速器类型
    /// 
    /// # 返回
    /// 硬件加速器信息，如果后端不支持该加速器则返回错误
    fn get_accelerator_info(&self, accelerator: AcceleratorType) -> Result<AcceleratorInfo>;
    
    /// 获取模型元数据
    /// 
    /// 在加载模型后，可以获取模型的元数据信息。
    /// 
    /// # 参数
    /// - `handle`: 模型句柄
    /// 
    /// # 返回
    /// 模型元数据
    async fn get_model_metadata(
        &self,
        handle: &Self::ModelHandle,
    ) -> Result<ModelMetadata>;
    
    /// 检查模型是否已加载
    /// 
    /// # 参数
    /// - `handle`: 模型句柄
    /// 
    /// # 返回
    /// 如果模型已加载返回 `true`，否则返回 `false`
    fn is_model_loaded(&self, handle: &Self::ModelHandle) -> bool;
}

/// 后端配置
/// 
/// 用于配置推理后端的各种参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// 线程数（用于 CPU 推理）
    pub num_threads: Option<usize>,
    
    /// 批处理大小
    pub batch_size: Option<usize>,
    
    /// 上下文大小（对于文本模型）
    pub context_size: Option<usize>,
    
    /// GPU 设备 ID（对于 GPU 加速）
    pub device_id: Option<u32>,
    
    /// 内存限制（字节）
    pub memory_limit: Option<u64>,
    
    /// 是否使用量化
    pub use_quantization: Option<bool>,
    
    /// 量化类型
    pub quantization: Option<QuantizationType>,
    
    /// 自定义配置项
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            num_threads: None,
            batch_size: Some(1),
            context_size: None,
            device_id: None,
            memory_limit: None,
            use_quantization: None,
            quantization: None,
            custom: HashMap::new(),
        }
    }
}

/// 后端输入
/// 
/// 统一的输入数据格式，支持多种数据类型。
#[derive(Debug, Clone)]
pub enum BackendInput {
    /// 文本输入
    Text(String),
    /// 文本列表（批处理）
    TextBatch(Vec<String>),
    /// 图像数据（字节）
    Image(Vec<u8>),
    /// 图像列表（批处理）
    ImageBatch(Vec<Vec<u8>>),
    /// 音频数据（字节）
    Audio(Vec<u8>),
    /// 音频列表（批处理）
    AudioBatch(Vec<Vec<u8>>),
    /// 张量数据
    Tensor(TensorData),
    /// 张量列表（批处理）
    TensorBatch(Vec<TensorData>),
    /// 多模态输入
    Multimodal(MultimodalInput),
}

/// 后端输出
/// 
/// 统一的输出数据格式，支持多种数据类型。
#[derive(Debug, Clone)]
pub enum BackendOutput {
    /// 文本输出
    Text(String),
    /// 文本列表（批处理）
    TextBatch(Vec<String>),
    /// 图像数据（字节）
    Image(Vec<u8>),
    /// 图像列表（批处理）
    ImageBatch(Vec<Vec<u8>>),
    /// 音频数据（字节）
    Audio(Vec<u8>),
    /// 音频列表（批处理）
    AudioBatch(Vec<Vec<u8>>),
    /// 张量数据
    Tensor(TensorData),
    /// 张量列表（批处理）
    TensorBatch(Vec<TensorData>),
    /// 嵌入向量
    Embedding(Vec<f32>),
    /// 嵌入向量列表（批处理）
    EmbeddingBatch(Vec<Vec<f32>>),
    /// 多模态输出
    Multimodal(MultimodalOutput),
}

/// 张量数据
#[derive(Debug, Clone)]
pub struct TensorData {
    /// 形状
    pub shape: Vec<usize>,
    /// 数据类型
    pub dtype: TensorDtype,
    /// 数据（扁平化）
    pub data: Vec<u8>,
}

/// 张量数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    /// 32位浮点数
    F32,
    /// 16位浮点数
    F16,
    /// 8位整数
    I8,
    /// 32位整数
    I32,
    /// 64位整数
    I64,
    /// 无符号8位整数
    U8,
}

/// 多模态输入
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    /// 文本部分
    pub text: Option<String>,
    /// 图像部分
    pub image: Option<Vec<u8>>,
    /// 音频部分
    pub audio: Option<Vec<u8>>,
    /// 其他数据
    pub other: HashMap<String, Vec<u8>>,
}

/// 多模态输出
#[derive(Debug, Clone)]
pub struct MultimodalOutput {
    /// 文本部分
    pub text: Option<String>,
    /// 图像部分
    pub image: Option<Vec<u8>>,
    /// 音频部分
    pub audio: Option<Vec<u8>>,
    /// 其他数据
    pub other: HashMap<String, Vec<u8>>,
}

/// 硬件加速器信息
#[derive(Debug, Clone)]
pub struct AcceleratorInfo {
    /// 加速器类型
    pub accelerator_type: AcceleratorType,
    /// 设备名称
    pub device_name: String,
    /// 总内存（字节）
    pub total_memory: Option<u64>,
    /// 可用内存（字节）
    pub available_memory: Option<u64>,
    /// 计算能力（对于 GPU）
    pub compute_capability: Option<String>,
    /// 是否可用
    pub available: bool,
    /// 其他信息
    pub metadata: HashMap<String, String>,
}

/// 后端信息
/// 
/// 存储后端的元数据信息，用于注册表管理。
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// 后端类型
    pub backend_type: BackendType,
    /// 后端名称
    pub name: String,
    /// 支持的模型格式
    pub supported_formats: Vec<ModelFormat>,
    /// 支持的硬件加速器
    pub supported_accelerators: Vec<AcceleratorType>,
}

/// 后端注册表
/// 
/// 管理所有可用的推理后端，提供后端注册和查找功能。
/// 
/// 注意：由于 Rust 的类型系统限制，我们不能直接将不同类型的 `ModelHandle`
/// 统一到一个 trait object 中。因此，注册表只存储后端的元数据信息。
/// 实际的后端实例应该由调用者管理，注册表只提供查找和匹配功能。
pub struct BackendRegistry {
    backends: Arc<RwLock<HashMap<BackendType, BackendInfo>>>,
}

impl BackendRegistry {
    /// 创建新的后端注册表
    pub fn new() -> Self {
        Self {
            backends: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 注册后端
    /// 
    /// # 参数
    /// - `backend_type`: 后端类型
    /// - `name`: 后端名称
    /// - `supported_formats`: 支持的模型格式
    /// - `supported_accelerators`: 支持的硬件加速器
    pub async fn register(
        &self,
        backend_type: BackendType,
        name: String,
        supported_formats: Vec<ModelFormat>,
        supported_accelerators: Vec<AcceleratorType>,
    ) {
        let backend_type_clone = backend_type.clone();
        let name_clone = name.clone();
        
        let info = BackendInfo {
            backend_type: backend_type_clone.clone(),
            name: name_clone.clone(),
            supported_formats,
            supported_accelerators,
        };
        
        self.backends.write().await.insert(backend_type_clone.clone(), info);
        tracing::info!("Registered inference backend: {} ({:?})", name_clone, backend_type_clone);
    }
    
    /// 注册后端（从后端实现自动提取信息）
    /// 
    /// # 参数
    /// - `backend`: 后端实现
    pub async fn register_backend<B>(&self, backend: &B)
    where
        B: InferenceBackend,
    {
        let backend_type = backend.backend_type();
        let name = backend.name().to_string();
        let supported_formats = backend.supported_formats().to_vec();
        let supported_accelerators = backend.supported_accelerators().to_vec();
        
        self.register(backend_type, name, supported_formats, supported_accelerators).await;
    }
    
    /// 获取后端信息
    /// 
    /// # 参数
    /// - `backend_type`: 后端类型
    /// 
    /// # 返回
    /// 后端信息，如果未找到则返回 `None`
    pub async fn get_info(&self, backend_type: &BackendType) -> Option<BackendInfo> {
        let backends = self.backends.read().await;
        backends.get(backend_type).cloned()
    }
    
    /// 列出所有已注册的后端
    pub async fn list(&self) -> Vec<BackendType> {
        let backends = self.backends.read().await;
        backends.keys().map(|k| k.clone()).collect()
    }
    
    /// 列出所有已注册的后端信息
    pub async fn list_info(&self) -> Vec<BackendInfo> {
        let backends = self.backends.read().await;
        backends.values().cloned().collect()
    }
    
    /// 根据模型格式查找支持的后端
    pub async fn find_by_format(&self, format: ModelFormat) -> Vec<BackendType> {
        let backends = self.backends.read().await;
        backends
            .iter()
            .filter(|(_, info)| info.supported_formats.contains(&format))
            .map(|(backend_type, _)| backend_type.clone())
            .collect()
    }
    
    /// 根据硬件加速器查找支持的后端
    pub async fn find_by_accelerator(&self, accelerator: AcceleratorType) -> Vec<BackendType> {
        let backends = self.backends.read().await;
        backends
            .iter()
            .filter(|(_, info)| info.supported_accelerators.contains(&accelerator))
            .map(|(backend_type, _)| backend_type.clone())
            .collect()
    }
    
    /// 检查后端是否已注册
    pub async fn is_registered(&self, backend_type: &BackendType) -> bool {
        let backends = self.backends.read().await;
        backends.contains_key(backend_type)
    }
    
    /// 移除后端注册
    pub async fn unregister(&self, backend_type: &BackendType) {
        self.backends.write().await.remove(backend_type);
        tracing::info!("Unregistered inference backend: {:?}", backend_type);
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::types::*;
    use crate::models::metadata::{ModelMetadata, ModelRequirements};
    
    // 测试后端配置
    #[test]
    fn test_backend_config() {
        let config = BackendConfig::default();
        assert_eq!(config.batch_size, Some(1));
        assert_eq!(config.num_threads, None);
    }
    
    // 测试后端输入/输出类型
    #[test]
    fn test_backend_input_output() {
        let text_input = BackendInput::Text("Hello".to_string());
        match text_input {
            BackendInput::Text(s) => assert_eq!(s, "Hello"),
            _ => panic!("Expected Text input"),
        }
        
        let text_output = BackendOutput::Text("World".to_string());
        match text_output {
            BackendOutput::Text(s) => assert_eq!(s, "World"),
            _ => panic!("Expected Text output"),
        }
    }
    
    // 测试张量数据
    #[test]
    fn test_tensor_data() {
        let tensor = TensorData {
            shape: vec![2, 3],
            dtype: TensorDtype::F32,
            data: vec![0; 24], // 2 * 3 * 4 bytes (f32)
        };
        
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.dtype, TensorDtype::F32);
    }
    
    // 测试后端注册表
    #[tokio::test]
    async fn test_backend_registry() {
        let registry = BackendRegistry::new();
        
        // 检查初始状态
        let backends = registry.list().await;
        assert_eq!(backends.len(), 0);
        
        // 检查是否已注册
        assert!(!registry.is_registered(&BackendType::Candle).await);
    }
}
