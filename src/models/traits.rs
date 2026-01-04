//! 模型 Trait 定义
//! 
//! 定义模型的基础 trait，包括 Model、Inferable、Streamable 等。

use crate::Result;
use crate::api::request::InferenceOptions;
use crate::models::types::*;
use crate::models::metadata::{ModelMetadata, MemoryUsage};
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::pin::Pin;
use futures::Stream;

/// 基础模型 trait
/// 
/// 所有模型都必须实现这个 trait，提供基本的模型信息和管理功能。
#[async_trait]
pub trait Model: Send + Sync {
    /// 配置类型
    type Config: DeserializeOwned + Send + Sync;
    
    /// 模型名称
    fn name(&self) -> &str;
    
    /// 模型类型
    fn model_type(&self) -> ModelType;
    
    /// 检查模型是否已加载
    fn is_loaded(&self) -> bool;
    
    /// 获取模型元数据
    fn metadata(&self) -> &ModelMetadata;
    
    /// 加载模型
    async fn load(&mut self, config: Self::Config) -> Result<()>;
    
    /// 卸载模型
    async fn unload(&mut self) -> Result<()>;
    
    /// 获取当前内存使用
    fn memory_usage(&self) -> Result<MemoryUsage>;
}

/// 可推理模型 trait
/// 
/// 实现此 trait 的模型可以进行推理操作。
#[async_trait]
pub trait Inferable: Model {
    /// 输入类型
    type Input: Send + Sync;
    /// 输出类型
    type Output: Send + Sync;
    
    /// 单次推理
    async fn infer(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Self::Output>;
    
    /// 批处理推理
    async fn infer_batch(
        &self,
        inputs: Vec<Self::Input>,
        options: InferenceOptions,
    ) -> Result<Vec<Self::Output>>;
}

/// 流式推理 trait
/// 
/// 实现此 trait 的模型支持流式输出。
#[async_trait]
pub trait Streamable: Inferable {
    /// 流类型
    type Stream: Stream<Item = Result<Self::Output>> + Send;
    
    /// 流式推理
    async fn infer_stream(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Self::Stream>;
}

/// 可训练模型 trait（未来扩展）
/// 
/// 实现此 trait 的模型可以进行训练和微调。
#[async_trait]
pub trait Trainable: Model {
    /// 训练数据集类型
    type Dataset: Send + Sync;
    
    /// 训练模型
    async fn train(&mut self, dataset: Self::Dataset) -> Result<()>;
    
    /// 微调模型
    async fn fine_tune(&mut self, dataset: Self::Dataset) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::types::*;
    use crate::models::metadata::{ModelMetadata, ModelRequirements};

    // 测试模型类型枚举
    #[test]
    fn test_model_type() {
        let text_type = ModelType::Text(TextModelType::LanguageModel);
        let image_type = ModelType::Image(ImageModelType::Generation);
        
        assert!(matches!(text_type, ModelType::Text(_)));
        assert!(matches!(image_type, ModelType::Image(_)));
    }

    #[test]
    fn test_backend_type() {
        let candle = BackendType::Candle;
        let llama = BackendType::LlamaCpp;
        let external = BackendType::External(ExternalBridgeType::PyTorch);
        
        assert_eq!(candle, BackendType::Candle);
        assert_eq!(llama, BackendType::LlamaCpp);
        assert!(matches!(external, BackendType::External(_)));
    }

    #[test]
    fn test_accelerator_type() {
        let cpu = AcceleratorType::Cpu;
        let cuda = AcceleratorType::Cuda { device_id: 0 };
        let metal = AcceleratorType::Metal { device_id: 0 };
        
        assert!(matches!(cpu, AcceleratorType::Cpu));
        assert!(matches!(cuda, AcceleratorType::Cuda { device_id: 0 }));
        assert!(matches!(metal, AcceleratorType::Metal { device_id: 0 }));
    }

    #[test]
    fn test_model_format() {
        let gguf = ModelFormat::Gguf;
        let safetensors = ModelFormat::Safetensors;
        
        assert_eq!(gguf, ModelFormat::Gguf);
        assert_eq!(safetensors, ModelFormat::Safetensors);
    }
}
