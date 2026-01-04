//! Mold 统一模型接口
//! 
//! Mold 是 FerrousForge 的核心抽象，提供统一的模型接口。
//! 所有模型类型都通过 Mold 接口暴露，无论底层使用什么后端。

use crate::Result;
use crate::api::request::InferenceOptions;
use crate::models::types::*;
use crate::models::metadata::ModelMetadata;
use crate::models::traits::{Model, Inferable, Streamable};
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::pin::Pin;
use futures::Stream;

/// 统一模型接口 "Mold"
/// 
/// Mold 是所有模型的统一接口，无论使用什么后端（Candle、llama.cpp、ONNX 等），
/// 都通过 Mold 接口暴露。
#[async_trait]
pub trait Mold: Send + Sync {
    /// 输入类型
    type Input: Send + Sync;
    /// 输出类型
    type Output: Send + Sync;
    /// 配置类型
    type Config: DeserializeOwned + Send + Sync;
    
    // ========== 模型标识 ==========
    
    /// 模型名称
    fn name(&self) -> &str;
    
    /// 模型类型（文本、图像、音频等）
    fn model_type(&self) -> ModelType;
    
    /// 后端类型（Candle、LlamaCpp 等）
    fn backend_type(&self) -> BackendType;
    
    /// 当前使用的硬件加速器
    fn hardware_accelerator(&self) -> AcceleratorType;
    
    // ========== 生命周期 ==========
    
    /// 加载模型
    async fn load(&mut self, config: Self::Config) -> Result<()>;
    
    /// 卸载模型
    async fn unload(&mut self) -> Result<()>;
    
    /// 检查模型是否已加载
    fn is_loaded(&self) -> bool;
    
    // ========== 元数据 ==========
    
    /// 获取模型元数据
    fn metadata(&self) -> &ModelMetadata;
    
    // ========== 推理接口 ==========
    
    /// 单次推理
    async fn infer(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Self::Output>;
    
    /// 流式推理
    async fn infer_stream(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output>> + Send>>>;
    
    /// 批处理推理
    async fn infer_batch(
        &self,
        inputs: Vec<Self::Input>,
        options: InferenceOptions,
    ) -> Result<Vec<Self::Output>>;
    
    // ========== 资源管理 ==========
    
    /// 获取当前内存使用
    fn memory_usage(&self) -> Result<crate::models::metadata::MemoryUsage>;
    
    /// 预热模型（可选）
    async fn warmup(&self) -> Result<()>;
}

/// Mold 实现类型枚举
/// 
/// 用于在运行时存储不同类型的 Mold 实现。
/// 
/// 注意：由于 Rust 的类型系统限制，这个枚举目前主要用于类型标记。
/// 实际使用时，应该使用具体的 Mold 实现类型。
pub enum MoldType {
    /// Candle Mold
    Candle,
    /// LlamaCpp Mold
    LlamaCpp,
    /// External Mold
    External(ExternalBridgeType),
}

/// CandleMold: 纯 Rust 实现
/// 
/// 使用 Candle 框架的 Mold 实现。
/// 
/// 注意：这是一个标记 trait，用于标识使用 Candle 后端的 Mold 实现。
pub trait CandleMold: Mold {
    /// 获取设备类型
    fn device(&self) -> String;  // CPU, CUDA, Metal
}

/// LlamaCppMold: C++ 高性能后端
/// 
/// 通过 FFI 调用 llama.cpp 的 Mold 实现。
/// 
/// 注意：这是一个标记 trait，用于标识使用 llama.cpp 后端的 Mold 实现。
pub trait LlamaCppMold: Mold {
    /// 获取 GPU 层数
    fn gpu_layers(&self) -> usize;
}

/// ExternalMold: 外部框架桥接
/// 
/// 桥接外部框架（PyTorch、TensorFlow 等）的 Mold 实现。
/// 
/// 注意：这是一个标记 trait，用于标识使用外部框架的 Mold 实现。
pub trait ExternalMold: Mold {
    /// 获取桥接类型
    fn bridge_type(&self) -> ExternalBridgeType;
}
