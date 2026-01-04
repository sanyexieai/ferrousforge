//! 模型元数据定义

use crate::models::types::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// 模型元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// 模型名称
    pub name: String,
    /// 模型版本
    pub version: String,
    /// 模型类型
    pub model_type: ModelType,
    /// 模型格式
    pub format: ModelFormat,
    /// 文件大小（字节）
    pub size: u64,
    /// 参数量
    pub parameters: Option<u64>,
    /// 量化类型
    pub quantization: Option<QuantizationType>,
    /// 架构名称
    pub architecture: String,
    /// 上下文大小（对于文本模型）
    pub context_size: Option<usize>,
    /// 输入形状
    pub input_shapes: Vec<Shape>,
    /// 输出形状
    pub output_shapes: Vec<Shape>,
    /// 模型需求
    pub requirements: ModelRequirements,
    /// 标签
    pub tags: Vec<String>,
    /// 描述
    pub description: Option<String>,
    /// 许可证
    pub license: Option<String>,
    /// 作者
    pub author: Option<String>,
    /// 创建时间
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    /// 更新时间
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// 模型需求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequirements {
    /// 最小内存（字节）
    pub min_memory: Option<u64>,
    /// 最小显存（字节）
    pub min_vram: Option<u64>,
    /// CPU 核心数
    pub cpu_cores: Option<usize>,
    /// 是否需要 GPU
    pub gpu_required: bool,
    /// GPU 计算能力要求（CUDA）
    pub gpu_compute_capability: Option<String>,
}

impl Default for ModelRequirements {
    fn default() -> Self {
        Self {
            min_memory: None,
            min_vram: None,
            cpu_cores: None,
            gpu_required: false,
            gpu_compute_capability: None,
        }
    }
}

/// 内存使用信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// 已使用内存（字节）
    pub used: u64,
    /// 峰值内存（字节）
    pub peak: u64,
    /// 已使用显存（字节，如果有 GPU）
    pub vram_used: Option<u64>,
    /// 峰值显存（字节，如果有 GPU）
    pub vram_peak: Option<u64>,
}

