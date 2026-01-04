//! 模型类型定义
//! 
//! 定义模型类型、后端类型、硬件加速器类型等枚举和结构。

use serde::{Deserialize, Serialize};

/// 模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", content = "subtype")]
pub enum ModelType {
    /// 文本模型
    Text(TextModelType),
    /// 图像模型
    Image(ImageModelType),
    /// 音频模型
    Audio(AudioModelType),
    /// 视频模型
    Video(VideoModelType),
    /// 多模态模型
    Multimodal(MultimodalModelType),
}

/// 文本模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TextModelType {
    /// 语言模型（LLM）
    LanguageModel,
    /// 嵌入模型
    Embedding,
    /// 分类模型
    Classification,
    /// 翻译模型
    Translation,
    /// 代码生成模型
    CodeGeneration,
}

/// 图像模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageModelType {
    /// 图像生成（Stable Diffusion等）
    Generation,
    /// 图像理解（CLIP等）
    Understanding,
    /// 图像分类
    Classification,
    /// 图像分割
    Segmentation,
    /// 图像编辑
    Editing,
}

/// 音频模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioModelType {
    /// 语音识别（Whisper等）
    SpeechRecognition,
    /// 语音合成
    SpeechSynthesis,
    /// 音乐生成
    MusicGeneration,
    /// 音频分析
    AudioAnalysis,
}

/// 视频模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoModelType {
    /// 视频生成
    Generation,
    /// 视频理解
    Understanding,
    /// 视频编辑
    Editing,
}

/// 多模态模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MultimodalModelType {
    /// 视觉语言模型（LLaVA等）
    VisionLanguage,
    /// 音频视觉模型
    AudioVisual,
    /// 通用多模态
    General,
}

/// 后端类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendType {
    /// Candle（纯 Rust）
    Candle,
    /// llama.cpp（C++）
    LlamaCpp,
    /// ONNX Runtime
    OnnxRuntime,
    /// TensorRT（NVIDIA）
    TensorRT,
    /// TVM（Apache TVM）
    Tvm,
    /// WebAssembly
    Wasm,
    /// 外部框架桥接
    External(ExternalBridgeType),
}

/// 外部框架桥接类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExternalBridgeType {
    /// PyTorch
    PyTorch,
    /// TensorFlow
    TensorFlow,
    /// JAX
    Jax,
    /// ONNX
    Onnx,
    /// 自定义
    Custom(String),
}

/// 硬件加速器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", content = "device_id")]
pub enum AcceleratorType {
    /// CPU
    Cpu,
    /// CUDA（NVIDIA GPU）
    Cuda { device_id: u32 },
    /// Metal（Apple Silicon）
    Metal { device_id: u32 },
    /// Vulkan
    Vulkan { device_id: u32 },
    /// WebGPU
    WebGpu,
    /// WebAssembly
    Wasm,
}

/// 模型格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFormat {
    /// GGUF 格式（llama.cpp）
    Gguf,
    /// Safetensors 格式（HuggingFace）
    Safetensors,
    /// ONNX 格式
    Onnx,
    /// TVM 格式
    Tvm,
    /// PyTorch 格式
    PyTorch,
    /// TensorFlow 格式
    TensorFlow,
    /// 原始二进制格式
    Binary,
    /// 未知格式
    Unknown,
}

/// 量化类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationType {
    /// FP32（全精度）
    Fp32,
    /// FP16（半精度）
    Fp16,
    /// BF16（Brain Float 16）
    Bf16,
    /// INT8
    Int8,
    /// INT4
    Int4,
    /// Q4_0
    Q4_0,
    /// Q4_1
    Q4_1,
    /// Q5_0
    Q5_0,
    /// Q5_1
    Q5_1,
    /// Q8_0
    Q8_0,
}

/// 形状定义（用于张量维度）
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    pub dimensions: Vec<usize>,
}

impl Shape {
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self { dimensions }
    }

    pub fn total_size(&self) -> usize {
        self.dimensions.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.total_size(), 24);
        assert_eq!(shape.dimensions, vec![2, 3, 4]);
    }

    #[test]
    fn test_model_type_serialization() {
        let model_type = ModelType::Text(TextModelType::LanguageModel);
        let json = serde_json::to_string(&model_type).unwrap();
        let deserialized: ModelType = serde_json::from_str(&json).unwrap();
        assert_eq!(model_type, deserialized);
    }
}

