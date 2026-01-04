use serde::Deserialize;
use std::time::Duration;

/// 生成请求
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub system: Option<String>,
    pub context: Option<Vec<u64>>,
    pub stream: Option<bool>,
    pub options: Option<InferenceOptions>,
}

/// 聊天请求
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub options: Option<InferenceOptions>,
}

/// 聊天消息
#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// 嵌入请求
#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    pub prompt: String,
    pub options: Option<InferenceOptions>,
}

/// 推理选项
#[derive(Debug, Clone, Deserialize)]
pub struct InferenceOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
    pub stop_sequences: Option<Vec<String>>,
    pub repetition_penalty: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub max_memory: Option<u64>,
    pub timeout: Option<Duration>,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            seed: None,
            stop_sequences: None,
            repetition_penalty: None,
            width: None,
            height: None,
            steps: None,
            guidance_scale: None,
            max_memory: None,
            timeout: None,
        }
    }
}

