use chrono::{DateTime, Utc};
use serde::Serialize;
use std::time::Duration;

/// 生成响应
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub response: String,
    pub done: bool,
    pub context: Option<Vec<u64>>,
    pub total_duration: Option<Duration>,
    pub load_duration: Option<Duration>,
    pub prompt_eval_count: Option<usize>,
    pub prompt_eval_duration: Option<Duration>,
    pub eval_count: Option<usize>,
    pub eval_duration: Option<Duration>,
}

/// 聊天响应
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub message: ChatMessage,
    pub done: bool,
    pub total_duration: Option<Duration>,
}

/// 聊天消息（响应）
#[derive(Debug, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// 嵌入响应
#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
}

/// 模型信息
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: String,
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
}

/// 模型详情
#[derive(Debug, Serialize)]
pub struct ModelDetails {
    pub format: String,
    pub family: String,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

