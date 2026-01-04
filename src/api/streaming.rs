use chrono::{DateTime, Utc};
use serde::Serialize;

/// 流式响应块
#[derive(Debug, Serialize)]
pub struct GenerateResponseChunk {
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub response: String,
    pub done: bool,
}

/// 聊天流式响应块
#[derive(Debug, Serialize)]
pub struct ChatResponseChunk {
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub message: Option<ChatMessageChunk>,
    pub done: bool,
}

/// 聊天消息块
#[derive(Debug, Serialize)]
pub struct ChatMessageChunk {
    pub role: String,
    pub content: String,
}

