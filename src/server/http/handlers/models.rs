use axum::{Json, response::IntoResponse};
use crate::api::response::ModelInfo;

/// 列出模型
pub async fn list_models() -> impl IntoResponse {
    // TODO: 实现模型列表逻辑
    Json::<Vec<ModelInfo>>(vec![])
}

