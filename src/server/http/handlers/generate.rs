use axum::{Json, response::IntoResponse};
use crate::api::request::GenerateRequest;
use crate::api::response::GenerateResponse;

/// 生成请求处理
pub async fn generate(Json(request): Json<GenerateRequest>) -> impl IntoResponse {
    tracing::info!("Generate request for model: {}", request.model);
    
    // TODO: 实现生成逻辑
    Json(GenerateResponse {
        model: request.model,
        created_at: chrono::Utc::now(),
        response: "Not implemented yet".to_string(),
        done: true,
        context: None,
        total_duration: None,
        load_duration: None,
        prompt_eval_count: None,
        prompt_eval_duration: None,
        eval_count: None,
        eval_duration: None,
    })
}

