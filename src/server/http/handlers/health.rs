use axum::Json;
use serde_json::json;

/// 健康检查端点
pub async fn health() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "version": crate::VERSION
    }))
}

