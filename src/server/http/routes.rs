use axum::Router;
use crate::server::http::handlers;

/// 定义路由
pub fn create_router() -> Router {
    Router::new()
        .route("/health", axum::routing::get(handlers::health))
        .route("/api/tags", axum::routing::get(handlers::list_models))
        .route("/api/generate", axum::routing::post(handlers::generate))
}

