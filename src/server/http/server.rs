use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use crate::Result;
use crate::config::Config;
use crate::server::http::routes;

/// 启动 HTTP 服务器
pub async fn serve(config: Config) -> Result<()> {
    let app = routes::create_router()
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        );

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await
        .map_err(|e| crate::api::error::ApiError::Internal(format!("Failed to bind to {}: {}", addr, e)))?;

    tracing::info!("Server listening on http://{}", addr);

    axum::serve(listener, app).await
        .map_err(|e| crate::api::error::ApiError::Internal(format!("Server error: {}", e)))?;

    Ok(())
}

