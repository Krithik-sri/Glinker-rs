//! OpenAI-compatible HTTP interface.
//!
//! * `POST /v1/chat/completions` — the last `user` message is the text to link. If its content
//!   is JSON `{"texts": [...]}` every text is linked in one batch. The assistant message content
//!   is the L0 output as JSON (`{"entities": [[...]], "stats": {...}}`). `stream: true` returns
//!   one SSE `chat.completion.chunk` followed by `data: [DONE]`.
//! * `GET /v1/models`, `GET /health`.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use crate::executor::Executor;

#[derive(Clone)]
pub struct AppState {
    pub executor: Arc<Mutex<Executor>>,
    pub model_name: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// Content may be a string or OpenAI "parts" (`[{"type":"text","text":"..."}]`).
fn content_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts.iter().filter_map(|p| p.get("text").and_then(|t| t.as_str())).collect::<Vec<_>>().join("\n"),
        other => other.to_string(),
    }
}

pub fn texts_from_request(req: &ChatRequest) -> Result<Vec<String>, String> {
    let msg = req.messages.iter().rev().find(|m| m.role == "user").ok_or("no user message")?;
    let raw = content_text(&msg.content);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
        if let Some(arr) = v.get("texts").and_then(|t| t.as_array()) {
            return Ok(arr.iter().filter_map(|x| x.as_str().map(str::to_string)).collect());
        }
    }
    Ok(vec![raw])
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}}))).into_response()
}

async fn chat(State(st): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let texts = match texts_from_request(&req) {
        Ok(t) => t,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let words: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
    let ex = st.executor.clone();
    let t2 = texts.clone();
    let out = tokio::task::spawn_blocking(move || {
        let mut g = ex.lock().map_err(|_| "executor poisoned".to_string())?;
        g.execute(&t2).map_err(|e| e.to_string())
    })
    .await;
    let out = match out {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let content = serde_json::to_string(&out).unwrap_or_default();
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let id = format!("chatcmpl-glinker-{created}");
    let model = req.model.clone().unwrap_or_else(|| st.model_name.clone());
    let links = out.stats.linked;
    if req.stream {
        let chunk = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
        });
        let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
        return ([(header::CONTENT_TYPE, "text/event-stream")], body).into_response();
    }
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": Usage { prompt_tokens: words, completion_tokens: links, total_tokens: words + links },
    }))
    .into_response()
}

async fn models(State(st): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({"object": "list", "data": [{"id": st.model_name, "object": "model", "owned_by": "glinker-rs"}]}))
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/health", get(|| async { "ok" }))
        .with_state(state)
}

pub async fn serve(state: AppState, addr: &str) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(state)).await
}
