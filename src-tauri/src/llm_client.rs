use crate::settings::{PostProcessProvider, GOOGLE_AI_STUDIO_PROVIDER_ID};
use log::{debug, info};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, REFERER, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct JsonSchema {
    name: String,
    strict: bool,
    schema: Value,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    json_schema: JsonSchema,
}

#[derive(Debug, Serialize, Clone, Default, PartialEq)]
struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude: Option<bool>,
}

/// Request fields used to ask an endpoint to skip reasoning/thinking.
/// Providers disagree on the field name and accepted values, so at most one of
/// these is set per request (see `reasoning_disable_params`).
#[derive(Debug, Serialize, Clone, Default, PartialEq)]
struct ReasoningParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Value>,
}

impl ReasoningParams {
    fn is_empty(&self) -> bool {
        self.reasoning_effort.is_none() && self.reasoning.is_none() && self.thinking.is_none()
    }
}

/// Pick the reasoning-disable request fields an endpoint understands.
/// Unknown endpoints get the common OpenAI-style field; if they reject it,
/// the request is retried without it (see `send_chat_completion_with_schema`).
fn reasoning_disable_params(provider: &PostProcessProvider) -> ReasoningParams {
    let base_url = provider.base_url.to_lowercase();
    if base_url.contains("api.deepseek.com") {
        // DeepSeek rejects reasoning_effort "none" and uses its own field:
        // https://api-docs.deepseek.com/guides/thinking_mode
        ReasoningParams {
            thinking: Some(serde_json::json!({ "type": "disabled" })),
            ..Default::default()
        }
    } else if provider.id == "openrouter" {
        // OpenRouter nested object; exclude:true also keeps reasoning text out
        // of the response so it can't pollute structured-output JSON parsing
        ReasoningParams {
            reasoning: Some(ReasoningConfig {
                effort: Some("none".to_string()),
                exclude: Some(true),
            }),
            ..Default::default()
        }
    } else {
        ReasoningParams {
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        }
    }
}

/// Endpoints (base_url|model) that rejected the reasoning-disable fields with a
/// 4xx. Remembered for the lifetime of the process so every dictation after the
/// first skips the doomed attempt and goes straight to a plain request.
fn reasoning_rejections() -> &'static Mutex<HashSet<String>> {
    static REJECTED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    REJECTED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn endpoint_key(provider: &PostProcessProvider, model: &str) -> String {
    format!("{}|{}", provider.base_url.trim_end_matches('/'), model)
}

fn is_known_rejected(key: &str) -> bool {
    reasoning_rejections()
        .lock()
        .map(|set| set.contains(key))
        .unwrap_or(false)
}

fn remember_rejection(key: String) {
    if let Ok(mut set) = reasoning_rejections().lock() {
        set.insert(key);
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(flatten)]
    reasoning: ReasoningParams,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiGenerateContentResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: Option<String>,
}

fn is_google_ai_studio(provider: &PostProcessProvider) -> bool {
    provider.id == GOOGLE_AI_STUDIO_PROVIDER_ID
}

/// Gemini's REST response schema uses the enum-style uppercase type names,
/// while Handy's provider-neutral JSON schema is lowercase. Normalize only
/// schema `type` values and preserve all other fields.
fn normalize_gemini_schema(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if let Some(Value::String(type_name)) = object.get_mut("type") {
                *type_name = type_name.to_uppercase();
            }
            for child in object.values_mut() {
                normalize_gemini_schema(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_gemini_schema(item);
            }
        }
        _ => {}
    }
}

/// Build headers for API requests based on provider type
fn build_headers(provider: &PostProcessProvider, api_key: &str) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();

    // Common headers
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        REFERER,
        HeaderValue::from_static("https://github.com/cjpais/Handy"),
    );
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("Handy/1.0 (+https://github.com/cjpais/Handy)"),
    );
    headers.insert("X-Title", HeaderValue::from_static("Handy"));

    // Provider-specific auth headers
    if !api_key.is_empty() {
        if is_google_ai_studio(provider) {
            headers.insert(
                "x-goog-api-key",
                HeaderValue::from_str(api_key)
                    .map_err(|e| format!("Invalid Google AI Studio API key: {}", e))?,
            );
        } else if provider.id == "anthropic" {
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(api_key)
                    .map_err(|e| format!("Invalid API key header value: {}", e))?,
            );
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        } else {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .map_err(|e| format!("Invalid authorization header value: {}", e))?,
            );
        }
    }

    Ok(headers)
}

/// Overall request timeout for chat completions (connect + response headers + body).
const CHAT_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
/// TCP connect timeout — fail fast on unreachable hosts.
const CHAT_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Create an HTTP client with provider-specific headers and explicit timeouts.
fn create_client(provider: &PostProcessProvider, api_key: &str) -> Result<reqwest::Client, String> {
    let headers = build_headers(provider, api_key)?;
    reqwest::Client::builder()
        .default_headers(headers)
        .timeout(CHAT_REQUEST_TIMEOUT)
        .connect_timeout(CHAT_CONNECT_TIMEOUT)
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))
}

/// Send a request through Google's native Gemini API. Gemini is not an
/// OpenAI-compatible endpoint: it uses `generateContent`, `contents`/`parts`,
/// a `systemInstruction`, and the `x-goog-api-key` header.
async fn send_google_ai_studio_completion(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    user_content: String,
    system_prompt: Option<String>,
    json_schema: Option<Value>,
) -> Result<Option<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');
    let model = model.strip_prefix("models/").unwrap_or(model);
    let url = format!("{}/models/{}:generateContent", base_url, model);

    debug!("Sending Gemini generateContent request to: {}", url);

    let client = create_client(provider, &api_key)?;
    let mut request_body = serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": user_content }]
        }]
    });

    if let Some(system) = system_prompt {
        request_body["systemInstruction"] = serde_json::json!({
            "parts": [{ "text": system }]
        });
    }

    if let Some(mut schema) = json_schema {
        normalize_gemini_schema(&mut schema);
        request_body["generationConfig"] = serde_json::json!({
            "responseMimeType": "application/json",
            "responseSchema": schema
        });
    }

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Gemini API request failed: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());
        return Err(format!(
            "Gemini API request failed with status {}: {}",
            status, error_text
        ));
    }

    let completion: GeminiGenerateContentResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Gemini API response: {}", e))?;

    Ok(completion
        .candidates
        .first()
        .and_then(|candidate| candidate.content.as_ref())
        .map(|content| {
            content
                .parts
                .iter()
                .filter_map(|part| part.text.as_deref())
                .collect::<String>()
        })
        .filter(|content| !content.is_empty()))
}

/// Send a chat completion request to an OpenAI-compatible API
/// Returns Ok(Some(content)) on success, Ok(None) if response has no content,
/// or Err on actual errors (HTTP, parsing, etc.)
pub async fn send_chat_completion(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    prompt: String,
    disable_reasoning: bool,
) -> Result<Option<String>, String> {
    send_chat_completion_with_schema(
        provider,
        api_key,
        model,
        prompt,
        None,
        None,
        disable_reasoning,
    )
    .await
}

/// Send a chat completion request with structured output support.
/// When json_schema is provided, uses structured outputs mode.
/// system_prompt is used as the system message when provided.
///
/// When disable_reasoning is set, the request carries the reasoning-disable
/// fields the endpoint is expected to understand. Not every OpenAI-compatible
/// endpoint accepts them (DeepSeek, Gemini's compat layer, and some OpenRouter
/// upstreams reject with 400), so a 400/422 answer to such a request triggers
/// one retry without the fields, and the rejection is remembered per
/// (base_url, model) so later requests skip the failing attempt entirely.
pub async fn send_chat_completion_with_schema(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    user_content: String,
    system_prompt: Option<String>,
    json_schema: Option<Value>,
    disable_reasoning: bool,
) -> Result<Option<String>, String> {
    if is_google_ai_studio(provider) {
        // Gemini exposes its own thinking controls and does not accept the
        // provider-neutral reasoning fields used by OpenAI-compatible APIs.
        return send_google_ai_studio_completion(
            provider,
            api_key,
            model,
            user_content,
            system_prompt,
            json_schema,
        )
        .await;
    }

    let base_url = provider.base_url.trim_end_matches('/');
    let url = format!("{}/chat/completions", base_url);

    debug!("Sending chat completion request to: {}", url);

    let client = create_client(provider, &api_key)?;

    // Build messages vector
    let mut messages = Vec::new();

    // Add system prompt if provided
    if let Some(system) = system_prompt {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system,
        });
    }

    // Add user message
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: user_content,
    });

    // Build response_format if schema is provided
    let response_format = json_schema.map(|schema| ResponseFormat {
        format_type: "json_schema".to_string(),
        json_schema: JsonSchema {
            name: "transcription_output".to_string(),
            strict: true,
            schema,
        },
    });

    let key = endpoint_key(provider, model);
    let reasoning = if disable_reasoning && !is_known_rejected(&key) {
        reasoning_disable_params(provider)
    } else {
        ReasoningParams::default()
    };

    let mut request_body = ChatCompletionRequest {
        model: model.to_string(),
        messages,
        response_format,
        reasoning,
    };

    let mut response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {}", e))?;
    let mut status = response.status();

    // A 400/422 on a request carrying reasoning-disable fields is almost always
    // the endpoint rejecting those fields — retry once without them.
    if !status.is_success()
        && matches!(status.as_u16(), 400 | 422)
        && !request_body.reasoning.is_empty()
    {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());
        info!(
            "Endpoint rejected request with reasoning disabled (status {}): {}. Retrying without reasoning fields",
            status, error_text
        );

        request_body.reasoning = ReasoningParams::default();
        response = client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;
        status = response.status();

        if status.is_success() {
            info!(
                "Retry without reasoning fields succeeded; '{}' (model '{}') will skip them from now on",
                base_url, model
            );
            remember_rejection(key);
        }
    }

    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());
        return Err(format!(
            "API request failed with status {}: {}",
            status, error_text
        ));
    }

    let completion: ChatCompletionResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse API response: {}", e))?;

    Ok(completion
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone()))
}

fn parse_model_ids(provider: &PostProcessProvider, parsed: &Value) -> Vec<String> {
    let mut models = Vec::new();

    if is_google_ai_studio(provider) {
        // Gemini returns { models: [{ name, baseModelId,
        // supportedGenerationMethods }] }. Only expose models that can serve
        // generateContent; embeddings and media-only models are not valid for
        // Handy's post-processing requests.
        if let Some(entries) = parsed.get("models").and_then(|value| value.as_array()) {
            for entry in entries {
                let supports_generation = entry
                    .get("supportedGenerationMethods")
                    .and_then(|methods| methods.as_array())
                    .map(|methods| {
                        methods
                            .iter()
                            .any(|method| method.as_str() == Some("generateContent"))
                    })
                    .unwrap_or(true);
                if !supports_generation {
                    continue;
                }

                let model = entry
                    .get("baseModelId")
                    .and_then(|value| value.as_str())
                    .or_else(|| entry.get("name").and_then(|value| value.as_str()))
                    .map(|model| model.trim_start_matches("models/").to_string());
                if let Some(model) = model.filter(|model| !model.is_empty()) {
                    models.push(model);
                }
            }
        }
        return models;
    }

    // Handle OpenAI format: { data: [ { id: "..." }, ... ] }
    if let Some(data) = parsed.get("data").and_then(|d| d.as_array()) {
        for entry in data {
            if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                models.push(id.to_string());
            } else if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
                models.push(name.to_string());
            }
        }
    }
    // Handle array format: [ "model1", "model2", ... ]
    else if let Some(array) = parsed.as_array() {
        for entry in array {
            if let Some(model) = entry.as_str() {
                models.push(model.to_string());
            }
        }
    }

    models
}

/// Fetch available models from a provider's model-list endpoint.
/// Returns a list of model IDs.
pub async fn fetch_models(
    provider: &PostProcessProvider,
    api_key: String,
) -> Result<Vec<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');
    let models_endpoint = provider.models_endpoint.as_deref().unwrap_or("/models");
    let url = format!("{}{}", base_url, models_endpoint);

    debug!("Fetching models from: {}", url);

    let client = create_client(provider, &api_key)?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch models: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!(
            "Model list request failed ({}): {}",
            status, error_text
        ));
    }

    let parsed: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    Ok(parse_model_ids(provider, &parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(id: &str, base_url: &str) -> PostProcessProvider {
        PostProcessProvider {
            id: id.to_string(),
            label: id.to_string(),
            base_url: base_url.to_string(),
            allow_base_url_edit: true,
            models_endpoint: None,
            supports_structured_output: false,
        }
    }

    fn request_json(reasoning: ReasoningParams) -> Value {
        let request = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            response_format: None,
            reasoning,
        };
        serde_json::to_value(&request).unwrap()
    }

    #[test]
    fn default_reasoning_params_serialize_to_no_fields() {
        let json = request_json(ReasoningParams::default());
        assert!(json.get("reasoning_effort").is_none());
        assert!(json.get("reasoning").is_none());
        assert!(json.get("thinking").is_none());
    }

    #[test]
    fn custom_provider_uses_top_level_reasoning_effort() {
        let params = reasoning_disable_params(&provider("custom", "http://localhost:11434/v1"));
        let json = request_json(params);
        assert_eq!(json["reasoning_effort"], "none");
        assert!(json.get("reasoning").is_none());
        assert!(json.get("thinking").is_none());
    }

    #[test]
    fn openrouter_uses_nested_reasoning_object() {
        let params =
            reasoning_disable_params(&provider("openrouter", "https://openrouter.ai/api/v1"));
        let json = request_json(params);
        assert!(json.get("reasoning_effort").is_none());
        assert_eq!(json["reasoning"]["effort"], "none");
        assert_eq!(json["reasoning"]["exclude"], true);
        assert!(json.get("thinking").is_none());
    }

    #[test]
    fn deepseek_base_url_uses_thinking_disabled() {
        let params = reasoning_disable_params(&provider("custom", "https://api.deepseek.com"));
        let json = request_json(params);
        assert!(json.get("reasoning_effort").is_none());
        assert!(json.get("reasoning").is_none());
        assert_eq!(json["thinking"]["type"], "disabled");
    }

    #[test]
    fn google_ai_studio_uses_native_api_key_header() {
        let google = provider(
            GOOGLE_AI_STUDIO_PROVIDER_ID,
            "https://generativelanguage.googleapis.com/v1beta",
        );
        let headers = build_headers(&google, "AIza-test-key").unwrap();

        assert_eq!(headers.get("x-goog-api-key").unwrap(), "AIza-test-key");
        assert!(headers.get(AUTHORIZATION).is_none());
    }

    #[test]
    fn gemini_schema_uses_uppercase_type_names() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "transcription": { "type": "string" }
            }
        });

        normalize_gemini_schema(&mut schema);

        assert_eq!(schema["type"], "OBJECT");
        assert_eq!(schema["properties"]["transcription"]["type"], "STRING");
    }

    #[test]
    fn google_model_list_uses_generate_content_models_only() {
        let google = provider(GOOGLE_AI_STUDIO_PROVIDER_ID, "https://example.com/v1beta");
        let parsed = serde_json::json!({
            "models": [
                {
                    "name": "models/gemini-2.5-flash-lite",
                    "baseModelId": "gemini-2.5-flash-lite",
                    "supportedGenerationMethods": ["generateContent"]
                },
                {
                    "name": "models/text-embedding-004",
                    "supportedGenerationMethods": ["embedContent"]
                }
            ]
        });

        assert_eq!(
            parse_model_ids(&google, &parsed),
            vec!["gemini-2.5-flash-lite"]
        );
    }

    #[test]
    fn reasoning_params_is_empty_tracks_all_fields() {
        assert!(ReasoningParams::default().is_empty());
        assert!(!ReasoningParams {
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        }
        .is_empty());
        assert!(!ReasoningParams {
            thinking: Some(serde_json::json!({ "type": "disabled" })),
            ..Default::default()
        }
        .is_empty());
    }

    #[test]
    fn rejection_memo_is_keyed_by_base_url_and_model() {
        let deepseek = provider("custom", "https://api.deepseek.com/");
        let key = endpoint_key(&deepseek, "deepseek-chat");
        assert_eq!(key, "https://api.deepseek.com|deepseek-chat");
        assert!(!is_known_rejected(&key));
        remember_rejection(key.clone());
        assert!(is_known_rejected(&key));
        // A different model on the same endpoint is tracked separately
        assert!(!is_known_rejected(&endpoint_key(&deepseek, "other-model")));
    }
}
