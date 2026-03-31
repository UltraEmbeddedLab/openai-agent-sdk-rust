//! `OpenAI` Chat Completions API model implementation.
//!
//! This module provides [`OpenAIChatCompletionsModel`], which implements the [`Model`] trait
//! using the older `/v1/chat/completions` endpoint. This endpoint is widely supported by
//! `OpenAI`-compatible providers (Azure `OpenAI`, local LLMs, third-party gateways, etc.).
//!
//! Internally the model converts between the Responses API format used by the SDK (the
//! `ResponseInputItem` / `ResponseOutputItem` JSON values) and the Chat Completions
//! message format (`messages` array with `role`/`content` objects).

use std::pin::Pin;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tokio_stream::Stream;
use tracing::{debug, warn};

use crate::config::{ModelSettings, ToolChoice};
use crate::error::{AgentError, Result};
use crate::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use crate::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use crate::usage::{InputTokensDetails, OutputTokensDetails, Usage};

/// Default base URL for the `OpenAI` Chat Completions API.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// A fake response ID used when the Chat Completions API does not provide one.
const FAKE_RESPONSES_ID: &str = "fake_id";

// ---------------------------------------------------------------------------
// Helper: convert tool choice
// ---------------------------------------------------------------------------

/// Convert the SDK [`ToolChoice`] enum into a JSON value suitable for the
/// Chat Completions `tool_choice` request parameter.
#[must_use]
fn convert_tool_choice(tool_choice: &ToolChoice) -> serde_json::Value {
    match tool_choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Required => json!("required"),
        ToolChoice::None => json!("none"),
        ToolChoice::Named(name) => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

// ---------------------------------------------------------------------------
// Helper: parse usage
// ---------------------------------------------------------------------------

/// Parse the `usage` object from a Chat Completions response into the SDK [`Usage`] type.
fn parse_usage(usage_value: Option<&serde_json::Value>) -> Usage {
    let Some(u) = usage_value else {
        return Usage::default();
    };

    let input_tokens = u
        .get("prompt_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let output_tokens = u
        .get("completion_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total_tokens = u
        .get("total_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(input_tokens + output_tokens);

    let cached_tokens = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    let reasoning_tokens = u
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    Usage {
        requests: 1,
        input_tokens,
        output_tokens,
        total_tokens,
        input_tokens_details: InputTokensDetails { cached_tokens },
        output_tokens_details: OutputTokensDetails { reasoning_tokens },
        request_usage_entries: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Helper: convert input items to Chat Completions messages
// ---------------------------------------------------------------------------

/// Convert Responses API format input items into Chat Completions `messages` array entries.
///
/// This handles `easy_input_message` style items (just `role` + `content`), typed items
/// (`type: "message"`, `type: "function_call"`, `type: "function_call_output"`), and
/// assistant messages with tool calls.
#[allow(clippy::too_many_lines)]
fn convert_input_to_messages(
    system_instructions: Option<&str>,
    input: &[ResponseInputItem],
) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();

    // System message from instructions.
    if let Some(instructions) = system_instructions {
        messages.push(json!({
            "role": "system",
            "content": instructions
        }));
    }

    // Track assistant messages so adjacent function_call items can be grouped.
    let mut pending_tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut pending_assistant_content: Option<String> = None;

    // Flush any accumulated assistant message with tool calls.
    let flush_assistant = |messages: &mut Vec<serde_json::Value>,
                           tool_calls: &mut Vec<serde_json::Value>,
                           content: &mut Option<String>| {
        if tool_calls.is_empty() && content.is_none() {
            return;
        }
        let mut msg = json!({ "role": "assistant" });
        if let Some(text) = content.take() {
            msg["content"] = json!(text);
        } else {
            msg["content"] = serde_json::Value::Null;
        }
        if !tool_calls.is_empty() {
            msg["tool_calls"] = json!(std::mem::take(tool_calls));
        }
        messages.push(msg);
    };

    for item in input {
        let item_type = item.get("type").and_then(serde_json::Value::as_str);
        let item_role = item.get("role").and_then(serde_json::Value::as_str);

        match item_type {
            // --- Typed items ---
            Some("message") => {
                // Flush any pending assistant state first.
                flush_assistant(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_assistant_content,
                );

                let role = item_role.unwrap_or("user");
                let content = extract_message_content(item);
                messages.push(json!({ "role": role, "content": content }));
            }

            Some("function_call") => {
                // This is an assistant tool call. Accumulate it.
                let call_id = item
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let name = item
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let arguments = item
                    .get("arguments")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("{}");

                pending_tool_calls.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }));
            }

            Some("function_call_output") => {
                // Flush pending assistant message before tool output.
                flush_assistant(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_assistant_content,
                );

                let call_id = item
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let output = item
                    .get("output")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output
                }));
            }

            // --- No type field: "easy input message" style (role + content only) ---
            None => {
                let role = item_role.unwrap_or("user");
                match role {
                    "assistant" | "model" => {
                        // Flush any prior pending assistant state.
                        flush_assistant(
                            &mut messages,
                            &mut pending_tool_calls,
                            &mut pending_assistant_content,
                        );

                        let content = item
                            .get("content")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("")
                            .to_owned();

                        // Check if this easy assistant message has tool_calls embedded.
                        if let Some(tool_calls) =
                            item.get("tool_calls").and_then(|tc| tc.as_array())
                        {
                            let mut msg = json!({ "role": "assistant", "content": content });
                            msg["tool_calls"] = json!(tool_calls);
                            messages.push(msg);
                        } else {
                            pending_assistant_content = Some(content);
                        }
                    }
                    "user" | "system" | "developer" => {
                        flush_assistant(
                            &mut messages,
                            &mut pending_tool_calls,
                            &mut pending_assistant_content,
                        );
                        let content = extract_easy_content(item);
                        messages.push(json!({ "role": role, "content": content }));
                    }
                    _ => {
                        // Unknown role, pass through best-effort.
                        flush_assistant(
                            &mut messages,
                            &mut pending_tool_calls,
                            &mut pending_assistant_content,
                        );
                        let content = extract_easy_content(item);
                        messages.push(json!({ "role": role, "content": content }));
                    }
                }
            }

            // --- Other typed items (reasoning, etc.) are skipped. ---
            Some(_) => {
                debug!(
                    "Skipping unrecognized item type in chat completions input: {:?}",
                    item_type
                );
            }
        }
    }

    // Flush any trailing assistant state.
    flush_assistant(
        &mut messages,
        &mut pending_tool_calls,
        &mut pending_assistant_content,
    );

    messages
}

/// Extract the text content from a Responses API message item.
///
/// Handles both string content and structured `content` arrays with `output_text` entries.
fn extract_message_content(item: &serde_json::Value) -> serde_json::Value {
    if let Some(content) = item.get("content") {
        // If content is a string, return it directly.
        if content.is_string() {
            return content.clone();
        }
        // If content is an array with output_text entries, concatenate texts.
        if let Some(arr) = content.as_array() {
            let texts: Vec<&str> = arr
                .iter()
                .filter(|c| {
                    c.get("type").and_then(serde_json::Value::as_str) == Some("output_text")
                })
                .filter_map(|c| c.get("text").and_then(serde_json::Value::as_str))
                .collect();
            if !texts.is_empty() {
                return json!(texts.join(""));
            }
            // If we have input_text entries instead.
            let input_texts: Vec<&str> = arr
                .iter()
                .filter(|c| c.get("type").and_then(serde_json::Value::as_str) == Some("input_text"))
                .filter_map(|c| c.get("text").and_then(serde_json::Value::as_str))
                .collect();
            if !input_texts.is_empty() {
                return json!(input_texts.join(""));
            }
            // Pass through the array as-is for complex content (images, etc.).
            return content.clone();
        }
        return content.clone();
    }
    json!("")
}

/// Extract content from an "easy input message" (role + content, no type field).
fn extract_easy_content(item: &serde_json::Value) -> serde_json::Value {
    item.get("content").cloned().unwrap_or_else(|| json!(""))
}

// ---------------------------------------------------------------------------
// Helper: convert Chat Completions response to ModelResponse
// ---------------------------------------------------------------------------

/// Convert a Chat Completions API JSON response into the SDK [`ModelResponse`].
///
/// Extracts text content and tool calls from `choices[0].message` and converts them
/// into Responses API format output items.
fn convert_response(response_json: &serde_json::Value) -> Result<ModelResponse> {
    let choice = response_json
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
        .ok_or_else(|| AgentError::ModelBehavior {
            message: "No choices in Chat Completions response".to_owned(),
        })?;

    let message = choice
        .get("message")
        .ok_or_else(|| AgentError::ModelBehavior {
            message: "No message in Chat Completions choice".to_owned(),
        })?;

    let mut output_items = Vec::new();

    // Extract text content as a message output item (Responses API format).
    if let Some(content) = message.get("content").and_then(serde_json::Value::as_str) {
        if !content.is_empty() {
            output_items.push(json!({
                "id": FAKE_RESPONSES_ID,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "output_text", "text": content, "annotations": [] }]
            }));
        }
    }

    // Extract refusal.
    if let Some(refusal) = message.get("refusal").and_then(serde_json::Value::as_str) {
        if !refusal.is_empty() {
            // If we already pushed a message item, add refusal to its content.
            if let Some(last) = output_items.last_mut() {
                if let Some(content_arr) = last.get_mut("content").and_then(|c| c.as_array_mut()) {
                    content_arr.push(json!({ "type": "refusal", "refusal": refusal }));
                }
            } else {
                output_items.push(json!({
                    "id": FAKE_RESPONSES_ID,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "refusal", "refusal": refusal }]
                }));
            }
        }
    }

    // Extract tool calls as function_call items (Responses API format).
    if let Some(tool_calls) = message
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
    {
        for tc in tool_calls {
            let function = tc.get("function").unwrap_or(&serde_json::Value::Null);
            let call_id = tc
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let name = function
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let arguments = function
                .get("arguments")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("{}");

            output_items.push(json!({
                "id": FAKE_RESPONSES_ID,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments
            }));
        }
    }

    let usage = parse_usage(response_json.get("usage"));

    let response_id = response_json
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(String::from);

    Ok(ModelResponse::new(output_items, usage, response_id, None))
}

// ---------------------------------------------------------------------------
// Helper: build request body
// ---------------------------------------------------------------------------

/// Build the full Chat Completions request body as a JSON value.
#[allow(clippy::too_many_arguments)]
fn build_request_body(
    model: &str,
    system_instructions: Option<&str>,
    input: &[ResponseInputItem],
    model_settings: &ModelSettings,
    tools: &[ToolSpec],
    output_schema: Option<&OutputSchemaSpec>,
    handoffs: &[HandoffToolSpec],
    stream: bool,
) -> serde_json::Value {
    let messages = convert_input_to_messages(system_instructions, input);

    let mut body = json!({
        "model": model,
        "messages": messages,
    });

    // Collect tool definitions.
    let mut tool_defs: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.params_json_schema,
                    "strict": t.strict,
                }
            })
        })
        .collect();

    for h in handoffs {
        tool_defs.push(json!({
            "type": "function",
            "function": {
                "name": h.tool_name,
                "description": h.tool_description,
                "parameters": h.input_json_schema,
                "strict": h.strict,
            }
        }));
    }

    if !tool_defs.is_empty() {
        body["tools"] = json!(tool_defs);
    }

    // Apply model settings.
    if let Some(temp) = model_settings.temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(top_p) = model_settings.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(max) = model_settings.max_tokens {
        body["max_tokens"] = json!(max);
    }
    if let Some(fp) = model_settings.frequency_penalty {
        body["frequency_penalty"] = json!(fp);
    }
    if let Some(pp) = model_settings.presence_penalty {
        body["presence_penalty"] = json!(pp);
    }
    if let Some(parallel) = model_settings.parallel_tool_calls {
        if !tool_defs.is_empty() {
            body["parallel_tool_calls"] = json!(parallel);
        }
    }

    // Tool choice.
    if let Some(ref tc) = model_settings.tool_choice {
        body["tool_choice"] = convert_tool_choice(tc);
    }

    // Structured output via response_format.
    if let Some(schema) = output_schema {
        body["response_format"] = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "final_output",
                "schema": schema.json_schema,
                "strict": schema.strict,
            }
        });
    }

    // Streaming.
    if stream {
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
    }

    // Store.
    if let Some(store) = model_settings.store {
        body["store"] = json!(store);
    }

    // Extra body fields.
    if let Some(ref extra) = model_settings.extra_body {
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                body[k] = v.clone();
            }
        }
    }

    // Extra args (arbitrary).
    if let Some(ref extra_args) = model_settings.extra_args {
        for (k, v) in extra_args {
            body[k] = v.clone();
        }
    }

    body
}

// ---------------------------------------------------------------------------
// OpenAIChatCompletionsModel
// ---------------------------------------------------------------------------

/// An `OpenAI` model using the Chat Completions API.
///
/// This provides compatibility with the older `/v1/chat/completions` endpoint,
/// which is widely supported by `OpenAI`-compatible providers such as Azure `OpenAI`,
/// local model servers, and third-party gateways.
///
/// The model converts between the Responses API item format used internally by the
/// SDK and the Chat Completions message format expected by the endpoint.
///
/// # Example
///
/// ```no_run
/// use openai_agents::models::openai_chatcompletions::OpenAIChatCompletionsModel;
///
/// let model = OpenAIChatCompletionsModel::new("gpt-4o", "sk-...");
/// ```
#[derive(Debug, Clone)]
pub struct OpenAIChatCompletionsModel {
    /// The model identifier (e.g. `"gpt-4o"`, `"gpt-4o-mini"`).
    model: String,
    /// The HTTP client used for API requests.
    client: Client,
    /// The API key for authentication.
    api_key: String,
    /// The base URL for the Chat Completions endpoint.
    base_url: String,
}

impl OpenAIChatCompletionsModel {
    /// Create a new Chat Completions model with the given model name and API key.
    ///
    /// Uses the default `OpenAI` base URL (`https://api.openai.com/v1`).
    #[must_use]
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            client: Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Create a new Chat Completions model with a custom base URL.
    ///
    /// This is useful for `OpenAI`-compatible providers or local model servers.
    #[must_use]
    pub fn with_base_url(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            client: Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Create a new Chat Completions model with a pre-configured HTTP client.
    ///
    /// This allows sharing a connection pool or configuring timeouts and proxies.
    #[must_use]
    pub fn with_client(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        client: Client,
    ) -> Self {
        Self {
            model: model.into(),
            client,
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Return the model name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Return the configured base URL.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Build the full endpoint URL for chat completions.
    fn endpoint_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{base}/chat/completions")
    }

    /// Merge extra headers from model settings with the authorization header.
    fn build_headers(&self, model_settings: &ModelSettings) -> Vec<(String, String)> {
        let mut headers = vec![
            (
                "Authorization".to_owned(),
                format!("Bearer {}", self.api_key),
            ),
            ("Content-Type".to_owned(), "application/json".to_owned()),
            ("User-Agent".to_owned(), "openai-agents-rs/0.1.0".to_owned()),
        ];

        if let Some(ref extra) = model_settings.extra_headers {
            for (k, v) in extra {
                headers.push((k.clone(), v.clone()));
            }
        }

        headers
    }

    /// Send the request to the Chat Completions endpoint and return the raw JSON response.
    async fn fetch_response(
        &self,
        system_instructions: Option<&str>,
        input: &[ResponseInputItem],
        model_settings: &ModelSettings,
        tools: &[ToolSpec],
        output_schema: Option<&OutputSchemaSpec>,
        handoffs: &[HandoffToolSpec],
    ) -> Result<serde_json::Value> {
        let body = build_request_body(
            &self.model,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            false,
        );

        debug!(
            model = %self.model,
            endpoint = %self.endpoint_url(),
            "Sending Chat Completions request"
        );
        let headers = self.build_headers(model_settings);
        let mut request = self.client.post(self.endpoint_url());
        for (key, value) in &headers {
            request = request.header(key.as_str(), value.as_str());
        }
        let request = request.json(&body);

        let response = request.send().await?;

        let status = response.status();
        let response_text = response.text().await?;

        if !status.is_success() {
            return Err(AgentError::ModelBehavior {
                message: format!("Chat Completions API returned status {status}: {response_text}"),
            });
        }

        let response_json: serde_json::Value = serde_json::from_str(&response_text)?;

        Ok(response_json)
    }
}

#[async_trait]
impl Model for OpenAIChatCompletionsModel {
    async fn get_response(
        &self,
        system_instructions: Option<&str>,
        input: &[ResponseInputItem],
        model_settings: &ModelSettings,
        tools: &[ToolSpec],
        output_schema: Option<&OutputSchemaSpec>,
        handoffs: &[HandoffToolSpec],
        _tracing: ModelTracing,
        _previous_response_id: Option<&str>,
    ) -> Result<ModelResponse> {
        let response_json = self
            .fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
            )
            .await?;

        convert_response(&response_json)
    }

    #[allow(clippy::too_many_lines)]
    fn stream_response<'a>(
        &'a self,
        system_instructions: Option<&'a str>,
        input: &'a [ResponseInputItem],
        model_settings: &'a ModelSettings,
        tools: &'a [ToolSpec],
        output_schema: Option<&'a OutputSchemaSpec>,
        handoffs: &'a [HandoffToolSpec],
        _tracing: ModelTracing,
        _previous_response_id: Option<&'a str>,
    ) -> Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent>> + Send + 'a>> {
        let body = build_request_body(
            &self.model,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            true,
        );

        let headers = self.build_headers(model_settings);
        let endpoint = self.endpoint_url();
        let client = self.client.clone();

        Box::pin(async_stream::try_stream! {
            let mut request = client.post(endpoint);
            for (key, value) in &headers {
                request = request.header(key.as_str(), value.as_str());
            }
            let request = request.json(&body);

            let response = request.send().await?;

            let status = response.status();
            let response = if status.is_success() {
                response
            } else {
                let error_text = response.text().await.unwrap_or_default();
                Err(AgentError::ModelBehavior {
                    message: format!(
                        "Chat Completions streaming API returned status {status}: {error_text}"
                    ),
                })?;
                // The Err above always diverges; this satisfies the type checker.
                return;
            };

            // State accumulators for streaming deltas.
            let mut accumulated_content = String::new();
            let mut accumulated_tool_calls: Vec<serde_json::Value> = Vec::new();
            let mut stream_usage: Option<serde_json::Value> = None;
            let mut response_id: Option<String> = None;

            // Read the SSE stream line-by-line.
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            use futures::StreamExt as _;

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk_bytes = chunk_result?;
                buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));

                // Process complete lines from the buffer.
                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_owned();
                    buffer = buffer[newline_pos + 1..].to_owned();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    if line == "data: [DONE]" {
                        // Stream finished.
                        break;
                    }

                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };

                    let chunk: serde_json::Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Failed to parse streaming chunk: {e}");
                            continue;
                        }
                    };

                    // Capture response ID.
                    if response_id.is_none() {
                        response_id = chunk.get("id")
                            .and_then(serde_json::Value::as_str)
                            .map(String::from);
                    }

                    // Capture usage from final chunk.
                    if let Some(usage_val) = chunk.get("usage") {
                        if !usage_val.is_null() {
                            stream_usage = Some(usage_val.clone());
                        }
                    }

                    // Process the delta.
                    let delta = chunk
                        .get("choices")
                        .and_then(serde_json::Value::as_array)
                        .and_then(|arr| arr.first())
                        .and_then(|choice| choice.get("delta"));

                    let Some(delta) = delta else { continue };

                    // Accumulate content.
                    if let Some(content) = delta.get("content").and_then(serde_json::Value::as_str)
                    {
                        accumulated_content.push_str(content);

                        // Yield a partial text event.
                        yield json!({
                            "type": "response.output_text.delta",
                            "delta": content,
                            "content_index": 0,
                            "output_index": 0
                        });
                    }

                    // Accumulate tool calls.
                    if let Some(tool_calls) =
                        delta.get("tool_calls").and_then(serde_json::Value::as_array)
                    {
                        for tc_delta in tool_calls {
                            #[allow(clippy::cast_possible_truncation)]
                            let index = tc_delta
                                .get("index")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0) as usize;

                            // Ensure we have enough slots.
                            while accumulated_tool_calls.len() <= index {
                                accumulated_tool_calls.push(json!({
                                    "id": "",
                                    "type": "function",
                                    "function": { "name": "", "arguments": "" }
                                }));
                            }

                            let tc = &mut accumulated_tool_calls[index];

                            if let Some(id) =
                                tc_delta.get("id").and_then(serde_json::Value::as_str)
                            {
                                tc["id"] = json!(id);
                            }
                            if let Some(func) = tc_delta.get("function") {
                                if let Some(name) =
                                    func.get("name").and_then(serde_json::Value::as_str)
                                {
                                    tc["function"]["name"] = json!(name);
                                }
                                if let Some(args) =
                                    func.get("arguments").and_then(serde_json::Value::as_str)
                                {
                                    let existing = tc["function"]["arguments"]
                                        .as_str()
                                        .unwrap_or("");
                                    tc["function"]["arguments"] =
                                        json!(format!("{existing}{args}"));
                                }
                            }
                        }
                    }
                }
            }

            // Build the final response and yield a completed event.
            let mut output_items = Vec::new();

            if !accumulated_content.is_empty() {
                output_items.push(json!({
                    "id": FAKE_RESPONSES_ID,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "output_text", "text": accumulated_content, "annotations": [] }]
                }));
            }

            for tc in &accumulated_tool_calls {
                let function = tc.get("function").unwrap_or(&serde_json::Value::Null);
                output_items.push(json!({
                    "id": FAKE_RESPONSES_ID,
                    "type": "function_call",
                    "call_id": tc.get("id").and_then(serde_json::Value::as_str).unwrap_or(""),
                    "name": function.get("name").and_then(serde_json::Value::as_str).unwrap_or(""),
                    "arguments": function.get("arguments").and_then(serde_json::Value::as_str).unwrap_or("{}")
                }));
            }

            let usage = parse_usage(stream_usage.as_ref());

            let final_response = ModelResponse::new(
                output_items.clone(),
                usage,
                response_id,
                None,
            );

            yield json!({
                "type": "response.completed",
                "response": {
                    "id": final_response.response_id,
                    "output": final_response.output,
                    "usage": {
                        "input_tokens": final_response.usage.input_tokens,
                        "output_tokens": final_response.usage.output_tokens,
                        "total_tokens": final_response.usage.total_tokens,
                    }
                }
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- convert_tool_choice ----

    #[test]
    fn tool_choice_auto() {
        let result = convert_tool_choice(&ToolChoice::Auto);
        assert_eq!(result, json!("auto"));
    }

    #[test]
    fn tool_choice_required() {
        let result = convert_tool_choice(&ToolChoice::Required);
        assert_eq!(result, json!("required"));
    }

    #[test]
    fn tool_choice_none() {
        let result = convert_tool_choice(&ToolChoice::None);
        assert_eq!(result, json!("none"));
    }

    #[test]
    fn tool_choice_named() {
        let result = convert_tool_choice(&ToolChoice::Named("my_func".to_owned()));
        assert_eq!(
            result,
            json!({ "type": "function", "function": { "name": "my_func" } })
        );
    }

    // ---- parse_usage ----

    #[test]
    fn parse_usage_from_full_response() {
        let usage_json = json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": { "cached_tokens": 20 },
            "completion_tokens_details": { "reasoning_tokens": 10 }
        });
        let usage = parse_usage(Some(&usage_json));
        assert_eq!(usage.requests, 1);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.input_tokens_details.cached_tokens, 20);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 10);
    }

    #[test]
    fn parse_usage_from_minimal_response() {
        let usage_json = json!({
            "prompt_tokens": 42,
            "completion_tokens": 17
        });
        let usage = parse_usage(Some(&usage_json));
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 17);
        assert_eq!(usage.total_tokens, 59);
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
    }

    #[test]
    fn parse_usage_none() {
        let usage = parse_usage(None);
        assert_eq!(usage.requests, 0);
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }

    // ---- convert_input_to_messages ----

    #[test]
    fn convert_empty_input_with_system() {
        let messages = convert_input_to_messages(Some("You are helpful."), &[]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
    }

    #[test]
    fn convert_empty_input_without_system() {
        let messages = convert_input_to_messages(None, &[]);
        assert!(messages.is_empty());
    }

    #[test]
    fn convert_easy_input_message() {
        let input = vec![json!({ "role": "user", "content": "Hello!" })];
        let messages = convert_input_to_messages(None, &input);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "Hello!");
    }

    #[test]
    fn convert_typed_message() {
        let input = vec![json!({
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": "Hello typed" }]
        })];
        let messages = convert_input_to_messages(None, &input);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "Hello typed");
    }

    #[test]
    fn convert_function_call_and_output() {
        let input = vec![
            json!({
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": "{\"city\":\"SF\"}"
            }),
            json!({
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Sunny, 72F"
            }),
        ];
        let messages = convert_input_to_messages(None, &input);

        // First message: assistant with tool_calls.
        assert_eq!(messages[0]["role"], "assistant");
        let tool_calls = messages[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call_123");
        assert_eq!(tool_calls[0]["function"]["name"], "get_weather");

        // Second message: tool response.
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_123");
        assert_eq!(messages[1]["content"], "Sunny, 72F");
    }

    #[test]
    fn convert_multiple_function_calls_grouped() {
        let input = vec![
            json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "tool_a",
                "arguments": "{}"
            }),
            json!({
                "type": "function_call",
                "call_id": "call_2",
                "name": "tool_b",
                "arguments": "{}"
            }),
            json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result_a"
            }),
            json!({
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "result_b"
            }),
        ];
        let messages = convert_input_to_messages(None, &input);

        // The two function calls should be grouped into one assistant message.
        assert_eq!(messages[0]["role"], "assistant");
        let tool_calls = messages[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 2);

        // Followed by two tool messages.
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[2]["role"], "tool");
    }

    #[test]
    fn convert_assistant_easy_message() {
        let input = vec![
            json!({ "role": "user", "content": "Hi" }),
            json!({ "role": "assistant", "content": "Hello there!" }),
            json!({ "role": "user", "content": "How are you?" }),
        ];
        let messages = convert_input_to_messages(None, &input);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "Hello there!");
        assert_eq!(messages[2]["role"], "user");
    }

    // ---- convert_response ----

    #[test]
    fn convert_text_only_response() {
        let response = json!({
            "id": "chatcmpl-abc123",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        });

        let model_response = convert_response(&response).unwrap();
        assert_eq!(model_response.output.len(), 1);
        assert_eq!(model_response.output[0]["type"], "message");

        let content = model_response.output[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "output_text");
        assert_eq!(content[0]["text"], "Hello, how can I help?");

        assert_eq!(model_response.usage.input_tokens, 10);
        assert_eq!(model_response.usage.output_tokens, 8);
        assert_eq!(
            model_response.response_id.as_deref(),
            Some("chatcmpl-abc123")
        );
    }

    #[test]
    fn convert_tool_call_response() {
        let response = json!({
            "id": "chatcmpl-xyz",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"San Francisco\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        });

        let model_response = convert_response(&response).unwrap();
        assert_eq!(model_response.output.len(), 1);
        assert_eq!(model_response.output[0]["type"], "function_call");
        assert_eq!(model_response.output[0]["call_id"], "call_abc");
        assert_eq!(model_response.output[0]["name"], "get_weather");
        assert_eq!(
            model_response.output[0]["arguments"],
            "{\"location\":\"San Francisco\"}"
        );
    }

    #[test]
    fn convert_text_and_tool_calls_response() {
        let response = json!({
            "id": "chatcmpl-both",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check that for you.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{\"q\":\"test\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 12,
                "total_tokens": 27
            }
        });

        let model_response = convert_response(&response).unwrap();
        // Should have both a message item and a function_call item.
        assert_eq!(model_response.output.len(), 2);
        assert_eq!(model_response.output[0]["type"], "message");
        assert_eq!(model_response.output[1]["type"], "function_call");
    }

    #[test]
    fn convert_response_no_choices_error() {
        let response = json!({ "id": "chatcmpl-empty", "choices": [] });
        let result = convert_response(&response);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AgentError::ModelBehavior { .. }));
    }

    #[test]
    fn convert_response_refusal() {
        let response = json!({
            "id": "chatcmpl-ref",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "refusal": "I cannot help with that."
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
        });

        let model_response = convert_response(&response).unwrap();
        assert_eq!(model_response.output.len(), 1);
        let content = model_response.output[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "refusal");
        assert_eq!(content[0]["refusal"], "I cannot help with that.");
    }

    // ---- build_request_body ----

    #[test]
    fn build_body_minimal() {
        let body = build_request_body(
            "gpt-4o",
            None,
            &[json!({ "role": "user", "content": "hi" })],
            &ModelSettings::default(),
            &[],
            None,
            &[],
            false,
        );

        assert_eq!(body["model"], "gpt-4o");
        assert!(body.get("tools").is_none());
        assert!(body.get("stream").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn build_body_with_tools() {
        let tools = vec![ToolSpec {
            name: "get_weather".to_owned(),
            description: "Gets weather info.".to_owned(),
            params_json_schema: json!({ "type": "object" }),
            strict: true,
        }];

        let body = build_request_body(
            "gpt-4o",
            Some("Be helpful."),
            &[],
            &ModelSettings::default(),
            &tools,
            None,
            &[],
            false,
        );

        let tools_arr = body["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["function"]["name"], "get_weather");
        assert_eq!(tools_arr[0]["function"]["strict"], true);
    }

    #[test]
    fn build_body_with_handoffs() {
        let handoffs = vec![HandoffToolSpec {
            tool_name: "transfer_to_support".to_owned(),
            tool_description: "Transfer to support agent.".to_owned(),
            input_json_schema: json!({ "type": "object" }),
            strict: false,
        }];

        let body = build_request_body(
            "gpt-4o",
            None,
            &[],
            &ModelSettings::default(),
            &[],
            None,
            &handoffs,
            false,
        );

        let tools_arr = body["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["function"]["name"], "transfer_to_support");
    }

    #[test]
    fn build_body_with_model_settings() {
        let settings = ModelSettings {
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(1024),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.3),
            tool_choice: Some(ToolChoice::Required),
            ..Default::default()
        };

        let body = build_request_body("gpt-4o", None, &[], &settings, &[], None, &[], false);

        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["frequency_penalty"], 0.5);
        assert_eq!(body["presence_penalty"], 0.3);
        assert_eq!(body["tool_choice"], "required");
    }

    #[test]
    fn build_body_with_output_schema() {
        let schema = OutputSchemaSpec {
            json_schema: json!({
                "type": "object",
                "properties": { "answer": { "type": "string" } }
            }),
            strict: true,
        };

        let body = build_request_body(
            "gpt-4o",
            None,
            &[],
            &ModelSettings::default(),
            &[],
            Some(&schema),
            &[],
            false,
        );

        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            "final_output"
        );
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn build_body_with_streaming() {
        let body = build_request_body(
            "gpt-4o",
            None,
            &[],
            &ModelSettings::default(),
            &[],
            None,
            &[],
            true,
        );

        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    // ---- OpenAIChatCompletionsModel construction ----

    #[test]
    fn model_construction_default_base_url() {
        let model = OpenAIChatCompletionsModel::new("gpt-4o", "sk-test");
        assert_eq!(model.model_name(), "gpt-4o");
        assert_eq!(model.base_url(), DEFAULT_BASE_URL);
    }

    #[test]
    fn model_construction_custom_base_url() {
        let model = OpenAIChatCompletionsModel::with_base_url(
            "local-model",
            "no-key",
            "http://localhost:8080/v1",
        );
        assert_eq!(model.model_name(), "local-model");
        assert_eq!(model.base_url(), "http://localhost:8080/v1");
    }

    #[test]
    fn model_endpoint_url() {
        let model = OpenAIChatCompletionsModel::new("gpt-4o", "sk-test");
        assert_eq!(
            model.endpoint_url(),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn model_endpoint_url_trailing_slash() {
        let model = OpenAIChatCompletionsModel::with_base_url(
            "gpt-4o",
            "sk-test",
            "https://api.openai.com/v1/",
        );
        assert_eq!(
            model.endpoint_url(),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn model_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OpenAIChatCompletionsModel>();
    }

    #[test]
    fn model_is_clone() {
        let original = OpenAIChatCompletionsModel::new("gpt-4o", "sk-test");
        let cloned = original.clone();
        // Verify both the clone and original are independently usable.
        assert_eq!(original.model_name(), "gpt-4o");
        assert_eq!(cloned.model_name(), "gpt-4o");
    }

    // ---- extract_message_content ----

    #[test]
    fn extract_content_from_string() {
        let item = json!({ "content": "Hello world" });
        let content = extract_message_content(&item);
        assert_eq!(content, "Hello world");
    }

    #[test]
    fn extract_content_from_output_text_array() {
        let item = json!({
            "content": [
                { "type": "output_text", "text": "First " },
                { "type": "output_text", "text": "Second" }
            ]
        });
        let content = extract_message_content(&item);
        assert_eq!(content, "First Second");
    }

    #[test]
    fn extract_content_from_input_text_array() {
        let item = json!({
            "content": [
                { "type": "input_text", "text": "typed input" }
            ]
        });
        let content = extract_message_content(&item);
        assert_eq!(content, "typed input");
    }

    #[test]
    fn extract_content_missing() {
        let item = json!({ "role": "user" });
        let content = extract_message_content(&item);
        assert_eq!(content, "");
    }

    // ---- Integration-like: full round-trip conversion ----

    #[test]
    fn round_trip_user_message_and_response() {
        // Simulate: user sends message, model responds with text.
        let input = vec![json!({ "role": "user", "content": "What is 2+2?" })];

        let messages = convert_input_to_messages(Some("You are a math tutor."), &input);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");

        let api_response = json!({
            "id": "chatcmpl-test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "2+2 equals 4."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        });

        let model_response = convert_response(&api_response).unwrap();
        assert_eq!(model_response.output.len(), 1);
        assert_eq!(model_response.usage.requests, 1);
        assert_eq!(model_response.usage.total_tokens, 30);
    }

    #[test]
    fn round_trip_tool_call_and_result() {
        // Simulate: user asks, model calls tool, tool returns, model responds.
        let input = vec![json!({ "role": "user", "content": "What is the weather in SF?" })];

        let messages = convert_input_to_messages(None, &input);
        assert_eq!(messages.len(), 1);

        // Model responds with a tool call.
        let tool_response = json!({
            "id": "chatcmpl-tool",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"San Francisco\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 25, "completion_tokens": 20, "total_tokens": 45 }
        });

        let first_response = convert_response(&tool_response).unwrap();
        assert_eq!(first_response.output.len(), 1);
        assert_eq!(first_response.output[0]["type"], "function_call");

        // Now feed back the tool result.
        let second_input = vec![
            json!({ "role": "user", "content": "What is the weather in SF?" }),
            json!({
                "type": "function_call",
                "call_id": "call_weather",
                "name": "get_weather",
                "arguments": "{\"city\":\"San Francisco\"}"
            }),
            json!({
                "type": "function_call_output",
                "call_id": "call_weather",
                "output": "Sunny, 68F"
            }),
        ];

        let messages2 = convert_input_to_messages(None, &second_input);
        assert_eq!(messages2.len(), 3);
        assert_eq!(messages2[0]["role"], "user");
        assert_eq!(messages2[1]["role"], "assistant");
        assert_eq!(messages2[2]["role"], "tool");
        assert_eq!(messages2[2]["content"], "Sunny, 68F");
    }
}
