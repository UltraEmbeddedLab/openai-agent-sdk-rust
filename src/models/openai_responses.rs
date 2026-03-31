//! `OpenAI` Responses API model implementation.
//!
//! This module provides [`OpenAIResponsesModel`], a concrete implementation of the
//! [`Model`] trait that calls the `OpenAI` `/v1/responses` endpoint. It supports both
//! synchronous (non-streaming) and streaming response modes, tool use, structured
//! output via JSON schemas, and handoff tools.
//!
//! Additionally, [`OpenAIProvider`] implements [`ModelProvider`] to create
//! [`OpenAIResponsesModel`] instances by model name.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;
use tokio_stream::Stream;
use tracing::debug;

use crate::config::ModelSettings;
use crate::error::{AgentError, Result};
use crate::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use crate::models::{
    HandoffToolSpec, Model, ModelProvider, ModelTracing, OutputSchemaSpec, ToolSpec,
};
use crate::usage::Usage;

/// User agent string sent with every request.
const USER_AGENT: &str = "openai-agent-sdk-rust/0.1.0";

/// Default base URL for the `OpenAI` API.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// An `OpenAI` model using the Responses API.
///
/// This implementation calls the `OpenAI` `/v1/responses` endpoint to generate
/// completions with tool use, structured output, and streaming support.
#[derive(Debug, Clone)]
pub struct OpenAIResponsesModel {
    /// The model identifier (e.g., "gpt-4o", "o3-mini").
    model: String,
    /// The HTTP client for making API calls.
    client: Client,
    /// The `OpenAI` API key.
    api_key: String,
    /// The base URL for the API (defaults to `https://api.openai.com/v1`).
    base_url: String,
}

impl OpenAIResponsesModel {
    /// Create a new Responses API model with the given model name.
    ///
    /// Reads the API key from the `OPENAI_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the `OPENAI_API_KEY` environment variable is not set.
    pub fn new(model: impl Into<String>) -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| AgentError::UserError {
            message: "OPENAI_API_KEY environment variable is not set".to_owned(),
        })?;
        Ok(Self {
            model: model.into(),
            client: Client::new(),
            api_key,
            base_url: DEFAULT_BASE_URL.to_owned(),
        })
    }

    /// Create a model with a custom API key and base URL.
    #[must_use]
    pub fn with_config(
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

    /// Return the model identifier.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Build the JSON request body for the Responses API.
    #[allow(clippy::too_many_arguments)]
    fn build_request_body(
        &self,
        system_instructions: Option<&str>,
        input: &[ResponseInputItem],
        model_settings: &ModelSettings,
        tools: &[ToolSpec],
        output_schema: Option<&OutputSchemaSpec>,
        handoffs: &[HandoffToolSpec],
        previous_response_id: Option<&str>,
    ) -> serde_json::Value {
        let mut body = json!({
            "model": self.model,
            "input": input,
        });

        // System instructions.
        if let Some(instructions) = system_instructions {
            body["instructions"] = json!(instructions);
        }

        // Convert tools to the Responses API format.
        let mut api_tools: Vec<serde_json::Value> = Vec::new();
        for tool in tools {
            api_tools.push(json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.params_json_schema,
                "strict": tool.strict,
            }));
        }
        // Convert handoff tools to function tools.
        for handoff in handoffs {
            api_tools.push(json!({
                "type": "function",
                "name": handoff.tool_name,
                "description": handoff.tool_description,
                "parameters": handoff.input_json_schema,
                "strict": handoff.strict,
            }));
        }
        if !api_tools.is_empty() {
            body["tools"] = json!(api_tools);
        }

        // Model settings.
        if let Some(temp) = model_settings.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(top_p) = model_settings.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(max_tokens) = model_settings.max_tokens {
            body["max_output_tokens"] = json!(max_tokens);
        }
        if let Some(ref truncation) = model_settings.truncation {
            body["truncation"] = serde_json::to_value(truncation).unwrap_or_else(|_| json!("auto"));
        }
        if let Some(ref tool_choice) = model_settings.tool_choice {
            body["tool_choice"] =
                serde_json::to_value(tool_choice).unwrap_or_else(|_| json!("auto"));
        }
        if let Some(parallel) = model_settings.parallel_tool_calls {
            body["parallel_tool_calls"] = json!(parallel);
        }
        if let Some(store) = model_settings.store {
            body["store"] = json!(store);
        }
        if let Some(ref metadata) = model_settings.metadata {
            body["metadata"] = json!(metadata);
        }
        if let Some(prev_id) = previous_response_id {
            body["previous_response_id"] = json!(prev_id);
        }

        // Output schema (structured output).
        //
        // The Responses API requires a `name` field inside `json_schema`.
        // We use the schema's own `"title"` property when available, falling back
        // to the generic name `"final_output"`.
        if let Some(schema) = output_schema {
            let schema_name = schema
                .json_schema
                .get("title")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("final_output")
                .to_owned();
            body["text"] = json!({
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema.json_schema,
                    "strict": schema.strict,
                }
            });
        }

        // Merge extra body fields.
        if let Some(ref extra_body) = model_settings.extra_body {
            if let Some(extra_obj) = extra_body.as_object() {
                if let Some(body_obj) = body.as_object_mut() {
                    for (key, value) in extra_obj {
                        body_obj.insert(key.clone(), value.clone());
                    }
                }
            }
        }

        // Merge extra args.
        if let Some(ref extra_args) = model_settings.extra_args {
            if let Some(body_obj) = body.as_object_mut() {
                for (key, value) in extra_args {
                    body_obj.insert(key.clone(), value.clone());
                }
            }
        }

        body
    }

    /// Parse a Responses API JSON response into a [`ModelResponse`].
    #[must_use]
    fn parse_response(response_json: &serde_json::Value) -> ModelResponse {
        let output = response_json
            .get("output")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();

        let usage = Self::parse_usage(response_json);

        let response_id = response_json
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(String::from);

        ModelResponse::new(output, usage, response_id, None)
    }

    /// Extract usage information from a Responses API JSON response.
    fn parse_usage(response_json: &serde_json::Value) -> Usage {
        let Some(usage_json) = response_json.get("usage") else {
            return Usage::default();
        };

        let input_tokens = usage_json
            .get("input_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let output_tokens = usage_json
            .get("output_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let total_tokens = usage_json
            .get("total_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);

        Usage {
            requests: 1,
            input_tokens,
            output_tokens,
            total_tokens,
            ..Usage::default()
        }
    }

    /// Build the responses endpoint URL.
    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url)
    }
}

/// Parse a single SSE event from the buffer, removing the consumed portion.
///
/// SSE events are separated by double newlines (`\n\n`). Each event line may
/// start with `data: `. Returns the data payload of the first complete event
/// in the buffer, or `None` if no complete event is available yet.
fn parse_sse_event(buffer: &mut String) -> Option<String> {
    let pos = buffer.find("\n\n")?;
    let event_block = buffer[..pos].to_string();
    // Remove the consumed event (including the double newline separator).
    *buffer = buffer[pos + 2..].to_string();

    // Extract the data field from the event block.
    for line in event_block.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            return Some(data.to_string());
        }
        // Handle `data:` with no space (edge case per SSE spec).
        if let Some(data) = line.strip_prefix("data:") {
            return Some(data.to_string());
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
#[async_trait]
impl Model for OpenAIResponsesModel {
    async fn get_response(
        &self,
        system_instructions: Option<&str>,
        input: &[ResponseInputItem],
        model_settings: &ModelSettings,
        tools: &[ToolSpec],
        output_schema: Option<&OutputSchemaSpec>,
        handoffs: &[HandoffToolSpec],
        tracing: ModelTracing,
        previous_response_id: Option<&str>,
    ) -> Result<ModelResponse> {
        let body = self.build_request_body(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            previous_response_id,
        );

        if tracing.include_data() {
            debug!(
                model = %self.model,
                input = %serde_json::to_string(input).unwrap_or_default(),
                "Calling OpenAI Responses API"
            );
        } else if !tracing.is_disabled() {
            debug!(model = %self.model, "Calling OpenAI Responses API");
        }

        let mut request = self
            .client
            .post(self.responses_url())
            .bearer_auth(&self.api_key)
            .header("User-Agent", USER_AGENT)
            .json(&body);

        // Apply extra headers from model settings.
        if let Some(ref extra_headers) = model_settings.extra_headers {
            for (key, value) in extra_headers {
                request = request.header(key.as_str(), value.as_str());
            }
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            return Err(AgentError::ModelBehavior {
                message: format!("OpenAI API error {status}: {error_body}"),
            });
        }

        let response_json: serde_json::Value = response.json().await?;

        if tracing.include_data() {
            debug!(
                response_id = response_json
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
                "OpenAI Responses API response received"
            );
        }

        Ok(Self::parse_response(&response_json))
    }

    fn stream_response<'a>(
        &'a self,
        system_instructions: Option<&'a str>,
        input: &'a [ResponseInputItem],
        model_settings: &'a ModelSettings,
        tools: &'a [ToolSpec],
        output_schema: Option<&'a OutputSchemaSpec>,
        handoffs: &'a [HandoffToolSpec],
        tracing: ModelTracing,
        previous_response_id: Option<&'a str>,
    ) -> Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent>> + Send + 'a>> {
        Box::pin(async_stream::try_stream! {
            let mut body = self.build_request_body(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                previous_response_id,
            );
            body["stream"] = json!(true);

            if tracing.include_data() {
                debug!(
                    model = %self.model,
                    "Calling OpenAI Responses API (streaming)"
                );
            }

            let mut request = self
                .client
                .post(self.responses_url())
                .bearer_auth(&self.api_key)
                .header("User-Agent", USER_AGENT)
                .json(&body);

            // Apply extra headers from model settings.
            if let Some(ref extra_headers) = model_settings.extra_headers {
                for (key, value) in extra_headers {
                    request = request.header(key.as_str(), value.as_str());
                }
            }

            let response = request.send().await?;

            let response = if response.status().is_success() {
                response
            } else {
                let status = response.status();
                let error_body = response.text().await.unwrap_or_default();
                Err(AgentError::ModelBehavior {
                    message: format!("OpenAI API error {status}: {error_body}"),
                })?;
                return;
            };

            // Read the SSE stream.
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = chunk_result?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Parse all complete SSE events from the buffer.
                while let Some(data) = parse_sse_event(&mut buffer) {
                    if data == "[DONE]" {
                        return;
                    }
                    let event_json: serde_json::Value =
                        serde_json::from_str(&data)?;
                    yield event_json;
                }
            }
        })
    }
}

/// An `OpenAI` model provider that creates Responses API models.
///
/// Reads the API key from the `OPENAI_API_KEY` environment variable on creation.
/// Use [`OpenAIProvider::with_config`] to supply credentials explicitly.
#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    /// The `OpenAI` API key.
    api_key: String,
    /// The base URL for the API.
    base_url: String,
}

impl OpenAIProvider {
    /// Create a new provider, reading the API key from the `OPENAI_API_KEY`
    /// environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment variable is not set.
    pub fn new() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| AgentError::UserError {
            message: "OPENAI_API_KEY environment variable is not set".to_owned(),
        })?;
        Ok(Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_owned(),
        })
    }

    /// Create a provider with explicit credentials.
    #[must_use]
    pub fn with_config(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }
}

#[async_trait]
impl ModelProvider for OpenAIProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let name = model_name.unwrap_or("gpt-4o");
        Ok(Arc::new(OpenAIResponsesModel::with_config(
            name,
            &self.api_key,
            &self.base_url,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- build_request_body ----

    fn make_model() -> OpenAIResponsesModel {
        OpenAIResponsesModel::with_config("gpt-4o", "test-key", "https://api.example.com/v1")
    }

    #[test]
    fn build_request_body_minimal() {
        let model = make_model();
        let body =
            model.build_request_body(None, &[], &ModelSettings::default(), &[], None, &[], None);

        assert_eq!(body["model"], "gpt-4o");
        assert!(body["input"].is_array());
        assert!(body.get("instructions").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn build_request_body_with_instructions() {
        let model = make_model();
        let body = model.build_request_body(
            Some("You are helpful."),
            &[],
            &ModelSettings::default(),
            &[],
            None,
            &[],
            None,
        );

        assert_eq!(body["instructions"], "You are helpful.");
    }

    #[test]
    fn build_request_body_with_tools() {
        let model = make_model();
        let tools = vec![ToolSpec {
            name: "get_weather".to_owned(),
            description: "Get weather for a location.".to_owned(),
            params_json_schema: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
            strict: true,
        }];

        let body = model.build_request_body(
            None,
            &[],
            &ModelSettings::default(),
            &tools,
            None,
            &[],
            None,
        );

        let api_tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(api_tools.len(), 1);
        assert_eq!(api_tools[0]["type"], "function");
        assert_eq!(api_tools[0]["name"], "get_weather");
        assert_eq!(api_tools[0]["strict"], true);
    }

    #[test]
    fn build_request_body_with_handoffs() {
        let model = make_model();
        let handoffs = vec![HandoffToolSpec {
            tool_name: "transfer_to_support".to_owned(),
            tool_description: "Transfer to support agent.".to_owned(),
            input_json_schema: json!({"type": "object"}),
            strict: false,
        }];

        let body = model.build_request_body(
            None,
            &[],
            &ModelSettings::default(),
            &[],
            None,
            &handoffs,
            None,
        );

        let api_tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(api_tools.len(), 1);
        assert_eq!(api_tools[0]["name"], "transfer_to_support");
        assert!(!api_tools[0]["strict"].as_bool().unwrap_or(true));
    }

    #[test]
    fn build_request_body_with_tools_and_handoffs() {
        let model = make_model();
        let tools = vec![ToolSpec {
            name: "search".to_owned(),
            description: "Search the web.".to_owned(),
            params_json_schema: json!({}),
            strict: false,
        }];
        let handoffs = vec![HandoffToolSpec {
            tool_name: "handoff".to_owned(),
            tool_description: "Hand off.".to_owned(),
            input_json_schema: json!({}),
            strict: true,
        }];

        let body = model.build_request_body(
            None,
            &[],
            &ModelSettings::default(),
            &tools,
            None,
            &handoffs,
            None,
        );

        let api_tools = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(api_tools.len(), 2);
        assert_eq!(api_tools[0]["name"], "search");
        assert_eq!(api_tools[1]["name"], "handoff");
    }

    #[test]
    fn build_request_body_with_model_settings() {
        let model = make_model();
        let settings = ModelSettings {
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(1024),
            store: Some(true),
            ..ModelSettings::default()
        };

        let body = model.build_request_body(None, &[], &settings, &[], None, &[], None);

        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["max_output_tokens"], 1024);
        assert_eq!(body["store"], true);
    }

    #[test]
    fn build_request_body_with_output_schema() {
        let model = make_model();
        let schema = OutputSchemaSpec {
            json_schema: json!({
                "name": "answer_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"}
                    }
                }
            }),
            strict: true,
        };

        let body = model.build_request_body(
            None,
            &[],
            &ModelSettings::default(),
            &[],
            Some(&schema),
            &[],
            None,
        );

        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert!(body["text"]["format"]["strict"].as_bool().unwrap_or(false));
    }

    #[test]
    fn build_request_body_with_previous_response_id() {
        let model = make_model();
        let body = model.build_request_body(
            None,
            &[],
            &ModelSettings::default(),
            &[],
            None,
            &[],
            Some("resp_abc123"),
        );

        assert_eq!(body["previous_response_id"], "resp_abc123");
    }

    #[test]
    fn build_request_body_with_input_items() {
        let model = make_model();
        let input = vec![json!({"role": "user", "content": "Hello!"})];
        let body = model.build_request_body(
            None,
            &input,
            &ModelSettings::default(),
            &[],
            None,
            &[],
            None,
        );

        let api_input = body["input"].as_array().expect("input should be an array");
        assert_eq!(api_input.len(), 1);
        assert_eq!(api_input[0]["role"], "user");
        assert_eq!(api_input[0]["content"], "Hello!");
    }

    #[test]
    fn build_request_body_merges_extra_body() {
        let model = make_model();
        let settings = ModelSettings {
            extra_body: Some(json!({"custom_field": "custom_value"})),
            ..ModelSettings::default()
        };

        let body = model.build_request_body(None, &[], &settings, &[], None, &[], None);

        assert_eq!(body["custom_field"], "custom_value");
    }

    #[test]
    fn build_request_body_merges_extra_args() {
        let model = make_model();
        let mut extra_args = std::collections::HashMap::new();
        extra_args.insert("reasoning".to_owned(), json!({"effort": "high"}));
        let settings = ModelSettings {
            extra_args: Some(extra_args),
            ..ModelSettings::default()
        };

        let body = model.build_request_body(None, &[], &settings, &[], None, &[], None);

        assert_eq!(body["reasoning"]["effort"], "high");
    }

    // ---- parse_response ----

    #[test]
    fn parse_response_full() {
        let response_json = json!({
            "id": "resp_001",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}]
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            }
        });

        let response = OpenAIResponsesModel::parse_response(&response_json);

        assert_eq!(response.response_id.as_deref(), Some("resp_001"));
        assert_eq!(response.output.len(), 1);
        assert_eq!(response.usage.requests, 1);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[test]
    fn parse_response_no_usage() {
        let response_json = json!({
            "id": "resp_002",
            "output": []
        });

        let response = OpenAIResponsesModel::parse_response(&response_json);

        assert_eq!(response.response_id.as_deref(), Some("resp_002"));
        assert!(response.output.is_empty());
        assert_eq!(response.usage.requests, 0);
        assert_eq!(response.usage.input_tokens, 0);
    }

    #[test]
    fn parse_response_no_output() {
        let response_json = json!({
            "id": "resp_003"
        });

        let response = OpenAIResponsesModel::parse_response(&response_json);

        assert!(response.output.is_empty());
    }

    #[test]
    fn parse_response_no_id() {
        let response_json = json!({
            "output": [{"type": "message"}],
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        });

        let response = OpenAIResponsesModel::parse_response(&response_json);

        assert!(response.response_id.is_none());
        assert_eq!(response.output.len(), 1);
    }

    #[test]
    fn parse_response_with_tool_calls() {
        let response_json = json!({
            "id": "resp_004",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_001",
                    "name": "get_weather",
                    "arguments": "{\"location\":\"NYC\"}"
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30
            }
        });

        let response = OpenAIResponsesModel::parse_response(&response_json);

        assert_eq!(response.output.len(), 1);
        assert_eq!(response.output[0]["type"], "function_call");
        assert_eq!(response.output[0]["name"], "get_weather");
    }

    // ---- parse_sse_event ----

    #[test]
    fn parse_sse_event_simple() {
        let mut buffer = "data: {\"type\":\"response.created\"}\n\n".to_string();
        let event = parse_sse_event(&mut buffer);
        assert_eq!(event, Some("{\"type\":\"response.created\"}".to_string()));
        assert!(buffer.is_empty());
    }

    #[test]
    fn parse_sse_event_done() {
        let mut buffer = "data: [DONE]\n\n".to_string();
        let event = parse_sse_event(&mut buffer);
        assert_eq!(event, Some("[DONE]".to_string()));
    }

    #[test]
    fn parse_sse_event_incomplete() {
        let mut buffer = "data: {\"partial\":true}".to_string();
        let event = parse_sse_event(&mut buffer);
        assert!(event.is_none());
        // Buffer should be unchanged.
        assert_eq!(buffer, "data: {\"partial\":true}");
    }

    #[test]
    fn parse_sse_event_multiple_events() {
        let mut buffer =
            "data: {\"type\":\"first\"}\n\ndata: {\"type\":\"second\"}\n\n".to_string();

        let first = parse_sse_event(&mut buffer);
        assert_eq!(first, Some("{\"type\":\"first\"}".to_string()));

        let second = parse_sse_event(&mut buffer);
        assert_eq!(second, Some("{\"type\":\"second\"}".to_string()));

        let third = parse_sse_event(&mut buffer);
        assert!(third.is_none());
    }

    #[test]
    fn parse_sse_event_with_event_type_line() {
        let mut buffer = "event: message\ndata: {\"hello\":true}\n\n".to_string();
        let event = parse_sse_event(&mut buffer);
        assert_eq!(event, Some("{\"hello\":true}".to_string()));
    }

    #[test]
    fn parse_sse_event_no_space_after_data_colon() {
        let mut buffer = "data:{\"compact\":true}\n\n".to_string();
        let event = parse_sse_event(&mut buffer);
        assert_eq!(event, Some("{\"compact\":true}".to_string()));
    }

    #[test]
    fn parse_sse_event_no_data_line() {
        let mut buffer = "event: ping\n\n".to_string();
        let event = parse_sse_event(&mut buffer);
        // No data field, so None.
        assert!(event.is_none());
    }

    // ---- parse_usage ----

    #[test]
    fn parse_usage_complete() {
        let json = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150
            }
        });
        let usage = OpenAIResponsesModel::parse_usage(&json);
        assert_eq!(usage.requests, 1);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn parse_usage_missing() {
        let json = json!({});
        let usage = OpenAIResponsesModel::parse_usage(&json);
        assert_eq!(usage.requests, 0);
        assert_eq!(usage.input_tokens, 0);
    }

    #[test]
    fn parse_usage_partial() {
        let json = json!({
            "usage": {
                "input_tokens": 42
            }
        });
        let usage = OpenAIResponsesModel::parse_usage(&json);
        assert_eq!(usage.requests, 1);
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    // ---- OpenAIResponsesModel construction ----

    #[test]
    fn with_config_sets_fields() {
        let model = OpenAIResponsesModel::with_config(
            "gpt-4o-mini",
            "sk-test-key",
            "https://custom.api.com/v1",
        );
        assert_eq!(model.model_name(), "gpt-4o-mini");
        assert_eq!(model.api_key, "sk-test-key");
        assert_eq!(model.base_url, "https://custom.api.com/v1");
    }

    #[test]
    fn responses_url_is_correct() {
        let model = make_model();
        assert_eq!(
            model.responses_url(),
            "https://api.example.com/v1/responses"
        );
    }

    #[test]
    fn new_without_env_var_returns_error() {
        // Ensure the env var is not set for this test.
        temp_env::with_var_unset("OPENAI_API_KEY", || {
            let result = OpenAIResponsesModel::new("gpt-4o");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                matches!(err, AgentError::UserError { .. }),
                "expected UserError, got {err:?}"
            );
        });
    }

    #[test]
    fn new_with_env_var_succeeds() {
        temp_env::with_var("OPENAI_API_KEY", Some("sk-env-key"), || {
            let model = OpenAIResponsesModel::new("gpt-4o").expect("should succeed");
            assert_eq!(model.model_name(), "gpt-4o");
            assert_eq!(model.api_key, "sk-env-key");
            assert_eq!(model.base_url, DEFAULT_BASE_URL);
        });
    }

    // ---- OpenAIProvider ----

    #[test]
    fn provider_with_config() {
        let provider = OpenAIProvider::with_config("sk-key", "https://api.example.com/v1");
        let model = provider
            .get_model(Some("gpt-4o-mini"))
            .expect("should succeed");
        // Verify we got a model back (type-erased).
        let _ = model;
    }

    #[test]
    fn provider_get_model_default_name() {
        let provider = OpenAIProvider::with_config("sk-key", "https://api.example.com/v1");
        let model = provider.get_model(None).expect("should succeed");
        let _ = model;
    }

    #[test]
    fn provider_new_without_env_var_returns_error() {
        temp_env::with_var_unset("OPENAI_API_KEY", || {
            let result = OpenAIProvider::new();
            assert!(result.is_err());
        });
    }

    #[test]
    fn provider_new_with_env_var_succeeds() {
        temp_env::with_var("OPENAI_API_KEY", Some("sk-provider-key"), || {
            let provider = OpenAIProvider::new().expect("should succeed");
            let model = provider.get_model(Some("o3-mini")).expect("should succeed");
            let _ = model;
        });
    }

    // ---- Model trait object safety for OpenAIResponsesModel ----

    #[test]
    fn model_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OpenAIResponsesModel>();
    }

    #[test]
    fn model_can_be_arc_dyn() {
        let model = make_model();
        let _arc: Arc<dyn Model> = Arc::new(model);
    }

    #[test]
    fn provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OpenAIProvider>();
    }
}
