//! Model interface traits and specification types.
//!
//! This module defines the core [`Model`] and [`ModelProvider`] traits that abstract
//! over different LLM backends, along with supporting types like [`ToolSpec`],
//! [`HandoffToolSpec`], [`OutputSchemaSpec`], and [`ModelTracing`].
//!
//! The SDK provides built-in implementations for `OpenAI`'s Responses API and Chat
//! Completions API in submodules. To add support for a custom model backend,
//! implement the [`Model`] trait.

pub mod any_provider;
pub mod litellm;
pub mod multi_provider;
pub mod openai_chatcompletions;
pub mod openai_responses;

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_stream::Stream;

use crate::config::ModelSettings;
use crate::error::Result;
use crate::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};

pub use any_provider::AnyProvider;
pub use litellm::{LiteLLMModel, LiteLLMProvider};
pub use multi_provider::MultiProvider;
pub use openai_responses::{OpenAIProvider, OpenAIResponsesModel};

/// Tracing configuration for model calls.
///
/// Controls whether tracing is active and whether request/response data
/// should be included in traces. The default is [`ModelTracing::Enabled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ModelTracing {
    /// Tracing is disabled entirely.
    Disabled,
    /// Tracing is enabled, including request/response data.
    #[default]
    Enabled,
    /// Tracing is enabled but sensitive request/response data is excluded.
    EnabledWithoutData,
}

impl ModelTracing {
    /// Returns `true` if tracing is disabled.
    #[must_use]
    pub const fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }

    /// Returns `true` if request/response data should be included in traces.
    #[must_use]
    pub const fn include_data(&self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Specification of a tool as seen by the model.
///
/// This is a type-erased representation of a tool that does not carry the
/// context generic `C`. It is passed to [`Model::get_response`] and
/// [`Model::stream_response`] so that model implementations do not need
/// to be generic over the agent context type.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolSpec {
    /// The tool's name.
    pub name: String,
    /// A human-readable description of what the tool does.
    pub description: String,
    /// JSON schema describing the tool's parameters.
    pub params_json_schema: serde_json::Value,
    /// Whether to enforce strict JSON schema validation on the tool's parameters.
    pub strict: bool,
}

/// Specification of a handoff tool as seen by the model.
///
/// Similar to [`ToolSpec`], but specifically for handoff tools that transfer
/// control between agents.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HandoffToolSpec {
    /// The handoff tool's name.
    pub tool_name: String,
    /// A human-readable description of the handoff tool.
    pub tool_description: String,
    /// JSON schema for structured handoff arguments.
    pub input_json_schema: serde_json::Value,
    /// Whether to enforce strict JSON schema validation.
    pub strict: bool,
}

/// The output schema specification passed to the model.
///
/// Tells the model to structure its output according to a specific JSON schema.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OutputSchemaSpec {
    /// The JSON schema for the expected output.
    pub json_schema: serde_json::Value,
    /// Whether to enforce strict schema validation on the output.
    pub strict: bool,
}

/// The core trait for calling an LLM.
///
/// Implement this trait to add support for a new model backend. The SDK provides
/// built-in implementations for `OpenAI`'s Responses API and Chat Completions API.
///
/// All methods receive type-erased tool specifications ([`ToolSpec`], [`HandoffToolSpec`])
/// so that model implementations do not need to be generic over the agent context type.
#[allow(clippy::too_many_arguments)]
#[async_trait]
pub trait Model: Send + Sync {
    /// Get a complete (non-streaming) response from the model.
    ///
    /// # Arguments
    ///
    /// * `system_instructions` - Optional system prompt for the model.
    /// * `input` - The input items in `OpenAI` Responses format.
    /// * `model_settings` - Configuration parameters for this model call.
    /// * `tools` - The tools available to the model.
    /// * `output_schema` - Optional schema constraining the model's output format.
    /// * `handoffs` - The handoff tools available to the model.
    /// * `tracing` - Tracing configuration for this call.
    /// * `previous_response_id` - ID of a previous response, used by the Responses API.
    ///
    /// # Errors
    ///
    /// Returns an error if the model call fails (network error, invalid response, etc.).
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
    ) -> Result<ModelResponse>;

    /// Stream a response from the model.
    ///
    /// Returns a stream of response events. The caller collects these events to
    /// build the final [`ModelResponse`].
    ///
    /// # Arguments
    ///
    /// * `system_instructions` - Optional system prompt for the model.
    /// * `input` - The input items in `OpenAI` Responses format.
    /// * `model_settings` - Configuration parameters for this model call.
    /// * `tools` - The tools available to the model.
    /// * `output_schema` - Optional schema constraining the model's output format.
    /// * `handoffs` - The handoff tools available to the model.
    /// * `tracing` - Tracing configuration for this call.
    /// * `previous_response_id` - ID of a previous response, used by the Responses API.
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
    ) -> Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent>> + Send + 'a>>;

    /// Release any resources held by this model.
    ///
    /// Models that maintain persistent connections can override this. The default
    /// implementation is a no-op.
    async fn close(&self) {}
}

/// Resolves model names to [`Model`] instances.
///
/// Implement this trait to provide custom model routing logic. For example, a
/// provider might map model names to different API endpoints or credentials.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Look up a model by name.
    ///
    /// If `model_name` is `None`, the provider should return its default model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model name is unknown or the model cannot be created.
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>>;

    /// Release any resources held by this provider.
    ///
    /// Providers that cache persistent models or network connections can override
    /// this. The default implementation is a no-op.
    async fn close(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage::Usage;

    // ---- ModelTracing ----

    #[test]
    fn model_tracing_default_is_enabled() {
        let tracing = ModelTracing::default();
        assert_eq!(tracing, ModelTracing::Enabled);
    }

    #[test]
    fn model_tracing_is_disabled() {
        assert!(ModelTracing::Disabled.is_disabled());
        assert!(!ModelTracing::Enabled.is_disabled());
        assert!(!ModelTracing::EnabledWithoutData.is_disabled());
    }

    #[test]
    fn model_tracing_include_data() {
        assert!(!ModelTracing::Disabled.include_data());
        assert!(ModelTracing::Enabled.include_data());
        assert!(!ModelTracing::EnabledWithoutData.include_data());
    }

    #[test]
    fn model_tracing_clone_and_copy() {
        let tracing = ModelTracing::Enabled;
        let cloned = tracing;
        assert_eq!(tracing, cloned);
    }

    #[test]
    fn model_tracing_debug() {
        let debug_str = format!("{:?}", ModelTracing::EnabledWithoutData);
        assert_eq!(debug_str, "EnabledWithoutData");
    }

    // ---- ToolSpec ----

    #[test]
    fn tool_spec_construction() {
        let spec = ToolSpec {
            name: "get_weather".to_owned(),
            description: "Fetches current weather for a location.".to_owned(),
            params_json_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
            strict: true,
        };
        assert_eq!(spec.name, "get_weather");
        assert_eq!(spec.description, "Fetches current weather for a location.");
        assert!(spec.strict);
        assert!(spec.params_json_schema.is_object());
    }

    #[test]
    fn tool_spec_clone() {
        let spec = ToolSpec {
            name: "search".to_owned(),
            description: "Search the web.".to_owned(),
            params_json_schema: serde_json::json!({}),
            strict: false,
        };
        let cloned = spec.clone();
        // Verify the original is still accessible after cloning.
        assert_eq!(spec.name, cloned.name);
        assert_eq!(cloned.name, "search");
        assert!(!cloned.strict);
    }

    // ---- HandoffToolSpec ----

    #[test]
    fn handoff_tool_spec_construction() {
        let spec = HandoffToolSpec {
            tool_name: "transfer_to_support".to_owned(),
            tool_description: "Transfer the conversation to a support agent.".to_owned(),
            input_json_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "reason": {"type": "string"}
                }
            }),
            strict: false,
        };
        assert_eq!(spec.tool_name, "transfer_to_support");
        assert!(!spec.strict);
    }

    #[test]
    fn handoff_tool_spec_clone() {
        let spec = HandoffToolSpec {
            tool_name: "handoff".to_owned(),
            tool_description: "Hand off.".to_owned(),
            input_json_schema: serde_json::json!({}),
            strict: true,
        };
        let cloned = spec.clone();
        assert_eq!(spec.tool_name, cloned.tool_name);
        assert_eq!(cloned.tool_name, "handoff");
        assert!(cloned.strict);
    }

    // ---- OutputSchemaSpec ----

    #[test]
    fn output_schema_spec_construction() {
        let spec = OutputSchemaSpec {
            json_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": {"type": "string"}
                }
            }),
            strict: true,
        };
        assert!(spec.json_schema.is_object());
        assert!(spec.strict);
    }

    #[test]
    fn output_schema_spec_clone() {
        let spec = OutputSchemaSpec {
            json_schema: serde_json::json!({"type": "string"}),
            strict: false,
        };
        let cloned = spec.clone();
        assert_eq!(spec.json_schema, cloned.json_schema);
        assert_eq!(cloned.json_schema, serde_json::json!({"type": "string"}));
        assert!(!cloned.strict);
    }

    // ---- Model trait object safety ----

    /// Verifies that the `Model` trait is object-safe and can be used as `dyn Model`.
    #[test]
    fn model_trait_is_object_safe() {
        fn _accepts_dyn_model(_model: &dyn Model) {}
        fn _accepts_boxed_model(_model: Box<dyn Model>) {}
        fn _accepts_arc_model(_model: Arc<dyn Model>) {}
    }

    /// Verifies that `dyn Model` satisfies `Send + Sync` bounds.
    #[test]
    fn model_trait_object_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn Model>>();
        assert_send_sync::<Arc<dyn Model>>();
    }

    // ---- ModelProvider trait object safety ----

    /// Verifies that the `ModelProvider` trait is object-safe.
    #[test]
    fn model_provider_trait_is_object_safe() {
        fn _accepts_dyn_provider(_provider: &dyn ModelProvider) {}
        fn _accepts_boxed_provider(_provider: Box<dyn ModelProvider>) {}
        fn _accepts_arc_provider(_provider: Arc<dyn ModelProvider>) {}
    }

    /// Verifies that `dyn ModelProvider` satisfies `Send + Sync` bounds.
    #[test]
    fn model_provider_trait_object_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn ModelProvider>>();
        assert_send_sync::<Arc<dyn ModelProvider>>();
    }

    // ---- Mock Model implementation ----

    /// A simple mock model for testing the trait interface.
    struct MockModel {
        response: ModelResponse,
    }

    #[async_trait]
    impl Model for MockModel {
        async fn get_response(
            &self,
            _system_instructions: Option<&str>,
            _input: &[ResponseInputItem],
            _model_settings: &ModelSettings,
            _tools: &[ToolSpec],
            _output_schema: Option<&OutputSchemaSpec>,
            _handoffs: &[HandoffToolSpec],
            _tracing: ModelTracing,
            _previous_response_id: Option<&str>,
        ) -> Result<ModelResponse> {
            Ok(self.response.clone())
        }

        fn stream_response<'a>(
            &'a self,
            _system_instructions: Option<&'a str>,
            _input: &'a [ResponseInputItem],
            _model_settings: &'a ModelSettings,
            _tools: &'a [ToolSpec],
            _output_schema: Option<&'a OutputSchemaSpec>,
            _handoffs: &'a [HandoffToolSpec],
            _tracing: ModelTracing,
            _previous_response_id: Option<&'a str>,
        ) -> Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent>> + Send + 'a>> {
            Box::pin(tokio_stream::empty())
        }
    }

    #[tokio::test]
    async fn mock_model_get_response() {
        let model = MockModel {
            response: ModelResponse {
                output: vec![serde_json::json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello!"}]
                })],
                usage: Usage::default(),
                response_id: Some("resp_001".to_owned()),
                request_id: None,
            },
        };

        let response = model
            .get_response(
                Some("You are a helpful assistant."),
                &[],
                &ModelSettings::default(),
                &[],
                None,
                &[],
                ModelTracing::Enabled,
                None,
            )
            .await
            .expect("mock should not fail");

        assert_eq!(response.output.len(), 1);
        assert_eq!(response.response_id.as_deref(), Some("resp_001"));
    }

    #[tokio::test]
    async fn mock_model_stream_response_is_empty() {
        use tokio_stream::StreamExt;

        let model = MockModel {
            response: ModelResponse {
                output: vec![],
                usage: Usage::default(),
                response_id: None,
                request_id: None,
            },
        };

        let settings = ModelSettings::default();
        let mut stream = model.stream_response(
            None,
            &[],
            &settings,
            &[],
            None,
            &[],
            ModelTracing::Disabled,
            None,
        );

        let next = stream.next().await;
        assert!(next.is_none(), "empty stream should yield no items");
    }

    #[tokio::test]
    async fn mock_model_as_dyn_trait() {
        let model: Arc<dyn Model> = Arc::new(MockModel {
            response: ModelResponse {
                output: vec![],
                usage: Usage::default(),
                response_id: None,
                request_id: None,
            },
        });

        let response = model
            .get_response(
                None,
                &[],
                &ModelSettings::default(),
                &[],
                None,
                &[],
                ModelTracing::default(),
                None,
            )
            .await
            .expect("should succeed");

        assert!(response.output.is_empty());
    }

    // ---- Mock ModelProvider implementation ----

    /// A simple mock provider for testing the trait interface.
    struct MockProvider;

    #[async_trait]
    impl ModelProvider for MockProvider {
        fn get_model(&self, _model_name: Option<&str>) -> Result<Arc<dyn Model>> {
            Ok(Arc::new(MockModel {
                response: ModelResponse {
                    output: vec![],
                    usage: Usage::default(),
                    response_id: None,
                    request_id: None,
                },
            }))
        }
    }

    #[test]
    fn mock_provider_get_model() {
        let provider = MockProvider;
        let model = provider.get_model(Some("gpt-4o")).expect("should succeed");
        // Just verify we get a valid Arc<dyn Model> back.
        let _ = model;
    }

    #[test]
    fn mock_provider_get_model_default() {
        let provider = MockProvider;
        let model = provider.get_model(None).expect("should succeed with None");
        let _ = model;
    }

    #[tokio::test]
    async fn mock_provider_as_dyn_trait() {
        let provider: Box<dyn ModelProvider> = Box::new(MockProvider);
        let model = provider.get_model(Some("test")).expect("should succeed");
        let response = model
            .get_response(
                None,
                &[],
                &ModelSettings::default(),
                &[],
                None,
                &[],
                ModelTracing::Enabled,
                None,
            )
            .await
            .expect("should succeed");
        assert!(response.output.is_empty());
    }

    #[tokio::test]
    async fn model_close_default_is_noop() {
        let model = MockModel {
            response: ModelResponse {
                output: vec![],
                usage: Usage::default(),
                response_id: None,
                request_id: None,
            },
        };
        // Should complete without error.
        model.close().await;
    }

    #[tokio::test]
    async fn provider_close_default_is_noop() {
        let provider = MockProvider;
        // Should complete without error.
        provider.close().await;
    }
}
