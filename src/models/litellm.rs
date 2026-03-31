//! `LiteLLM`-compatible model provider.
//!
//! Provides a model that can connect to any `LiteLLM`-compatible proxy server,
//! enabling access to models from Anthropic, Google, Azure, AWS Bedrock,
//! and many other providers through a single unified API.
//!
//! `LiteLLM` proxies expose an `OpenAI`-compatible Chat Completions endpoint, so
//! this module delegates to [`OpenAIChatCompletionsModel`] under the hood.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_stream::Stream;

use crate::config::ModelSettings;
use crate::error::{AgentError, Result};
use crate::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use crate::models::openai_chatcompletions::OpenAIChatCompletionsModel;
use crate::models::{
    HandoffToolSpec, Model, ModelProvider, ModelTracing, OutputSchemaSpec, ToolSpec,
};

/// Default base URL for a local `LiteLLM` proxy server.
const DEFAULT_LITELLM_BASE_URL: &str = "http://localhost:4000/v1";

/// A model that connects to a `LiteLLM` proxy server.
///
/// `LiteLLM` provides an `OpenAI`-compatible API that routes to various
/// LLM providers. This model uses the Chat Completions format to
/// communicate with the proxy.
///
/// # Example
///
/// ```no_run
/// use openai_agents::models::litellm::LiteLLMModel;
///
/// let model = LiteLLMModel::new(
///     "anthropic/claude-sonnet-4-20250514",
///     "your-litellm-api-key",
///     "http://localhost:4000/v1",
/// );
/// ```
#[derive(Clone)]
pub struct LiteLLMModel {
    /// The delegated chat completions model.
    inner: OpenAIChatCompletionsModel,
    /// The original model name (e.g. `"anthropic/claude-sonnet-4-20250514"`).
    model_name: String,
}

impl fmt::Debug for LiteLLMModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteLLMModel")
            .field("model_name", &self.model_name)
            .finish_non_exhaustive()
    }
}

impl LiteLLMModel {
    /// Create a new `LiteLLM` model targeting a proxy server.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier including provider prefix (e.g. `"anthropic/claude-sonnet-4-20250514"`).
    /// * `api_key` - The API key for authenticating with the `LiteLLM` proxy.
    /// * `base_url` - The base URL of the `LiteLLM` proxy (e.g. `"http://localhost:4000/v1"`).
    #[must_use]
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let model_name = model.into();
        Self {
            inner: OpenAIChatCompletionsModel::with_base_url(&model_name, api_key, base_url),
            model_name,
        }
    }

    /// Create a model targeting the default local `LiteLLM` proxy at `localhost:4000`.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier including provider prefix.
    /// * `api_key` - The API key for authenticating with the proxy.
    #[must_use]
    pub fn local(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::new(model, api_key, DEFAULT_LITELLM_BASE_URL)
    }

    /// Returns the model name this instance was created with.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[async_trait]
impl Model for LiteLLMModel {
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
        self.inner
            .get_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id,
            )
            .await
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
        self.inner.stream_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
        )
    }

    async fn close(&self) {
        self.inner.close().await;
    }
}

/// A model provider that routes through a `LiteLLM` proxy.
///
/// Use this to connect to a `LiteLLM` proxy server that provides a unified
/// API for accessing models from many different providers. API keys for
/// the underlying providers are typically configured on the `LiteLLM` proxy
/// itself rather than passed through by the client.
///
/// # Example
///
/// ```no_run
/// use openai_agents::models::litellm::LiteLLMProvider;
/// use openai_agents::models::ModelProvider;
///
/// let provider = LiteLLMProvider::new("lk-...", "http://localhost:4000/v1");
/// let model = provider.get_model(Some("anthropic/claude-sonnet-4-20250514")).unwrap();
/// ```
pub struct LiteLLMProvider {
    /// The API key for authenticating with the `LiteLLM` proxy.
    api_key: String,
    /// The base URL of the `LiteLLM` proxy.
    base_url: String,
}

impl fmt::Debug for LiteLLMProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiteLLMProvider")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl LiteLLMProvider {
    /// Create a new `LiteLLM` provider with the given API key and base URL.
    #[must_use]
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Create a provider targeting the default local `LiteLLM` proxy at `localhost:4000`.
    #[must_use]
    pub fn local(api_key: impl Into<String>) -> Self {
        Self::new(api_key, DEFAULT_LITELLM_BASE_URL)
    }
}

#[async_trait]
impl ModelProvider for LiteLLMProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let name = model_name.ok_or_else(|| AgentError::UserError {
            message:
                "LiteLLM requires an explicit model name (e.g. \"anthropic/claude-sonnet-4-20250514\")"
                    .to_owned(),
        })?;
        Ok(Arc::new(LiteLLMModel::new(
            name,
            &self.api_key,
            &self.base_url,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- LiteLLMModel construction ----

    #[test]
    fn litellm_model_new() {
        let model = LiteLLMModel::new(
            "anthropic/claude-sonnet-4-20250514",
            "test-key",
            "http://localhost:4000/v1",
        );
        assert_eq!(model.model_name(), "anthropic/claude-sonnet-4-20250514");
    }

    #[test]
    fn litellm_model_local() {
        let model = LiteLLMModel::local("anthropic/claude-sonnet-4-20250514", "test-key");
        assert_eq!(model.model_name(), "anthropic/claude-sonnet-4-20250514");
    }

    #[test]
    fn litellm_model_debug() {
        let model = LiteLLMModel::new("test-model", "secret-key", "http://localhost:4000/v1");
        let debug = format!("{model:?}");
        assert!(debug.contains("LiteLLMModel"));
        assert!(debug.contains("test-model"));
        assert!(
            !debug.contains("secret-key"),
            "API key should not appear in Debug output"
        );
    }

    #[test]
    fn litellm_model_clone() {
        let model = LiteLLMModel::new("test-model", "key", "http://localhost:4000/v1");
        let cloned = model.clone();
        assert_eq!(model.model_name(), cloned.model_name());
    }

    // ---- LiteLLMProvider construction ----

    #[test]
    fn litellm_provider_new() {
        let provider = LiteLLMProvider::new("test-key", "http://proxy:4000/v1");
        let debug = format!("{provider:?}");
        assert!(debug.contains("LiteLLMProvider"));
        assert!(debug.contains("[REDACTED]"));
        assert!(
            !debug.contains("test-key"),
            "API key should be redacted in Debug output"
        );
    }

    #[test]
    fn litellm_provider_local() {
        let provider = LiteLLMProvider::local("test-key");
        let debug = format!("{provider:?}");
        assert!(debug.contains("localhost:4000"));
    }

    #[test]
    fn litellm_provider_get_model_requires_name() {
        let provider = LiteLLMProvider::new("key", "http://localhost:4000/v1");
        let result = provider.get_model(None);
        assert!(result.is_err());
        let err = result.err().expect("should be an error");
        assert!(
            matches!(err, AgentError::UserError { .. }),
            "expected UserError, got {err:?}"
        );
        assert!(err.to_string().contains("explicit model name"));
    }

    #[test]
    fn litellm_provider_get_model_with_name() {
        let provider = LiteLLMProvider::new("key", "http://localhost:4000/v1");
        let model = provider
            .get_model(Some("anthropic/claude-sonnet-4-20250514"))
            .expect("should succeed");
        let _ = model;
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn litellm_model_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LiteLLMModel>();
    }

    #[test]
    fn litellm_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LiteLLMProvider>();
    }

    #[test]
    fn litellm_model_as_arc_dyn_model() {
        let model: Arc<dyn Model> =
            Arc::new(LiteLLMModel::new("test", "key", "http://localhost:4000/v1"));
        let _ = model;
    }

    #[test]
    fn litellm_provider_as_arc_dyn_model_provider() {
        let provider: Arc<dyn ModelProvider> =
            Arc::new(LiteLLMProvider::new("key", "http://localhost:4000/v1"));
        let _ = provider;
    }
}
