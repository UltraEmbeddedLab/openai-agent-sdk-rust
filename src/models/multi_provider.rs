//! Multi-provider model routing.
//!
//! This module provides [`MultiProvider`], a [`ModelProvider`] implementation that
//! routes model name strings to the correct backend. By default it handles:
//!
//! - Standard `OpenAI` model names (e.g. `"gpt-4o"`, `"o3-mini"`) via the Responses API.
//! - Names prefixed with `"openai/"` (e.g. `"openai/gpt-4o"`) stripped and routed to the Responses API.
//! - Names prefixed with `"chatcompletions/"` (e.g. `"chatcompletions/gpt-4o"`) via the Chat Completions API.
//! - Custom provider prefixes registered at runtime.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{AgentError, Result};
use crate::models::openai_chatcompletions::OpenAIChatCompletionsModel;
use crate::models::openai_responses::OpenAIResponsesModel;
use crate::models::{Model, ModelProvider};

/// Default base URL for the `OpenAI` API.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// The default model name used when no model name is provided.
const DEFAULT_MODEL: &str = "gpt-4o";

/// Controls how `openai/...` model strings are interpreted by [`MultiProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OpenAiPrefixMode {
    /// Strip the `openai/` prefix before calling the `OpenAI` provider (default, historical behavior).
    #[default]
    Alias,
    /// Keep the full `openai/...` string as a literal model ID.
    ModelId,
}

/// Controls behavior for unrecognized prefixes in [`MultiProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum UnknownPrefixMode {
    /// Return an error for unknown prefixes (default, historical behavior).
    #[default]
    Error,
    /// Pass the full string through to the `OpenAI` provider.
    ModelId,
}

/// A model provider that routes model names to the appropriate backend.
///
/// By default, it routes standard `OpenAI` model names to the Responses API
/// and names prefixed with `chatcompletions/` to the Chat Completions API.
/// Names prefixed with `openai/` have that prefix stripped and are routed
/// to the Responses API.
///
/// Custom providers can be registered for arbitrary prefixes. When a model
/// name matches a registered prefix (i.e. starts with `"prefix/"`), the
/// corresponding [`ModelProvider`] is used instead of the defaults.
pub struct MultiProvider {
    /// The `OpenAI` API key used for built-in providers.
    api_key: String,
    /// The base URL used for built-in `OpenAI` providers.
    base_url: String,
    /// Custom provider overrides keyed by model name prefix.
    custom_providers: HashMap<String, Arc<dyn ModelProvider>>,
    /// Controls how `openai/...` model strings are interpreted.
    openai_prefix_mode: OpenAiPrefixMode,
    /// Controls behavior for unrecognized prefixes.
    unknown_prefix_mode: UnknownPrefixMode,
}

impl fmt::Debug for MultiProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiProvider")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field(
                "custom_providers",
                &self.custom_providers.keys().collect::<Vec<_>>(),
            )
            .field("openai_prefix_mode", &self.openai_prefix_mode)
            .field("unknown_prefix_mode", &self.unknown_prefix_mode)
            .finish()
    }
}

impl MultiProvider {
    /// Create a provider from the `OPENAI_API_KEY` environment variable.
    ///
    /// Uses the default `OpenAI` base URL (`https://api.openai.com/v1`).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the `OPENAI_API_KEY` environment
    /// variable is not set.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| AgentError::UserError {
            message: "OPENAI_API_KEY environment variable is not set".to_owned(),
        })?;
        Ok(Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_owned(),
            custom_providers: HashMap::new(),
            openai_prefix_mode: OpenAiPrefixMode::default(),
            unknown_prefix_mode: UnknownPrefixMode::default(),
        })
    }

    /// Create a provider with an explicit API key and the default `OpenAI` base URL.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            custom_providers: HashMap::new(),
            openai_prefix_mode: OpenAiPrefixMode::default(),
            unknown_prefix_mode: UnknownPrefixMode::default(),
        }
    }

    /// Create a provider with an explicit API key and custom base URL.
    #[must_use]
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            custom_providers: HashMap::new(),
            openai_prefix_mode: OpenAiPrefixMode::default(),
            unknown_prefix_mode: UnknownPrefixMode::default(),
        }
    }

    /// Register a custom provider for a specific prefix.
    ///
    /// When [`get_model`](ModelProvider::get_model) is called with a model name
    /// that starts with `"prefix/"`, the registered provider receives the full
    /// model name (including the prefix).
    pub fn register_provider(
        &mut self,
        prefix: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
    ) {
        self.custom_providers.insert(prefix.into(), provider);
    }

    /// Set the `OpenAI` prefix mode.
    #[must_use]
    pub const fn with_openai_prefix_mode(mut self, mode: OpenAiPrefixMode) -> Self {
        self.openai_prefix_mode = mode;
        self
    }

    /// Set the unknown prefix mode.
    #[must_use]
    pub const fn with_unknown_prefix_mode(mut self, mode: UnknownPrefixMode) -> Self {
        self.unknown_prefix_mode = mode;
        self
    }
}

#[async_trait]
impl ModelProvider for MultiProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let name = model_name.unwrap_or(DEFAULT_MODEL);

        // Check for a prefix by splitting on the first '/'.
        if let Some((prefix, rest)) = name.split_once('/') {
            // Check custom providers first.
            if let Some(provider) = self.custom_providers.get(prefix) {
                return provider.get_model(Some(name));
            }

            // Built-in: "openai/" prefix behavior depends on `openai_prefix_mode`.
            if prefix == "openai" {
                let model_id = match self.openai_prefix_mode {
                    OpenAiPrefixMode::Alias => rest,
                    OpenAiPrefixMode::ModelId => name,
                };
                return Ok(Arc::new(OpenAIResponsesModel::with_config(
                    model_id,
                    &self.api_key,
                    &self.base_url,
                )));
            }

            // Built-in: "chatcompletions/" prefix strips and routes to Chat Completions API.
            if prefix == "chatcompletions" {
                return Ok(Arc::new(OpenAIChatCompletionsModel::new(
                    rest,
                    &self.api_key,
                )));
            }

            // Unknown prefix behavior depends on `unknown_prefix_mode`.
            return match self.unknown_prefix_mode {
                UnknownPrefixMode::Error => Err(AgentError::UserError {
                    message: format!("Unknown model prefix: '{prefix}'"),
                }),
                UnknownPrefixMode::ModelId => Ok(Arc::new(OpenAIResponsesModel::with_config(
                    name,
                    &self.api_key,
                    &self.base_url,
                ))),
            };
        }

        // No prefix: route to the Responses API.
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
    use std::pin::Pin;

    use async_trait::async_trait;
    use tokio_stream::Stream;

    use crate::config::ModelSettings;
    use crate::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
    use crate::models::{HandoffToolSpec, ModelTracing, OutputSchemaSpec, ToolSpec};
    use crate::usage::Usage;

    // ---- Stub types for testing ----

    /// A minimal mock model that records the model name it was created with.
    struct StubModel {
        name: String,
    }

    #[async_trait]
    impl Model for StubModel {
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
            Ok(ModelResponse {
                output: vec![],
                usage: Usage::default(),
                response_id: Some(self.name.clone()),
                request_id: None,
            })
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

    /// A test provider that always creates a `StubModel` whose `response_id`
    /// is set to the model name it received.
    struct StubProvider;

    #[async_trait]
    impl ModelProvider for StubProvider {
        fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
            Ok(Arc::new(StubModel {
                name: model_name.unwrap_or("default").to_owned(),
            }))
        }
    }

    // ---- Default routing ----

    #[test]
    fn default_model_name_when_none() {
        let provider = MultiProvider::new("sk-test");
        // Should not error; routes to Responses API with "gpt-4o".
        let model = provider.get_model(None).expect("should succeed");
        let _ = model;
    }

    #[test]
    fn routes_bare_name_to_responses_api() {
        let provider = MultiProvider::new("sk-test");
        let model = provider.get_model(Some("gpt-4o")).expect("should succeed");
        let _ = model;
    }

    #[test]
    fn routes_o3_mini_to_responses_api() {
        let provider = MultiProvider::new("sk-test");
        let model = provider.get_model(Some("o3-mini")).expect("should succeed");
        let _ = model;
    }

    // ---- openai/ prefix ----

    #[test]
    fn routes_openai_prefix_to_responses_api() {
        let provider = MultiProvider::new("sk-test");
        let model = provider
            .get_model(Some("openai/gpt-4o"))
            .expect("should succeed");
        let _ = model;
    }

    // ---- chatcompletions/ prefix ----

    #[test]
    fn routes_chatcompletions_prefix_to_chat_completions_api() {
        let provider = MultiProvider::new("sk-test");
        let model = provider
            .get_model(Some("chatcompletions/gpt-4o"))
            .expect("should succeed");
        let _ = model;
    }

    #[test]
    fn chatcompletions_prefix_strips_prefix() {
        let provider = MultiProvider::new("sk-test");
        // Just verify we can get a model; the stripped name is used internally.
        let model = provider
            .get_model(Some("chatcompletions/gpt-4o-mini"))
            .expect("should succeed");
        let _ = model;
    }

    // ---- Unknown prefix ----

    #[test]
    fn unknown_prefix_returns_error() {
        let provider = MultiProvider::new("sk-test");
        let result = provider.get_model(Some("unknown/some-model"));
        let err = result.err().expect("should be an error");
        assert!(
            matches!(err, AgentError::UserError { .. }),
            "expected UserError, got {err:?}"
        );
        assert!(
            err.to_string().contains("Unknown model prefix"),
            "error message should mention unknown prefix"
        );
    }

    // ---- Custom provider registration ----

    #[tokio::test]
    async fn custom_provider_takes_priority() {
        let mut provider = MultiProvider::new("sk-test");
        provider.register_provider("custom", Arc::new(StubProvider));

        let model = provider
            .get_model(Some("custom/my-model"))
            .expect("should succeed");

        // The StubProvider receives the full name and stores it as response_id.
        let response = model
            .get_response(
                None,
                &[],
                &ModelSettings::default(),
                &[],
                None,
                &[],
                ModelTracing::Disabled,
                None,
            )
            .await
            .expect("should succeed");

        assert_eq!(
            response.response_id.as_deref(),
            Some("custom/my-model"),
            "custom provider should receive the full model name"
        );
    }

    #[test]
    fn custom_provider_overrides_builtin_prefix() {
        let mut provider = MultiProvider::new("sk-test");
        provider.register_provider("openai", Arc::new(StubProvider));

        // Now "openai/gpt-4o" should route to StubProvider, not the built-in.
        let model = provider
            .get_model(Some("openai/gpt-4o"))
            .expect("should succeed");
        let _ = model;
    }

    // ---- from_env ----

    #[test]
    fn from_env_without_key_returns_error() {
        temp_env::with_var_unset("OPENAI_API_KEY", || {
            let result = MultiProvider::from_env();
            assert!(result.is_err());
            assert!(result.is_err());
            let err = result.err().unwrap();
            assert!(
                matches!(err, AgentError::UserError { .. }),
                "expected UserError, got {err:?}"
            );
        });
    }

    #[test]
    fn from_env_with_key_succeeds() {
        temp_env::with_var("OPENAI_API_KEY", Some("sk-env-test"), || {
            let provider = MultiProvider::from_env().expect("should succeed");
            let model = provider.get_model(Some("gpt-4o")).expect("should succeed");
            let _ = model;
        });
    }

    // ---- with_base_url ----

    #[test]
    fn with_base_url_sets_fields() {
        let provider = MultiProvider::with_base_url("sk-test", "https://custom.api.com/v1");
        // Verify it can create models without error.
        let model = provider.get_model(Some("gpt-4o")).expect("should succeed");
        let _ = model;
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn multi_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MultiProvider>();
    }

    #[test]
    fn multi_provider_as_dyn_model_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn ModelProvider>>();

        // Also verify it can be wrapped in Arc<dyn ModelProvider>.
        let provider: Arc<dyn ModelProvider> = Arc::new(MultiProvider::new("sk-test"));
        let _ = provider;
    }

    // ---- Debug ----

    #[test]
    fn debug_redacts_api_key() {
        let provider = MultiProvider::new("sk-secret-key");
        let debug = format!("{provider:?}");
        assert!(
            !debug.contains("sk-secret-key"),
            "API key should be redacted in Debug output"
        );
        assert!(
            debug.contains("[REDACTED]"),
            "Debug output should show [REDACTED] for the API key"
        );
    }

    // ---- OpenAI prefix mode ----

    #[test]
    fn openai_prefix_mode_alias_strips_prefix() {
        // Default mode (Alias) strips the "openai/" prefix.
        let provider = MultiProvider::new("sk-test");
        assert_eq!(provider.openai_prefix_mode, OpenAiPrefixMode::Alias);
        let model = provider
            .get_model(Some("openai/gpt-4o"))
            .expect("should succeed");
        let _ = model;
    }

    #[tokio::test]
    async fn openai_prefix_mode_model_id_keeps_full_name() {
        let mut provider =
            MultiProvider::new("sk-test").with_openai_prefix_mode(OpenAiPrefixMode::ModelId);
        // Register a custom provider for "openai" to verify what model name it receives.
        provider.register_provider("openai", Arc::new(StubProvider));
        let model = provider
            .get_model(Some("openai/gpt-4o"))
            .expect("should succeed");
        // StubProvider receives the full name because custom providers take priority.
        let response = model
            .get_response(
                None,
                &[],
                &ModelSettings::default(),
                &[],
                None,
                &[],
                ModelTracing::Disabled,
                None,
            )
            .await
            .expect("should succeed");
        assert_eq!(
            response.response_id.as_deref(),
            Some("openai/gpt-4o"),
            "custom provider should receive full name"
        );
    }

    #[test]
    fn openai_prefix_mode_model_id_routes_to_responses_api() {
        // Without a custom provider override, ModelId mode should pass the full name
        // to the Responses API model.
        let provider =
            MultiProvider::new("sk-test").with_openai_prefix_mode(OpenAiPrefixMode::ModelId);
        let model = provider
            .get_model(Some("openai/gpt-4o"))
            .expect("should succeed");
        let _ = model;
    }

    // ---- Unknown prefix mode ----

    #[test]
    fn unknown_prefix_mode_error_returns_error() {
        // Default mode (Error) returns an error for unknown prefixes.
        let provider = MultiProvider::new("sk-test");
        assert_eq!(provider.unknown_prefix_mode, UnknownPrefixMode::Error);
        let result = provider.get_model(Some("anthropic/claude-3"));
        let err = result.err().expect("should be an error");
        assert!(
            matches!(err, AgentError::UserError { .. }),
            "expected UserError, got {err:?}"
        );
    }

    #[test]
    fn unknown_prefix_mode_model_id_passes_through() {
        // ModelId mode passes the full name to the Responses API.
        let provider =
            MultiProvider::new("sk-test").with_unknown_prefix_mode(UnknownPrefixMode::ModelId);
        let model = provider
            .get_model(Some("anthropic/claude-3"))
            .expect("should succeed with ModelId mode");
        let _ = model;
    }

    // ---- Debug includes new fields ----

    #[test]
    fn debug_includes_prefix_modes() {
        let provider = MultiProvider::new("sk-test")
            .with_openai_prefix_mode(OpenAiPrefixMode::ModelId)
            .with_unknown_prefix_mode(UnknownPrefixMode::ModelId);
        let debug = format!("{provider:?}");
        assert!(
            debug.contains("openai_prefix_mode: ModelId"),
            "Debug output should include openai_prefix_mode"
        );
        assert!(
            debug.contains("unknown_prefix_mode: ModelId"),
            "Debug output should include unknown_prefix_mode"
        );
    }
}
