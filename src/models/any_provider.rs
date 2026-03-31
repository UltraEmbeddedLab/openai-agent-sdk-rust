//! Generic `OpenAI`-compatible model provider.
//!
//! Works with any API that implements the `OpenAI` Chat Completions format,
//! including vLLM, TGI, Ollama, and other local inference servers.
//!
//! This is the simplest way to connect to alternative LLM backends. For
//! `LiteLLM` proxies specifically, see [`super::litellm`].

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::models::openai_chatcompletions::OpenAIChatCompletionsModel;
use crate::models::{Model, ModelProvider};

/// A provider for any `OpenAI`-compatible API endpoint.
///
/// Creates [`OpenAIChatCompletionsModel`] instances that communicate with the
/// given base URL. This is the simplest way to connect to alternative LLM
/// backends that implement the Chat Completions API format.
///
/// # Example
///
/// ```no_run
/// use openai_agents::models::any_provider::AnyProvider;
/// use openai_agents::models::ModelProvider;
///
/// // Connect to a local Ollama server.
/// let provider = AnyProvider::unauthenticated("http://localhost:11434/v1");
/// let model = provider.get_model(Some("llama3")).unwrap();
///
/// // Connect to a vLLM server with an API key.
/// let provider = AnyProvider::new("my-api-key", "http://localhost:8000/v1");
/// let model = provider.get_model(Some("meta-llama/Llama-3-8B")).unwrap();
/// ```
pub struct AnyProvider {
    /// The API key for authentication.
    api_key: String,
    /// The base URL of the `OpenAI`-compatible endpoint.
    base_url: String,
}

impl fmt::Debug for AnyProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyProvider")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl AnyProvider {
    /// Create a new provider with the given API key and base URL.
    #[must_use]
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Create a provider with no API key for local servers that do not require authentication.
    ///
    /// Sends `"no-key"` as the bearer token, which most local servers ignore.
    #[must_use]
    pub fn unauthenticated(base_url: impl Into<String>) -> Self {
        Self::new("no-key", base_url)
    }
}

#[async_trait]
impl ModelProvider for AnyProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let name = model_name.unwrap_or("default");
        Ok(Arc::new(OpenAIChatCompletionsModel::with_base_url(
            name,
            &self.api_key,
            &self.base_url,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AnyProvider construction ----

    #[test]
    fn any_provider_new() {
        let provider = AnyProvider::new("my-key", "http://localhost:8000/v1");
        let debug = format!("{provider:?}");
        assert!(debug.contains("AnyProvider"));
        assert!(debug.contains("[REDACTED]"));
        assert!(
            !debug.contains("my-key"),
            "API key should be redacted in Debug output"
        );
        assert!(debug.contains("localhost:8000"));
    }

    #[test]
    fn any_provider_unauthenticated() {
        let provider = AnyProvider::unauthenticated("http://localhost:11434/v1");
        let debug = format!("{provider:?}");
        assert!(debug.contains("localhost:11434"));
    }

    // ---- get_model ----

    #[test]
    fn get_model_with_name() {
        let provider = AnyProvider::new("key", "http://localhost:8000/v1");
        let model = provider.get_model(Some("llama3")).expect("should succeed");
        let _ = model;
    }

    #[test]
    fn get_model_with_none_uses_default() {
        let provider = AnyProvider::new("key", "http://localhost:8000/v1");
        let model = provider.get_model(None).expect("should succeed with None");
        let _ = model;
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn any_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AnyProvider>();
    }

    #[test]
    fn any_provider_as_arc_dyn_model_provider() {
        let provider: Arc<dyn ModelProvider> =
            Arc::new(AnyProvider::new("key", "http://localhost:8000/v1"));
        let _ = provider;
    }
}
