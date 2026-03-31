//! Global SDK configuration.
//!
//! Provides functions for setting and retrieving default values used across the
//! SDK, such as the default `OpenAI` API key, model name, base URL, and API
//! selection. This mirrors the Python SDK's `_config.py` module.

use std::sync::RwLock;

/// The `OpenAI` API variant to use by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpenAiApi {
    /// Use the Responses API (default).
    Responses,
    /// Use the Chat Completions API.
    ChatCompletions,
}

/// The transport to use for the Responses API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponsesTransport {
    /// HTTP transport (default).
    Http,
    /// WebSocket transport.
    WebSocket,
}

static DEFAULT_API_KEY: RwLock<Option<String>> = RwLock::new(None);
static DEFAULT_MODEL: RwLock<Option<String>> = RwLock::new(None);
static DEFAULT_BASE_URL: RwLock<Option<String>> = RwLock::new(None);
static DEFAULT_API: RwLock<Option<OpenAiApi>> = RwLock::new(None);
static DEFAULT_TRANSPORT: RwLock<Option<ResponsesTransport>> = RwLock::new(None);

/// Set the default `OpenAI` API key used by model constructors.
///
/// If not set, the SDK falls back to the `OPENAI_API_KEY` environment variable.
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
pub fn set_default_openai_key(key: impl Into<String>) {
    *DEFAULT_API_KEY
        .write()
        .expect("DEFAULT_API_KEY lock poisoned") = Some(key.into());
}

/// Get the default `OpenAI` API key.
///
/// Returns the key set via [`set_default_openai_key`], or falls back to the
/// `OPENAI_API_KEY` environment variable.
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
#[must_use]
pub fn get_default_openai_key() -> Option<String> {
    DEFAULT_API_KEY
        .read()
        .expect("DEFAULT_API_KEY lock poisoned")
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
}

/// Set the default model name (e.g., `"gpt-4o"`).
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
pub fn set_default_model(model: impl Into<String>) {
    *DEFAULT_MODEL.write().expect("DEFAULT_MODEL lock poisoned") = Some(model.into());
}

/// Get the default model name.
///
/// Returns the model set via [`set_default_model`], or `"gpt-4o"` as the
/// fallback default.
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
#[must_use]
pub fn get_default_model() -> String {
    DEFAULT_MODEL
        .read()
        .expect("DEFAULT_MODEL lock poisoned")
        .clone()
        .unwrap_or_else(|| "gpt-4o".to_string())
}

/// Set the default base URL for the `OpenAI` API.
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
pub fn set_default_base_url(url: impl Into<String>) {
    *DEFAULT_BASE_URL
        .write()
        .expect("DEFAULT_BASE_URL lock poisoned") = Some(url.into());
}

/// Get the default base URL.
///
/// Returns the URL set via [`set_default_base_url`], or falls back to the
/// `OPENAI_BASE_URL` environment variable, or `"https://api.openai.com/v1"`.
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
#[must_use]
pub fn get_default_base_url() -> String {
    DEFAULT_BASE_URL
        .read()
        .expect("DEFAULT_BASE_URL lock poisoned")
        .clone()
        .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
}

/// Set the default `OpenAI` API variant.
///
/// Choose between [`OpenAiApi::Responses`] (default) and
/// [`OpenAiApi::ChatCompletions`].
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
pub fn set_default_openai_api(api: OpenAiApi) {
    *DEFAULT_API.write().expect("DEFAULT_API lock poisoned") = Some(api);
}

/// Get the default `OpenAI` API variant.
///
/// Returns the API set via [`set_default_openai_api`], or
/// [`OpenAiApi::Responses`] as the fallback.
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
#[must_use]
pub fn get_default_openai_api() -> OpenAiApi {
    DEFAULT_API
        .read()
        .expect("DEFAULT_API lock poisoned")
        .unwrap_or(OpenAiApi::Responses)
}

/// Set the default transport for the Responses API.
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
pub fn set_default_responses_transport(transport: ResponsesTransport) {
    *DEFAULT_TRANSPORT
        .write()
        .expect("DEFAULT_TRANSPORT lock poisoned") = Some(transport);
}

/// Get the default transport for the Responses API.
///
/// Returns the transport set via [`set_default_responses_transport`], or
/// [`ResponsesTransport::Http`] as the fallback.
///
/// # Panics
///
/// Panics if the internal lock is poisoned.
#[must_use]
pub fn get_default_responses_transport() -> ResponsesTransport {
    DEFAULT_TRANSPORT
        .read()
        .expect("DEFAULT_TRANSPORT lock poisoned")
        .unwrap_or(ResponsesTransport::Http)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Because these tests mutate global state, they must be run
    // sequentially within this module. Each test resets the state it touches.

    #[test]
    fn set_and_get_api_key() {
        set_default_openai_key("sk-test-key-123");
        let key = get_default_openai_key();
        assert_eq!(key, Some("sk-test-key-123".to_string()));
        // Reset.
        *DEFAULT_API_KEY.write().unwrap() = None;
    }

    #[test]
    fn get_api_key_falls_back_to_env() {
        // Ensure the global is cleared.
        *DEFAULT_API_KEY.write().unwrap() = None;
        // If OPENAI_API_KEY env var is set, it should be returned.
        // We cannot reliably set env vars in parallel tests, so just
        // verify the function returns None when neither is set (or
        // the env value if it happens to be set).
        let result = get_default_openai_key();
        if std::env::var("OPENAI_API_KEY").is_ok() {
            assert!(result.is_some());
        } else {
            assert!(result.is_none());
        }
    }

    #[test]
    fn set_and_get_model() {
        set_default_model("gpt-4o-mini");
        assert_eq!(get_default_model(), "gpt-4o-mini");
        // Reset.
        *DEFAULT_MODEL.write().unwrap() = None;
    }

    #[test]
    fn get_model_default_fallback() {
        *DEFAULT_MODEL.write().unwrap() = None;
        assert_eq!(get_default_model(), "gpt-4o");
    }

    #[test]
    fn set_and_get_base_url() {
        set_default_base_url("https://custom.api.com/v1");
        assert_eq!(get_default_base_url(), "https://custom.api.com/v1");
        // Reset.
        *DEFAULT_BASE_URL.write().unwrap() = None;
    }

    #[test]
    fn get_base_url_default_fallback() {
        *DEFAULT_BASE_URL.write().unwrap() = None;
        // If OPENAI_BASE_URL is not set, should return the default.
        if std::env::var("OPENAI_BASE_URL").is_err() {
            assert_eq!(get_default_base_url(), "https://api.openai.com/v1");
        }
    }

    #[test]
    fn set_and_get_api() {
        set_default_openai_api(OpenAiApi::ChatCompletions);
        assert_eq!(get_default_openai_api(), OpenAiApi::ChatCompletions);
        // Reset.
        *DEFAULT_API.write().unwrap() = None;
    }

    #[test]
    fn get_api_default_fallback() {
        *DEFAULT_API.write().unwrap() = None;
        assert_eq!(get_default_openai_api(), OpenAiApi::Responses);
    }

    #[test]
    fn set_and_get_transport() {
        set_default_responses_transport(ResponsesTransport::WebSocket);
        assert_eq!(
            get_default_responses_transport(),
            ResponsesTransport::WebSocket
        );
        // Reset.
        *DEFAULT_TRANSPORT.write().unwrap() = None;
    }

    #[test]
    fn get_transport_default_fallback() {
        *DEFAULT_TRANSPORT.write().unwrap() = None;
        assert_eq!(get_default_responses_transport(), ResponsesTransport::Http);
    }
}
