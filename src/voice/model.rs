//! Realtime model trait and related types.
//!
//! The [`RealtimeModel`] trait defines the interface for connecting to a realtime
//! backend (e.g. the `OpenAI` Realtime API over WebSocket).  Implementations
//! manage the connection lifecycle, send events, and dispatch received events to
//! registered listeners.
//!
//! This module mirrors the Python SDK's `realtime/model.py`.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::config::RealtimeModelSettings;

// ---------------------------------------------------------------------------
// RealtimeModelConfig
// ---------------------------------------------------------------------------

/// Options for connecting to a realtime model.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RealtimeModelConfig {
    /// The API key to use when connecting.  If unset, the model will try to
    /// use the `OPENAI_API_KEY` environment variable.
    pub api_key: Option<String>,
    /// The URL to use when connecting.  If unset, the model will use a sane
    /// default (e.g. the `OpenAI` WebSocket URL).
    pub url: Option<String>,
    /// Additional headers to send when connecting.
    pub headers: Option<HashMap<String, String>>,
    /// The initial model settings to use when connecting.
    pub initial_model_settings: Option<RealtimeModelSettings>,
    /// Attach to an existing realtime call instead of creating a new session.
    pub call_id: Option<String>,
}

// ---------------------------------------------------------------------------
// RealtimeModelEvent
// ---------------------------------------------------------------------------

/// An event received from the realtime model layer.
///
/// This is a simplified representation used as the boundary between the model
/// transport and the session logic.  The full event data is stored as a JSON
/// value to allow for extensibility without breaking changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RealtimeModelEvent {
    /// The event type string (e.g. `"response.audio.delta"`).
    pub event_type: String,
    /// The full event payload as a JSON value.
    pub data: serde_json::Value,
}

// ---------------------------------------------------------------------------
// RealtimeModelListener
// ---------------------------------------------------------------------------

/// A listener for realtime transport events.
///
/// Implement this trait to receive events from a [`RealtimeModel`] connection.
#[async_trait::async_trait]
pub trait RealtimeModelListener: Send + Sync + fmt::Debug {
    /// Called when an event is emitted by the realtime transport.
    async fn on_event(&self, event: RealtimeModelEvent) -> Result<()>;
}

// ---------------------------------------------------------------------------
// RealtimeModel
// ---------------------------------------------------------------------------

/// Interface for connecting to a realtime model and sending/receiving events.
///
/// This is the Rust equivalent of the Python SDK's `RealtimeModel` abstract
/// base class.  Implement this trait to provide a custom realtime transport
/// (e.g. WebSocket, SIP, or a mock for testing).
#[async_trait::async_trait]
pub trait RealtimeModel: Send + Sync + fmt::Debug {
    /// Establish a connection to the model and keep it alive.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection cannot be established.
    async fn connect(&mut self, options: RealtimeModelConfig) -> Result<()>;

    /// Add a listener that will receive events from the model.
    fn add_listener(&mut self, listener: Box<dyn RealtimeModelListener>);

    /// Send a JSON event to the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be sent (e.g. disconnected).
    async fn send_event(&self, event: serde_json::Value) -> Result<()>;

    /// Close the connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection cannot be closed cleanly.
    async fn close(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- RealtimeModelConfig ----

    #[test]
    fn model_config_default() {
        let config = RealtimeModelConfig::default();
        assert!(config.api_key.is_none());
        assert!(config.url.is_none());
        assert!(config.headers.is_none());
        assert!(config.initial_model_settings.is_none());
        assert!(config.call_id.is_none());
    }

    #[test]
    fn model_config_with_fields() {
        let config = RealtimeModelConfig {
            api_key: Some("sk-test".to_owned()),
            url: Some("wss://example.com".to_owned()),
            headers: Some(HashMap::from([("X-Custom".to_owned(), "val".to_owned())])),
            initial_model_settings: None,
            call_id: None,
        };
        assert_eq!(config.api_key.as_deref(), Some("sk-test"));
        assert_eq!(config.url.as_deref(), Some("wss://example.com"));
        assert!(config.headers.unwrap().contains_key("X-Custom"));
    }

    // ---- RealtimeModelEvent ----

    #[test]
    fn model_event_construction() {
        let event = RealtimeModelEvent {
            event_type: "response.audio.delta".to_owned(),
            data: serde_json::json!({"delta": "base64data"}),
        };
        assert_eq!(event.event_type, "response.audio.delta");
    }

    #[test]
    fn model_event_serialization() {
        let event = RealtimeModelEvent {
            event_type: "session.created".to_owned(),
            data: serde_json::json!({"session_id": "sess-1"}),
        };
        let json = serde_json::to_string(&event).expect("serialize");
        let deserialized: RealtimeModelEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.event_type, "session.created");
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn model_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RealtimeModelConfig>();
        assert_send_sync::<RealtimeModelEvent>();
    }

    #[test]
    fn model_trait_is_object_safe() {
        // Verify that RealtimeModel can be used as a trait object.
        fn _accept_model(_m: &dyn RealtimeModel) {}
    }

    #[test]
    fn listener_trait_is_object_safe() {
        // Verify that RealtimeModelListener can be used as a trait object.
        fn _accept_listener(_l: &dyn RealtimeModelListener) {}
    }
}
