//! Realtime session for managing a live voice conversation.
//!
//! A [`RealtimeSession`] manages the WebSocket connection lifecycle and provides
//! methods for sending audio, text, and tool outputs to the model, and for
//! receiving events.
//!
//! This module mirrors the Python SDK's `realtime/session.py`.

use crate::error::{AgentError, Result};

use super::config::RealtimeConfig;
use super::events::RealtimeSessionEvent;

// ---------------------------------------------------------------------------
// RealtimeSession
// ---------------------------------------------------------------------------

/// A WebSocket session to the `OpenAI` Realtime API.
///
/// This manages the WebSocket connection lifecycle and provides methods for
/// sending audio data and receiving events.  The actual WebSocket transport
/// is not yet implemented -- the public API surface is complete so that
/// downstream code can be written against it.
///
/// # Example
///
/// ```no_run
/// use openai_agents::voice::session::RealtimeSession;
/// use openai_agents::voice::config::RealtimeConfig;
///
/// # async fn example() -> openai_agents::Result<()> {
/// let mut session = RealtimeSession::new(RealtimeConfig::default());
/// session.connect().await?;
/// session.send_text("Hello!").await?;
/// if let Some(event) = session.recv_event().await? {
///     println!("{event:?}");
/// }
/// session.disconnect().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RealtimeSession {
    config: RealtimeConfig,
    is_connected: bool,
}

// All async methods are intentionally async: they will perform I/O once the
// WebSocket transport is implemented.
#[allow(clippy::unused_async)]
impl RealtimeSession {
    /// Create a new session with the given configuration.
    ///
    /// The session is not connected until [`connect`](Self::connect) is called.
    #[must_use]
    pub const fn new(config: RealtimeConfig) -> Self {
        Self {
            config,
            is_connected: false,
        }
    }

    /// Connect to the Realtime API via WebSocket.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection cannot be established.
    ///
    /// # Note
    ///
    /// The WebSocket transport is not yet implemented.  This method currently
    /// sets the connection flag to `true` and returns `Ok(())`.
    pub async fn connect(&mut self) -> Result<()> {
        // TODO: Implement WebSocket connection using tokio-tungstenite.
        self.is_connected = true;
        Ok(())
    }

    /// Send audio data to the model.
    ///
    /// The data should be in the format specified by the session's
    /// [`AudioFormat`](super::config::AudioFormat) configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    pub async fn send_audio(&self, _data: &[u8]) -> Result<()> {
        self.ensure_connected()?;
        // TODO: Send audio via WebSocket.
        Ok(())
    }

    /// Send a text message to the model (for text-mode realtime interaction).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    pub async fn send_text(&self, _text: &str) -> Result<()> {
        self.ensure_connected()?;
        // TODO: Send text message via WebSocket.
        Ok(())
    }

    /// Send a tool call output back to the model.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    pub async fn send_tool_output(&self, _call_id: &str, _output: &str) -> Result<()> {
        self.ensure_connected()?;
        // TODO: Send tool output via WebSocket.
        Ok(())
    }

    /// Request the model to generate a response.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    pub async fn create_response(&self) -> Result<()> {
        self.ensure_connected()?;
        // TODO: Send response.create event via WebSocket.
        Ok(())
    }

    /// Interrupt the current model response (e.g. because the user started
    /// speaking).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    pub async fn interrupt(&self) -> Result<()> {
        self.ensure_connected()?;
        // TODO: Send interrupt event via WebSocket.
        Ok(())
    }

    /// Receive the next event from the session.
    ///
    /// Returns `Ok(None)` if there are no more events (session ended).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    pub async fn recv_event(&self) -> Result<Option<RealtimeSessionEvent>> {
        self.ensure_connected()?;
        // TODO: Receive from WebSocket and translate to RealtimeSessionEvent.
        Ok(None)
    }

    /// Disconnect from the Realtime API.
    ///
    /// # Errors
    ///
    /// Returns an error if the disconnection fails.
    pub async fn disconnect(&mut self) -> Result<()> {
        self.is_connected = false;
        // TODO: Close WebSocket connection.
        Ok(())
    }

    /// Whether the session is currently connected.
    #[must_use]
    pub const fn is_connected(&self) -> bool {
        self.is_connected
    }

    /// Get a reference to the session's configuration.
    #[must_use]
    pub const fn config(&self) -> &RealtimeConfig {
        &self.config
    }

    /// Ensure the session is connected, returning an error if not.
    fn ensure_connected(&self) -> Result<()> {
        if self.is_connected {
            Ok(())
        } else {
            Err(AgentError::UserError {
                message: "Realtime session is not connected".to_owned(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Construction ----

    #[test]
    fn session_starts_disconnected() {
        let session = RealtimeSession::new(RealtimeConfig::default());
        assert!(!session.is_connected());
    }

    #[test]
    fn session_config_accessible() {
        let session = RealtimeSession::new(RealtimeConfig::default());
        assert_eq!(session.config().model, "gpt-4o-realtime-preview");
    }

    // ---- Connect / disconnect ----

    #[tokio::test]
    async fn connect_and_disconnect() {
        let mut session = RealtimeSession::new(RealtimeConfig::default());
        assert!(!session.is_connected());

        session.connect().await.unwrap();
        assert!(session.is_connected());

        session.disconnect().await.unwrap();
        assert!(!session.is_connected());
    }

    // ---- Error on not connected ----

    #[tokio::test]
    async fn send_audio_not_connected() {
        let session = RealtimeSession::new(RealtimeConfig::default());
        let result = session.send_audio(&[1, 2, 3]).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AgentError::UserError { .. }));
        assert!(err.to_string().contains("not connected"));
    }

    #[tokio::test]
    async fn send_text_not_connected() {
        let session = RealtimeSession::new(RealtimeConfig::default());
        let result = session.send_text("hello").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn send_tool_output_not_connected() {
        let session = RealtimeSession::new(RealtimeConfig::default());
        let result = session.send_tool_output("call-1", "output").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn create_response_not_connected() {
        let session = RealtimeSession::new(RealtimeConfig::default());
        let result = session.create_response().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn interrupt_not_connected() {
        let session = RealtimeSession::new(RealtimeConfig::default());
        let result = session.interrupt().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn recv_event_not_connected() {
        let session = RealtimeSession::new(RealtimeConfig::default());
        let result = session.recv_event().await;
        assert!(result.is_err());
    }

    // ---- Operations succeed when connected ----

    #[tokio::test]
    async fn send_audio_when_connected() {
        let mut session = RealtimeSession::new(RealtimeConfig::default());
        session.connect().await.unwrap();
        let result = session.send_audio(&[1, 2, 3]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn send_text_when_connected() {
        let mut session = RealtimeSession::new(RealtimeConfig::default());
        session.connect().await.unwrap();
        let result = session.send_text("hello").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn recv_event_returns_none_when_connected() {
        let mut session = RealtimeSession::new(RealtimeConfig::default());
        session.connect().await.unwrap();
        let event = session.recv_event().await.unwrap();
        assert!(event.is_none());
    }

    // ---- Debug ----

    #[test]
    fn session_debug() {
        let session = RealtimeSession::new(RealtimeConfig::default());
        let debug_str = format!("{session:?}");
        assert!(debug_str.contains("RealtimeSession"));
        assert!(debug_str.contains("is_connected: false"));
    }

    // ---- Send + Sync ----

    #[test]
    fn session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RealtimeSession>();
    }
}
