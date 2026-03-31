//! Realtime session for managing a live voice conversation.
//!
//! A [`RealtimeSession`] manages the WebSocket connection lifecycle and provides
//! methods for sending audio, text, and tool outputs to the model, and for
//! receiving events.
//!
//! When the `voice` feature is enabled, the session uses `tokio-tungstenite` for
//! real WebSocket transport to the `OpenAI` Realtime API.  Without the feature,
//! methods return stub errors or no-ops so the type can still be referenced.
//!
//! This module mirrors the Python SDK's `realtime/session.py`.

use crate::error::{AgentError, Result};

use super::config::RealtimeConfig;
use super::events::RealtimeSessionEvent;

#[cfg(feature = "voice")]
use super::events::{
    AudioDeltaEvent, RealtimeErrorEvent, ResponseDoneEvent, SessionCreatedEvent,
    SpeechStartedEvent, SpeechStoppedEvent, ToolCallCreatedEvent, TranscriptDeltaEvent,
    TranscriptRole,
};

#[cfg(feature = "voice")]
use futures::{SinkExt, StreamExt};
#[cfg(feature = "voice")]
use tokio::sync::Mutex;
#[cfg(feature = "voice")]
use tokio_tungstenite::tungstenite::Message;

#[cfg(feature = "voice")]
use base64::Engine as _;

// ---------------------------------------------------------------------------
// Type aliases for the WebSocket stream halves (voice feature only).
// ---------------------------------------------------------------------------

#[cfg(feature = "voice")]
type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

#[cfg(feature = "voice")]
type WsWriter = futures::stream::SplitSink<WsStream, Message>;

#[cfg(feature = "voice")]
type WsReader = futures::stream::SplitStream<WsStream>;

// ---------------------------------------------------------------------------
// RealtimeSession
// ---------------------------------------------------------------------------

/// A WebSocket session to the `OpenAI` Realtime API.
///
/// This manages the WebSocket connection lifecycle and provides methods for
/// sending audio data and receiving events.
///
/// When the `voice` feature is enabled, [`connect`](Self::connect) establishes
/// a real WebSocket connection to `wss://api.openai.com/v1/realtime`.
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
pub struct RealtimeSession {
    config: RealtimeConfig,
    is_connected: bool,
    #[cfg(feature = "voice")]
    ws_writer: Option<Mutex<WsWriter>>,
    #[cfg(feature = "voice")]
    ws_reader: Option<Mutex<WsReader>>,
}

impl std::fmt::Debug for RealtimeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeSession")
            .field("config", &self.config)
            .field("is_connected", &self.is_connected)
            .finish_non_exhaustive()
    }
}

impl RealtimeSession {
    /// Create a new session with the given configuration.
    ///
    /// The session is not connected until [`connect`](Self::connect) is called.
    #[must_use]
    pub const fn new(config: RealtimeConfig) -> Self {
        Self {
            config,
            is_connected: false,
            #[cfg(feature = "voice")]
            ws_writer: None,
            #[cfg(feature = "voice")]
            ws_reader: None,
        }
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

    // =======================================================================
    // Real WebSocket implementation (voice feature enabled)
    // =======================================================================

    /// Connect to the Realtime API via WebSocket.
    ///
    /// Reads the API key from [`RealtimeConfig::api_key`] or falls back to the
    /// `OPENAI_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if no API key is available or if the WebSocket handshake
    /// fails.
    #[cfg(feature = "voice")]
    pub async fn connect(&mut self) -> Result<()> {
        let api_key = self
            .config
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| AgentError::UserError {
                message: "No API key for realtime session. Set OPENAI_API_KEY or \
                          provide one in RealtimeConfig."
                    .into(),
            })?;

        let url = format!("{}?model={}", self.config.base_url, self.config.model);

        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("OpenAI-Beta", "realtime=v1")
            .header("Host", "api.openai.com")
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .body(())
            .map_err(|e| AgentError::UserError {
                message: format!("Failed to build WebSocket request: {e}"),
            })?;

        let (ws_stream, _response) =
            tokio_tungstenite::connect_async(request)
                .await
                .map_err(|e| AgentError::UserError {
                    message: format!("WebSocket connection failed: {e}"),
                })?;

        let (writer, reader) = ws_stream.split();
        self.ws_writer = Some(Mutex::new(writer));
        self.ws_reader = Some(Mutex::new(reader));
        self.is_connected = true;

        // Send a session.update to configure the session with model settings.
        self.send_session_update().await?;

        Ok(())
    }

    /// Connect to the Realtime API via WebSocket.
    ///
    /// # Errors
    ///
    /// Without the `voice` feature, this always returns an error.
    #[cfg(not(feature = "voice"))]
    #[allow(clippy::unused_async)]
    pub async fn connect(&mut self) -> Result<()> {
        Err(AgentError::UserError {
            message: "WebSocket transport requires the `voice` feature flag".to_owned(),
        })
    }

    /// Send a `session.update` event to configure the session.
    #[cfg(feature = "voice")]
    async fn send_session_update(&self) -> Result<()> {
        let settings = &self.config.model_settings;

        let input_format = settings.input_audio_format.unwrap_or_default().to_string();
        let output_format = settings.output_audio_format.unwrap_or_default().to_string();

        let turn_detection = turn_detection_to_json(&settings.turn_detection);

        let mut session = serde_json::json!({
            "modalities": ["text", "audio"],
            "voice": settings.voice.to_string(),
            "input_audio_format": input_format,
            "output_audio_format": output_format,
        });

        // Only include turn detection if not null.
        if !turn_detection.is_null() {
            session["turn_detection"] = turn_detection;
        }

        if let Some(ref instructions) = settings.instructions {
            session["instructions"] = serde_json::Value::String(instructions.clone());
        }
        if let Some(temp) = settings.temperature {
            session["temperature"] = serde_json::json!(temp);
        }
        if let Some(max_tokens) = settings.max_response_output_tokens {
            session["max_response_output_tokens"] = serde_json::json!(max_tokens);
        }

        let update = serde_json::json!({
            "type": "session.update",
            "session": session,
        });
        self.send_json(&update).await
    }

    /// Send a JSON message over the WebSocket.
    #[cfg(feature = "voice")]
    async fn send_json(&self, value: &serde_json::Value) -> Result<()> {
        let json_str = serde_json::to_string(value)?;
        if let Some(ref writer) = self.ws_writer {
            writer
                .lock()
                .await
                .send(Message::Text(json_str.into()))
                .await
                .map_err(|e| AgentError::UserError {
                    message: format!("WebSocket send error: {e}"),
                })?;
        }
        Ok(())
    }

    /// Send audio data to the model.
    ///
    /// The data should be in the format specified by the session's
    /// [`AudioFormat`](super::config::AudioFormat) configuration.
    /// It is base64-encoded and sent as an `input_audio_buffer.append` event.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    #[cfg(feature = "voice")]
    pub async fn send_audio(&self, data: &[u8]) -> Result<()> {
        self.ensure_connected()?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(data);
        let event = serde_json::json!({
            "type": "input_audio_buffer.append",
            "audio": encoded,
        });
        self.send_json(&event).await
    }

    /// Send audio data to the model.
    ///
    /// # Errors
    ///
    /// Without the `voice` feature, returns [`AgentError::UserError`] if not
    /// connected. With the feature disabled the session can never connect, so
    /// this always errors.
    #[cfg(not(feature = "voice"))]
    #[allow(clippy::unused_async)]
    pub async fn send_audio(&self, _data: &[u8]) -> Result<()> {
        self.ensure_connected()
    }

    /// Send a text message to the model (for text-mode realtime interaction).
    ///
    /// Creates a `conversation.item.create` event followed by a
    /// `response.create` event.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    #[cfg(feature = "voice")]
    pub async fn send_text(&self, text: &str) -> Result<()> {
        self.ensure_connected()?;
        let event = serde_json::json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        });
        self.send_json(&event).await?;
        self.create_response().await
    }

    /// Send a text message to the model.
    ///
    /// # Errors
    ///
    /// Without the `voice` feature, always returns an error.
    #[cfg(not(feature = "voice"))]
    #[allow(clippy::unused_async)]
    pub async fn send_text(&self, _text: &str) -> Result<()> {
        self.ensure_connected()
    }

    /// Send a tool call output back to the model.
    ///
    /// Creates a `conversation.item.create` event with `function_call_output`
    /// type.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    #[cfg(feature = "voice")]
    pub async fn send_tool_output(&self, call_id: &str, output: &str) -> Result<()> {
        self.ensure_connected()?;
        let event = serde_json::json!({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        });
        self.send_json(&event).await
    }

    /// Send a tool call output back to the model.
    ///
    /// # Errors
    ///
    /// Without the `voice` feature, always returns an error.
    #[cfg(not(feature = "voice"))]
    #[allow(clippy::unused_async)]
    pub async fn send_tool_output(&self, _call_id: &str, _output: &str) -> Result<()> {
        self.ensure_connected()
    }

    /// Request the model to generate a response.
    ///
    /// Sends a `response.create` event.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    #[cfg(feature = "voice")]
    pub async fn create_response(&self) -> Result<()> {
        self.ensure_connected()?;
        let event = serde_json::json!({"type": "response.create"});
        self.send_json(&event).await
    }

    /// Request the model to generate a response.
    ///
    /// # Errors
    ///
    /// Without the `voice` feature, always returns an error.
    #[cfg(not(feature = "voice"))]
    #[allow(clippy::unused_async)]
    pub async fn create_response(&self) -> Result<()> {
        self.ensure_connected()
    }

    /// Interrupt the current model response (e.g. because the user started
    /// speaking).
    ///
    /// Sends a `response.cancel` event.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected.
    #[cfg(feature = "voice")]
    pub async fn interrupt(&self) -> Result<()> {
        self.ensure_connected()?;
        let event = serde_json::json!({"type": "response.cancel"});
        self.send_json(&event).await
    }

    /// Interrupt the current model response.
    ///
    /// # Errors
    ///
    /// Without the `voice` feature, always returns an error.
    #[cfg(not(feature = "voice"))]
    #[allow(clippy::unused_async)]
    pub async fn interrupt(&self) -> Result<()> {
        self.ensure_connected()
    }

    /// Receive the next event from the session.
    ///
    /// Reads the next WebSocket text message, parses it as JSON, and converts
    /// it into a [`RealtimeSessionEvent`].  Returns `Ok(None)` when the stream
    /// ends.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the session is not connected or if
    /// a WebSocket error occurs.
    #[cfg(feature = "voice")]
    pub async fn recv_event(&self) -> Result<Option<RealtimeSessionEvent>> {
        self.ensure_connected()?;

        if let Some(ref reader) = self.ws_reader {
            let mut reader = reader.lock().await;
            match reader.next().await {
                Some(Ok(Message::Text(text))) => {
                    let json: serde_json::Value = serde_json::from_str(&text)?;
                    Ok(Some(parse_realtime_event(&json)))
                }
                Some(Ok(Message::Binary(data))) => {
                    // Binary frames are not expected from the Realtime API;
                    // attempt to parse as JSON anyway.
                    let text = String::from_utf8_lossy(&data);
                    Ok(Some(
                        serde_json::from_str::<serde_json::Value>(&text).map_or_else(
                            |_| {
                                RealtimeSessionEvent::Unknown(
                                    serde_json::json!({"type": "binary", "length": data.len()}),
                                )
                            },
                            |json| parse_realtime_event(&json),
                        ),
                    ))
                }
                Some(Ok(Message::Close(_))) => Ok(Some(RealtimeSessionEvent::SessionClosed)),
                Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {
                    // Control frames -- skip and try the next message.
                    drop(reader);
                    // Re-enter without holding the lock.
                    Box::pin(self.recv_event()).await
                }
                Some(Err(e)) => Err(AgentError::UserError {
                    message: format!("WebSocket receive error: {e}"),
                }),
                None => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    /// Receive the next event from the session.
    ///
    /// # Errors
    ///
    /// Without the `voice` feature, always returns an error.
    #[cfg(not(feature = "voice"))]
    #[allow(clippy::unused_async)]
    pub async fn recv_event(&self) -> Result<Option<RealtimeSessionEvent>> {
        self.ensure_connected().map(|()| None)
    }

    /// Disconnect from the Realtime API.
    ///
    /// Sends a close frame and drops the WebSocket halves.
    ///
    /// # Errors
    ///
    /// Returns an error if the disconnection fails.
    #[cfg(feature = "voice")]
    pub async fn disconnect(&mut self) -> Result<()> {
        if let Some(ref writer) = self.ws_writer {
            let _ = writer.lock().await.close().await;
        }
        self.ws_writer = None;
        self.ws_reader = None;
        self.is_connected = false;
        Ok(())
    }

    /// Disconnect from the Realtime API.
    ///
    /// # Errors
    ///
    /// Without the `voice` feature, this is a no-op that clears the
    /// connected flag.
    #[cfg(not(feature = "voice"))]
    #[allow(clippy::unused_async)]
    pub async fn disconnect(&mut self) -> Result<()> {
        self.is_connected = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Turn detection JSON serialization helper.
// ---------------------------------------------------------------------------

/// Convert a [`TurnDetection`] to a JSON value for the `session.update` event.
#[cfg(feature = "voice")]
fn turn_detection_to_json(td: &super::config::TurnDetection) -> serde_json::Value {
    use super::config::TurnDetection;
    match td {
        TurnDetection::SemanticVad {
            eagerness,
            create_response,
            interrupt_response,
        } => {
            let mut obj = serde_json::json!({"type": "semantic_vad"});
            if let Some(e) = eagerness {
                obj["eagerness"] = serde_json::Value::String(e.clone());
            }
            if let Some(cr) = create_response {
                obj["create_response"] = serde_json::json!(cr);
            }
            if let Some(ir) = interrupt_response {
                obj["interrupt_response"] = serde_json::json!(ir);
            }
            obj
        }
        TurnDetection::ServerVad {
            threshold,
            prefix_padding_ms,
            silence_duration_ms,
            create_response,
            interrupt_response,
        } => {
            let mut obj = serde_json::json!({
                "type": "server_vad",
                "threshold": threshold,
                "prefix_padding_ms": prefix_padding_ms,
                "silence_duration_ms": silence_duration_ms,
            });
            if let Some(cr) = create_response {
                obj["create_response"] = serde_json::json!(cr);
            }
            if let Some(ir) = interrupt_response {
                obj["interrupt_response"] = serde_json::json!(ir);
            }
            obj
        }
        TurnDetection::Disabled => serde_json::Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Event parsing
// ---------------------------------------------------------------------------

/// Parse a raw JSON event from the Realtime API into a [`RealtimeSessionEvent`].
///
/// Unknown event types are returned as [`RealtimeSessionEvent::Unknown`].
#[cfg(feature = "voice")]
#[allow(clippy::too_many_lines)]
fn parse_realtime_event(json: &serde_json::Value) -> RealtimeSessionEvent {
    let event_type = json
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    match event_type {
        "session.created" => {
            let session_id = json
                .get("session")
                .and_then(|s| s.get("id"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            RealtimeSessionEvent::SessionCreated(SessionCreatedEvent { session_id })
        }
        "response.audio.delta" => {
            let audio_b64 = json
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let data = base64_decode(audio_b64);
            let response_id = json
                .get("response_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            RealtimeSessionEvent::AudioDelta(AudioDeltaEvent { data, response_id })
        }
        "response.audio_transcript.delta" => {
            let text = json
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            RealtimeSessionEvent::Transcript(TranscriptDeltaEvent {
                text,
                is_final: false,
                role: TranscriptRole::Assistant,
                item_id: json
                    .get("item_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
            })
        }
        "response.audio_transcript.done" => {
            let text = json
                .get("transcript")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            RealtimeSessionEvent::Transcript(TranscriptDeltaEvent {
                text,
                is_final: true,
                role: TranscriptRole::Assistant,
                item_id: json
                    .get("item_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
            })
        }
        "conversation.item.input_audio_transcription.completed" => {
            let text = json
                .get("transcript")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            RealtimeSessionEvent::Transcript(TranscriptDeltaEvent {
                text,
                is_final: true,
                role: TranscriptRole::User,
                item_id: json
                    .get("item_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
            })
        }
        "input_audio_buffer.speech_started" => {
            RealtimeSessionEvent::SpeechStarted(SpeechStartedEvent)
        }
        "input_audio_buffer.speech_stopped" => {
            RealtimeSessionEvent::SpeechStopped(SpeechStoppedEvent)
        }
        "response.function_call_arguments.done" => {
            let tool_name = json
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            let call_id = json
                .get("call_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            let arguments = json
                .get("arguments")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("{}")
                .to_owned();
            RealtimeSessionEvent::ToolCallCreated(ToolCallCreatedEvent {
                tool_name,
                call_id,
                arguments,
            })
        }
        "response.done" => {
            let response_id = json
                .get("response")
                .and_then(|r| r.get("id"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            RealtimeSessionEvent::ResponseDone(ResponseDoneEvent { response_id })
        }
        "error" => {
            let message = json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown error")
                .to_owned();
            let code = json
                .get("error")
                .and_then(|e| e.get("code"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            RealtimeSessionEvent::Error(RealtimeErrorEvent { message, code })
        }
        _ => RealtimeSessionEvent::Unknown(json.clone()),
    }
}

/// Decode a base64-encoded string into bytes.
#[cfg(feature = "voice")]
fn base64_decode(input: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(input)
        .unwrap_or_default()
}

// ==========================================================================
// Tests
// ==========================================================================

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

    // ---- Disconnect when not connected is a no-op ----

    #[tokio::test]
    async fn disconnect_when_not_connected() {
        let mut session = RealtimeSession::new(RealtimeConfig::default());
        let result = session.disconnect().await;
        assert!(result.is_ok());
        assert!(!session.is_connected());
    }

    // ---- Connect without voice feature returns error ----

    #[cfg(not(feature = "voice"))]
    #[tokio::test]
    async fn connect_without_voice_feature() {
        let mut session = RealtimeSession::new(RealtimeConfig::default());
        let result = session.connect().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("voice"));
    }

    // ---- Event parsing (voice feature only) ----

    #[cfg(feature = "voice")]
    mod voice_tests {
        use super::*;

        #[test]
        fn parse_session_created() {
            let json = serde_json::json!({
                "type": "session.created",
                "session": {"id": "sess-abc123"}
            });
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::SessionCreated(e) = event {
                assert_eq!(e.session_id, "sess-abc123");
            } else {
                panic!("expected SessionCreated, got {event:?}");
            }
        }

        #[test]
        fn parse_audio_delta() {
            use base64::Engine;
            let audio_data = vec![0x01, 0x02, 0x03];
            let encoded = base64::engine::general_purpose::STANDARD.encode(&audio_data);
            let json = serde_json::json!({
                "type": "response.audio.delta",
                "delta": encoded,
                "response_id": "resp-1",
            });
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::AudioDelta(e) = event {
                assert_eq!(e.data, audio_data);
                assert_eq!(e.response_id, "resp-1");
            } else {
                panic!("expected AudioDelta, got {event:?}");
            }
        }

        #[test]
        fn parse_transcript_delta() {
            let json = serde_json::json!({
                "type": "response.audio_transcript.delta",
                "delta": "Hello ",
                "item_id": "item-1",
            });
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::Transcript(e) = event {
                assert_eq!(e.text, "Hello ");
                assert!(!e.is_final);
                assert_eq!(e.role, TranscriptRole::Assistant);
                assert_eq!(e.item_id.as_deref(), Some("item-1"));
            } else {
                panic!("expected Transcript, got {event:?}");
            }
        }

        #[test]
        fn parse_transcript_done() {
            let json = serde_json::json!({
                "type": "response.audio_transcript.done",
                "transcript": "Hello world!",
                "item_id": "item-2",
            });
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::Transcript(e) = event {
                assert_eq!(e.text, "Hello world!");
                assert!(e.is_final);
                assert_eq!(e.role, TranscriptRole::Assistant);
            } else {
                panic!("expected Transcript, got {event:?}");
            }
        }

        #[test]
        fn parse_user_audio_transcription() {
            let json = serde_json::json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "What is the weather?",
                "item_id": "item-3",
            });
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::Transcript(e) = event {
                assert_eq!(e.text, "What is the weather?");
                assert!(e.is_final);
                assert_eq!(e.role, TranscriptRole::User);
            } else {
                panic!("expected Transcript, got {event:?}");
            }
        }

        #[test]
        fn parse_speech_started() {
            let json = serde_json::json!({"type": "input_audio_buffer.speech_started"});
            let event = parse_realtime_event(&json);
            assert!(matches!(event, RealtimeSessionEvent::SpeechStarted(_)));
        }

        #[test]
        fn parse_speech_stopped() {
            let json = serde_json::json!({"type": "input_audio_buffer.speech_stopped"});
            let event = parse_realtime_event(&json);
            assert!(matches!(event, RealtimeSessionEvent::SpeechStopped(_)));
        }

        #[test]
        fn parse_tool_call_created() {
            let json = serde_json::json!({
                "type": "response.function_call_arguments.done",
                "name": "get_weather",
                "call_id": "call-42",
                "arguments": "{\"city\": \"NYC\"}",
            });
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::ToolCallCreated(e) = event {
                assert_eq!(e.tool_name, "get_weather");
                assert_eq!(e.call_id, "call-42");
                assert_eq!(e.arguments, "{\"city\": \"NYC\"}");
            } else {
                panic!("expected ToolCallCreated, got {event:?}");
            }
        }

        #[test]
        fn parse_response_done() {
            let json = serde_json::json!({
                "type": "response.done",
                "response": {"id": "resp-xyz"}
            });
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::ResponseDone(e) = event {
                assert_eq!(e.response_id, "resp-xyz");
            } else {
                panic!("expected ResponseDone, got {event:?}");
            }
        }

        #[test]
        fn parse_error() {
            let json = serde_json::json!({
                "type": "error",
                "error": {
                    "message": "rate limited",
                    "code": "rate_limit_exceeded"
                }
            });
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::Error(e) = event {
                assert_eq!(e.message, "rate limited");
                assert_eq!(e.code.as_deref(), Some("rate_limit_exceeded"));
            } else {
                panic!("expected Error, got {event:?}");
            }
        }

        #[test]
        fn parse_error_without_code() {
            let json = serde_json::json!({
                "type": "error",
                "error": {"message": "something went wrong"}
            });
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::Error(e) = event {
                assert_eq!(e.message, "something went wrong");
                assert!(e.code.is_none());
            } else {
                panic!("expected Error, got {event:?}");
            }
        }

        #[test]
        fn parse_unknown_event() {
            let json = serde_json::json!({"type": "some.future.event", "data": 42});
            let event = parse_realtime_event(&json);
            if let RealtimeSessionEvent::Unknown(v) = event {
                assert_eq!(
                    v.get("type").and_then(serde_json::Value::as_str),
                    Some("some.future.event")
                );
            } else {
                panic!("expected Unknown, got {event:?}");
            }
        }

        #[test]
        fn parse_event_missing_type() {
            let json = serde_json::json!({"data": "no type field"});
            let event = parse_realtime_event(&json);
            assert!(matches!(event, RealtimeSessionEvent::Unknown(_)));
        }

        // ---- Base64 round-trip ----

        #[test]
        fn base64_round_trip() {
            use base64::Engine;
            let original = vec![0u8, 1, 2, 127, 128, 255];
            let encoded = base64::engine::general_purpose::STANDARD.encode(&original);
            let decoded = base64_decode(&encoded);
            assert_eq!(decoded, original);
        }

        #[test]
        fn base64_decode_empty() {
            let decoded = base64_decode("");
            assert!(decoded.is_empty());
        }

        #[test]
        fn base64_decode_invalid_returns_empty() {
            let decoded = base64_decode("!@#$%^&*");
            assert!(decoded.is_empty());
        }

        // ---- Turn detection JSON ----

        #[test]
        fn turn_detection_server_vad_json() {
            use crate::voice::config::TurnDetection;
            let td = TurnDetection::default();
            let json = turn_detection_to_json(&td);
            assert_eq!(
                json.get("type").and_then(serde_json::Value::as_str),
                Some("server_vad")
            );
            assert!(json.get("threshold").is_some());
        }

        #[test]
        fn turn_detection_disabled_json() {
            use crate::voice::config::TurnDetection;
            let td = TurnDetection::Disabled;
            let json = turn_detection_to_json(&td);
            assert!(json.is_null());
        }

        #[test]
        fn turn_detection_semantic_vad_json() {
            use crate::voice::config::TurnDetection;
            let td = TurnDetection::SemanticVad {
                eagerness: Some("high".to_owned()),
                create_response: Some(true),
                interrupt_response: None,
            };
            let json = turn_detection_to_json(&td);
            assert_eq!(
                json.get("type").and_then(serde_json::Value::as_str),
                Some("semantic_vad")
            );
            assert_eq!(
                json.get("eagerness").and_then(serde_json::Value::as_str),
                Some("high")
            );
        }
    }
}
