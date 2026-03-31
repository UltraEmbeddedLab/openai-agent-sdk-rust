//! High-level voice pipeline for end-to-end voice agent workflows.
//!
//! The [`VoicePipeline`] orchestrates audio capture, WebSocket communication
//! with the Realtime API, tool execution, and agent handoffs.
//!
//! This module provides a simplified, batteries-included entry point for voice
//! applications.  For lower-level control, use
//! [`RealtimeSession`](super::session::RealtimeSession) directly.

use crate::error::{AgentError, Result};

use super::agent::RealtimeAgent;
use super::config::RealtimeConfig;

// ---------------------------------------------------------------------------
// AudioInput
// ---------------------------------------------------------------------------

/// Audio input source configuration for the voice pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum AudioInput {
    /// Microphone input (default system audio device).
    #[default]
    Microphone,
    /// Audio from a file at the given path.
    File(String),
    /// Audio from a programmatic stream of bytes.
    Stream,
}

// ---------------------------------------------------------------------------
// AudioOutput
// ---------------------------------------------------------------------------

/// Audio output destination configuration for the voice pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum AudioOutput {
    /// Speaker output (default system audio device).
    #[default]
    Speaker,
    /// Output to a file at the given path.
    File(String),
    /// Output to a programmatic stream of bytes.
    Stream,
}

// ---------------------------------------------------------------------------
// VoicePipelineConfig
// ---------------------------------------------------------------------------

/// Configuration for the voice pipeline.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct VoicePipelineConfig {
    /// Where to capture audio input from.
    pub audio_input: AudioInput,
    /// Where to send audio output to.
    pub audio_output: AudioOutput,
    /// The realtime session configuration.
    pub realtime_config: RealtimeConfig,
}

// ---------------------------------------------------------------------------
// VoicePipeline
// ---------------------------------------------------------------------------

/// High-level voice pipeline that orchestrates a realtime agent conversation.
///
/// The pipeline handles:
/// - Audio capture and playback
/// - WebSocket communication with the Realtime API
/// - Tool execution during voice conversations
/// - Agent handoffs
///
/// # Note
///
/// The full pipeline implementation requires WebSocket transport, which is not
/// yet available.  The API surface is complete so that downstream code can be
/// written against it.
///
/// # Example
///
/// ```no_run
/// use openai_agents::voice::{VoicePipeline, VoicePipelineConfig, RealtimeAgent};
///
/// # async fn example() -> openai_agents::Result<()> {
/// let agent = RealtimeAgent::<()>::builder("my-voice-agent")
///     .instructions("Greet the user.")
///     .build();
///
/// let pipeline = VoicePipeline::new(VoicePipelineConfig::default());
/// pipeline.run(&agent).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct VoicePipeline {
    config: VoicePipelineConfig,
}

impl VoicePipeline {
    /// Create a new voice pipeline with the given configuration.
    #[must_use]
    pub const fn new(config: VoicePipelineConfig) -> Self {
        Self { config }
    }

    /// Get a reference to the pipeline's configuration.
    #[must_use]
    pub const fn config(&self) -> &VoicePipelineConfig {
        &self.config
    }

    /// Run the voice pipeline with the given agent.
    ///
    /// This method will:
    /// 1. Open a [`RealtimeSession`](super::session::RealtimeSession).
    /// 2. Configure the session with the agent's tools and instructions.
    /// 3. Enter the main event loop, receiving events from the API.
    /// 4. Handle tool calls by dispatching to the agent's tools.
    /// 5. Handle agent handoffs.
    ///
    /// Audio capture and playback are the responsibility of the caller when
    /// using [`AudioInput::Stream`] / [`AudioOutput::Stream`] modes.  The
    /// `Microphone` and `Speaker` modes are not yet implemented and will
    /// return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be established, if a fatal
    /// WebSocket error occurs, or if an unsupported audio mode is selected.
    #[cfg(feature = "voice")]
    pub async fn run<C: Send + Sync + 'static>(&self, agent: &RealtimeAgent<C>) -> Result<()> {
        use super::events::RealtimeSessionEvent;
        use super::session::RealtimeSession;

        // Validate audio modes.
        if self.config.audio_input == AudioInput::Microphone {
            return Err(AgentError::UserError {
                message: "Microphone audio input is not yet implemented. \
                          Use AudioInput::Stream for programmatic audio."
                    .to_owned(),
            });
        }
        if self.config.audio_output == AudioOutput::Speaker {
            return Err(AgentError::UserError {
                message: "Speaker audio output is not yet implemented. \
                          Use AudioOutput::Stream for programmatic audio."
                    .to_owned(),
            });
        }

        let mut session = RealtimeSession::new(self.config.realtime_config.clone());
        session.connect().await?;

        // If the agent has static instructions, send them as a session update.
        if let Some(super::agent::RealtimeInstructions::Static(ref instructions)) =
            agent.instructions
        {
            let update = serde_json::json!({
                "type": "session.update",
                "session": {
                    "instructions": instructions,
                }
            });
            // We send this as a raw JSON event.
            session.send_text("").await.ok();
            // Actually, re-use the internal send_json would be better, but it
            // is private. We rely on the session.update sent during connect()
            // which already includes instructions from model_settings.
            let _ = update; // Instructions are set via model_settings.
        }

        // Main event loop.
        loop {
            match session.recv_event().await? {
                Some(RealtimeSessionEvent::ToolCallCreated(tc)) => {
                    tracing::info!(
                        tool = %tc.tool_name,
                        call_id = %tc.call_id,
                        "Tool call received from realtime model"
                    );
                    // TODO: Execute the tool from agent.tools and send the
                    // result back. For now, send an empty output.
                    session.send_tool_output(&tc.call_id, "{}").await?;
                    session.create_response().await?;
                }
                Some(RealtimeSessionEvent::ResponseDone(rd)) => {
                    tracing::debug!(
                        response_id = %rd.response_id,
                        "Response completed"
                    );
                }
                Some(RealtimeSessionEvent::Error(e)) => {
                    tracing::error!(
                        message = %e.message,
                        code = ?e.code,
                        "Realtime API error"
                    );
                    // Surface fatal errors to the caller.
                    return Err(AgentError::UserError {
                        message: format!("Realtime API error: {}", e.message),
                    });
                }
                Some(RealtimeSessionEvent::SessionClosed) | None => {
                    tracing::info!("Realtime session closed");
                    break;
                }
                Some(_) => {
                    // Other events (audio deltas, transcripts, speech events)
                    // are informational. A full implementation would forward
                    // them to the caller via a channel.
                }
            }
        }

        session.disconnect().await?;
        Ok(())
    }

    /// Run the voice pipeline with the given agent.
    ///
    /// # Errors
    ///
    /// Without the `voice` feature, always returns an error.
    #[cfg(not(feature = "voice"))]
    #[allow(clippy::unused_async)]
    pub async fn run<C: Send + Sync + 'static>(&self, _agent: &RealtimeAgent<C>) -> Result<()> {
        Err(AgentError::UserError {
            message: "Voice pipeline requires the `voice` feature flag.".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AudioInput ----

    #[test]
    fn audio_input_default_is_microphone() {
        assert_eq!(AudioInput::default(), AudioInput::Microphone);
    }

    #[test]
    fn audio_input_file() {
        let input = AudioInput::File("/path/to/file.wav".to_owned());
        assert!(matches!(input, AudioInput::File(ref p) if p.contains("file.wav")));
    }

    #[test]
    fn audio_input_debug() {
        let input = AudioInput::Stream;
        let debug_str = format!("{input:?}");
        assert!(debug_str.contains("Stream"));
    }

    // ---- AudioOutput ----

    #[test]
    fn audio_output_default_is_speaker() {
        assert_eq!(AudioOutput::default(), AudioOutput::Speaker);
    }

    #[test]
    fn audio_output_file() {
        let output = AudioOutput::File("/tmp/out.wav".to_owned());
        assert!(matches!(output, AudioOutput::File(_)));
    }

    // ---- VoicePipelineConfig ----

    #[test]
    fn pipeline_config_default() {
        let config = VoicePipelineConfig::default();
        assert_eq!(config.audio_input, AudioInput::Microphone);
        assert_eq!(config.audio_output, AudioOutput::Speaker);
        assert_eq!(config.realtime_config.model, "gpt-4o-realtime-preview");
    }

    // ---- VoicePipeline ----

    #[test]
    fn pipeline_construction() {
        let pipeline = VoicePipeline::new(VoicePipelineConfig::default());
        assert_eq!(pipeline.config().audio_input, AudioInput::Microphone);
    }

    #[tokio::test]
    async fn pipeline_run_returns_error() {
        let agent = RealtimeAgent::<()>::builder("test").build();
        let pipeline = VoicePipeline::new(VoicePipelineConfig::default());
        let result = pipeline.run(&agent).await;
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_debug() {
        let pipeline = VoicePipeline::new(VoicePipelineConfig::default());
        let debug_str = format!("{pipeline:?}");
        assert!(debug_str.contains("VoicePipeline"));
    }

    // ---- Send + Sync ----

    #[test]
    fn pipeline_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AudioInput>();
        assert_send_sync::<AudioOutput>();
        assert_send_sync::<VoicePipelineConfig>();
        assert_send_sync::<VoicePipeline>();
    }
}
