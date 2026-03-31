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
    /// 3. Start audio capture from the configured input.
    /// 4. Forward audio to the API.
    /// 5. Receive and play back audio from the model.
    /// 6. Handle tool calls.
    /// 7. Handle handoffs between agents.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline fails to start or encounters an
    /// unrecoverable error during execution.
    ///
    /// # Note
    ///
    /// The full implementation requires WebSocket transport.  This currently
    /// returns an error indicating the pipeline is not yet fully implemented.
    #[allow(clippy::unused_async)]
    pub async fn run<C: Send + Sync + 'static>(&self, _agent: &RealtimeAgent<C>) -> Result<()> {
        // TODO: Implement full voice pipeline:
        // 1. Open RealtimeSession
        // 2. Configure session with agent's tools/instructions
        // 3. Start audio capture
        // 4. Forward audio to API
        // 5. Receive and play back audio
        // 6. Handle tool calls
        // 7. Handle handoffs
        Err(AgentError::UserError {
            message: "Voice pipeline not yet fully implemented. \
                      WebSocket transport required."
                .to_owned(),
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
    async fn pipeline_run_returns_not_implemented_error() {
        let agent = RealtimeAgent::<()>::builder("test").build();
        let pipeline = VoicePipeline::new(VoicePipelineConfig::default());
        let result = pipeline.run(&agent).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not yet fully implemented"));
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
