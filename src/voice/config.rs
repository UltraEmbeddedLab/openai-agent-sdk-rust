//! Configuration types for realtime voice sessions.
//!
//! This module defines the configuration types used to set up a realtime session,
//! including audio format, voice selection, turn detection, model settings, and
//! run configuration.  It mirrors the Python SDK's `realtime/config.py`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// AudioFormat
// ---------------------------------------------------------------------------

/// Audio format for realtime audio streams.
///
/// Controls how audio data is encoded when sent to and received from the
/// Realtime API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum AudioFormat {
    /// PCM 16-bit audio at 24 kHz (default).
    #[default]
    #[serde(rename = "pcm16")]
    Pcm16,
    /// G.711 mu-law encoded audio.
    #[serde(rename = "g711_ulaw")]
    G711Ulaw,
    /// G.711 a-law encoded audio.
    #[serde(rename = "g711_alaw")]
    G711Alaw,
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pcm16 => write!(f, "pcm16"),
            Self::G711Ulaw => write!(f, "g711_ulaw"),
            Self::G711Alaw => write!(f, "g711_alaw"),
        }
    }
}

// ---------------------------------------------------------------------------
// Voice
// ---------------------------------------------------------------------------

/// Voice options for text-to-speech in realtime sessions.
///
/// Specifies which voice the model should use when generating audio output.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum Voice {
    /// The "alloy" voice (default).
    #[default]
    #[serde(rename = "alloy")]
    Alloy,
    /// The "ash" voice.
    #[serde(rename = "ash")]
    Ash,
    /// The "ballad" voice.
    #[serde(rename = "ballad")]
    Ballad,
    /// The "coral" voice.
    #[serde(rename = "coral")]
    Coral,
    /// The "echo" voice.
    #[serde(rename = "echo")]
    Echo,
    /// The "sage" voice.
    #[serde(rename = "sage")]
    Sage,
    /// The "shimmer" voice.
    #[serde(rename = "shimmer")]
    Shimmer,
    /// The "verse" voice.
    #[serde(rename = "verse")]
    Verse,
    /// A custom voice identifier not covered by the known variants.
    Custom(String),
}

impl std::fmt::Display for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Alloy => write!(f, "alloy"),
            Self::Ash => write!(f, "ash"),
            Self::Ballad => write!(f, "ballad"),
            Self::Coral => write!(f, "coral"),
            Self::Echo => write!(f, "echo"),
            Self::Sage => write!(f, "sage"),
            Self::Shimmer => write!(f, "shimmer"),
            Self::Verse => write!(f, "verse"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

// ---------------------------------------------------------------------------
// TurnDetection
// ---------------------------------------------------------------------------

/// Turn detection mode for realtime sessions.
///
/// Controls how the model detects when the user has finished speaking and it
/// should begin generating a response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TurnDetection {
    /// Semantic voice activity detection provided by the server.
    SemanticVad {
        /// How eagerly to detect turn boundaries.
        eagerness: Option<String>,
        /// Whether to create a response when a turn is detected.
        create_response: Option<bool>,
        /// Whether to allow interrupting the assistant's response.
        interrupt_response: Option<bool>,
    },
    /// Server-side voice activity detection.
    ServerVad {
        /// Activation threshold for voice activity detection (0.0 to 1.0).
        threshold: f64,
        /// Padding time in milliseconds before the detected speech start.
        prefix_padding_ms: u32,
        /// Duration of silence in milliseconds required to trigger end of turn.
        silence_duration_ms: u32,
        /// Whether to create a response when a turn is detected.
        create_response: Option<bool>,
        /// Whether to allow interrupting the assistant's response.
        interrupt_response: Option<bool>,
    },
    /// No automatic turn detection (manual push-to-talk mode).
    Disabled,
}

impl Default for TurnDetection {
    fn default() -> Self {
        Self::ServerVad {
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
            create_response: None,
            interrupt_response: None,
        }
    }
}

// ---------------------------------------------------------------------------
// RealtimeAudioInputConfig
// ---------------------------------------------------------------------------

/// Configuration for audio input in a realtime session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct RealtimeAudioInputConfig {
    /// The audio format for input streams.
    pub format: Option<AudioFormat>,
    /// Turn detection configuration.
    pub turn_detection: Option<TurnDetection>,
    /// Language code for input audio transcription.
    pub transcription_language: Option<String>,
    /// The transcription model to use (e.g. `"whisper-1"`).
    pub transcription_model: Option<String>,
}

// ---------------------------------------------------------------------------
// RealtimeAudioOutputConfig
// ---------------------------------------------------------------------------

/// Configuration for audio output in a realtime session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct RealtimeAudioOutputConfig {
    /// The audio format for output streams.
    pub format: Option<AudioFormat>,
    /// The voice to use for text-to-speech.
    pub voice: Option<Voice>,
    /// The playback speed multiplier.
    pub speed: Option<f64>,
}

// ---------------------------------------------------------------------------
// RealtimeAudioConfig
// ---------------------------------------------------------------------------

/// Combined audio input and output configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct RealtimeAudioConfig {
    /// Input audio configuration.
    pub input: Option<RealtimeAudioInputConfig>,
    /// Output audio configuration.
    pub output: Option<RealtimeAudioOutputConfig>,
}

// ---------------------------------------------------------------------------
// RealtimeModelTracingConfig
// ---------------------------------------------------------------------------

/// Tracing configuration for realtime model sessions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct RealtimeModelTracingConfig {
    /// The workflow name to use for tracing.
    pub workflow_name: Option<String>,
    /// A group identifier to link multiple traces together.
    pub group_id: Option<String>,
    /// Additional metadata to include with the trace.
    pub metadata: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// RealtimeModelSettings
// ---------------------------------------------------------------------------

/// Model settings specific to the Realtime API.
///
/// Controls voice, audio format, turn detection, and other parameters for a
/// realtime model session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct RealtimeModelSettings {
    /// The name of the realtime model to use.
    pub model_name: Option<String>,
    /// System instructions for the model.
    pub instructions: Option<String>,
    /// The modalities the model should support (e.g. `["text", "audio"]`).
    pub modalities: Option<Vec<String>>,
    /// The voice to use for audio output.
    pub voice: Voice,
    /// Audio configuration for input and output.
    pub audio: Option<RealtimeAudioConfig>,
    /// The format for input audio streams (legacy, prefer `audio.input.format`).
    pub input_audio_format: Option<AudioFormat>,
    /// The format for output audio streams (legacy, prefer `audio.output.format`).
    pub output_audio_format: Option<AudioFormat>,
    /// Turn detection configuration.
    pub turn_detection: TurnDetection,
    /// Sampling temperature for model responses.
    pub temperature: Option<f64>,
    /// Maximum number of output tokens per response.
    pub max_response_output_tokens: Option<u32>,
    /// Tracing configuration.
    pub tracing: Option<RealtimeModelTracingConfig>,
}

// ---------------------------------------------------------------------------
// RealtimeGuardrailsSettings
// ---------------------------------------------------------------------------

/// Settings for output guardrails in realtime sessions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RealtimeGuardrailsSettings {
    /// The minimum number of characters to accumulate before running guardrails
    /// on transcript deltas.  Defaults to 100.
    pub debounce_text_length: u32,
}

impl Default for RealtimeGuardrailsSettings {
    fn default() -> Self {
        Self {
            debounce_text_length: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// RealtimeRunConfig
// ---------------------------------------------------------------------------

/// Configuration for running a realtime agent session.
///
/// This is the realtime equivalent of [`RunConfig`](crate::config::RunConfig).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeRunConfig {
    /// Settings for the realtime model session.
    pub model_settings: Option<RealtimeModelSettings>,
    /// Settings for guardrail execution.
    pub guardrails_settings: Option<RealtimeGuardrailsSettings>,
    /// Whether tracing is disabled for this run.
    pub tracing_disabled: bool,
    /// Whether function tool calls should run asynchronously.  Defaults to `true`.
    pub async_tool_calls: bool,
}

impl Default for RealtimeRunConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl RealtimeRunConfig {
    /// Create a new default run configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model_settings: None,
            guardrails_settings: None,
            tracing_disabled: false,
            async_tool_calls: true,
        }
    }
}

// ---------------------------------------------------------------------------
// RealtimeConfig
// ---------------------------------------------------------------------------

/// Top-level configuration for establishing a realtime session connection.
///
/// Includes the model name, API credentials, and model settings.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeConfig {
    /// The realtime model name to connect to.
    pub model: String,
    /// Model settings for the session.
    pub model_settings: RealtimeModelSettings,
    /// API key for authentication.  If `None`, the `OPENAI_API_KEY` environment
    /// variable is used.
    pub api_key: Option<String>,
    /// The WebSocket base URL for the Realtime API.
    pub base_url: String,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o-realtime-preview".to_owned(),
            model_settings: RealtimeModelSettings::default(),
            api_key: None,
            base_url: "wss://api.openai.com/v1/realtime".to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AudioFormat ----

    #[test]
    fn audio_format_default_is_pcm16() {
        assert_eq!(AudioFormat::default(), AudioFormat::Pcm16);
    }

    #[test]
    fn audio_format_display() {
        assert_eq!(AudioFormat::Pcm16.to_string(), "pcm16");
        assert_eq!(AudioFormat::G711Ulaw.to_string(), "g711_ulaw");
        assert_eq!(AudioFormat::G711Alaw.to_string(), "g711_alaw");
    }

    #[test]
    fn audio_format_serialization_roundtrip() {
        let format = AudioFormat::G711Ulaw;
        let json = serde_json::to_string(&format).expect("serialize");
        assert_eq!(json, r#""g711_ulaw""#);
        let deserialized: AudioFormat = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, format);
    }

    // ---- Voice ----

    #[test]
    fn voice_default_is_alloy() {
        assert_eq!(Voice::default(), Voice::Alloy);
    }

    #[test]
    fn voice_display() {
        assert_eq!(Voice::Alloy.to_string(), "alloy");
        assert_eq!(Voice::Sage.to_string(), "sage");
        assert_eq!(Voice::Custom("my-voice".to_owned()).to_string(), "my-voice");
    }

    #[test]
    fn voice_serialization_roundtrip() {
        let voice = Voice::Coral;
        let json = serde_json::to_string(&voice).expect("serialize");
        let deserialized: Voice = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, voice);
    }

    #[test]
    fn voice_custom_serialization() {
        let voice = Voice::Custom("custom-v1".to_owned());
        let json = serde_json::to_string(&voice).expect("serialize");
        let deserialized: Voice = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, voice);
    }

    // ---- TurnDetection ----

    #[test]
    fn turn_detection_default_is_server_vad() {
        let td = TurnDetection::default();
        assert!(matches!(td, TurnDetection::ServerVad { .. }));
    }

    #[test]
    fn turn_detection_server_vad_values() {
        if let TurnDetection::ServerVad {
            threshold,
            prefix_padding_ms,
            silence_duration_ms,
            ..
        } = TurnDetection::default()
        {
            assert!((threshold - 0.5).abs() < f64::EPSILON);
            assert_eq!(prefix_padding_ms, 300);
            assert_eq!(silence_duration_ms, 500);
        } else {
            panic!("expected ServerVad");
        }
    }

    // ---- RealtimeModelSettings ----

    #[test]
    fn model_settings_default() {
        let settings = RealtimeModelSettings::default();
        assert_eq!(settings.voice, Voice::Alloy);
        assert!(settings.temperature.is_none());
        assert!(settings.max_response_output_tokens.is_none());
        assert!(settings.model_name.is_none());
        assert!(settings.instructions.is_none());
    }

    // ---- RealtimeGuardrailsSettings ----

    #[test]
    fn guardrails_settings_default() {
        let gs = RealtimeGuardrailsSettings::default();
        assert_eq!(gs.debounce_text_length, 100);
    }

    // ---- RealtimeRunConfig ----

    #[test]
    fn run_config_default() {
        let rc = RealtimeRunConfig::default();
        assert!(rc.model_settings.is_none());
        assert!(rc.guardrails_settings.is_none());
        assert!(!rc.tracing_disabled);
        assert!(rc.async_tool_calls);
    }

    #[test]
    fn run_config_new_matches_default() {
        let a = RealtimeRunConfig::new();
        let b = RealtimeRunConfig::default();
        assert_eq!(a.tracing_disabled, b.tracing_disabled);
        assert_eq!(a.async_tool_calls, b.async_tool_calls);
    }

    // ---- RealtimeConfig ----

    #[test]
    fn realtime_config_default() {
        let cfg = RealtimeConfig::default();
        assert_eq!(cfg.model, "gpt-4o-realtime-preview");
        assert!(cfg.api_key.is_none());
        assert!(cfg.base_url.starts_with("wss://"));
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn config_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AudioFormat>();
        assert_send_sync::<Voice>();
        assert_send_sync::<TurnDetection>();
        assert_send_sync::<RealtimeModelSettings>();
        assert_send_sync::<RealtimeGuardrailsSettings>();
        assert_send_sync::<RealtimeRunConfig>();
        assert_send_sync::<RealtimeConfig>();
        assert_send_sync::<RealtimeAudioConfig>();
        assert_send_sync::<RealtimeAudioInputConfig>();
        assert_send_sync::<RealtimeAudioOutputConfig>();
        assert_send_sync::<RealtimeModelTracingConfig>();
    }
}
