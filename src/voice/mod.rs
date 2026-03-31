//! Voice and realtime agent support.
//!
//! This module provides types for building agents that interact via voice (audio)
//! using `OpenAI`'s Realtime API over WebSocket connections.  It mirrors the Python
//! SDK's `realtime/` package.
//!
//! Enable the `voice` feature flag to use this module.
//!
//! # Architecture
//!
//! The voice module is organized into several sub-modules:
//!
//! - [`config`] -- configuration types for audio format, voice, turn detection,
//!   and model settings.
//! - [`agent`] -- the [`RealtimeAgent`] type, a specialized agent for voice
//!   interactions.
//! - [`events`] -- event types emitted during a realtime session.
//! - [`items`] -- data types representing messages, tool calls, and responses in
//!   realtime conversations.
//! - [`model`] -- the [`RealtimeModel`] trait for connecting to a realtime backend.
//! - [`session`] -- the [`RealtimeSession`] that orchestrates a live voice
//!   conversation.
//! - [`pipeline`] -- high-level [`VoicePipeline`] for end-to-end voice workflows.

pub mod agent;
pub mod config;
pub mod events;
pub mod items;
pub mod model;
pub mod pipeline;
pub mod session;

pub use agent::{RealtimeAgent, RealtimeAgentBuilder};
pub use config::{
    AudioFormat, RealtimeAudioConfig, RealtimeAudioInputConfig, RealtimeAudioOutputConfig,
    RealtimeConfig, RealtimeGuardrailsSettings, RealtimeModelSettings, RealtimeModelTracingConfig,
    RealtimeRunConfig, TurnDetection, Voice,
};
pub use events::{
    RealtimeAgentEndEvent, RealtimeAgentStartEvent, RealtimeAudioEndEvent, RealtimeAudioEvent,
    RealtimeAudioInterruptedEvent, RealtimeErrorEvent, RealtimeGuardrailTrippedEvent,
    RealtimeHandoffEvent, RealtimeHistoryAddedEvent, RealtimeHistoryUpdatedEvent,
    RealtimeSessionEvent, RealtimeToolEndEvent, RealtimeToolStartEvent, TranscriptDeltaEvent,
    TranscriptRole,
};
pub use items::{
    AssistantAudioContent, AssistantMessageItem, AssistantTextContent, InputAudioContent,
    InputTextContent, RealtimeItem, RealtimeMessageItem, RealtimeResponse, RealtimeToolCallItem,
    SystemMessageItem, ToolCallStatus, UserMessageItem,
};
pub use model::{RealtimeModel, RealtimeModelConfig, RealtimeModelEvent, RealtimeModelListener};
pub use pipeline::{AudioInput, AudioOutput, VoicePipeline, VoicePipelineConfig};
pub use session::RealtimeSession;
