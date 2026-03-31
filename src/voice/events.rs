//! Event types emitted during a realtime voice session.
//!
//! These events are yielded by [`RealtimeSession`](super::session::RealtimeSession)
//! and represent lifecycle milestones, audio data, transcript updates, tool calls,
//! errors, and more.
//!
//! This module mirrors the Python SDK's `realtime/events.py`.

use serde::{Deserialize, Serialize};

use super::config::AudioFormat;

// ---------------------------------------------------------------------------
// TranscriptRole
// ---------------------------------------------------------------------------

/// The role associated with a transcript event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TranscriptRole {
    /// Transcript of user speech.
    User,
    /// Transcript of assistant speech.
    Assistant,
}

impl std::fmt::Display for TranscriptRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

// ---------------------------------------------------------------------------
// Individual event structs
// ---------------------------------------------------------------------------

/// A new agent has started handling the session.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeAgentStartEvent {
    /// The name of the agent that started.
    pub agent_name: String,
}

/// An agent has stopped handling the session.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeAgentEndEvent {
    /// The name of the agent that ended.
    pub agent_name: String,
}

/// An agent handed off to another agent.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeHandoffEvent {
    /// The name of the agent that initiated the handoff.
    pub from_agent: String,
    /// The name of the agent that received the handoff.
    pub to_agent: String,
}

/// A tool call has started executing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeToolStartEvent {
    /// The name of the agent invoking the tool.
    pub agent_name: String,
    /// The name of the tool being called.
    pub tool_name: String,
    /// The JSON-encoded arguments for the tool call.
    pub arguments: String,
}

/// A tool call has finished executing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeToolEndEvent {
    /// The name of the agent that invoked the tool.
    pub agent_name: String,
    /// The name of the tool that was called.
    pub tool_name: String,
    /// The JSON-encoded arguments that were passed to the tool.
    pub arguments: String,
    /// The output produced by the tool.
    pub output: String,
}

/// Audio data received from the model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeAudioEvent {
    /// The raw audio data bytes.
    pub data: Vec<u8>,
    /// The audio format of the data.
    pub format: AudioFormat,
    /// The item ID this audio belongs to.
    pub item_id: String,
    /// The content index within the item.
    pub content_index: u32,
}

/// The model finished generating audio for an item.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeAudioEndEvent {
    /// The item ID whose audio is complete.
    pub item_id: String,
    /// The content index within the item.
    pub content_index: u32,
}

/// Audio playback was interrupted (e.g. the user spoke while the assistant was
/// speaking).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeAudioInterruptedEvent {
    /// The item ID that was interrupted.
    pub item_id: String,
    /// The content index within the item.
    pub content_index: u32,
}

/// A transcript delta or final transcript event.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TranscriptDeltaEvent {
    /// The transcript text.
    pub text: String,
    /// Whether this is the final transcript for the utterance.
    pub is_final: bool,
    /// Whether this is from the user or assistant.
    pub role: TranscriptRole,
    /// The item ID this transcript belongs to.
    pub item_id: Option<String>,
}

/// An error occurred during the realtime session.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeErrorEvent {
    /// A human-readable error message.
    pub message: String,
}

/// A guardrail was tripped during the session.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeGuardrailTrippedEvent {
    /// The message that was being generated when the guardrail triggered.
    pub message: String,
}

/// The conversation history was updated.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeHistoryUpdatedEvent {
    /// The number of items now in the history.
    pub item_count: usize,
}

/// A new item was added to the conversation history.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RealtimeHistoryAddedEvent {
    /// The ID of the item that was added.
    pub item_id: String,
}

// ---------------------------------------------------------------------------
// RealtimeSessionEvent
// ---------------------------------------------------------------------------

/// An event emitted by a realtime session.
///
/// This is the primary event type that consumers iterate over while a
/// [`RealtimeSession`](super::session::RealtimeSession) is active.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RealtimeSessionEvent {
    /// A new agent started handling the session.
    AgentStart(RealtimeAgentStartEvent),
    /// An agent stopped handling the session.
    AgentEnd(RealtimeAgentEndEvent),
    /// An agent handed off to another agent.
    Handoff(RealtimeHandoffEvent),
    /// A tool call started executing.
    ToolStart(RealtimeToolStartEvent),
    /// A tool call finished executing.
    ToolEnd(RealtimeToolEndEvent),
    /// Audio data from the model.
    Audio(RealtimeAudioEvent),
    /// The model finished generating audio for an item.
    AudioEnd(RealtimeAudioEndEvent),
    /// Audio playback was interrupted.
    AudioInterrupted(RealtimeAudioInterruptedEvent),
    /// A transcript delta or final transcript.
    Transcript(TranscriptDeltaEvent),
    /// An error occurred.
    Error(RealtimeErrorEvent),
    /// A guardrail was tripped.
    GuardrailTripped(RealtimeGuardrailTrippedEvent),
    /// The conversation history was updated.
    HistoryUpdated(RealtimeHistoryUpdatedEvent),
    /// A new item was added to the conversation history.
    HistoryAdded(RealtimeHistoryAddedEvent),
    /// The model finished its response turn.
    TurnEnded,
    /// The session was closed.
    SessionClosed,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TranscriptRole ----

    #[test]
    fn transcript_role_display() {
        assert_eq!(TranscriptRole::User.to_string(), "user");
        assert_eq!(TranscriptRole::Assistant.to_string(), "assistant");
    }

    #[test]
    fn transcript_role_equality() {
        assert_eq!(TranscriptRole::User, TranscriptRole::User);
        assert_ne!(TranscriptRole::User, TranscriptRole::Assistant);
    }

    // ---- Event construction ----

    #[test]
    fn agent_start_event() {
        let event = RealtimeAgentStartEvent {
            agent_name: "voice-bot".to_owned(),
        };
        let debug_str = format!("{event:?}");
        assert!(debug_str.contains("voice-bot"));
    }

    #[test]
    fn handoff_event() {
        let event = RealtimeHandoffEvent {
            from_agent: "a".to_owned(),
            to_agent: "b".to_owned(),
        };
        let debug_str = format!("{event:?}");
        assert!(debug_str.contains("a"));
        assert!(debug_str.contains("b"));
    }

    #[test]
    fn audio_event() {
        let event = RealtimeAudioEvent {
            data: vec![1, 2, 3],
            format: AudioFormat::Pcm16,
            item_id: "item-1".to_owned(),
            content_index: 0,
        };
        assert_eq!(event.data.len(), 3);
        assert_eq!(event.format, AudioFormat::Pcm16);
    }

    #[test]
    fn transcript_delta_event() {
        let event = TranscriptDeltaEvent {
            text: "Hello there".to_owned(),
            is_final: false,
            role: TranscriptRole::Assistant,
            item_id: Some("item-2".to_owned()),
        };
        assert!(!event.is_final);
        assert_eq!(event.role, TranscriptRole::Assistant);
    }

    #[test]
    fn error_event() {
        let event = RealtimeErrorEvent {
            message: "connection lost".to_owned(),
        };
        let debug_str = format!("{event:?}");
        assert!(debug_str.contains("connection lost"));
    }

    // ---- Session event variants ----

    #[test]
    fn session_event_variants() {
        let events: Vec<RealtimeSessionEvent> = vec![
            RealtimeSessionEvent::AgentStart(RealtimeAgentStartEvent {
                agent_name: "a".to_owned(),
            }),
            RealtimeSessionEvent::AgentEnd(RealtimeAgentEndEvent {
                agent_name: "a".to_owned(),
            }),
            RealtimeSessionEvent::TurnEnded,
            RealtimeSessionEvent::SessionClosed,
        ];
        assert_eq!(events.len(), 4);
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn event_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TranscriptRole>();
        assert_send_sync::<RealtimeAgentStartEvent>();
        assert_send_sync::<RealtimeAgentEndEvent>();
        assert_send_sync::<RealtimeHandoffEvent>();
        assert_send_sync::<RealtimeToolStartEvent>();
        assert_send_sync::<RealtimeToolEndEvent>();
        assert_send_sync::<RealtimeAudioEvent>();
        assert_send_sync::<RealtimeAudioEndEvent>();
        assert_send_sync::<RealtimeAudioInterruptedEvent>();
        assert_send_sync::<TranscriptDeltaEvent>();
        assert_send_sync::<RealtimeErrorEvent>();
        assert_send_sync::<RealtimeGuardrailTrippedEvent>();
        assert_send_sync::<RealtimeHistoryUpdatedEvent>();
        assert_send_sync::<RealtimeHistoryAddedEvent>();
        assert_send_sync::<RealtimeSessionEvent>();
    }
}
