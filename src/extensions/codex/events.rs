//! Codex thread event types.
//!
//! This module defines the event types emitted during Codex thread and turn
//! execution. Events are streamed from the Codex CLI JSONL output and parsed
//! into strongly typed Rust enums.
//!
//! Mirrors the Python SDK's `events.py` in the Codex extension.

use serde::{Deserialize, Serialize};

use super::items::{ThreadItem, coerce_thread_item};

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// Token usage information for a Codex turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CodexUsage {
    /// Number of input tokens consumed.
    pub input_tokens: u64,
    /// Number of cached input tokens.
    pub cached_input_tokens: u64,
    /// Number of output tokens generated.
    pub output_tokens: u64,
}

impl CodexUsage {
    /// Create a new usage record.
    #[must_use]
    pub const fn new(input_tokens: u64, cached_input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            cached_input_tokens,
            output_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// ThreadError
// ---------------------------------------------------------------------------

/// An error reported during Codex thread execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ThreadError {
    /// The error message.
    pub message: String,
}

// ---------------------------------------------------------------------------
// Event structs
// ---------------------------------------------------------------------------

/// Emitted when a Codex thread starts.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ThreadStartedEvent {
    /// The unique identifier for the thread.
    pub thread_id: String,
}

/// Emitted when a new turn starts within a thread.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TurnStartedEvent;

/// Emitted when a turn completes successfully.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TurnCompletedEvent {
    /// Token usage for this turn, if available.
    pub usage: Option<CodexUsage>,
}

/// Emitted when a turn fails.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TurnFailedEvent {
    /// The error that caused the turn to fail.
    pub error: ThreadError,
}

/// Emitted when an item starts being processed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ItemStartedEvent {
    /// The item that started.
    pub item: ThreadItem,
}

/// Emitted when an item is updated during processing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ItemUpdatedEvent {
    /// The updated item.
    pub item: ThreadItem,
}

/// Emitted when an item finishes processing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ItemCompletedEvent {
    /// The completed item.
    pub item: ThreadItem,
}

/// A stream-level error event.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ThreadErrorEvent {
    /// The error message.
    pub message: String,
}

// ---------------------------------------------------------------------------
// ThreadEvent enum
// ---------------------------------------------------------------------------

/// Events emitted during Codex thread and turn execution.
///
/// This enum represents the full set of events that the Codex CLI can emit
/// via its JSONL stream. An `Unknown` variant provides forward compatibility
/// with future event types.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ThreadEvent {
    /// A thread has started.
    ThreadStarted(ThreadStartedEvent),
    /// A turn has started.
    TurnStarted(TurnStartedEvent),
    /// A turn completed successfully.
    TurnCompleted(TurnCompletedEvent),
    /// A turn failed.
    TurnFailed(TurnFailedEvent),
    /// An item started processing.
    ItemStarted(ItemStartedEvent),
    /// An item was updated.
    ItemUpdated(ItemUpdatedEvent),
    /// An item completed processing.
    ItemCompleted(ItemCompletedEvent),
    /// A stream-level error occurred.
    Error(ThreadErrorEvent),
    /// An unknown event type for forward compatibility.
    Unknown {
        /// The raw event type string.
        event_type: String,
        /// The raw event payload.
        payload: serde_json::Value,
    },
}

// ---------------------------------------------------------------------------
// coerce_thread_event
// ---------------------------------------------------------------------------

/// Deserialize a [`ThreadEvent`] from a raw JSON mapping.
///
/// This function handles the same coercion logic as the Python SDK's
/// `coerce_thread_event`, constructing the appropriate variant based on
/// the `"type"` field.
///
/// # Errors
///
/// Returns a `serde_json::Error` if the mapping cannot be deserialized into
/// a valid event.
pub fn coerce_thread_event(raw: &serde_json::Value) -> Result<ThreadEvent, serde_json::Error> {
    let event_type = raw
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");

    match event_type {
        "thread.started" => {
            let thread_id = raw
                .get("thread_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            Ok(ThreadEvent::ThreadStarted(ThreadStartedEvent { thread_id }))
        }
        "turn.started" => Ok(ThreadEvent::TurnStarted(TurnStartedEvent)),
        "turn.completed" => {
            let usage = raw.get("usage").and_then(|u| {
                Some(CodexUsage {
                    input_tokens: u.get("input_tokens")?.as_u64()?,
                    cached_input_tokens: u.get("cached_input_tokens")?.as_u64()?,
                    output_tokens: u.get("output_tokens")?.as_u64()?,
                })
            });
            Ok(ThreadEvent::TurnCompleted(TurnCompletedEvent { usage }))
        }
        "turn.failed" => {
            let error_raw = raw.get("error").cloned().unwrap_or_default();
            let message = error_raw
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            Ok(ThreadEvent::TurnFailed(TurnFailedEvent {
                error: ThreadError { message },
            }))
        }
        "item.started" => {
            let item = raw
                .get("item")
                .map(coerce_thread_item)
                .transpose()?
                .unwrap_or(ThreadItem::Unknown);
            Ok(ThreadEvent::ItemStarted(ItemStartedEvent { item }))
        }
        "item.updated" => {
            let item = raw
                .get("item")
                .map(coerce_thread_item)
                .transpose()?
                .unwrap_or(ThreadItem::Unknown);
            Ok(ThreadEvent::ItemUpdated(ItemUpdatedEvent { item }))
        }
        "item.completed" => {
            let item = raw
                .get("item")
                .map(coerce_thread_item)
                .transpose()?
                .unwrap_or(ThreadItem::Unknown);
            Ok(ThreadEvent::ItemCompleted(ItemCompletedEvent { item }))
        }
        "error" => {
            let message = raw
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            Ok(ThreadEvent::Error(ThreadErrorEvent { message }))
        }
        _ => Ok(ThreadEvent::Unknown {
            event_type: event_type.to_owned(),
            payload: raw.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- CodexUsage ----

    #[test]
    fn codex_usage_new() {
        let usage = CodexUsage::new(100, 50, 200);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.cached_input_tokens, 50);
        assert_eq!(usage.output_tokens, 200);
    }

    #[test]
    fn codex_usage_serde_round_trip() {
        let usage = CodexUsage::new(100, 50, 200);
        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: CodexUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage, deserialized);
    }

    // ---- coerce_thread_event ----

    #[test]
    fn coerce_thread_started() {
        let raw = json!({"type": "thread.started", "thread_id": "t-123"});
        let event = coerce_thread_event(&raw).unwrap();
        if let ThreadEvent::ThreadStarted(e) = event {
            assert_eq!(e.thread_id, "t-123");
        } else {
            panic!("expected ThreadStarted");
        }
    }

    #[test]
    fn coerce_turn_started() {
        let raw = json!({"type": "turn.started"});
        let event = coerce_thread_event(&raw).unwrap();
        assert!(matches!(event, ThreadEvent::TurnStarted(_)));
    }

    #[test]
    fn coerce_turn_completed_with_usage() {
        let raw = json!({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 50,
                "output_tokens": 200
            }
        });
        let event = coerce_thread_event(&raw).unwrap();
        if let ThreadEvent::TurnCompleted(e) = event {
            let usage = e.usage.unwrap();
            assert_eq!(usage.input_tokens, 100);
            assert_eq!(usage.cached_input_tokens, 50);
            assert_eq!(usage.output_tokens, 200);
        } else {
            panic!("expected TurnCompleted");
        }
    }

    #[test]
    fn coerce_turn_completed_without_usage() {
        let raw = json!({"type": "turn.completed"});
        let event = coerce_thread_event(&raw).unwrap();
        if let ThreadEvent::TurnCompleted(e) = event {
            assert!(e.usage.is_none());
        } else {
            panic!("expected TurnCompleted");
        }
    }

    #[test]
    fn coerce_turn_failed() {
        let raw = json!({
            "type": "turn.failed",
            "error": {"message": "something went wrong"}
        });
        let event = coerce_thread_event(&raw).unwrap();
        if let ThreadEvent::TurnFailed(e) = event {
            assert_eq!(e.error.message, "something went wrong");
        } else {
            panic!("expected TurnFailed");
        }
    }

    #[test]
    fn coerce_item_started() {
        let raw = json!({
            "type": "item.started",
            "item": {
                "type": "agent_message",
                "id": "msg-1",
                "text": "hello"
            }
        });
        let event = coerce_thread_event(&raw).unwrap();
        if let ThreadEvent::ItemStarted(e) = event {
            assert!(e.item.is_agent_message());
        } else {
            panic!("expected ItemStarted");
        }
    }

    #[test]
    fn coerce_item_updated() {
        let raw = json!({
            "type": "item.updated",
            "item": {
                "type": "agent_message",
                "id": "msg-1",
                "text": "hello world"
            }
        });
        let event = coerce_thread_event(&raw).unwrap();
        assert!(matches!(event, ThreadEvent::ItemUpdated(_)));
    }

    #[test]
    fn coerce_item_completed() {
        let raw = json!({
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "id": "cmd-1",
                "command": "echo hello",
                "status": "completed",
                "aggregated_output": "hello\n",
                "exit_code": 0
            }
        });
        let event = coerce_thread_event(&raw).unwrap();
        if let ThreadEvent::ItemCompleted(e) = event {
            assert!(matches!(e.item, ThreadItem::CommandExecution(_)));
        } else {
            panic!("expected ItemCompleted");
        }
    }

    #[test]
    fn coerce_error_event() {
        let raw = json!({"type": "error", "message": "stream error"});
        let event = coerce_thread_event(&raw).unwrap();
        if let ThreadEvent::Error(e) = event {
            assert_eq!(e.message, "stream error");
        } else {
            panic!("expected Error");
        }
    }

    #[test]
    fn coerce_unknown_event() {
        let raw = json!({"type": "future.event", "data": 42});
        let event = coerce_thread_event(&raw).unwrap();
        if let ThreadEvent::Unknown {
            event_type,
            payload,
        } = event
        {
            assert_eq!(event_type, "future.event");
            assert_eq!(payload["data"], 42);
        } else {
            panic!("expected Unknown");
        }
    }

    #[test]
    fn coerce_item_started_without_item() {
        let raw = json!({"type": "item.started"});
        let event = coerce_thread_event(&raw).unwrap();
        if let ThreadEvent::ItemStarted(e) = event {
            assert!(matches!(e.item, ThreadItem::Unknown));
        } else {
            panic!("expected ItemStarted");
        }
    }

    // ---- Send + Sync ----

    #[test]
    fn events_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ThreadEvent>();
        assert_send_sync::<CodexUsage>();
        assert_send_sync::<ThreadError>();
        assert_send_sync::<ThreadStartedEvent>();
        assert_send_sync::<TurnStartedEvent>();
        assert_send_sync::<TurnCompletedEvent>();
        assert_send_sync::<TurnFailedEvent>();
        assert_send_sync::<ItemStartedEvent>();
        assert_send_sync::<ItemUpdatedEvent>();
        assert_send_sync::<ItemCompletedEvent>();
        assert_send_sync::<ThreadErrorEvent>();
    }
}
