//! Streaming events emitted during an agent run.
//!
//! This module defines the [`StreamEvent`] enum, which represents all possible events
//! that can be emitted while streaming an agent execution. Events include raw LLM
//! responses, run item creation notifications, and agent change notifications.
//!
//! The types here mirror the Python SDK's `stream_events.py` module.

use crate::items::{ResponseStreamEvent, RunItem};

/// A streaming event emitted during an agent run.
///
/// As the runner processes the agent loop, it yields these events to the caller
/// so that intermediate progress can be observed in real time. The three categories
/// are raw model events, structured run-item events, and agent-change notifications.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamEvent {
    /// Raw LLM streaming event from the model.
    ///
    /// These are passed through directly from the underlying model provider
    /// without any additional processing.
    RawResponse(ResponseStreamEvent),

    /// A [`RunItem`] was created during the run.
    ///
    /// As the agent processes the LLM response it will generate these events
    /// for new messages, tool calls, tool outputs, handoffs, and reasoning steps.
    RunItemCreated {
        /// The kind of item event that occurred.
        name: RunItemEventName,
        /// The item that was created.
        item: RunItem,
    },

    /// The active agent changed (for example, due to a handoff).
    AgentUpdated {
        /// The name of the new active agent.
        new_agent_name: String,
    },
}

/// Names for [`RunItem`] stream events, identifying what kind of item was created.
///
/// Each variant corresponds to a specific stage in the agent execution loop. The
/// [`Display`](std::fmt::Display) implementation produces `snake_case` strings that
/// match the Python SDK's event name literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RunItemEventName {
    /// A message output was created by the LLM.
    MessageOutputCreated,
    /// A handoff was requested by the current agent.
    HandoffRequested,
    /// A handoff was completed to a new agent.
    HandoffOccurred,
    /// A tool was called by the LLM.
    ToolCalled,
    /// A tool produced output.
    ToolOutput,
    /// A reasoning step was created.
    ReasoningItemCreated,
}

impl std::fmt::Display for RunItemEventName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MessageOutputCreated => write!(f, "message_output_created"),
            Self::HandoffRequested => write!(f, "handoff_requested"),
            Self::HandoffOccurred => write!(f, "handoff_occurred"),
            Self::ToolCalled => write!(f, "tool_called"),
            Self::ToolOutput => write!(f, "tool_output"),
            Self::ReasoningItemCreated => write!(f, "reasoning_item_created"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::MessageOutputItem;
    use serde_json::json;
    use std::collections::HashSet;

    // ---- StreamEvent construction for each variant ----

    #[test]
    fn stream_event_raw_response() {
        let event = StreamEvent::RawResponse(json!({"type": "response.output_text.delta"}));
        assert!(matches!(event, StreamEvent::RawResponse(_)));
    }

    #[test]
    fn stream_event_run_item_created() {
        let item = RunItem::MessageOutput(MessageOutputItem {
            agent_name: "assistant".to_owned(),
            raw_item: json!({"type": "message", "content": []}),
        });
        let event = StreamEvent::RunItemCreated {
            name: RunItemEventName::MessageOutputCreated,
            item,
        };
        assert!(matches!(
            event,
            StreamEvent::RunItemCreated {
                name: RunItemEventName::MessageOutputCreated,
                ..
            }
        ));
    }

    #[test]
    fn stream_event_agent_updated() {
        let event = StreamEvent::AgentUpdated {
            new_agent_name: "specialist_agent".to_owned(),
        };
        if let StreamEvent::AgentUpdated { new_agent_name } = &event {
            assert_eq!(new_agent_name, "specialist_agent");
        } else {
            panic!("expected AgentUpdated variant");
        }
    }

    // ---- RunItemEventName Display output ----

    #[test]
    fn display_message_output_created() {
        assert_eq!(
            RunItemEventName::MessageOutputCreated.to_string(),
            "message_output_created"
        );
    }

    #[test]
    fn display_handoff_requested() {
        assert_eq!(
            RunItemEventName::HandoffRequested.to_string(),
            "handoff_requested"
        );
    }

    #[test]
    fn display_handoff_occurred() {
        assert_eq!(
            RunItemEventName::HandoffOccurred.to_string(),
            "handoff_occurred"
        );
    }

    #[test]
    fn display_tool_called() {
        assert_eq!(RunItemEventName::ToolCalled.to_string(), "tool_called");
    }

    #[test]
    fn display_tool_output() {
        assert_eq!(RunItemEventName::ToolOutput.to_string(), "tool_output");
    }

    #[test]
    fn display_reasoning_item_created() {
        assert_eq!(
            RunItemEventName::ReasoningItemCreated.to_string(),
            "reasoning_item_created"
        );
    }

    // ---- RunItemEventName equality ----

    #[test]
    fn event_name_equality() {
        assert_eq!(
            RunItemEventName::MessageOutputCreated,
            RunItemEventName::MessageOutputCreated
        );
        assert_ne!(RunItemEventName::ToolCalled, RunItemEventName::ToolOutput);
    }

    // ---- RunItemEventName hash ----

    #[test]
    fn event_name_hash_in_set() {
        let mut set = HashSet::new();
        set.insert(RunItemEventName::MessageOutputCreated);
        set.insert(RunItemEventName::HandoffRequested);
        set.insert(RunItemEventName::HandoffOccurred);
        set.insert(RunItemEventName::ToolCalled);
        set.insert(RunItemEventName::ToolOutput);
        set.insert(RunItemEventName::ReasoningItemCreated);
        assert_eq!(set.len(), 6);

        // Inserting a duplicate does not increase the size.
        set.insert(RunItemEventName::ToolCalled);
        assert_eq!(set.len(), 6);
    }

    // ---- Pattern matching coverage ----

    #[test]
    fn stream_event_pattern_matching() {
        let events = vec![
            StreamEvent::RawResponse(json!({"type": "delta"})),
            StreamEvent::RunItemCreated {
                name: RunItemEventName::ToolCalled,
                item: RunItem::ToolCall(crate::items::ToolCallItem {
                    agent_name: "a".to_owned(),
                    raw_item: json!({"type": "function_call"}),
                    tool_origin: None,
                }),
            },
            StreamEvent::AgentUpdated {
                new_agent_name: "b".to_owned(),
            },
        ];

        let mut raw_count = 0u32;
        let mut item_count = 0u32;
        let mut agent_count = 0u32;

        for event in &events {
            match event {
                StreamEvent::RawResponse(_) => raw_count += 1,
                StreamEvent::RunItemCreated { name, item: _ } => {
                    assert_eq!(*name, RunItemEventName::ToolCalled);
                    item_count += 1;
                }
                StreamEvent::AgentUpdated { new_agent_name } => {
                    assert_eq!(new_agent_name, "b");
                    agent_count += 1;
                }
            }
        }

        assert_eq!(raw_count, 1);
        assert_eq!(item_count, 1);
        assert_eq!(agent_count, 1);
    }

    #[test]
    fn run_item_event_name_is_copy() {
        let name = RunItemEventName::ToolOutput;
        let copy = name;
        // Both the original and the copy are usable, proving Copy.
        assert_eq!(name, copy);
    }

    #[test]
    fn stream_event_debug_format() {
        let event = StreamEvent::AgentUpdated {
            new_agent_name: "test".to_owned(),
        };
        let debug_str = format!("{event:?}");
        assert!(debug_str.contains("AgentUpdated"));
        assert!(debug_str.contains("test"));
    }
}
