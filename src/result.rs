//! Result types for agent runs.
//!
//! This module defines [`RunResult`] and [`RunResultStreaming`], the two main result
//! types returned by the runner after executing an agent. `RunResult` is produced by
//! non-streaming runs, while `RunResultStreaming` is used for streaming runs where
//! events are delivered incrementally via a channel.
//!
//! These types mirror the Python SDK's `result.py` module.

use std::fmt;
use std::pin::Pin;

use serde::de::DeserializeOwned;
use tokio_stream::Stream;

use crate::error::{AgentError, Result};
use crate::guardrail::{InputGuardrailResult, OutputGuardrailResult};
use crate::items::{InputContent, ModelResponse, ResponseInputItem, RunItem};
use crate::stream_events::StreamEvent;
use crate::usage::Usage;

/// The result of a non-streaming agent run.
///
/// Contains all items generated during the run, raw model responses, the final output,
/// guardrail results, and accumulated token usage. Use [`final_output_as`](RunResult::final_output_as)
/// to deserialize the final output into a concrete type, and [`to_input_list`](RunResult::to_input_list)
/// to build a follow-up input for multi-turn conversations.
#[derive(Debug)]
#[non_exhaustive]
pub struct RunResult {
    /// The original input provided to the run.
    pub input: InputContent,
    /// All items generated during the run (messages, tool calls, handoffs, etc.).
    pub new_items: Vec<RunItem>,
    /// Raw model responses from each LLM call.
    pub raw_responses: Vec<ModelResponse>,
    /// The final output value from the agent (JSON).
    pub final_output: serde_json::Value,
    /// Name of the agent that produced the final output.
    pub last_agent_name: String,
    /// Results from input guardrail checks.
    pub input_guardrail_results: Vec<InputGuardrailResult>,
    /// Results from output guardrail checks.
    pub output_guardrail_results: Vec<OutputGuardrailResult>,
    /// Accumulated token usage across all model calls.
    pub usage: Usage,
}

impl RunResult {
    /// Deserialize the final output into a specific type.
    ///
    /// This is useful when the agent's output schema is known at compile time.
    /// The method clones the JSON value and attempts deserialization via `serde_json`.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Serialization` if the output cannot be deserialized
    /// into `T`.
    pub fn final_output_as<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_value(self.final_output.clone()).map_err(AgentError::from)
    }

    /// Convert this result's items into an input list for a follow-up run.
    ///
    /// This is useful for multi-turn conversations where the output of one run
    /// becomes the input to the next. The returned list contains the original input
    /// items followed by all output items from the model responses.
    #[must_use]
    pub fn to_input_list(&self) -> Vec<ResponseInputItem> {
        self.raw_responses
            .iter()
            .flat_map(|r| r.output.clone())
            .collect()
    }

    /// Get the last response ID from the model responses (if any).
    ///
    /// This can be used for referencing the last model response in subsequent
    /// API calls or for logging purposes.
    #[must_use]
    pub fn last_response_id(&self) -> Option<&str> {
        self.raw_responses
            .last()
            .and_then(|r| r.response_id.as_deref())
    }
}

/// The result of a streaming agent run.
///
/// Use [`stream_events`](RunResultStreaming::stream_events) to consume events
/// as they are produced by the runner. The struct fields are populated as
/// streaming progresses, so `final_output` and `is_complete` are only meaningful
/// after the stream has been fully consumed.
#[non_exhaustive]
pub struct RunResultStreaming {
    /// The original input provided to the run.
    pub input: InputContent,
    /// Items generated so far (populated as streaming progresses).
    pub new_items: Vec<RunItem>,
    /// Raw model responses collected so far.
    pub raw_responses: Vec<ModelResponse>,
    /// The final output value (populated when streaming completes).
    pub final_output: serde_json::Value,
    /// Name of the currently active agent.
    pub current_agent_name: String,
    /// The current turn number.
    pub current_turn: u32,
    /// Maximum number of turns allowed.
    pub max_turns: u32,
    /// Whether the streaming run has completed.
    pub is_complete: bool,
    /// Input guardrail results.
    pub input_guardrail_results: Vec<InputGuardrailResult>,
    /// Output guardrail results.
    pub output_guardrail_results: Vec<OutputGuardrailResult>,
    /// Accumulated token usage.
    pub usage: Usage,
    /// The channel receiver for streaming events.
    event_rx: Option<tokio::sync::mpsc::Receiver<StreamEvent>>,
    /// Cancellation token for the background streaming task.
    cancel_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl RunResultStreaming {
    /// Create a new streaming result (used internally by the runner).
    #[allow(dead_code)]
    pub(crate) fn new(
        input: InputContent,
        agent_name: String,
        max_turns: u32,
        event_rx: tokio::sync::mpsc::Receiver<StreamEvent>,
        cancel_tx: tokio::sync::oneshot::Sender<()>,
    ) -> Self {
        Self {
            input,
            new_items: Vec::new(),
            raw_responses: Vec::new(),
            final_output: serde_json::Value::Null,
            current_agent_name: agent_name,
            current_turn: 0,
            max_turns,
            is_complete: false,
            input_guardrail_results: Vec::new(),
            output_guardrail_results: Vec::new(),
            usage: Usage::default(),
            event_rx: Some(event_rx),
            cancel_tx: Some(cancel_tx),
        }
    }

    /// Stream events as they are produced by the runner.
    ///
    /// This returns a `Stream` that yields [`crate::stream_events::StreamEvent`] values until the
    /// run completes or is cancelled. The receiver is taken from the struct
    /// on the first call; subsequent calls will return an empty stream.
    pub fn stream_events(&mut self) -> Pin<Box<dyn Stream<Item = StreamEvent> + Send + '_>> {
        if let Some(rx) = self.event_rx.take() {
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
        } else {
            Box::pin(tokio_stream::empty())
        }
    }

    /// Cancel the streaming run.
    ///
    /// Sends a cancellation signal to the background task. If the cancellation
    /// token has already been consumed (e.g., the run completed or was already
    /// cancelled), this is a no-op.
    pub fn cancel(&mut self) {
        if let Some(tx) = self.cancel_tx.take() {
            let _ = tx.send(());
        }
    }

    /// Deserialize the final output into a specific type.
    ///
    /// Should only be called after the stream has been fully consumed
    /// (i.e., after `is_complete` is `true`).
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Serialization` if the output cannot be deserialized
    /// into `T`.
    pub fn final_output_as<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_value(self.final_output.clone()).map_err(AgentError::from)
    }
}

impl fmt::Debug for RunResultStreaming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunResultStreaming")
            .field("input", &self.input)
            .field("new_items", &self.new_items)
            .field("raw_responses", &self.raw_responses)
            .field("final_output", &self.final_output)
            .field("current_agent_name", &self.current_agent_name)
            .field("current_turn", &self.current_turn)
            .field("max_turns", &self.max_turns)
            .field("is_complete", &self.is_complete)
            .field("input_guardrail_results", &self.input_guardrail_results)
            .field("output_guardrail_results", &self.output_guardrail_results)
            .field("usage", &self.usage)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;
    use tokio_stream::StreamExt;

    use crate::guardrail::GuardrailFunctionOutput;
    use crate::items::MessageOutputItem;

    // ---- Helper to build a ModelResponse ----

    fn make_model_response(
        output_items: Vec<serde_json::Value>,
        response_id: Option<&str>,
    ) -> ModelResponse {
        ModelResponse {
            output: output_items,
            usage: Usage::default(),
            response_id: response_id.map(String::from),
            request_id: None,
        }
    }

    // ---- RunResult construction and field access ----

    #[test]
    fn run_result_field_access() {
        let result = RunResult {
            input: InputContent::Text("hello".to_owned()),
            new_items: vec![RunItem::MessageOutput(MessageOutputItem {
                agent_name: "agent".to_owned(),
                raw_item: json!({"type": "message", "content": []}),
            })],
            raw_responses: vec![make_model_response(
                vec![json!({"type": "message"})],
                Some("resp_1"),
            )],
            final_output: json!({"answer": 42}),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        assert_eq!(result.last_agent_name, "agent");
        assert_eq!(result.final_output, json!({"answer": 42}));
        assert_eq!(result.new_items.len(), 1);
        assert_eq!(result.raw_responses.len(), 1);
    }

    // ---- final_output_as with correct type ----

    #[test]
    fn final_output_as_correct_type() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Output {
            answer: i32,
        }

        let result = RunResult {
            input: InputContent::Text("q".to_owned()),
            new_items: vec![],
            raw_responses: vec![],
            final_output: json!({"answer": 42}),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        let output: Output = result.final_output_as().expect("should deserialize");
        assert_eq!(output, Output { answer: 42 });
    }

    // ---- final_output_as with wrong type returns error ----

    #[test]
    fn final_output_as_wrong_type() {
        #[derive(Debug, Deserialize)]
        #[allow(dead_code)]
        struct WrongType {
            name: String,
            required_field: Vec<i32>,
        }

        let result = RunResult {
            input: InputContent::Text("q".to_owned()),
            new_items: vec![],
            raw_responses: vec![],
            final_output: json!(42),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        let err = result.final_output_as::<WrongType>();
        assert!(err.is_err());
        assert!(matches!(err.unwrap_err(), AgentError::Serialization(_)));
    }

    // ---- to_input_list collects from raw_responses ----

    #[test]
    fn to_input_list_collects_from_raw_responses() {
        let result = RunResult {
            input: InputContent::Text("hello".to_owned()),
            new_items: vec![],
            raw_responses: vec![
                make_model_response(
                    vec![
                        json!({"type": "message", "id": "m1"}),
                        json!({"type": "function_call", "id": "f1"}),
                    ],
                    Some("resp_1"),
                ),
                make_model_response(vec![json!({"type": "message", "id": "m2"})], Some("resp_2")),
            ],
            final_output: json!("done"),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        let input_list = result.to_input_list();
        assert_eq!(input_list.len(), 3);
        assert_eq!(input_list[0]["id"], "m1");
        assert_eq!(input_list[1]["id"], "f1");
        assert_eq!(input_list[2]["id"], "m2");
    }

    #[test]
    fn to_input_list_empty_responses() {
        let result = RunResult {
            input: InputContent::Text("hello".to_owned()),
            new_items: vec![],
            raw_responses: vec![],
            final_output: json!(null),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        assert!(result.to_input_list().is_empty());
    }

    // ---- last_response_id with and without responses ----

    #[test]
    fn last_response_id_with_responses() {
        let result = RunResult {
            input: InputContent::Text("q".to_owned()),
            new_items: vec![],
            raw_responses: vec![
                make_model_response(vec![], Some("resp_1")),
                make_model_response(vec![], Some("resp_2")),
            ],
            final_output: json!(null),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        assert_eq!(result.last_response_id(), Some("resp_2"));
    }

    #[test]
    fn last_response_id_without_responses() {
        let result = RunResult {
            input: InputContent::Text("q".to_owned()),
            new_items: vec![],
            raw_responses: vec![],
            final_output: json!(null),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        assert_eq!(result.last_response_id(), None);
    }

    #[test]
    fn last_response_id_none_in_response() {
        let result = RunResult {
            input: InputContent::Text("q".to_owned()),
            new_items: vec![],
            raw_responses: vec![make_model_response(vec![], None)],
            final_output: json!(null),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        assert_eq!(result.last_response_id(), None);
    }

    // ---- RunResultStreaming construction ----

    #[test]
    fn streaming_result_construction() {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let (cancel_tx, _cancel_rx) = tokio::sync::oneshot::channel();

        let result = RunResultStreaming::new(
            InputContent::Text("hello".to_owned()),
            "my_agent".to_owned(),
            10,
            rx,
            cancel_tx,
        );

        assert_eq!(result.current_agent_name, "my_agent");
        assert_eq!(result.max_turns, 10);
        assert_eq!(result.current_turn, 0);
        assert!(!result.is_complete);
        assert_eq!(result.final_output, json!(null));
        assert!(result.new_items.is_empty());
        assert!(result.raw_responses.is_empty());

        // Drop the sender so we don't leak.
        drop(tx);
    }

    // ---- Streaming event delivery via channel ----

    #[tokio::test]
    async fn streaming_event_delivery() {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let (cancel_tx, _cancel_rx) = tokio::sync::oneshot::channel();

        let mut result = RunResultStreaming::new(
            InputContent::Text("hello".to_owned()),
            "agent".to_owned(),
            10,
            rx,
            cancel_tx,
        );

        // Send some events.
        tx.send(StreamEvent::AgentUpdated {
            new_agent_name: "new_agent".to_owned(),
        })
        .await
        .expect("send should succeed");

        tx.send(StreamEvent::RawResponse(json!({"type": "delta"})))
            .await
            .expect("send should succeed");

        // Drop sender to signal end of stream.
        drop(tx);

        let mut stream = result.stream_events();
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event);
        }

        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[0], StreamEvent::AgentUpdated { new_agent_name } if new_agent_name == "new_agent")
        );
        assert!(matches!(&events[1], StreamEvent::RawResponse(_)));
    }

    // ---- Second call to stream_events returns empty stream ----

    #[tokio::test]
    async fn streaming_events_second_call_empty() {
        let (_tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
        let (cancel_tx, _cancel_rx) = tokio::sync::oneshot::channel();

        let mut result = RunResultStreaming::new(
            InputContent::Text("hello".to_owned()),
            "agent".to_owned(),
            10,
            rx,
            cancel_tx,
        );

        // Take the stream once, then drop it.
        {
            let _stream1 = result.stream_events();
        }

        // Second call should return an empty stream.
        let mut stream2 = result.stream_events();
        let next = stream2.next().await;
        assert!(next.is_none(), "second stream should be empty");
    }

    // ---- Cancel stops the stream ----

    #[tokio::test]
    async fn cancel_stops_stream() {
        let (_tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();

        let mut result = RunResultStreaming::new(
            InputContent::Text("hello".to_owned()),
            "agent".to_owned(),
            10,
            rx,
            cancel_tx,
        );

        // Cancel should send the signal.
        result.cancel();

        // The cancel receiver should have received the signal.
        let received = cancel_rx.await;
        assert!(received.is_ok(), "cancel signal should be received");

        // Calling cancel again is a no-op (no panic).
        result.cancel();
    }

    // ---- RunResultStreaming final_output_as ----

    #[test]
    fn streaming_final_output_as() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Output {
            value: String,
        }

        let (_tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(1);
        let (cancel_tx, _cancel_rx) = tokio::sync::oneshot::channel();

        let mut result = RunResultStreaming::new(
            InputContent::Text("q".to_owned()),
            "agent".to_owned(),
            10,
            rx,
            cancel_tx,
        );

        // Simulate the runner setting the final output.
        result.final_output = json!({"value": "hello"});

        let output: Output = result.final_output_as().expect("should deserialize");
        assert_eq!(
            output,
            Output {
                value: "hello".to_owned()
            }
        );
    }

    // ---- Debug impls ----

    #[test]
    fn run_result_debug() {
        let result = RunResult {
            input: InputContent::Text("hi".to_owned()),
            new_items: vec![],
            raw_responses: vec![],
            final_output: json!("done"),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("RunResult"));
        assert!(debug_str.contains("agent"));
        assert!(debug_str.contains("done"));
    }

    #[test]
    fn run_result_streaming_debug() {
        let (_tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(1);
        let (cancel_tx, _cancel_rx) = tokio::sync::oneshot::channel();

        let result = RunResultStreaming::new(
            InputContent::Text("hi".to_owned()),
            "streaming_agent".to_owned(),
            5,
            rx,
            cancel_tx,
        );

        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("RunResultStreaming"));
        assert!(debug_str.contains("streaming_agent"));
        assert!(debug_str.contains("max_turns"));
        // Should NOT contain channel fields.
        assert!(!debug_str.contains("event_rx"));
        assert!(!debug_str.contains("cancel_tx"));
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn run_result_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<RunResult>();
    }

    #[test]
    fn run_result_streaming_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<RunResultStreaming>();
    }

    // ---- Guardrail results stored correctly ----

    #[test]
    fn run_result_with_guardrail_results() {
        let result = RunResult {
            input: InputContent::Text("test".to_owned()),
            new_items: vec![],
            raw_responses: vec![],
            final_output: json!(null),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![InputGuardrailResult {
                guardrail_name: "input_check".to_owned(),
                output: GuardrailFunctionOutput::passed(json!("ok")),
            }],
            output_guardrail_results: vec![OutputGuardrailResult {
                guardrail_name: "output_check".to_owned(),
                agent_name: "agent".to_owned(),
                agent_output: json!("response"),
                output: GuardrailFunctionOutput::passed(json!("ok")),
            }],
            usage: Usage::default(),
        };

        assert_eq!(result.input_guardrail_results.len(), 1);
        assert_eq!(
            result.input_guardrail_results[0].guardrail_name,
            "input_check"
        );
        assert_eq!(result.output_guardrail_results.len(), 1);
        assert_eq!(
            result.output_guardrail_results[0].guardrail_name,
            "output_check"
        );
    }
}
