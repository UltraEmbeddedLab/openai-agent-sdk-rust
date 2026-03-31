//! Serializable run state for pause/resume workflows.
//!
//! This module provides [`RunState`], a serializable snapshot of an agent run that
//! can be persisted and later restored to resume execution. This is the primary
//! building block for human-in-the-loop flows where a run must be interrupted
//! (e.g., for tool call approval) and then continued.
//!
//! The types here mirror the Python SDK's `run_state.py` module, using a simpler
//! design suited to the Rust type system and `serde` serialization.

use serde::{Deserialize, Serialize};

use crate::error::{AgentError, Result};
use crate::guardrail::{InputGuardrailResult, OutputGuardrailResult};
use crate::items::{InputContent, ItemHelpers, ModelResponse, ResponseInputItem, RunItem};
use crate::usage::Usage;

/// Schema version for serialization compatibility.
///
/// This version is emitted by [`RunState::to_json`] and checked by
/// [`RunState::from_json`] to ensure forward compatibility.
pub const CURRENT_SCHEMA_VERSION: &str = "1.0";

/// A serializable snapshot of a run's state, used for pause/resume workflows.
///
/// When a run is interrupted (e.g., for tool approval), the state can be
/// serialized with [`to_json`](RunState::to_json) and stored persistently,
/// then deserialized later with [`from_json`](RunState::from_json) to resume
/// execution from where it left off.
///
/// # Example
///
/// ```
/// use openai_agents::run_state::{RunState, NextStep};
/// use openai_agents::items::InputContent;
///
/// let state = RunState::new("my_agent", InputContent::Text("Hello".to_owned()));
/// let json = state.to_json().expect("serialization should succeed");
/// let restored = RunState::from_json(&json).expect("deserialization should succeed");
/// assert_eq!(restored.agent_name, "my_agent");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RunState {
    /// Schema version for forward compatibility.
    pub schema_version: String,
    /// Name of the currently active agent.
    pub agent_name: String,
    /// The original input to the run.
    pub input: InputContent,
    /// Items generated so far.
    pub new_items: Vec<RunItem>,
    /// Raw model responses collected so far.
    pub raw_responses: Vec<ModelResponse>,
    /// Accumulated token usage.
    pub usage: Usage,
    /// The current turn number.
    pub turn: u32,
    /// The next step to execute when resuming.
    pub next_step: NextStep,
    /// Optional conversation ID for API continuity.
    pub conversation_id: Option<String>,
    /// Optional previous response ID for streaming continuity.
    pub previous_response_id: Option<String>,
    /// Input guardrail results collected so far.
    pub input_guardrail_results: Vec<InputGuardrailResult>,
    /// Output guardrail results collected so far.
    pub output_guardrail_results: Vec<OutputGuardrailResult>,
}

impl RunState {
    /// Create a new `RunState` for the start of a run.
    ///
    /// Initialises all collections to empty, sets the turn to zero, and marks
    /// the next step as [`NextStep::ContinueLoop`].
    #[must_use]
    pub fn new(agent_name: impl Into<String>, input: InputContent) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION.to_owned(),
            agent_name: agent_name.into(),
            input,
            new_items: Vec::new(),
            raw_responses: Vec::new(),
            usage: Usage::default(),
            turn: 0,
            next_step: NextStep::ContinueLoop,
            conversation_id: None,
            previous_response_id: None,
            input_guardrail_results: Vec::new(),
            output_guardrail_results: Vec::new(),
        }
    }

    /// Serialize this state to a pretty-printed JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Serialization`] if serialization fails.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(AgentError::from)
    }

    /// Deserialize a `RunState` from a JSON string.
    ///
    /// Validates that the schema version in the JSON matches
    /// [`CURRENT_SCHEMA_VERSION`]. If it does not, a [`AgentError::UserError`]
    /// is returned.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Serialization`] if the JSON is malformed, or
    /// [`AgentError::UserError`] if the schema version is unsupported.
    pub fn from_json(json: &str) -> Result<Self> {
        let state: Self = serde_json::from_str(json).map_err(AgentError::from)?;
        if state.schema_version != CURRENT_SCHEMA_VERSION {
            return Err(AgentError::UserError {
                message: format!(
                    "Unsupported schema version '{}', expected '{CURRENT_SCHEMA_VERSION}'",
                    state.schema_version
                ),
            });
        }
        Ok(state)
    }

    /// Approve a pending tool call by its call ID.
    ///
    /// If the current [`next_step`](RunState::next_step) is
    /// [`NextStep::ExecuteTools`], the tool call matching `call_id` is marked
    /// as approved. Otherwise, or if no matching call is found, an error is
    /// returned.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the next step is not
    /// `ExecuteTools` or if the call ID is not found.
    pub fn approve_tool(&mut self, call_id: &str) -> Result<()> {
        let NextStep::ExecuteTools(ref mut calls) = self.next_step else {
            return Err(AgentError::UserError {
                message: "Cannot approve tool: next step is not ExecuteTools".to_owned(),
            });
        };

        let call = calls.iter_mut().find(|c| c.call_id == call_id);
        match call {
            Some(c) => {
                c.approved = true;
                c.rejection_message = None;
                Ok(())
            }
            None => Err(AgentError::UserError {
                message: format!("Tool call with ID '{call_id}' not found"),
            }),
        }
    }

    /// Reject a pending tool call by its call ID.
    ///
    /// Marks the tool call as not approved and records the given rejection
    /// message.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the next step is not
    /// `ExecuteTools` or if the call ID is not found.
    pub fn reject_tool(&mut self, call_id: &str, message: impl Into<String>) -> Result<()> {
        let NextStep::ExecuteTools(ref mut calls) = self.next_step else {
            return Err(AgentError::UserError {
                message: "Cannot reject tool: next step is not ExecuteTools".to_owned(),
            });
        };

        let call = calls.iter_mut().find(|c| c.call_id == call_id);
        match call {
            Some(c) => {
                c.approved = false;
                c.rejection_message = Some(message.into());
                Ok(())
            }
            None => Err(AgentError::UserError {
                message: format!("Tool call with ID '{call_id}' not found"),
            }),
        }
    }

    /// Convert this state's input and generated items into an input list for the next model call.
    ///
    /// The returned list starts with the original input (converted via
    /// [`ItemHelpers::input_to_new_input_list`]) and appends all output items
    /// from the raw model responses collected so far.
    #[must_use]
    pub fn to_input_list(&self) -> Vec<ResponseInputItem> {
        let mut items = ItemHelpers::input_to_new_input_list(&self.input);
        for response in &self.raw_responses {
            items.extend(response.to_input_items());
        }
        items
    }
}

/// What the runner should do next when resuming from a saved state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NextStep {
    /// Continue with the next turn of the run loop.
    ContinueLoop,
    /// Execute pending tool calls that may require approval.
    ExecuteTools(Vec<PendingToolCall>),
    /// The run is complete with the given final output.
    FinalOutput(serde_json::Value),
}

/// A tool call awaiting approval or execution.
///
/// When a run is paused for human-in-the-loop approval, each pending tool call
/// is stored in this struct. The caller can approve or reject individual calls
/// before resuming the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PendingToolCall {
    /// The tool call ID from the model response.
    pub call_id: String,
    /// The name of the tool being called.
    pub tool_name: String,
    /// The raw JSON arguments string for the tool call.
    pub arguments: String,
    /// Whether this call has been approved for execution.
    pub approved: bool,
    /// Optional message explaining why the call was rejected.
    pub rejection_message: Option<String>,
}

impl PendingToolCall {
    /// Create a new pending tool call with the given identifiers.
    ///
    /// The call starts unapproved with no rejection message.
    #[must_use]
    pub fn new(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            arguments: arguments.into(),
            approved: false,
            rejection_message: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- RunState creation ----

    #[test]
    fn run_state_new_has_correct_defaults() {
        let state = RunState::new("agent_1", InputContent::Text("hello".to_owned()));

        assert_eq!(state.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(state.agent_name, "agent_1");
        assert_eq!(state.input, InputContent::Text("hello".to_owned()));
        assert!(state.new_items.is_empty());
        assert!(state.raw_responses.is_empty());
        assert_eq!(state.usage, Usage::default());
        assert_eq!(state.turn, 0);
        assert!(matches!(state.next_step, NextStep::ContinueLoop));
        assert!(state.conversation_id.is_none());
        assert!(state.previous_response_id.is_none());
        assert!(state.input_guardrail_results.is_empty());
        assert!(state.output_guardrail_results.is_empty());
    }

    #[test]
    fn run_state_new_with_items_input() {
        let items = vec![json!({"role": "user", "content": "hi"})];
        let state = RunState::new("agent_2", InputContent::Items(items.clone()));

        assert_eq!(state.agent_name, "agent_2");
        assert_eq!(state.input, InputContent::Items(items));
    }

    // ---- JSON round-trip ----

    #[test]
    fn run_state_to_json_from_json_round_trip() {
        let state = RunState::new("test_agent", InputContent::Text("hello world".to_owned()));
        let json_str = state.to_json().expect("serialization should succeed");
        let restored = RunState::from_json(&json_str).expect("deserialization should succeed");

        assert_eq!(restored.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(restored.agent_name, "test_agent");
        assert_eq!(restored.input, InputContent::Text("hello world".to_owned()));
        assert_eq!(restored.turn, 0);
        assert!(matches!(restored.next_step, NextStep::ContinueLoop));
    }

    #[test]
    fn run_state_round_trip_with_items() {
        use crate::items::MessageOutputItem;

        let mut state = RunState::new("agent", InputContent::Text("hi".to_owned()));
        state.new_items.push(RunItem::MessageOutput(MessageOutputItem {
            agent_name: "agent".to_owned(),
            raw_item: json!({"type": "message", "content": [{"type": "output_text", "text": "Hello!"}]}),
        }));
        state.turn = 3;
        state.conversation_id = Some("conv_123".to_owned());

        let json_str = state.to_json().expect("serialize");
        let restored = RunState::from_json(&json_str).expect("deserialize");

        assert_eq!(restored.new_items.len(), 1);
        assert_eq!(restored.turn, 3);
        assert_eq!(restored.conversation_id, Some("conv_123".to_owned()));
    }

    // ---- Version check ----

    #[test]
    fn from_json_rejects_wrong_schema_version() {
        let mut state = RunState::new("agent", InputContent::Text("hi".to_owned()));
        state.schema_version = "99.0".to_owned();
        let json_str = serde_json::to_string_pretty(&state).expect("serialize");

        let err = RunState::from_json(&json_str).expect_err("should reject wrong version");
        assert!(matches!(err, AgentError::UserError { .. }));
        let msg = err.to_string();
        assert!(msg.contains("99.0"), "error should mention the bad version");
        assert!(
            msg.contains(CURRENT_SCHEMA_VERSION),
            "error should mention the expected version"
        );
    }

    #[test]
    fn from_json_rejects_malformed_json() {
        let err = RunState::from_json("not valid json").expect_err("should fail");
        assert!(matches!(err, AgentError::Serialization(_)));
    }

    // ---- approve_tool ----

    #[test]
    fn approve_tool_succeeds() {
        let mut state = RunState::new("agent", InputContent::Text("hi".to_owned()));
        state.next_step = NextStep::ExecuteTools(vec![
            PendingToolCall::new("call_1", "get_weather", r#"{"city":"NYC"}"#),
            PendingToolCall::new("call_2", "get_time", r#"{"tz":"UTC"}"#),
        ]);

        state.approve_tool("call_1").expect("should approve");

        if let NextStep::ExecuteTools(ref calls) = state.next_step {
            assert!(calls[0].approved);
            assert!(calls[0].rejection_message.is_none());
            assert!(!calls[1].approved);
        } else {
            panic!("expected ExecuteTools");
        }
    }

    #[test]
    fn approve_tool_fails_for_unknown_id() {
        let mut state = RunState::new("agent", InputContent::Text("hi".to_owned()));
        state.next_step =
            NextStep::ExecuteTools(vec![PendingToolCall::new("call_1", "tool", "{}")]);

        let err = state.approve_tool("nonexistent").expect_err("should fail");
        assert!(matches!(err, AgentError::UserError { .. }));
        assert!(err.to_string().contains("nonexistent"));
    }

    #[test]
    fn approve_tool_fails_when_not_execute_tools() {
        let mut state = RunState::new("agent", InputContent::Text("hi".to_owned()));
        // next_step is ContinueLoop by default.
        let err = state.approve_tool("call_1").expect_err("should fail");
        assert!(matches!(err, AgentError::UserError { .. }));
        assert!(err.to_string().contains("not ExecuteTools"));
    }

    // ---- reject_tool ----

    #[test]
    fn reject_tool_succeeds() {
        let mut state = RunState::new("agent", InputContent::Text("hi".to_owned()));
        state.next_step =
            NextStep::ExecuteTools(vec![PendingToolCall::new("call_1", "dangerous_tool", "{}")]);

        state
            .reject_tool("call_1", "Not safe to run")
            .expect("should reject");

        if let NextStep::ExecuteTools(ref calls) = state.next_step {
            assert!(!calls[0].approved);
            assert_eq!(
                calls[0].rejection_message.as_deref(),
                Some("Not safe to run")
            );
        } else {
            panic!("expected ExecuteTools");
        }
    }

    #[test]
    fn reject_tool_fails_for_unknown_id() {
        let mut state = RunState::new("agent", InputContent::Text("hi".to_owned()));
        state.next_step =
            NextStep::ExecuteTools(vec![PendingToolCall::new("call_1", "tool", "{}")]);

        let err = state
            .reject_tool("nonexistent", "reason")
            .expect_err("should fail");
        assert!(matches!(err, AgentError::UserError { .. }));
    }

    #[test]
    fn reject_tool_fails_when_not_execute_tools() {
        let mut state = RunState::new("agent", InputContent::Text("hi".to_owned()));
        state.next_step = NextStep::FinalOutput(json!("done"));

        let err = state
            .reject_tool("call_1", "reason")
            .expect_err("should fail");
        assert!(matches!(err, AgentError::UserError { .. }));
        assert!(err.to_string().contains("not ExecuteTools"));
    }

    // ---- PendingToolCall creation ----

    #[test]
    fn pending_tool_call_new_defaults() {
        let call = PendingToolCall::new("call_abc", "my_tool", r#"{"key": "val"}"#);

        assert_eq!(call.call_id, "call_abc");
        assert_eq!(call.tool_name, "my_tool");
        assert_eq!(call.arguments, r#"{"key": "val"}"#);
        assert!(!call.approved);
        assert!(call.rejection_message.is_none());
    }

    // ---- NextStep serialization ----

    #[test]
    fn next_step_continue_loop_serializes() {
        let step = NextStep::ContinueLoop;
        let json_str = serde_json::to_string(&step).expect("serialize");
        let restored: NextStep = serde_json::from_str(&json_str).expect("deserialize");
        assert!(matches!(restored, NextStep::ContinueLoop));
    }

    #[test]
    fn next_step_execute_tools_serializes() {
        let step = NextStep::ExecuteTools(vec![PendingToolCall::new("c1", "tool", "{}")]);
        let json_str = serde_json::to_string(&step).expect("serialize");
        let restored: NextStep = serde_json::from_str(&json_str).expect("deserialize");

        if let NextStep::ExecuteTools(calls) = restored {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].call_id, "c1");
            assert_eq!(calls[0].tool_name, "tool");
        } else {
            panic!("expected ExecuteTools variant");
        }
    }

    #[test]
    fn next_step_final_output_serializes() {
        let step = NextStep::FinalOutput(json!({"result": 42}));
        let json_str = serde_json::to_string(&step).expect("serialize");
        let restored: NextStep = serde_json::from_str(&json_str).expect("deserialize");

        if let NextStep::FinalOutput(val) = restored {
            assert_eq!(val, json!({"result": 42}));
        } else {
            panic!("expected FinalOutput variant");
        }
    }

    // ---- to_input_list ----

    #[test]
    fn to_input_list_with_text_input_and_no_responses() {
        let state = RunState::new("agent", InputContent::Text("hello".to_owned()));
        let list = state.to_input_list();

        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["role"], "user");
        assert_eq!(list[0]["content"], "hello");
    }

    #[test]
    fn to_input_list_includes_response_outputs() {
        let mut state = RunState::new("agent", InputContent::Text("hi".to_owned()));
        state.raw_responses.push(ModelResponse::new(
            vec![json!({"type": "message", "id": "m1"})],
            Usage::default(),
            None,
            None,
        ));

        let list = state.to_input_list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0]["role"], "user");
        assert_eq!(list[1]["id"], "m1");
    }

    // ---- RunState with ExecuteTools round-trip ----

    #[test]
    fn run_state_with_pending_tools_round_trip() {
        let mut state = RunState::new("agent", InputContent::Text("do stuff".to_owned()));
        state.next_step = NextStep::ExecuteTools(vec![
            PendingToolCall::new("c1", "tool_a", r#"{"x":1}"#),
            PendingToolCall::new("c2", "tool_b", r#"{"y":2}"#),
        ]);
        state.turn = 5;

        let json_str = state.to_json().expect("serialize");
        let restored = RunState::from_json(&json_str).expect("deserialize");

        assert_eq!(restored.turn, 5);
        if let NextStep::ExecuteTools(calls) = &restored.next_step {
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].tool_name, "tool_a");
            assert_eq!(calls[1].tool_name, "tool_b");
        } else {
            panic!("expected ExecuteTools");
        }
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn run_state_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RunState>();
        assert_send_sync::<NextStep>();
        assert_send_sync::<PendingToolCall>();
    }
}
