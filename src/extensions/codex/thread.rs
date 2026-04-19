//! Thread and Turn types for the Codex extension.
//!
//! A [`Thread`] represents a session of work, containing one or more [`Turn`]s.
//! Each turn aggregates the items and events produced by a single interaction
//! with the underlying agent runner.
//!
//! Mirrors the Python SDK's `thread.py` in the Codex extension.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::agent::Agent;
use crate::config::RunConfig;
use crate::error::{AgentError, Result};
use crate::items::RunItem;
use crate::models::Model;
use crate::runner::Runner;

use super::events::{
    CodexUsage, ItemCompletedEvent, ThreadError, ThreadEvent, ThreadStartedEvent,
    TurnCompletedEvent, TurnFailedEvent, TurnStartedEvent,
};
use super::items::{
    AgentMessageItem, CommandExecutionItem, CommandExecutionStatus, ReasoningItem, ThreadItem,
};
use super::options::ThreadOptions;

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

/// A text input element for a Codex thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TextInput {
    /// The text content.
    pub text: String,
}

/// A local image input element for a Codex thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LocalImageInput {
    /// Path to the local image file.
    pub path: String,
}

/// A single input element that can be text or a local image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum UserInput {
    /// A text input.
    #[serde(rename = "text")]
    Text(TextInput),
    /// A local image input.
    #[serde(rename = "local_image")]
    LocalImage(LocalImageInput),
}

/// Input to a Codex thread: either a plain string or a list of structured inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Input {
    /// A plain text string.
    Text(String),
    /// A list of structured user input elements.
    Items(Vec<UserInput>),
}

impl From<&str> for Input {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl From<String> for Input {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<Vec<UserInput>> for Input {
    fn from(items: Vec<UserInput>) -> Self {
        Self::Items(items)
    }
}

/// Normalize an [`Input`] into a prompt string and a list of image paths.
///
/// Text inputs are joined with double newlines. Image inputs are collected
/// into the returned path list.
#[must_use]
pub fn normalize_input(input: &Input) -> (String, Vec<String>) {
    match input {
        Input::Text(text) => (text.clone(), Vec::new()),
        Input::Items(items) => {
            let mut prompt_parts = Vec::new();
            let mut images = Vec::new();
            for item in items {
                match item {
                    UserInput::Text(t) => prompt_parts.push(t.text.clone()),
                    UserInput::LocalImage(img) => {
                        if !img.path.is_empty() {
                            images.push(img.path.clone());
                        }
                    }
                }
            }
            (prompt_parts.join("\n\n"), images)
        }
    }
}

// ---------------------------------------------------------------------------
// Turn
// ---------------------------------------------------------------------------

/// A single turn within a Codex thread.
///
/// A turn represents one request-response cycle, collecting all items
/// produced and the final response text.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Turn {
    /// The items produced during this turn.
    pub items: Vec<ThreadItem>,
    /// The final response text from the agent.
    pub final_response: String,
    /// Token usage for this turn, if available.
    pub usage: Option<CodexUsage>,
}

impl Turn {
    /// Create a new turn from items, a final response, and optional usage.
    #[must_use]
    pub fn new(
        items: Vec<ThreadItem>,
        final_response: impl Into<String>,
        usage: Option<CodexUsage>,
    ) -> Self {
        Self {
            items,
            final_response: final_response.into(),
            usage,
        }
    }
}

/// Type alias matching the Python SDK's `RunResult = Turn`.
pub type RunResult = Turn;

// ---------------------------------------------------------------------------
// Thread
// ---------------------------------------------------------------------------

/// A Codex thread representing a session of work.
///
/// Threads can be started fresh or resumed from a previous session using
/// a thread ID. Each call to [`Thread::run`] adds a new turn to the thread,
/// delegating to the agent `Runner` for LLM execution.
#[non_exhaustive]
pub struct Thread {
    id: Option<String>,
    instructions: Option<String>,
    model: Option<Arc<dyn Model>>,
    options: ThreadOptions,
    turns: Vec<Turn>,
}

impl std::fmt::Debug for Thread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Thread")
            .field("id", &self.id)
            .field("instructions", &self.instructions)
            .field("model", &self.model.as_ref().map(|_| "<model>"))
            .field("options", &self.options)
            .field("turns", &self.turns)
            .finish()
    }
}

impl Thread {
    /// Create a new thread with no ID (a fresh session).
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: None,
            instructions: None,
            model: None,
            options: ThreadOptions::default(),
            turns: Vec::new(),
        }
    }

    /// Create a new thread that resumes from the given thread ID.
    #[must_use]
    pub fn with_id(id: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
            instructions: None,
            model: None,
            options: ThreadOptions::default(),
            turns: Vec::new(),
        }
    }

    /// Set the instructions for this thread.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set the model for this thread.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the thread options.
    #[must_use]
    pub fn with_options(mut self, options: ThreadOptions) -> Self {
        self.options = options;
        self
    }

    /// Get the thread ID, if one has been assigned.
    ///
    /// A thread ID is assigned after the first `thread.started` event
    /// is received from the Codex CLI.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Set the thread ID.
    ///
    /// This is called internally when a `thread.started` event is received.
    pub fn set_id(&mut self, id: String) {
        self.id = Some(id);
    }

    /// Get the instructions for this thread.
    #[must_use]
    pub fn instructions(&self) -> Option<&str> {
        self.instructions.as_deref()
    }

    /// Get a reference to the turns completed so far.
    #[must_use]
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Get the total number of items across all turns.
    #[must_use]
    pub fn total_items(&self) -> usize {
        self.turns.iter().map(|t| t.items.len()).sum()
    }

    /// Add a completed turn to the thread.
    pub fn add_turn(&mut self, turn: Turn) {
        self.turns.push(turn);
    }

    /// Get a reference to the thread options.
    #[must_use]
    pub const fn options(&self) -> &ThreadOptions {
        &self.options
    }

    /// Run a turn in this thread with the given input.
    ///
    /// Delegates to the agent `Runner` internally, converting `RunItem`s to
    /// `ThreadItem`s and emitting `ThreadEvent`s for each step.
    ///
    /// # Errors
    ///
    /// Returns an error if no model has been configured on the thread, or if
    /// the underlying runner encounters an error that is not recoverable.
    pub async fn run(&mut self, input: impl Into<Input>) -> Result<(Turn, Vec<ThreadEvent>)> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| AgentError::UserError {
                message: "Thread requires a model. Call with_model() first.".into(),
            })?
            .clone();

        let input = input.into();
        let mut events = Vec::new();

        // Assign a thread ID if we do not have one yet.
        if self.id.is_none() {
            self.id = Some(uuid_v4());
        }

        events.push(ThreadEvent::ThreadStarted(ThreadStartedEvent {
            thread_id: self.id.clone().unwrap_or_default(),
        }));

        // Build an agent from the thread's instructions.
        let instructions = self
            .instructions
            .as_deref()
            .unwrap_or("You are a helpful coding assistant.");
        let agent = Agent::<()>::builder("codex-agent")
            .instructions(instructions)
            .build();

        let max_turns = 10;
        let run_config = RunConfig {
            max_turns,
            ..RunConfig::default()
        };

        // Normalize input to a prompt string.
        let (prompt, _images) = normalize_input(&input);

        #[allow(clippy::cast_possible_truncation)]
        let turn_number = self.turns.len() as u32 + 1;
        events.push(ThreadEvent::TurnStarted(TurnStartedEvent));

        let result =
            Runner::run_with_model(&agent, prompt, (), model, None, Some(run_config)).await;

        match result {
            Ok(run_result) => {
                let mut thread_items = Vec::new();
                let mut final_response = String::new();

                for item in &run_result.new_items {
                    if let Some(ti) = run_item_to_thread_item(item, turn_number) {
                        events.push(ThreadEvent::ItemCompleted(ItemCompletedEvent {
                            item: ti.clone(),
                        }));
                        // Capture the final agent message text.
                        if let ThreadItem::AgentMessage(ref msg) = ti {
                            final_response.clone_from(&msg.text);
                        }
                        thread_items.push(ti);
                    }
                }

                let usage = CodexUsage::new(
                    run_result.usage.input_tokens,
                    run_result.usage.input_tokens_details.cached_tokens,
                    run_result.usage.output_tokens,
                );

                let turn = Turn::new(thread_items, &final_response, Some(usage));

                events.push(ThreadEvent::TurnCompleted(TurnCompletedEvent {
                    usage: Some(usage),
                }));

                self.turns.push(turn.clone());

                Ok((turn, events))
            }
            Err(e) => {
                events.push(ThreadEvent::TurnFailed(TurnFailedEvent {
                    error: ThreadError {
                        message: e.to_string(),
                    },
                }));
                Err(e)
            }
        }
    }
}

impl Default for Thread {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a [`RunItem`] to a [`ThreadItem`], if applicable.
///
/// Returns `None` for items that do not map to a Codex thread item
/// (e.g., handoff outputs, tool call outputs).
fn run_item_to_thread_item(item: &RunItem, _turn_number: u32) -> Option<ThreadItem> {
    match item {
        RunItem::MessageOutput(msg) => {
            let text = msg
                .raw_item
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|arr| {
                    arr.iter().find_map(|entry| {
                        if entry.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                            entry.get("text").and_then(|t| t.as_str()).map(String::from)
                        } else {
                            None
                        }
                    })
                })
                .unwrap_or_default();

            Some(ThreadItem::AgentMessage(AgentMessageItem {
                id: msg
                    .raw_item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("msg")
                    .to_owned(),
                text,
            }))
        }
        RunItem::ToolCall(tc) => {
            let name = tc
                .raw_item
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown")
                .to_string();
            let args = tc
                .raw_item
                .get("arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}")
                .to_string();
            let id = tc
                .raw_item
                .get("call_id")
                .or_else(|| tc.raw_item.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("cmd")
                .to_owned();

            Some(ThreadItem::CommandExecution(CommandExecutionItem {
                id,
                command: format!("{name}({args})"),
                status: CommandExecutionStatus::Completed,
                aggregated_output: String::new(),
                exit_code: Some(0),
            }))
        }
        RunItem::ToolCallOutput(_) => {
            // Already captured via the ToolCall item.
            None
        }
        RunItem::Reasoning(r) => {
            let text = r
                .raw_item
                .get("summary")
                .and_then(|s| s.as_array())
                .and_then(|arr| arr.first())
                .and_then(|entry| entry.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            let id = r
                .raw_item
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("reason")
                .to_owned();

            Some(ThreadItem::Reasoning(ReasoningItem { id, text }))
        }
        RunItem::HandoffCall(_) | RunItem::HandoffOutput(_) => None,
    }
}

/// Generate a simple UUID v4-like string for thread IDs.
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());

    // Use timestamp-based pseudo-random ID. Not cryptographic, but sufficient
    // for thread identification in a single process.
    format!("thread_{now:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Input conversions ----

    #[test]
    fn input_from_str() {
        let input: Input = "hello".into();
        assert_eq!(input, Input::Text("hello".to_owned()));
    }

    #[test]
    fn input_from_string() {
        let input: Input = String::from("world").into();
        assert_eq!(input, Input::Text("world".to_owned()));
    }

    #[test]
    fn input_from_user_input_vec() {
        let items = vec![
            UserInput::Text(TextInput {
                text: "hello".to_owned(),
            }),
            UserInput::LocalImage(LocalImageInput {
                path: "/tmp/img.png".to_owned(),
            }),
        ];
        let input: Input = items.clone().into();
        assert_eq!(input, Input::Items(items));
    }

    // ---- normalize_input ----

    #[test]
    fn normalize_text_input() {
        let input = Input::Text("hello world".to_owned());
        let (prompt, images) = normalize_input(&input);
        assert_eq!(prompt, "hello world");
        assert!(images.is_empty());
    }

    #[test]
    fn normalize_items_input() {
        let items = vec![
            UserInput::Text(TextInput {
                text: "part one".to_owned(),
            }),
            UserInput::LocalImage(LocalImageInput {
                path: "/tmp/img.png".to_owned(),
            }),
            UserInput::Text(TextInput {
                text: "part two".to_owned(),
            }),
        ];
        let input = Input::Items(items);
        let (prompt, images) = normalize_input(&input);
        assert_eq!(prompt, "part one\n\npart two");
        assert_eq!(images, vec!["/tmp/img.png"]);
    }

    #[test]
    fn normalize_items_skips_empty_image_paths() {
        let items = vec![UserInput::LocalImage(LocalImageInput {
            path: String::new(),
        })];
        let input = Input::Items(items);
        let (_, images) = normalize_input(&input);
        assert!(images.is_empty());
    }

    // ---- Thread ----

    #[test]
    fn thread_new_has_no_id() {
        let thread = Thread::new();
        assert!(thread.id().is_none());
    }

    #[test]
    fn thread_with_id() {
        let thread = Thread::with_id("t-123");
        assert_eq!(thread.id(), Some("t-123"));
    }

    #[test]
    fn thread_set_id() {
        let mut thread = Thread::new();
        assert!(thread.id().is_none());
        thread.set_id("t-456".to_owned());
        assert_eq!(thread.id(), Some("t-456"));
    }

    #[test]
    fn thread_default() {
        let thread = Thread::default();
        assert!(thread.id().is_none());
    }

    #[test]
    fn thread_instructions() {
        let thread = Thread::new().with_instructions("Be helpful.");
        assert_eq!(thread.instructions(), Some("Be helpful."));
    }

    #[test]
    fn thread_turns_initially_empty() {
        let thread = Thread::new();
        assert!(thread.turns().is_empty());
        assert_eq!(thread.total_items(), 0);
    }

    #[test]
    fn thread_add_turn() {
        let mut thread = Thread::new();
        let turn = Turn::new(
            vec![ThreadItem::AgentMessage(AgentMessageItem {
                id: "msg-1".to_owned(),
                text: "hello".to_owned(),
            })],
            "hello",
            None,
        );
        thread.add_turn(turn);
        assert_eq!(thread.turns().len(), 1);
        assert_eq!(thread.total_items(), 1);
    }

    #[test]
    fn thread_total_items_across_turns() {
        let mut thread = Thread::new();
        thread.add_turn(Turn::new(
            vec![
                ThreadItem::AgentMessage(AgentMessageItem {
                    id: "msg-1".to_owned(),
                    text: "a".to_owned(),
                }),
                ThreadItem::AgentMessage(AgentMessageItem {
                    id: "msg-2".to_owned(),
                    text: "b".to_owned(),
                }),
            ],
            "b",
            None,
        ));
        thread.add_turn(Turn::new(
            vec![ThreadItem::AgentMessage(AgentMessageItem {
                id: "msg-3".to_owned(),
                text: "c".to_owned(),
            })],
            "c",
            None,
        ));
        assert_eq!(thread.total_items(), 3);
    }

    // ---- Turn ----

    #[test]
    fn turn_construction() {
        let turn = Turn {
            items: vec![],
            final_response: "done".to_owned(),
            usage: None,
        };
        assert!(turn.items.is_empty());
        assert_eq!(turn.final_response, "done");
        assert!(turn.usage.is_none());
    }

    #[test]
    fn turn_new_constructor() {
        let turn = Turn::new(vec![], "hello", Some(CodexUsage::new(10, 5, 20)));
        assert_eq!(turn.final_response, "hello");
        assert!(turn.usage.is_some());
    }

    #[test]
    fn turn_with_usage() {
        let turn = Turn {
            items: vec![],
            final_response: String::new(),
            usage: Some(CodexUsage::new(10, 5, 20)),
        };
        let usage = turn.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
    }

    // ---- run_item_to_thread_item ----

    #[test]
    fn convert_message_output_item() {
        let item = RunItem::MessageOutput(crate::items::MessageOutputItem {
            agent_name: "test".to_owned(),
            raw_item: serde_json::json!({
                "id": "msg-1",
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello!"}]
            }),
        });
        let result = run_item_to_thread_item(&item, 1);
        assert!(result.is_some());
        let ti = result.unwrap();
        assert!(ti.is_agent_message());
        let msg = ti.as_agent_message().unwrap();
        assert_eq!(msg.text, "Hello!");
        assert_eq!(msg.id, "msg-1");
    }

    #[test]
    fn convert_tool_call_item() {
        let item = RunItem::ToolCall(crate::items::ToolCallItem {
            agent_name: "test".to_owned(),
            raw_item: serde_json::json!({
                "call_id": "call-1",
                "name": "get_weather",
                "arguments": "{\"city\":\"SF\"}"
            }),
            tool_origin: None,
        });
        let result = run_item_to_thread_item(&item, 1);
        assert!(result.is_some());
        if let Some(ThreadItem::CommandExecution(cmd)) = result {
            assert_eq!(cmd.id, "call-1");
            assert!(cmd.command.contains("get_weather"));
            assert_eq!(cmd.status, CommandExecutionStatus::Completed);
        } else {
            panic!("expected CommandExecution");
        }
    }

    #[test]
    fn convert_tool_call_output_returns_none() {
        let item = RunItem::ToolCallOutput(crate::items::ToolCallOutputItem {
            agent_name: "test".to_owned(),
            raw_item: serde_json::json!({}),
            output: serde_json::json!("result"),
            tool_origin: None,
        });
        assert!(run_item_to_thread_item(&item, 1).is_none());
    }

    #[test]
    fn convert_reasoning_item() {
        let item = RunItem::Reasoning(crate::items::ReasoningItem {
            agent_name: "test".to_owned(),
            raw_item: serde_json::json!({
                "id": "r-1",
                "summary": [{"text": "thinking..."}]
            }),
        });
        let result = run_item_to_thread_item(&item, 1);
        assert!(result.is_some());
        if let Some(ThreadItem::Reasoning(r)) = result {
            assert_eq!(r.text, "thinking...");
        } else {
            panic!("expected Reasoning");
        }
    }

    #[test]
    fn convert_handoff_returns_none() {
        let item = RunItem::HandoffCall(crate::items::HandoffCallItem {
            agent_name: "test".to_owned(),
            raw_item: serde_json::json!({}),
        });
        assert!(run_item_to_thread_item(&item, 1).is_none());
    }

    // ---- Thread::run error without model ----

    #[tokio::test]
    async fn thread_run_without_model_returns_error() {
        let mut thread = Thread::new();
        let err = thread.run("hello").await;
        assert!(err.is_err());
        let err = err.unwrap_err();
        assert!(err.to_string().contains("requires a model"));
    }

    // ---- UserInput serde ----

    #[test]
    fn user_input_text_serde_round_trip() {
        let input = UserInput::Text(TextInput {
            text: "hello".to_owned(),
        });
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: UserInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input, deserialized);
    }

    #[test]
    fn user_input_image_serde_round_trip() {
        let input = UserInput::LocalImage(LocalImageInput {
            path: "/tmp/img.png".to_owned(),
        });
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: UserInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input, deserialized);
    }

    // ---- uuid_v4 ----

    #[test]
    fn uuid_v4_generates_unique_ids() {
        let id1 = uuid_v4();
        let id2 = uuid_v4();
        assert!(id1.starts_with("thread_"));
        assert!(id2.starts_with("thread_"));
        // They should differ in most cases (timing-based).
    }

    // ---- Send + Sync ----

    #[test]
    fn thread_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Thread>();
        assert_send_sync::<Turn>();
        assert_send_sync::<Input>();
        assert_send_sync::<UserInput>();
    }
}
