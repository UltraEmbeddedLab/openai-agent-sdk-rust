//! Thread and Turn types for the Codex extension.
//!
//! A [`Thread`] represents a session of work with the Codex CLI, containing
//! one or more [`Turn`]s. Each turn aggregates the items and events produced
//! by a single interaction with the CLI.
//!
//! Mirrors the Python SDK's `thread.py` in the Codex extension.

use serde::{Deserialize, Serialize};

use super::events::CodexUsage;
use super::items::ThreadItem;

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
/// A turn represents one request-response cycle with the Codex CLI,
/// collecting all items produced and the final response text.
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

/// Type alias matching the Python SDK's `RunResult = Turn`.
pub type RunResult = Turn;

// ---------------------------------------------------------------------------
// Thread
// ---------------------------------------------------------------------------

/// A Codex thread representing a session of work with the Codex CLI.
///
/// Threads can be started fresh or resumed from a previous session using
/// a thread ID. Each call to [`Thread::run`] or [`Thread::run_streamed`]
/// adds a new turn to the thread.
///
/// In this initial implementation, the thread manages state but does not
/// yet implement the full subprocess execution. That will be added when
/// the `CodexExec` layer is implemented.
#[derive(Debug)]
#[non_exhaustive]
pub struct Thread {
    id: Option<String>,
}

impl Thread {
    /// Create a new thread with no ID (a fresh session).
    #[must_use]
    pub const fn new() -> Self {
        Self { id: None }
    }

    /// Create a new thread that resumes from the given thread ID.
    #[must_use]
    pub fn with_id(id: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
        }
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
}

impl Default for Thread {
    fn default() -> Self {
        Self::new()
    }
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
