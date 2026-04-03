//! Voice workflow orchestration for single and multi-agent conversations.
//!
//! Provides [`VoiceWorkflowBase`], an abstract trait for voice workflows, and
//! [`SingleAgentVoiceWorkflow`], a concrete implementation that manages a voice
//! conversation with a single starting agent.  These mirror the Python SDK's
//! `voice/workflow.py`.
//!
//! # Architecture
//!
//! A voice workflow receives user transcriptions (text) and yields response text
//! that will be turned into speech by a TTS model.  The simplest case is
//! [`SingleAgentVoiceWorkflow`], which feeds each transcription to a single
//! [`Agent`] via [`Runner::run_streamed`](crate::Runner) and
//! streams back the text deltas.
//!
//! For more complex workflows (multiple agents, custom logic, branching
//! conversations), implement [`VoiceWorkflowBase`] directly.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio_stream::Stream;

use crate::agent::Agent;
use crate::error::Result;

// ---------------------------------------------------------------------------
// VoiceWorkflowBase
// ---------------------------------------------------------------------------

/// Abstract base for voice workflows.
///
/// A voice workflow receives a user transcription and produces a stream of
/// response text chunks that will be spoken via TTS.  Implement this trait
/// to create custom voice interaction patterns.
///
/// For a batteries-included single-agent workflow, see
/// [`SingleAgentVoiceWorkflow`].
#[async_trait]
pub trait VoiceWorkflowBase: Send + Sync {
    /// Run the workflow for a single user transcription.
    ///
    /// Returns a stream of text chunks that should be spoken to the user.
    /// Each chunk is a partial sentence or word group suitable for TTS
    /// streaming.
    async fn run(
        &self,
        transcription: &str,
    ) -> Result<Box<dyn Stream<Item = String> + Send + Unpin>>;

    /// Optional hook called before any user input is received.
    ///
    /// Can be used to deliver a greeting or instruction via TTS.  The
    /// default implementation returns an empty stream.
    async fn on_start(&self) -> Result<Box<dyn Stream<Item = String> + Send + Unpin>> {
        Ok(Box::new(tokio_stream::empty()))
    }
}

// ---------------------------------------------------------------------------
// SingleAgentWorkflowCallbacks
// ---------------------------------------------------------------------------

/// Callbacks for [`SingleAgentVoiceWorkflow`] lifecycle events.
///
/// Implement this trait to receive notifications when the workflow processes
/// a transcription.
#[async_trait]
pub trait SingleAgentWorkflowCallbacks: Send + Sync {
    /// Called when the workflow begins processing a transcription.
    fn on_run(&self, _agent_name: &str, _transcription: &str) {}
}

// ---------------------------------------------------------------------------
// InputHistoryItem
// ---------------------------------------------------------------------------

/// A single item in the conversation history maintained by the workflow.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InputHistoryItem {
    /// The role of the message author (`"user"` or `"assistant"`).
    pub role: String,
    /// The text content of the message.
    pub content: String,
}

impl InputHistoryItem {
    /// Create a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_owned(),
            content: content.into(),
        }
    }

    /// Create an assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_owned(),
            content: content.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// SingleAgentVoiceWorkflow
// ---------------------------------------------------------------------------

/// A voice workflow that runs a single agent per transcription turn.
///
/// Each transcription is added to an internal input history, the agent is run
/// with the full history, and the resulting text deltas are yielded.  After
/// each run, the history is updated with the agent's response.
///
/// For more complex workflows (multiple `Runner` calls, custom message
/// history, branching), implement [`VoiceWorkflowBase`] directly.
///
/// # Example
///
/// ```
/// use openai_agents::voice::workflow::SingleAgentVoiceWorkflow;
/// use openai_agents::Agent;
///
/// let agent = Agent::<()>::builder("voice-assistant")
///     .instructions("Greet the user warmly.")
///     .build();
///
/// let workflow = SingleAgentVoiceWorkflow::new(agent);
/// assert_eq!(workflow.agent_name(), "voice-assistant");
/// ```
pub struct SingleAgentVoiceWorkflow<C: Send + Sync + 'static = ()> {
    /// The current agent handling the conversation.
    agent: Arc<Mutex<Agent<C>>>,
    /// Conversation history.
    input_history: Arc<Mutex<Vec<InputHistoryItem>>>,
    /// Optional callbacks.
    callbacks: Option<Arc<dyn SingleAgentWorkflowCallbacks>>,
}

impl<C: Send + Sync + 'static> SingleAgentVoiceWorkflow<C> {
    /// Create a new single-agent voice workflow.
    ///
    /// # Arguments
    ///
    /// * `agent` -- The agent to run for each transcription turn.
    #[must_use]
    pub fn new(agent: Agent<C>) -> Self {
        Self {
            agent: Arc::new(Mutex::new(agent)),
            input_history: Arc::new(Mutex::new(Vec::new())),
            callbacks: None,
        }
    }

    /// Attach callbacks to the workflow.
    #[must_use]
    pub fn with_callbacks(
        mut self,
        callbacks: impl SingleAgentWorkflowCallbacks + 'static,
    ) -> Self {
        self.callbacks = Some(Arc::new(callbacks));
        self
    }

    /// Get the name of the current agent.
    #[must_use]
    pub fn agent_name(&self) -> String {
        // This blocks briefly to read the agent name.  In practice the lock
        // is uncontested.
        self.agent.blocking_lock().name.clone()
    }

    /// Get a snapshot of the current input history.
    pub async fn input_history(&self) -> Vec<InputHistoryItem> {
        self.input_history.lock().await.clone()
    }

    /// Add a user transcription to the history without running the agent.
    ///
    /// This is useful for seeding the conversation with prior context.
    pub async fn push_user_message(&self, text: impl Into<String>) {
        self.input_history
            .lock()
            .await
            .push(InputHistoryItem::user(text));
    }

    /// Add an assistant message to the history without running the agent.
    ///
    /// This is useful for seeding the conversation with prior context.
    pub async fn push_assistant_message(&self, text: impl Into<String>) {
        self.input_history
            .lock()
            .await
            .push(InputHistoryItem::assistant(text));
    }

    /// Process a transcription turn.
    ///
    /// 1. Fires the `on_run` callback (if any).
    /// 2. Appends the transcription to the input history.
    /// 3. Runs the agent via `Runner::run_streamed`.
    /// 4. Streams text deltas back.
    /// 5. Updates the history with the assistant's response.
    ///
    /// # Errors
    ///
    /// Returns an error if the agent run fails.
    ///
    /// # Note
    ///
    /// The full integration with `Runner::run_streamed` for this workflow
    /// is still TODO. At present, this method only adds the transcription to
    /// history and returns an error.
    #[allow(clippy::unused_async)]
    pub async fn process_transcription(&self, transcription: &str) -> Result<()> {
        // Fire callback.
        if let Some(cb) = &self.callbacks {
            let name = self.agent.lock().await.name.clone();
            cb.on_run(&name, transcription);
        }

        // Append user message to history.
        self.input_history
            .lock()
            .await
            .push(InputHistoryItem::user(transcription));

        // TODO: Full implementation:
        // 1. Build input list from history
        // 2. Call Runner::run_streamed(agent, input_history)
        // 3. Use VoiceWorkflowHelper::stream_text_from(result)
        // 4. Yield text chunks
        // 5. Update input_history = result.to_input_list()
        // 6. Update current_agent = result.last_agent

        Err(crate::error::AgentError::UserError {
            message: "Voice workflow processing is not yet integrated with Runner::run_streamed"
                .into(),
        })
    }
}

impl<C: Send + Sync + 'static> std::fmt::Debug for SingleAgentVoiceWorkflow<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleAgentVoiceWorkflow")
            .field("callbacks", &self.callbacks.is_some())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// VoiceWorkflowHelper
// ---------------------------------------------------------------------------

/// Helper utilities for extracting text from streamed agent results.
///
/// This mirrors the Python SDK's `VoiceWorkflowHelper` class.
pub struct VoiceWorkflowHelper;

impl VoiceWorkflowHelper {
    // TODO: Implement `stream_text_from` once `RunResultStreaming` is available.
    //
    // pub async fn stream_text_from(result: RunResultStreaming)
    //     -> impl Stream<Item = String> { ... }
}

// ---------------------------------------------------------------------------
// TranscriptEntry (for workflow result reporting)
// ---------------------------------------------------------------------------

/// A single entry in a conversation transcript.
///
/// Used by voice workflows to report the full conversation after a session
/// ends.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TranscriptEntry {
    /// The role of the speaker (`"user"` or `"assistant"`).
    pub role: String,
    /// The spoken or transcribed text.
    pub text: String,
    /// Timestamp in milliseconds from the start of the session.
    pub timestamp_ms: u64,
}

impl TranscriptEntry {
    /// Create a new transcript entry.
    #[must_use]
    pub fn new(role: impl Into<String>, text: impl Into<String>, timestamp_ms: u64) -> Self {
        Self {
            role: role.into(),
            text: text.into(),
            timestamp_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// VoiceWorkflowResult
// ---------------------------------------------------------------------------

/// Summary result of a completed voice workflow session.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct VoiceWorkflowResult {
    /// Full transcript of the conversation.
    pub transcript: Vec<TranscriptEntry>,
    /// Number of tool calls made during the session.
    pub tool_call_count: u32,
    /// Total duration of the session in seconds.
    pub duration_seconds: f64,
}

impl VoiceWorkflowResult {
    /// Create a new empty result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- InputHistoryItem ----

    #[test]
    fn input_history_item_user() {
        let item = InputHistoryItem::user("hello");
        assert_eq!(item.role, "user");
        assert_eq!(item.content, "hello");
    }

    #[test]
    fn input_history_item_assistant() {
        let item = InputHistoryItem::assistant("hi there");
        assert_eq!(item.role, "assistant");
        assert_eq!(item.content, "hi there");
    }

    // ---- TranscriptEntry ----

    #[test]
    fn transcript_entry_construction() {
        let entry = TranscriptEntry::new("user", "Hello", 1000);
        assert_eq!(entry.role, "user");
        assert_eq!(entry.text, "Hello");
        assert_eq!(entry.timestamp_ms, 1000);
    }

    #[test]
    fn transcript_entry_equality() {
        let a = TranscriptEntry::new("user", "Hello", 1000);
        let b = TranscriptEntry::new("user", "Hello", 1000);
        assert_eq!(a, b);
    }

    #[test]
    fn transcript_entry_clone() {
        let entry = TranscriptEntry::new("assistant", "World", 2000);
        #[allow(clippy::redundant_clone)]
        let cloned = entry.clone();
        assert_eq!(cloned.role, "assistant");
        assert_eq!(cloned.text, "World");
        assert_eq!(cloned.timestamp_ms, 2000);
    }

    // ---- VoiceWorkflowResult ----

    #[test]
    fn workflow_result_default() {
        let result = VoiceWorkflowResult::new();
        assert!(result.transcript.is_empty());
        assert_eq!(result.tool_call_count, 0);
        assert!((result.duration_seconds - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn workflow_result_with_entries() {
        let mut result = VoiceWorkflowResult::new();
        result
            .transcript
            .push(TranscriptEntry::new("user", "Hi", 0));
        result
            .transcript
            .push(TranscriptEntry::new("assistant", "Hello!", 500));
        result.tool_call_count = 1;
        result.duration_seconds = 3.5;

        assert_eq!(result.transcript.len(), 2);
        assert_eq!(result.tool_call_count, 1);
        assert!((result.duration_seconds - 3.5).abs() < f64::EPSILON);
    }

    // ---- SingleAgentVoiceWorkflow ----

    #[test]
    fn workflow_construction() {
        let agent = Agent::<()>::builder("voice-bot")
            .instructions("Be helpful.")
            .build();
        let workflow = SingleAgentVoiceWorkflow::new(agent);
        assert_eq!(workflow.agent_name(), "voice-bot");
    }

    #[test]
    fn workflow_debug_impl() {
        let agent = Agent::<()>::builder("test").build();
        let workflow = SingleAgentVoiceWorkflow::new(agent);
        let debug_str = format!("{workflow:?}");
        assert!(debug_str.contains("SingleAgentVoiceWorkflow"));
    }

    #[test]
    fn workflow_with_callbacks() {
        struct TestCb;
        impl SingleAgentWorkflowCallbacks for TestCb {
            fn on_run(&self, agent_name: &str, transcription: &str) {
                assert!(!agent_name.is_empty());
                assert!(!transcription.is_empty());
            }
        }

        let agent = Agent::<()>::builder("test").build();
        let workflow = SingleAgentVoiceWorkflow::new(agent).with_callbacks(TestCb);
        let debug_str = format!("{workflow:?}");
        assert!(debug_str.contains("true")); // callbacks: true
    }

    #[tokio::test]
    async fn workflow_push_messages() {
        let agent = Agent::<()>::builder("test").build();
        let workflow = SingleAgentVoiceWorkflow::new(agent);

        workflow.push_user_message("Hello").await;
        workflow.push_assistant_message("Hi there!").await;

        let history = workflow.input_history().await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[0].content, "Hello");
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[1].content, "Hi there!");
    }

    #[tokio::test]
    async fn workflow_process_transcription_returns_error() {
        let agent = Agent::<()>::builder("test").build();
        let workflow = SingleAgentVoiceWorkflow::new(agent);

        let result = workflow.process_transcription("Hello").await;
        assert!(result.is_err());

        // The user message should still have been added to history.
        let history = workflow.input_history().await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "Hello");
    }

    #[tokio::test]
    async fn workflow_process_with_callback() {
        use std::sync::atomic::{AtomicBool, Ordering};

        // Use a static to track whether the callback was invoked.
        static CALLED: AtomicBool = AtomicBool::new(false);
        CALLED.store(false, Ordering::Relaxed);

        #[allow(clippy::items_after_statements)]
        struct TrackingCb;
        #[allow(clippy::items_after_statements)]
        impl SingleAgentWorkflowCallbacks for TrackingCb {
            fn on_run(&self, _agent_name: &str, _transcription: &str) {
                CALLED.store(true, Ordering::Relaxed);
            }
        }

        let agent = Agent::<()>::builder("test").build();
        let workflow = SingleAgentVoiceWorkflow::new(agent).with_callbacks(TrackingCb);

        // The run will fail but the callback should have been invoked.
        let _ = workflow.process_transcription("Test").await;
        assert!(CALLED.load(Ordering::Relaxed));
    }

    // ---- VoiceWorkflowHelper ----

    #[test]
    fn workflow_helper_exists() {
        // Just verify the type exists and can be referenced.
        fn assert_type_exists<T>() {}
        assert_type_exists::<VoiceWorkflowHelper>();
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn workflow_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InputHistoryItem>();
        assert_send_sync::<TranscriptEntry>();
        assert_send_sync::<VoiceWorkflowResult>();
        assert_send_sync::<VoiceWorkflowHelper>();
        assert_send_sync::<SingleAgentVoiceWorkflow<()>>();
    }
}
