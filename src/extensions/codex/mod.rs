//! Codex -- experimental thread-based agent execution engine.
//!
//! Codex provides a higher-level abstraction over the base Agent/Runner
//! system, organizing work into threads and turns with approval workflows,
//! sandbox execution, and rich event streaming.
//!
//! This module mirrors the Python SDK's
//! `agents.extensions.experimental.codex` package, translating its
//! dataclass-heavy design into idiomatic Rust enums and structs with
//! serde support.

#[allow(clippy::module_inception)]
pub mod codex;
pub mod codex_tool;
pub mod events;
pub mod items;
pub mod options;
pub mod thread;

pub use codex::Codex;
pub use codex_tool::CodexTool;
pub use events::{
    CodexUsage, ItemCompletedEvent, ItemStartedEvent, ItemUpdatedEvent, ThreadErrorEvent,
    ThreadEvent, ThreadStartedEvent, TurnCompletedEvent, TurnFailedEvent, TurnStartedEvent,
};
pub use items::{
    AgentMessageItem, CommandExecutionItem, CommandExecutionStatus, ErrorItem, FileChangeItem,
    FileUpdateChange, McpToolCallError, McpToolCallItem, McpToolCallResult, McpToolCallStatus,
    PatchApplyStatus, PatchChangeKind, ReasoningItem, ThreadItem, TodoItem, TodoListItem,
    WebSearchItem,
};
pub use options::{
    ApprovalMode, CodexOptions, ModelReasoningEffort, SandboxMode, ThreadOptions, TurnOptions,
    WebSearchMode,
};
pub use thread::{Input, RunResult, Thread, Turn, UserInput};
