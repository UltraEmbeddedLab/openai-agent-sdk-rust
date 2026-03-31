//! Codex thread item types.
//!
//! This module defines the item types produced during Codex thread execution.
//! Each item represents a discrete unit of work or output, such as a command
//! execution, file change, agent message, or MCP tool call.
//!
//! Mirrors the Python SDK's `items.py` in the Codex extension.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Status and kind enums
// ---------------------------------------------------------------------------

/// Status of a command execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CommandExecutionStatus {
    /// The command is still running.
    #[serde(rename = "in_progress")]
    InProgress,
    /// The command completed successfully.
    #[serde(rename = "completed")]
    Completed,
    /// The command failed.
    #[serde(rename = "failed")]
    Failed,
}

/// The kind of change applied to a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PatchChangeKind {
    /// A new file was added.
    #[serde(rename = "add")]
    Add,
    /// An existing file was deleted.
    #[serde(rename = "delete")]
    Delete,
    /// An existing file was updated.
    #[serde(rename = "update")]
    Update,
}

/// Status of a patch application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PatchApplyStatus {
    /// The patch was applied successfully.
    #[serde(rename = "completed")]
    Completed,
    /// The patch application failed.
    #[serde(rename = "failed")]
    Failed,
}

/// Status of an MCP tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum McpToolCallStatus {
    /// The MCP tool call is in progress.
    #[serde(rename = "in_progress")]
    InProgress,
    /// The MCP tool call completed.
    #[serde(rename = "completed")]
    Completed,
    /// The MCP tool call failed.
    #[serde(rename = "failed")]
    Failed,
}

// ---------------------------------------------------------------------------
// Item structs
// ---------------------------------------------------------------------------

/// A command that was executed during a Codex thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CommandExecutionItem {
    /// Unique identifier for this item.
    pub id: String,
    /// The command that was executed.
    pub command: String,
    /// Current execution status.
    pub status: CommandExecutionStatus,
    /// Aggregated stdout/stderr output from the command.
    #[serde(default)]
    pub aggregated_output: String,
    /// Exit code of the command, if completed.
    pub exit_code: Option<i32>,
}

/// A single file change within a [`FileChangeItem`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FileUpdateChange {
    /// Path of the changed file.
    pub path: String,
    /// Kind of change (add, delete, or update).
    pub kind: PatchChangeKind,
}

/// A set of file changes applied during a Codex thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FileChangeItem {
    /// Unique identifier for this item.
    pub id: String,
    /// List of individual file changes.
    pub changes: Vec<FileUpdateChange>,
    /// Status of the patch application.
    pub status: PatchApplyStatus,
}

/// Result of a successful MCP tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpToolCallResult {
    /// Content blocks returned by the MCP tool.
    pub content: Vec<serde_json::Value>,
    /// Structured content returned by the MCP tool, if any.
    pub structured_content: Option<serde_json::Value>,
}

/// Error from a failed MCP tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpToolCallError {
    /// Error message from the MCP tool.
    pub message: String,
}

/// An MCP tool call item from a Codex thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpToolCallItem {
    /// Unique identifier for this item.
    pub id: String,
    /// Name of the MCP server.
    pub server: String,
    /// Name of the tool that was called.
    pub tool: String,
    /// Arguments passed to the tool.
    pub arguments: Option<serde_json::Value>,
    /// Current execution status.
    pub status: McpToolCallStatus,
    /// Result of the tool call, if completed successfully.
    pub result: Option<McpToolCallResult>,
    /// Error from the tool call, if it failed.
    pub error: Option<McpToolCallError>,
}

/// A message from the agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentMessageItem {
    /// Unique identifier for this item.
    pub id: String,
    /// The text content of the message.
    pub text: String,
}

/// A reasoning step produced by the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ReasoningItem {
    /// Unique identifier for this item.
    pub id: String,
    /// The reasoning text.
    pub text: String,
}

/// A web search performed during a Codex thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WebSearchItem {
    /// Unique identifier for this item.
    pub id: String,
    /// The search query.
    pub query: String,
}

/// An error item from a Codex thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ErrorItem {
    /// Unique identifier for this item.
    pub id: String,
    /// Error message.
    pub message: String,
}

/// A single to-do entry within a [`TodoListItem`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TodoItem {
    /// Description of the to-do.
    pub text: String,
    /// Whether this to-do is complete.
    pub completed: bool,
}

/// A to-do list item from a Codex thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TodoListItem {
    /// Unique identifier for this item.
    pub id: String,
    /// The individual to-do entries.
    pub items: Vec<TodoItem>,
}

// ---------------------------------------------------------------------------
// ThreadItem enum
// ---------------------------------------------------------------------------

/// An item generated during a Codex thread execution.
///
/// This is a closed enum of all known item types, plus an `Unknown` variant
/// for forward compatibility with future Codex CLI versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum ThreadItem {
    /// A command execution.
    #[serde(rename = "command_execution")]
    CommandExecution(CommandExecutionItem),
    /// A set of file changes.
    #[serde(rename = "file_change")]
    FileChange(FileChangeItem),
    /// An MCP tool call.
    #[serde(rename = "mcp_tool_call")]
    McpToolCall(McpToolCallItem),
    /// A message from the agent.
    #[serde(rename = "agent_message")]
    AgentMessage(AgentMessageItem),
    /// A reasoning step.
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningItem),
    /// A web search.
    #[serde(rename = "web_search")]
    WebSearch(WebSearchItem),
    /// A to-do list.
    #[serde(rename = "todo_list")]
    TodoList(TodoListItem),
    /// An error.
    #[serde(rename = "error")]
    Error(ErrorItem),
    /// An unknown item type for forward compatibility.
    #[serde(other)]
    Unknown,
}

impl ThreadItem {
    /// Returns `true` if this item is an [`AgentMessage`](ThreadItem::AgentMessage).
    #[must_use]
    pub const fn is_agent_message(&self) -> bool {
        matches!(self, Self::AgentMessage(_))
    }

    /// If this item is an [`AgentMessage`](ThreadItem::AgentMessage), return a
    /// reference to the inner [`AgentMessageItem`].
    #[must_use]
    pub const fn as_agent_message(&self) -> Option<&AgentMessageItem> {
        match self {
            Self::AgentMessage(item) => Some(item),
            _ => None,
        }
    }
}

/// Deserialize a [`ThreadItem`] from a raw JSON mapping.
///
/// This function handles the same coercion logic as the Python SDK's
/// `coerce_thread_item`, constructing the appropriate variant based on
/// the `"type"` field.
///
/// # Errors
///
/// Returns a `serde_json::Error` if the mapping cannot be deserialized.
pub fn coerce_thread_item(raw: &serde_json::Value) -> Result<ThreadItem, serde_json::Error> {
    // Use the tag-based deserialization for known types.
    // For unknown types, fall back gracefully.
    let item_type = raw
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");

    match item_type {
        "command_execution" => {
            let item: CommandExecutionItem = serde_json::from_value(raw.clone())?;
            Ok(ThreadItem::CommandExecution(item))
        }
        "file_change" => {
            let item: FileChangeItem = serde_json::from_value(raw.clone())?;
            Ok(ThreadItem::FileChange(item))
        }
        "mcp_tool_call" => {
            let item: McpToolCallItem = serde_json::from_value(raw.clone())?;
            Ok(ThreadItem::McpToolCall(item))
        }
        "agent_message" => {
            let item: AgentMessageItem = serde_json::from_value(raw.clone())?;
            Ok(ThreadItem::AgentMessage(item))
        }
        "reasoning" => {
            let item: ReasoningItem = serde_json::from_value(raw.clone())?;
            Ok(ThreadItem::Reasoning(item))
        }
        "web_search" => {
            let item: WebSearchItem = serde_json::from_value(raw.clone())?;
            Ok(ThreadItem::WebSearch(item))
        }
        "todo_list" => {
            let item: TodoListItem = serde_json::from_value(raw.clone())?;
            Ok(ThreadItem::TodoList(item))
        }
        "error" => {
            let item: ErrorItem = serde_json::from_value(raw.clone())?;
            Ok(ThreadItem::Error(item))
        }
        _ => Ok(ThreadItem::Unknown),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- Status enum serde ----

    #[test]
    fn command_execution_status_serde() {
        let statuses = [
            (CommandExecutionStatus::InProgress, "\"in_progress\""),
            (CommandExecutionStatus::Completed, "\"completed\""),
            (CommandExecutionStatus::Failed, "\"failed\""),
        ];
        for (status, expected) in &statuses {
            let json = serde_json::to_string(status).unwrap();
            assert_eq!(&json, expected);
            let deserialized: CommandExecutionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, deserialized);
        }
    }

    #[test]
    fn patch_change_kind_serde() {
        let kinds = [
            (PatchChangeKind::Add, "\"add\""),
            (PatchChangeKind::Delete, "\"delete\""),
            (PatchChangeKind::Update, "\"update\""),
        ];
        for (kind, expected) in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            assert_eq!(&json, expected);
        }
    }

    #[test]
    fn mcp_tool_call_status_serde() {
        let statuses = [
            (McpToolCallStatus::InProgress, "\"in_progress\""),
            (McpToolCallStatus::Completed, "\"completed\""),
            (McpToolCallStatus::Failed, "\"failed\""),
        ];
        for (status, expected) in &statuses {
            let json = serde_json::to_string(status).unwrap();
            assert_eq!(&json, expected);
        }
    }

    // ---- ThreadItem deserialization ----

    #[test]
    fn deserialize_command_execution_item() {
        let raw = json!({
            "type": "command_execution",
            "id": "cmd-1",
            "command": "ls -la",
            "status": "completed",
            "aggregated_output": "total 0\n",
            "exit_code": 0
        });
        let item = coerce_thread_item(&raw).unwrap();
        if let ThreadItem::CommandExecution(cmd) = item {
            assert_eq!(cmd.id, "cmd-1");
            assert_eq!(cmd.command, "ls -la");
            assert_eq!(cmd.status, CommandExecutionStatus::Completed);
            assert_eq!(cmd.aggregated_output, "total 0\n");
            assert_eq!(cmd.exit_code, Some(0));
        } else {
            panic!("expected CommandExecution variant");
        }
    }

    #[test]
    fn deserialize_agent_message_item() {
        let raw = json!({
            "type": "agent_message",
            "id": "msg-1",
            "text": "Hello, world!"
        });
        let item = coerce_thread_item(&raw).unwrap();
        assert!(item.is_agent_message());
        let msg = item.as_agent_message().unwrap();
        assert_eq!(msg.id, "msg-1");
        assert_eq!(msg.text, "Hello, world!");
    }

    #[test]
    fn deserialize_file_change_item() {
        let raw = json!({
            "type": "file_change",
            "id": "fc-1",
            "changes": [
                {"path": "src/main.rs", "kind": "update"},
                {"path": "src/new.rs", "kind": "add"}
            ],
            "status": "completed"
        });
        let item = coerce_thread_item(&raw).unwrap();
        if let ThreadItem::FileChange(fc) = item {
            assert_eq!(fc.id, "fc-1");
            assert_eq!(fc.changes.len(), 2);
            assert_eq!(fc.changes[0].path, "src/main.rs");
            assert_eq!(fc.changes[0].kind, PatchChangeKind::Update);
            assert_eq!(fc.changes[1].kind, PatchChangeKind::Add);
            assert_eq!(fc.status, PatchApplyStatus::Completed);
        } else {
            panic!("expected FileChange variant");
        }
    }

    #[test]
    fn deserialize_reasoning_item() {
        let raw = json!({
            "type": "reasoning",
            "id": "r-1",
            "text": "thinking about this..."
        });
        let item = coerce_thread_item(&raw).unwrap();
        if let ThreadItem::Reasoning(r) = item {
            assert_eq!(r.id, "r-1");
            assert_eq!(r.text, "thinking about this...");
        } else {
            panic!("expected Reasoning variant");
        }
    }

    #[test]
    fn deserialize_web_search_item() {
        let raw = json!({
            "type": "web_search",
            "id": "ws-1",
            "query": "Rust async patterns"
        });
        let item = coerce_thread_item(&raw).unwrap();
        if let ThreadItem::WebSearch(ws) = item {
            assert_eq!(ws.id, "ws-1");
            assert_eq!(ws.query, "Rust async patterns");
        } else {
            panic!("expected WebSearch variant");
        }
    }

    #[test]
    fn deserialize_error_item() {
        let raw = json!({
            "type": "error",
            "id": "err-1",
            "message": "something went wrong"
        });
        let item = coerce_thread_item(&raw).unwrap();
        if let ThreadItem::Error(err) = item {
            assert_eq!(err.id, "err-1");
            assert_eq!(err.message, "something went wrong");
        } else {
            panic!("expected Error variant");
        }
    }

    #[test]
    fn deserialize_todo_list_item() {
        let raw = json!({
            "type": "todo_list",
            "id": "todo-1",
            "items": [
                {"text": "Write tests", "completed": false},
                {"text": "Fix bug", "completed": true}
            ]
        });
        let item = coerce_thread_item(&raw).unwrap();
        if let ThreadItem::TodoList(todo) = item {
            assert_eq!(todo.id, "todo-1");
            assert_eq!(todo.items.len(), 2);
            assert_eq!(todo.items[0].text, "Write tests");
            assert!(!todo.items[0].completed);
            assert!(todo.items[1].completed);
        } else {
            panic!("expected TodoList variant");
        }
    }

    #[test]
    fn deserialize_mcp_tool_call_item() {
        let raw = json!({
            "type": "mcp_tool_call",
            "id": "mcp-1",
            "server": "my-server",
            "tool": "my-tool",
            "arguments": {"key": "value"},
            "status": "completed",
            "result": {
                "content": [{"type": "text", "text": "result"}],
                "structured_content": null
            },
            "error": null
        });
        let item = coerce_thread_item(&raw).unwrap();
        if let ThreadItem::McpToolCall(mcp) = item {
            assert_eq!(mcp.id, "mcp-1");
            assert_eq!(mcp.server, "my-server");
            assert_eq!(mcp.tool, "my-tool");
            assert_eq!(mcp.status, McpToolCallStatus::Completed);
            assert!(mcp.result.is_some());
            assert!(mcp.error.is_none());
        } else {
            panic!("expected McpToolCall variant");
        }
    }

    #[test]
    fn deserialize_unknown_item_type() {
        let raw = json!({
            "type": "future_item_type",
            "id": "x-1",
            "data": "some data"
        });
        let item = coerce_thread_item(&raw).unwrap();
        assert!(matches!(item, ThreadItem::Unknown));
    }

    // ---- ThreadItem helper methods ----

    #[test]
    fn is_agent_message_returns_true_for_agent_message() {
        let item = ThreadItem::AgentMessage(AgentMessageItem {
            id: "msg-1".to_owned(),
            text: "hello".to_owned(),
        });
        assert!(item.is_agent_message());
    }

    #[test]
    fn is_agent_message_returns_false_for_other() {
        let item = ThreadItem::Error(ErrorItem {
            id: "err-1".to_owned(),
            message: "oops".to_owned(),
        });
        assert!(!item.is_agent_message());
    }

    #[test]
    fn as_agent_message_returns_some_for_agent_message() {
        let item = ThreadItem::AgentMessage(AgentMessageItem {
            id: "msg-1".to_owned(),
            text: "hello".to_owned(),
        });
        let msg = item.as_agent_message().unwrap();
        assert_eq!(msg.text, "hello");
    }

    #[test]
    fn as_agent_message_returns_none_for_other() {
        let item = ThreadItem::Unknown;
        assert!(item.as_agent_message().is_none());
    }

    // ---- Send + Sync ----

    #[test]
    fn items_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ThreadItem>();
        assert_send_sync::<CommandExecutionItem>();
        assert_send_sync::<FileChangeItem>();
        assert_send_sync::<McpToolCallItem>();
        assert_send_sync::<AgentMessageItem>();
        assert_send_sync::<ReasoningItem>();
        assert_send_sync::<WebSearchItem>();
        assert_send_sync::<ErrorItem>();
        assert_send_sync::<TodoListItem>();
        assert_send_sync::<TodoItem>();
    }
}
