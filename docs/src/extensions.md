# Extensions and Experimental Features

This page covers the optional and experimental parts of the SDK:

- **Codex** — thread-based agent execution engine with approval workflows and rich event
  streaming.
- **Computer control** — desktop and browser automation via the `Computer` trait.
- **Editor tools** — applying code diffs from agents (`ApplyPatchTool`, `apply_diff`).
- **Retry policies** — configurable exponential backoff for API calls.

---

## Codex (Experimental)

Codex is a higher-level orchestration system that organizes agent work into *threads* and
*turns* with support for tool approval, sandbox execution, and streaming events. It mirrors
the Python SDK's `extensions/experimental/codex` package.

> **Note.** Codex is marked experimental. APIs may change before stabilization.

### Entry point: `Codex`

`Codex` is the top-level executor. Create one instance per application and use it to start
or resume threads.

```rust,no_run
use openai_agents::extensions::codex::{Codex, CodexOptions};

let codex = Codex::new(
    CodexOptions::new()
        .with_api_key("sk-...")
        .with_base_url("https://api.openai.com/v1"),
);

// Attach a custom model (optional; uses OPENAI_API_KEY by default).
// let codex = codex.with_model(Arc::new(my_model));

let thread = codex.start_thread(None);
```

### CodexOptions

`CodexOptions` configures the Codex CLI process environment:

```rust
use openai_agents::extensions::codex::CodexOptions;
use std::collections::HashMap;

let mut env = HashMap::new();
env.insert("EXTRA_VAR".into(), "value".into());

let options = CodexOptions::new()
    .with_api_key("sk-...")
    .with_base_url("https://api.openai.com/v1")
    .with_codex_path_override("/usr/local/bin/codex")
    .with_env(env)
    .with_subprocess_stream_limit_bytes(1024 * 1024); // 1 MiB
```

### ApprovalMode

Controls when the user must approve tool invocations:

| Variant | Meaning |
|---|---|
| `ApprovalMode::Never` | No approval required (default) |
| `ApprovalMode::OnRequest` | Require approval when the model requests it |
| `ApprovalMode::OnFailure` | Require approval when a tool call fails |
| `ApprovalMode::Untrusted` | Require approval for untrusted operations |

### SandboxMode

Controls filesystem and network access granted to the Codex subprocess:

| Variant | Meaning |
|---|---|
| `SandboxMode::ReadOnly` | Read-only access (default) |
| `SandboxMode::WorkspaceWrite` | Write access within the working directory |
| `SandboxMode::DangerFullAccess` | Full filesystem access (use with caution) |

### ModelReasoningEffort

```rust
use openai_agents::extensions::codex::ModelReasoningEffort;

let effort = ModelReasoningEffort::High; // minimal, low, medium (default), high, xhigh
```

### WebSearchMode

```rust
use openai_agents::extensions::codex::WebSearchMode;

let search = WebSearchMode::Live; // disabled (default), cached, live
```

### ThreadOptions

`ThreadOptions` controls a single thread's model, sandbox, and working directory:

```rust
use openai_agents::extensions::codex::{
    ThreadOptions, ApprovalMode, SandboxMode, ModelReasoningEffort, WebSearchMode,
};

let thread_opts = ThreadOptions {
    model: Some("o3".to_owned()),
    approval_mode: Some(ApprovalMode::OnRequest),
    sandbox_mode: Some(SandboxMode::WorkspaceWrite),
    cwd: Some("/workspace".to_owned()),
    reasoning_effort: Some(ModelReasoningEffort::High),
    web_search: Some(WebSearchMode::Live),
    ..Default::default()
};
```

### Threads and Turns

A `Thread` holds the conversation history and state for a sequence of `Turn`s. Each call to
`thread.run(input)` creates a new turn and streams back events.

```rust,no_run
use openai_agents::extensions::codex::{Codex, CodexOptions, Input, ThreadOptions};
use openai_agents::extensions::codex::ThreadEvent;
use tokio_stream::StreamExt;

let codex = Codex::new(CodexOptions::new().with_api_key("sk-..."));
let mut thread = codex.start_thread(None);

let mut stream = thread.run(Input::from("Write a function that reverses a string in Rust."), None).await?;

while let Some(event) = stream.next().await {
    match event? {
        ThreadEvent::ThreadStarted(e) => println!("Thread: {}", e.thread_id),
        ThreadEvent::TurnStarted(_) => println!("[turn started]"),
        ThreadEvent::ItemCompleted(e) => println!("Item: {:?}", e.item),
        ThreadEvent::TurnCompleted(e) => {
            if let Some(usage) = e.usage {
                println!("Tokens: in={} out={}", usage.input_tokens, usage.output_tokens);
            }
        }
        ThreadEvent::TurnFailed(e) => eprintln!("Error: {}", e.error.message),
        _ => {}
    }
}
```

### ThreadItem variants

Items produced during a thread turn represent discrete actions taken by the agent:

| Type | Description |
|---|---|
| `AgentMessageItem` | A text message from the agent |
| `CommandExecutionItem` | A shell command that was executed |
| `FileChangeItem` | A file created, updated, or deleted |
| `ReasoningItem` | Internal reasoning text (when tracing is enabled) |
| `McpToolCallItem` | A call to an MCP tool |
| `WebSearchItem` | A web search query and result |
| `TodoItem` / `TodoListItem` | A task item or task list |
| `ErrorItem` | An error encountered during execution |

```rust
use openai_agents::extensions::codex::items::{
    ThreadItem, CommandExecutionItem, FileChangeItem, AgentMessageItem,
    CommandExecutionStatus, PatchChangeKind,
};

// Pattern match on thread items.
fn handle_item(item: &ThreadItem) {
    match item {
        ThreadItem::AgentMessage(msg) => println!("Agent: {}", msg.content),
        ThreadItem::CommandExecution(cmd) => {
            println!("Command: {}", cmd.command.join(" "));
            match cmd.status {
                CommandExecutionStatus::Completed => println!("  -> ok"),
                CommandExecutionStatus::Failed => println!("  -> failed"),
                CommandExecutionStatus::InProgress => println!("  -> running"),
            }
        }
        ThreadItem::FileChange(change) => {
            for update in &change.changes {
                println!("File {:?}: {}", update.kind, update.path);
            }
        }
        _ => {}
    }
}
```

### ThreadEvent variants

| Event | Description |
|---|---|
| `ThreadEvent::ThreadStarted(ThreadStartedEvent)` | Thread ID assigned |
| `ThreadEvent::TurnStarted(TurnStartedEvent)` | New turn beginning |
| `ThreadEvent::ItemStarted(ItemStartedEvent)` | Item processing started |
| `ThreadEvent::ItemUpdated(ItemUpdatedEvent)` | Item partially updated |
| `ThreadEvent::ItemCompleted(ItemCompletedEvent)` | Item fully produced |
| `ThreadEvent::TurnCompleted(TurnCompletedEvent)` | Turn finished with usage stats |
| `ThreadEvent::TurnFailed(TurnFailedEvent)` | Turn failed with an error |
| `ThreadEvent::ThreadError(ThreadErrorEvent)` | Unrecoverable thread error |

### CodexTool

`CodexTool` wraps a Codex thread as a standard function tool that a regular `Agent` can call.
This lets you compose a fast-response agent with a slow-but-powerful Codex thread:

```rust,no_run
use openai_agents::extensions::codex::{Codex, CodexOptions, CodexTool};
use openai_agents::{Agent, Tool};

let codex = Codex::new(CodexOptions::new().with_api_key("sk-..."));
let codex_tool = CodexTool::from_codex(codex, "codex_task", "Delegate complex tasks to Codex.");

let agent = Agent::<()>::builder("assistant")
    .instructions("You handle simple questions. For complex tasks, delegate to codex_task.")
    .tool(Tool::Codex(codex_tool))
    .build();
```

---

## Computer Control

The `Computer` trait enables agents to control a desktop or browser environment.

### Implementing `Computer`

```rust,no_run
use openai_agents::computer::{Computer, Environment, Button};
use openai_agents::error::Result;
use async_trait::async_trait;

pub struct MyDesktop;

#[async_trait]
impl Computer for MyDesktop {
    fn environment(&self) -> Option<Environment> {
        Some(Environment::Mac)
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((1920, 1080))
    }

    async fn screenshot(&self) -> Result<String> {
        // Capture a screenshot and return base64-encoded PNG.
        todo!()
    }

    async fn click(&self, x: i32, y: i32, button: Button) -> Result<()> {
        println!("Click {:?} at ({x}, {y})", button);
        Ok(())
    }

    async fn double_click(&self, x: i32, y: i32) -> Result<()> { todo!() }
    async fn scroll(&self, x: i32, y: i32, dx: i32, dy: i32) -> Result<()> { todo!() }
    async fn type_text(&self, text: &str) -> Result<()> { todo!() }
    async fn wait(&self) -> Result<()> { todo!() }
    async fn move_cursor(&self, x: i32, y: i32) -> Result<()> { todo!() }
    async fn keypress(&self, keys: &[String]) -> Result<()> { todo!() }
    async fn drag(&self, path: &[(i32, i32)]) -> Result<()> { todo!() }
}
```

### Environment and Button

```rust
use openai_agents::computer::{Environment, Button};

let env = Environment::Browser; // Mac, Windows, Ubuntu, Browser
let btn = Button::Left;         // Left, Right, Wheel, Back, Forward
```

### ComputerTool

`ComputerTool` wraps a `Computer` implementation as a hosted tool:

```rust,no_run
use openai_agents::computer::{ComputerTool, Environment};

let tool = ComputerTool {
    display_width: 1920,
    display_height: 1080,
    environment: Environment::Mac,
};
```

---

## Editor Tools

The editor module provides tools for agents that need to read and modify files using V4A-
format diffs, the same format used by `codex edit` and similar tools.

### ApplyPatchTool

`ApplyPatchTool` is a hosted tool that lets an agent apply a patch to files:

```rust
use openai_agents::editor::ApplyPatchTool;

let tool = ApplyPatchTool::new();
```

### ApplyPatchEditor trait

Implement `ApplyPatchEditor` to provide custom file I/O:

```rust,no_run
use openai_agents::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents::error::Result;
use async_trait::async_trait;

pub struct DiskEditor {
    base_path: String,
}

#[async_trait]
impl ApplyPatchEditor for DiskEditor {
    async fn create_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let full_path = format!("{}/{}", self.base_path, op.path);
        let content = op.diff.as_deref().unwrap_or("");
        std::fs::write(&full_path, content)?;
        Ok(ApplyPatchResult::completed(format!("Created {full_path}")))
    }

    async fn update_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        // Apply the diff and write back.
        todo!()
    }

    async fn delete_file(&self, op: &ApplyPatchOperation) -> Result<ApplyPatchResult> {
        let full_path = format!("{}/{}", self.base_path, op.path);
        std::fs::remove_file(&full_path)?;
        Ok(ApplyPatchResult::completed(format!("Deleted {full_path}")))
    }
}
```

### ApplyPatchOperation

```rust
use openai_agents::editor::{ApplyPatchOperation, ApplyPatchOperationType};

let op = ApplyPatchOperation::new(
    ApplyPatchOperationType::UpdateFile,
    "src/main.rs",
    Some("--- a/src/main.rs\n+++ b/src/main.rs\n...".to_owned()),
);

println!("{}: {}", op.operation_type, op.path);
// => "update_file: src/main.rs"
```

### apply_diff function

The lower-level `apply_diff` function applies a V4A-format diff string to a `&str` of file
content and returns the modified content:

```rust,no_run
use openai_agents::editor::apply_diff;

let original = "line 1\nline 2\nline 3\n";
let diff = "*** Begin Patch\n*** Update File: example.txt\n@@\n line 1\n-line 2\n+line TWO\n line 3\n*** End Patch";
let modified = apply_diff(original, diff)?;
assert!(modified.contains("line TWO"));
```

---

## Retry Policies

`RetryPolicy` provides configurable exponential backoff with optional jitter for any
fallible async operation.

```rust
use openai_agents::retry::RetryPolicy;
use std::time::Duration;

// Default: 3 retries, 1s initial delay, 2x backoff, 30s cap, jitter enabled.
let policy = RetryPolicy::default();

// Custom policy.
let policy = RetryPolicy::new(5)
    .with_initial_delay(Duration::from_millis(500))
    .with_backoff_factor(3.0)
    .with_max_delay(Duration::from_secs(60))
    .with_jitter(true);

// Disable retries entirely.
let no_retry = RetryPolicy::none();
```

### Calculating delays

```rust
use openai_agents::retry::RetryPolicy;
use std::time::Duration;

let policy = RetryPolicy::new(3)
    .with_initial_delay(Duration::from_secs(1))
    .with_backoff_factor(2.0)
    .with_jitter(false);

// Without jitter, delays are deterministic: 1s, 2s, 4s.
let d0 = policy.delay_for_attempt(0); // 1 second
let d1 = policy.delay_for_attempt(1); // 2 seconds
let d2 = policy.delay_for_attempt(2); // 4 seconds
```

### Executing with retry

```rust,no_run
use openai_agents::retry::RetryPolicy;
use std::time::Duration;

let policy = RetryPolicy::new(3).with_jitter(false);

let result = policy.execute(|| async {
    // Any fallible async operation.
    make_api_call().await
}).await?;
```

The `execute` method retries automatically on transient errors (HTTP 429, 5xx, network
timeouts). Non-retryable errors (HTTP 400, authentication failures) are returned immediately.

### Attaching a policy to RunConfig

```rust,no_run
use openai_agents::config::RunConfig;
use openai_agents::retry::RetryPolicy;
use std::time::Duration;

let config = RunConfig::default()
    .with_retry_policy(RetryPolicy::new(5).with_initial_delay(Duration::from_millis(200)));
```

---

## See Also

- [Models and Providers](./models.md) — custom model backends.
- [Tools](./tools.md) — function tools and hosted tools.
- [MCP](./mcp.md) — MCP tool servers.
- [Voice / Realtime](./voice.md) — voice agent support.
