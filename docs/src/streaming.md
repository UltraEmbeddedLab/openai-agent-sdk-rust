# Streaming

`Runner::run_streamed` drives the agent loop in a background Tokio task and
delivers events through a channel. This lets you render text as it arrives,
show progress indicators for tool calls, and react to agent handoffs in real
time.

## Starting a streaming run

```rust
pub fn run_streamed<C: Send + Sync + 'static>(
    agent: &'static Agent<C>,
    input: impl Into<InputContent> + Send + 'static,
    context: C,
    model: Arc<dyn Model>,
    hooks: Option<Arc<dyn RunHooks<C>>>,
    config: Option<RunConfig>,
) -> RunResultStreaming
```

The function returns immediately. The background task begins as soon as
`run_streamed` is called; you do not need to await it.

The `agent` parameter must be `'static` because it is moved into the spawned
task. Use `Box::leak` for quick prototypes or store agents in a
`once_cell::sync::Lazy` / `std::sync::OnceLock` for production code:

```rust
use std::sync::Arc;
use openai_agents::{Agent, Runner};
use openai_agents::models::openai_responses::OpenAIResponsesModel;

let agent: &'static Agent<()> = Box::leak(Box::new(
    Agent::<()>::builder("streaming-demo")
        .instructions("You are a concise assistant. Answer in 2–3 sentences.")
        .build(),
));

let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

let mut streaming = Runner::run_streamed(
    agent,
    "What is Rust?",
    (),
    model,
    None,
    None,
);
```

## Consuming events

Call `streaming.stream_events()` to get a `Pin<Box<dyn Stream<Item = StreamEvent>>>`,
then iterate with `StreamExt::next` from the `tokio-stream` crate:

```toml
# Cargo.toml
[dependencies]
tokio-stream = "0.1"
```

```rust
use tokio_stream::StreamExt;
use openai_agents::stream_events::{StreamEvent, RunItemEventName};

let mut stream = streaming.stream_events();
while let Some(event) = stream.next().await {
    match event {
        StreamEvent::RawResponse(raw) => {
            // Raw model delta — useful for token-by-token text rendering.
            let ty = raw.get("type").and_then(|v| v.as_str()).unwrap_or("?");
            print!("[raw:{ty}] ");
        }
        StreamEvent::RunItemCreated { name, item } => {
            println!("[{name}] {item:?}");
        }
        StreamEvent::AgentUpdated { new_agent_name } => {
            println!("Active agent is now: {new_agent_name}");
        }
        _ => {}
    }
}
// Drop the stream before accessing other fields on streaming.
drop(stream);
```

`stream_events()` takes the underlying channel receiver out of the struct on the
first call. Subsequent calls return an empty stream without panicking.

## StreamEvent variants

| Variant | When it occurs | Key fields |
|---------|---------------|------------|
| `RawResponse(serde_json::Value)` | Every raw event from the model provider. | The raw JSON object. |
| `RunItemCreated { name, item }` | Whenever the runner produces a new `RunItem`. | `name: RunItemEventName`, `item: RunItem`. |
| `AgentUpdated { new_agent_name }` | When a handoff causes the active agent to change. | `new_agent_name: String`. |

### RunItemEventName values

`RunItemEventName` implements `Display` with the same `snake_case` strings as
the Python SDK:

| Variant | Display string |
|---------|---------------|
| `MessageOutputCreated` | `message_output_created` |
| `HandoffRequested` | `handoff_requested` |
| `HandoffOccurred` | `handoff_occurred` |
| `ToolCalled` | `tool_called` |
| `ToolOutput` | `tool_output` |
| `ReasoningItemCreated` | `reasoning_item_created` |

## Reading the result after streaming

After the stream is exhausted, the struct fields are populated by the background
task. Check `is_complete` to confirm the run finished successfully:

```rust
drop(stream); // required before re-borrowing streaming_result

if streaming.is_complete {
    println!("Final output: {}", streaming.final_output);
    println!(
        "Tokens: input={} output={}",
        streaming.usage.input_tokens,
        streaming.usage.output_tokens,
    );
} else {
    println!("Run did not complete (cancelled or error).");
}
```

## Cancellation

Send a cancellation signal at any time:

```rust
streaming.cancel();
```

The background task checks for the signal between turns and exits early. The
stream will drain any already-sent events and then close. Calling `cancel()` a
second time is a no-op.

Alternatively, just drop the `RunResultStreaming` — the background task will
receive a send error on its next event and exit.

## Complete example

The following mirrors `examples/streaming.rs` in the repository:

```rust
use std::sync::Arc;
use tokio_stream::StreamExt;
use openai_agents::{Agent, Runner};
use openai_agents::models::openai_responses::OpenAIResponsesModel;
use openai_agents::stream_events::StreamEvent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent: &'static Agent<()> = Box::leak(Box::new(
        Agent::<()>::builder("assistant")
            .instructions("Answer concisely.")
            .build(),
    ));

    let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

    let mut streaming = Runner::run_streamed(
        agent,
        "What is Rust (the programming language)?",
        (),
        model,
        None,
        None,
    );

    let mut event_count = 0usize;
    let mut stream = streaming.stream_events();
    while let Some(event) = stream.next().await {
        event_count += 1;
        match &event {
            StreamEvent::RawResponse(raw) => {
                let ty = raw.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
                println!("[{event_count}] RawResponse: {ty}");
            }
            StreamEvent::RunItemCreated { name, .. } => {
                println!("[{event_count}] RunItemCreated: {name}");
            }
            StreamEvent::AgentUpdated { new_agent_name } => {
                println!("[{event_count}] AgentUpdated: {new_agent_name}");
            }
            _ => {}
        }
    }
    drop(stream);

    println!("\nFinal output: {}", streaming.final_output);
    println!("Complete: {}", streaming.is_complete);
    println!("Total events: {event_count}");

    Ok(())
}
```

Run this example (from the repository) with:

```bash
OPENAI_API_KEY=sk-... cargo run --example streaming
```
