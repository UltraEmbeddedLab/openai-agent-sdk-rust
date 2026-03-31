# Results

After `Runner::run_with_model` or `Runner::run_with_agents` completes, you
receive a `RunResult`. For streaming runs `Runner::run_streamed` returns a
`RunResultStreaming` immediately; its fields are populated as the stream is
consumed.

## RunResult fields

```rust
pub struct RunResult {
    /// The original input provided to the run.
    pub input: InputContent,
    /// All items generated during the run (messages, tool calls, handoffs, etc.).
    pub new_items: Vec<RunItem>,
    /// Raw model responses from each LLM call.
    pub raw_responses: Vec<ModelResponse>,
    /// The final output value (JSON).
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
```

### Reading the final output

`final_output` is `serde_json::Value`. For plain-text agents it holds the
response as a JSON string value:

```rust
println!("{}", result.final_output);
// or, as a plain Rust String:
if let Some(text) = result.final_output.as_str() {
    println!("{text}");
}
```

## final_output_as\<T\>()

When an agent has a structured `output_type`, deserialize the result into your
type:

```rust
use schemars::JsonSchema;
use serde::Deserialize;
use openai_agents::Agent;

#[derive(Deserialize, JsonSchema, Debug)]
struct CalendarEvent {
    name: String,
    date: String,
}

let agent = Agent::<()>::builder("extractor")
    .instructions("Extract calendar events.")
    .output_type::<CalendarEvent>()
    .build();

// After running...
let event: CalendarEvent = result.final_output_as()?;
println!("{} on {}", event.name, event.date);
```

`final_output_as` clones the JSON value and calls `serde_json::from_value`.
It returns `AgentError::Serialization` if deserialization fails.

## to_input_list() for multi-turn conversations

`to_input_list()` converts the raw model responses into a `Vec<ResponseInputItem>`
you can pass back as input in the next turn:

```rust
// Turn 1.
let result1 = Runner::run_with_model(
    &agent, "My name is Alice.", (), Arc::clone(&model), None, None,
).await?;

// Build turn 2 input: history + new user message.
let mut history = result1.to_input_list();
history.push(serde_json::json!({
    "role": "user",
    "content": "What is my name?"
}));

let result2 = Runner::run_with_model(
    &agent,
    openai_agents::items::InputContent::Items(history),
    (),
    Arc::clone(&model),
    None,
    None,
).await?;
```

The list contains all output items from `raw_responses` in order. The original
user input from `result1.input` is not included; append it yourself if you need
it.

## last_response_id()

Returns the `response_id` from the last model response, useful for referencing
the exchange in subsequent API calls:

```rust
if let Some(id) = result.last_response_id() {
    println!("Last response id: {id}");
}
```

## Usage tracking

`result.usage` accumulates token counts across every LLM call in the run:

```rust
println!(
    "input={} output={} total={} requests={}",
    result.usage.input_tokens,
    result.usage.output_tokens,
    result.usage.total_tokens,
    result.usage.requests,
);
```

The `Usage` struct also has `input_tokens_details.cached_tokens` and
`output_tokens_details.reasoning_tokens` for models that expose those metrics.

## RunResultStreaming

`RunResultStreaming` is returned by `Runner::run_streamed`. Its fields are
populated as the background task processes the agent loop:

```rust
pub struct RunResultStreaming {
    pub input:                    InputContent,
    pub new_items:                Vec<RunItem>,
    pub raw_responses:            Vec<ModelResponse>,
    pub final_output:             serde_json::Value,
    pub current_agent_name:       String,
    pub current_turn:             u32,
    pub max_turns:                u32,
    pub is_complete:              bool,
    pub input_guardrail_results:  Vec<InputGuardrailResult>,
    pub output_guardrail_results: Vec<OutputGuardrailResult>,
    pub usage:                    Usage,
}
```

### stream_events()

Consumes a `Pin<Box<dyn Stream<Item = StreamEvent>>>`. The stream ends when
the background task finishes or when you cancel it. Only one call to
`stream_events()` is meaningful; subsequent calls return an empty stream:

```rust
use tokio_stream::StreamExt;
use openai_agents::stream_events::StreamEvent;

let mut stream = streaming_result.stream_events();
while let Some(event) = stream.next().await {
    match event {
        StreamEvent::RunItemCreated { name, .. } => println!("{name}"),
        _ => {}
    }
}
drop(stream);

// Fields are now populated.
println!("{}", streaming_result.final_output);
println!("complete: {}", streaming_result.is_complete);
```

### cancel()

Send a cancellation signal to the background task:

```rust
streaming_result.cancel();
```

After cancellation the stream ends. Calling `cancel()` again is a no-op.

### final_output_as\<T\>()

Works the same as on `RunResult`, but should only be called after the stream is
fully consumed:

```rust
let event: CalendarEvent = streaming_result.final_output_as()?;
```

## RunItem enum

`result.new_items` is a `Vec<RunItem>`. Each variant wraps a specific item type
carrying the agent name and raw API item:

| Variant | Contents |
|---------|----------|
| `MessageOutput(MessageOutputItem)` | A message from the model. |
| `ToolCall(ToolCallItem)` | A tool call requested by the model. |
| `ToolCallOutput(ToolCallOutputItem)` | The result of executing a tool. |
| `HandoffCall(HandoffCallItem)` | A handoff requested by the model. |
| `HandoffOutput(HandoffOutputItem)` | Confirmation a handoff occurred. |
| `Reasoning(ReasoningItem)` | A reasoning step from the model. |

```rust
for item in &result.new_items {
    match item {
        openai_agents::items::RunItem::MessageOutput(m) => {
            println!("[{}] message", m.agent_name);
        }
        openai_agents::items::RunItem::ToolCall(t) => {
            println!("[{}] tool call", t.agent_name);
        }
        _ => {}
    }
}
```
