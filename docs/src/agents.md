# Agents

An `Agent<C>` is the core building block of the SDK. It bundles an LLM with a
system prompt, a set of tools, handoff targets, guardrails, and lifecycle hooks.
The generic parameter `C` is your custom context type — use `()` if you do not
need one.

## Creating an agent

All agents are built with `Agent::builder`, which returns an `AgentBuilder<C>`:

```rust
use openai_agents::Agent;

let agent = Agent::<()>::builder("assistant")
    .instructions("You are a helpful assistant.")
    .build();

assert_eq!(agent.name, "assistant");
```

The only required argument is the agent name. Every other field has a sensible
default and can be set on the builder.

## Static instructions

Pass a `&str` or `String` to set a fixed system prompt:

```rust
let agent = Agent::<()>::builder("summariser")
    .instructions("Summarise the user's input in one paragraph.")
    .build();
```

## Dynamic instructions

Use `dynamic_instructions` when the system prompt should change at runtime based
on the context value or the agent's own fields. The closure receives a
`&RunContextWrapper<C>` and a `&Agent<C>`, and must return a boxed async future
resolving to `Result<String>`:

```rust
use openai_agents::Agent;

let agent = Agent::<String>::builder("personalised-agent")
    .dynamic_instructions(|ctx, _agent| {
        let language = ctx.context.clone();
        Box::pin(async move {
            Ok(format!("You are a helpful assistant. Respond in {language}."))
        })
    })
    .build();
```

When the runner calls `get_instructions`, it evaluates the closure with the
current context. For static instructions the closure is never created.

## Model selection

By default an agent uses whichever model is passed to the runner. Override it
per-agent with a model name string or a `ModelRef`:

```rust
use openai_agents::Agent;

let agent = Agent::<()>::builder("gpt4-agent")
    .instructions("You are a reasoning assistant.")
    .model("gpt-4o")   // overrides the model passed to Runner
    .build();
```

`ModelRef` converts automatically from `&str` and `String`, so you can pass
either directly to `.model()`.

## Model settings

Fine-tune sampling parameters for this agent with a `ModelSettings` value. Any
`None` field falls back to the run-level `ModelSettings` in `RunConfig`:

```rust
use openai_agents::{Agent, ModelSettings};

let agent = Agent::<()>::builder("creative-writer")
    .instructions("Write imaginative short stories.")
    .model_settings(ModelSettings::new().with_temperature(0.9))
    .build();
```

See the [Configuration](./config.md) page for all `ModelSettings` fields.

## Tools

Add tools one at a time with `.tool()` or replace the whole list with `.tools()`:

```rust
use openai_agents::{Agent, Tool};
use openai_agents::tool::{WebSearchTool, FileSearchTool};

let agent = Agent::<()>::builder("researcher")
    .instructions("Answer questions using web search and file search.")
    .tool(Tool::WebSearch(WebSearchTool::default()))
    .tool(Tool::FileSearch(FileSearchTool {
        vector_store_ids: vec!["vs_abc123".to_string()],
        max_num_results: Some(5),
    }))
    .build();
```

See [Tools](./tools.md) for function tools and hosted tool details.

## Handoffs

Handoffs let one agent delegate to another. Add target agents with `.handoff()`:

```rust
use openai_agents::{Agent, Handoff};

let billing_agent = Agent::<()>::builder("billing")
    .instructions("Handle billing questions.")
    .build();

let triage_agent = Agent::<()>::builder("triage")
    .instructions("Route the user to the right specialist.")
    .handoff(Handoff::to_agent("billing").build())
    .build();
```

See [Handoffs](./handoffs.md) for details on multi-agent routing.

## Guardrails

Guardrails validate input before the agent runs and output after it finishes:

```rust
use openai_agents::{Agent, InputGuardrail, GuardrailFunctionOutput};

let agent = Agent::<()>::builder("safe-agent")
    .instructions("You are a safe assistant.")
    .input_guardrail(InputGuardrail::new(
        "profanity-check",
        |_ctx, _agent, input| {
            Box::pin(async move {
                // Inspect input; call tripwire if unsafe.
                Ok(GuardrailFunctionOutput::passed(serde_json::json!(null)))
            })
        },
    ))
    .build();
```

See [Guardrails](./guardrails.md) for the full guardrail API.

## Structured output

When you need the agent to produce JSON matching a specific schema, derive
`schemars::JsonSchema` on your output type and call `.output_type::<T>()`:

```rust
use openai_agents::Agent;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct CalendarEvent {
    name: String,
    date: String,
    participants: Vec<String>,
}

let agent = Agent::<()>::builder("event-extractor")
    .instructions("Extract calendar events from the user's message.")
    .output_type::<CalendarEvent>()
    .build();

assert!(agent.output_type.is_some());
```

The SDK generates a strict JSON schema from your type and passes it to the model
so the output is guaranteed to match the schema. After the run, call
`result.final_output_as::<CalendarEvent>()` to deserialize the result.

You can also supply a schema directly:

```rust
use openai_agents::{Agent, agent::OutputSchema};

let schema = OutputSchema::new(
    serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    }),
    true, // strict
);

let agent = Agent::<()>::builder("structured")
    .output_schema(schema)
    .build();
```

## ToolUseBehavior

After the model calls a tool, by default the SDK passes the results back to the
model for a follow-up response. This is `ToolUseBehavior::RunLlmAgain`. The two
alternatives are:

| Variant | Behaviour |
|---------|-----------|
| `RunLlmAgain` | (default) Call the LLM again after tool execution. |
| `StopOnFirstTool` | Use the first tool output directly as the final result. |
| `StopAtTools(names)` | Stop only when one of the named tools is called. |

```rust
use openai_agents::{Agent, ToolUseBehavior};

// Return the tool result directly without a second LLM call.
let agent = Agent::<()>::builder("extractor")
    .instructions("Extract the requested data and return it.")
    .tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
    .build();

// Stop only when the "save_record" tool is called.
let agent = Agent::<()>::builder("pipeline")
    .instructions("Process data, then save it.")
    .tool_use_behavior(ToolUseBehavior::StopAtTools(vec![
        "save_record".to_string(),
    ]))
    .build();
```

## Lifecycle hooks

Implement the `AgentHooks<C>` trait to receive callbacks at each stage of agent
execution (before/after each turn, before/after tool calls, on handoffs, etc.):

```rust
use openai_agents::{Agent, agent::AgentHooks};
use async_trait::async_trait;

struct LoggingHooks;

#[async_trait]
impl AgentHooks<()> for LoggingHooks {
    // Override individual methods to add logging or metrics.
    // All methods have default no-op implementations.
}

let agent = Agent::<()>::builder("observable")
    .instructions("Be helpful.")
    .hooks(LoggingHooks)
    .build();
```

## Agent fields reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `String` | (required) | Identifies the agent in logs and handoffs. |
| `instructions` | `Option<Instructions<C>>` | `None` | System prompt (static or dynamic). |
| `handoff_description` | `Option<String>` | `None` | Shown to a routing agent when selecting this target. |
| `model` | `Option<ModelRef>` | `None` | Overrides the model passed to the runner. |
| `model_settings` | `ModelSettings` | all `None` | Temperature, top-p, etc. |
| `tools` | `Vec<Tool<C>>` | empty | Function and hosted tools. |
| `handoffs` | `Vec<Handoff<C>>` | empty | Agents this agent may hand off to. |
| `input_guardrails` | `Vec<InputGuardrail<C>>` | empty | Pre-run validation. |
| `output_guardrails` | `Vec<OutputGuardrail<C>>` | empty | Post-run validation. |
| `output_type` | `Option<OutputSchema>` | `None` | Structured output schema. |
| `hooks` | `Option<Box<dyn AgentHooks<C>>>` | `None` | Per-agent lifecycle hooks. |
| `tool_use_behavior` | `ToolUseBehavior` | `RunLlmAgain` | Controls how tool results are processed. |
| `reset_tool_choice` | `bool` | `true` | Resets `tool_choice` after the first LLM call. |
