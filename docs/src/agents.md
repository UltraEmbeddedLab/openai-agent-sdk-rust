# Agents

An agent is the core building block of the SDK. It wraps an LLM with instructions, tools, guardrails, and handoffs.

## Creating an Agent

Use the builder pattern to create agents:

```rust
use openai_agents::Agent;

let agent = Agent::builder()
    .name("My Agent")
    .instructions("You are a helpful assistant.")
    .build();
```

## Configuration

| Field | Type | Description |
|-------|------|-------------|
| `name` | `String` | Display name for the agent |
| `instructions` | `Prompt` | System prompt (static string or dynamic function) |
| `model` | `Option<String>` | Model to use (defaults to config) |
| `tools` | `Vec<Box<dyn Tool<C>>>` | Tools available to the agent |
| `handoffs` | `Vec<Handoff<C>>` | Agents this agent can delegate to |
| `input_guardrails` | `Vec<InputGuardrail<C>>` | Input validation |
| `output_guardrails` | `Vec<OutputGuardrail<C>>` | Output validation |
| `model_settings` | `ModelSettings` | Temperature, top_p, etc. |
| `hooks` | `Option<Box<dyn AgentHooks<C>>>` | Lifecycle callbacks |

## Dynamic Instructions

Instructions can be generated dynamically based on context:

```rust
let agent = Agent::builder()
    .name("Dynamic Agent")
    .instructions_fn(|ctx| {
        format!("You are helping user {}. Be concise.", ctx.user_id)
    })
    .build();
```

## Output Types

By default, agents return plain text. You can specify a structured output type:

```rust
use serde::Deserialize;
use schemars::JsonSchema;

#[derive(Deserialize, JsonSchema)]
struct WeatherReport {
    location: String,
    temperature: f64,
    conditions: String,
}

let agent = Agent::<(), WeatherReport>::builder()
    .name("Weather Agent")
    .instructions("Return weather as structured JSON.")
    .build();
```
