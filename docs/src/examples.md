# Examples

All examples live in the `examples/` directory of the repository. Each is a standalone binary
with its own `#[tokio::main]` entry point. Every example supports two modes:

- **Real API mode** (default) — uses `gpt-4o-mini` via the OpenAI Responses API. Requires the
  `OPENAI_API_KEY` environment variable.
- **Mock fallback** — when no API key is set, a built-in mock model produces deterministic
  output so the example runs without network access.

Run any example with:

```bash
OPENAI_API_KEY=sk-... cargo run --example <name>
```

---

## hello_world

**File:** `examples/hello_world.rs`

The simplest possible agent: one instruction, one question, one answer. Use this as a
template when starting a new project.

**Demonstrates:**
- Creating an agent with `Agent::<()>::builder`.
- Running it to completion with `Runner::run_with_model`.
- Reading `result.final_output` and `result.usage`.

```bash
OPENAI_API_KEY=sk-... cargo run --example hello_world
```

**Key snippet:**

```rust,no_run
use openai_agents::{Agent, Runner};
use openai_agents::models::openai_responses::OpenAIResponsesModel;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::<()>::builder("greeter")
        .instructions("You are a cheerful assistant. Greet the user warmly in one sentence.")
        .build();

    let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

    let result = Runner::run_with_model(&agent, "Hello!", (), model, None, None).await?;

    println!("Final output: {}", result.final_output);
    println!(
        "Tokens used: input={}, output={}, total={}",
        result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens
    );

    Ok(())
}
```

---

## tools

**File:** `examples/tools.rs`

Attach custom function tools to an agent and let the model decide when to call them. Two
tools are defined: `get_weather` (returns fake weather data) and `calculate` (evaluates a
math expression stub).

**Demonstrates:**
- Defining typed tool inputs with `#[derive(Deserialize, JsonSchema)]`.
- Creating a tool with `function_tool`.
- Attaching tools to an agent with `.tool(Tool::Function(...))`.
- Inspecting `result.new_items` to see tool calls in the trace.

```bash
OPENAI_API_KEY=sk-... cargo run --example tools
```

**Key snippet:**

```rust,no_run
use openai_agents::{Agent, Runner, Tool};
use openai_agents::items::ToolOutput;
use openai_agents::tool::{ToolContext, function_tool};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherInput {
    /// The city to get weather for.
    city: String,
}

let weather_tool = function_tool::<(), WeatherInput, _, _>(
    "get_weather",
    "Get the current weather for a city.",
    |_ctx: ToolContext<()>, input: WeatherInput| async move {
        Ok(ToolOutput::Text(format!(
            "Weather in {}: Sunny, 22°C",
            input.city
        )))
    },
)?;

let agent = Agent::<()>::builder("assistant")
    .instructions("Use get_weather for weather questions.")
    .tool(Tool::Function(weather_tool))
    .build();
```

---

## handoffs

**File:** `examples/handoffs.rs`

A triage agent that delegates to a billing agent or a support agent based on the user's
question. This demonstrates the multi-agent routing pattern at its most basic level.

**Demonstrates:**
- Creating multiple agents with different specializations.
- Building a `Handoff` and attaching it with `.handoff(...)`.
- How the runner transparently switches to the target agent when a handoff tool is called.
- Inspecting which agent produced the final output.

```bash
OPENAI_API_KEY=sk-... cargo run --example handoffs
```

**Key snippet:**

```rust,no_run
use openai_agents::{Agent, Runner};
use openai_agents::handoffs::Handoff;

// Specialist agents.
let billing_agent = Agent::<()>::builder("billing-agent")
    .instructions("You handle billing questions.")
    .build();

let support_agent = Agent::<()>::builder("support-agent")
    .instructions("You handle technical support questions.")
    .build();

// Triage agent with handoffs to specialists.
let triage_agent = Agent::<()>::builder("triage-agent")
    .instructions("Route to billing or support based on the user's question.")
    .handoff(Handoff::new("billing-agent", billing_agent))
    .handoff(Handoff::new("support-agent", support_agent))
    .build();

let result = Runner::run(&triage_agent, "I was charged twice.", ()).await?;
println!("Answered by: {}", result.last_agent.name);
println!("Answer: {}", result.final_output);
```

---

## guardrails

**File:** `examples/guardrails.rs`

Attach input and output guardrails to an agent. Guardrails are validation functions that run
before the model processes a request (input guardrail) or after it generates a response
(output guardrail). A guardrail can raise a "tripwire" to block the request.

**Demonstrates:**
- Implementing `InputGuardrail<C>` to reject sensitive input.
- Implementing `OutputGuardrail<C>` to validate model output.
- How `AgentError::InputGuardrailTriggered` and `AgentError::OutputGuardrailTriggered` are
  returned when a tripwire fires.
- Using `GuardrailFunctionOutput` to carry optional rejection messages.

```bash
OPENAI_API_KEY=sk-... cargo run --example guardrails
```

**Key snippet:**

```rust,no_run
use openai_agents::{Agent, AgentError, Runner};
use openai_agents::guardrail::{GuardrailFunctionOutput, InputGuardrail};
use openai_agents::context::RunContextWrapper;
use async_trait::async_trait;

struct NoProfanityGuardrail;

#[async_trait]
impl InputGuardrail<()> for NoProfanityGuardrail {
    async fn run(
        &self,
        _ctx: &RunContextWrapper<()>,
        _agent: &openai_agents::Agent<()>,
        input: &str,
    ) -> openai_agents::Result<GuardrailFunctionOutput> {
        if input.contains("badword") {
            return Ok(GuardrailFunctionOutput::tripwire("Profanity detected"));
        }
        Ok(GuardrailFunctionOutput::pass())
    }
}

let agent = Agent::<()>::builder("assistant")
    .instructions("You are a helpful assistant.")
    .input_guardrail(NoProfanityGuardrail)
    .build();

match Runner::run(&agent, "How's the weather?", ()).await {
    Ok(result) => println!("{}", result.final_output),
    Err(AgentError::InputGuardrailTriggered { guardrail_name, .. }) => {
        println!("Blocked by guardrail: {}", guardrail_name);
    }
    Err(e) => return Err(e.into()),
}
```

---

## streaming

**File:** `examples/streaming.rs`

Use `Runner::run_streamed` to receive events in real time as the agent processes. Events
include raw model token deltas, notifications when run items are created (message, tool call,
tool output), and agent-change events during handoffs.

**Demonstrates:**
- Calling `Runner::run_streamed` to get a `RunResultStreaming`.
- Iterating the event stream with `stream.next().await`.
- The `StreamEvent` enum variants: `RawModelStreamEvent`, `RunItemCreated`,
  `RunItemCompleted`, `AgentChanged`.
- Collecting final output after the stream is exhausted.

```bash
OPENAI_API_KEY=sk-... cargo run --example streaming
```

**Key snippet:**

```rust,no_run
use openai_agents::{Agent, Runner};
use openai_agents::stream_events::StreamEvent;
use tokio_stream::StreamExt;

let agent: &'static Agent<()> = Box::leak(Box::new(
    Agent::<()>::builder("assistant")
        .instructions("Be concise.")
        .build(),
));

let mut streaming = Runner::run_streamed(agent, "Tell me a fact.", ()).await?;

while let Some(event) = streaming.stream.next().await {
    match event? {
        StreamEvent::RawModelStreamEvent { data, .. } => {
            print!("{}", data); // stream tokens to stdout
        }
        StreamEvent::RunItemCreated { item } => {
            println!("\n[Item created: {:?}]", item);
        }
        _ => {}
    }
}

let result = streaming.final_result().await?;
println!("\nFinal: {}", result.final_output);
```

---

## See Also

- [Quick Start](./quickstart.md) — first steps with the SDK.
- [Agents](./agents.md) — full agent configuration reference.
- [Tools](./tools.md) — detailed tool documentation.
- [Handoffs](./handoffs.md) — multi-agent routing patterns.
- [Guardrails](./guardrails.md) — validation and safety.
- [Streaming](./streaming.md) — streaming execution.
