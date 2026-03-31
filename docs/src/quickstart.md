# Quick Start

Get the OpenAI Agents Rust SDK running in a few minutes.

## Prerequisites

- Rust 1.85+ (install via [rustup.rs](https://rustup.rs))
- An OpenAI API key

## Add the dependency

Add the following to your `Cargo.toml`:

```toml
[dependencies]
openai-agents = "0.1"
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

## Set your API key

The SDK reads the key from the `OPENAI_API_KEY` environment variable by default:

```bash
export OPENAI_API_KEY="sk-..."
```

You can also set it in code before constructing any model:

```rust
openai_agents::set_default_openai_key("sk-...");
```

## Hello world

Create `src/main.rs`:

```rust
use std::sync::Arc;
use openai_agents::{Agent, Runner};
use openai_agents::models::openai_responses::OpenAIResponsesModel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::<()>::builder("assistant")
        .instructions("You are a helpful assistant.")
        .build();

    let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

    let result = Runner::run_with_model(
        &agent,
        "Say hello!",
        (),    // context — () when no custom context is needed
        model,
        None,  // hooks
        None,  // config (uses defaults: 10 max turns)
    )
    .await?;

    println!("{}", result.final_output);
    Ok(())
}
```

Run it:

```bash
cargo run
```

## Adding a tool

Tools let the agent call your Rust functions during execution. Define the input
type with `serde::Deserialize` and `schemars::JsonSchema`, then register the tool
with the agent builder:

```toml
# Cargo.toml
[dependencies]
schemars = "1"
serde = { version = "1", features = ["derive"] }
```

```rust
use std::sync::Arc;
use schemars::JsonSchema;
use serde::Deserialize;
use openai_agents::{Agent, Runner, Tool};
use openai_agents::items::ToolOutput;
use openai_agents::tool::{ToolContext, function_tool};
use openai_agents::models::openai_responses::OpenAIResponsesModel;

#[derive(Deserialize, JsonSchema)]
struct WeatherInput {
    /// The city to look up.
    city: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let weather_tool = function_tool::<(), WeatherInput, _, _>(
        "get_weather",
        "Get the current weather for a city.",
        |_ctx: ToolContext<()>, input: WeatherInput| async move {
            Ok(ToolOutput::Text(format!("Sunny, 22 °C in {}", input.city)))
        },
    )?;

    let agent = Agent::<()>::builder("weather-bot")
        .instructions("Help users check the weather.")
        .tool(Tool::Function(weather_tool))
        .build();

    let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

    let result = Runner::run_with_model(
        &agent,
        "What is the weather in Paris?",
        (),
        model,
        None,
        None,
    )
    .await?;

    println!("{}", result.final_output);
    Ok(())
}
```

## What's next

- [Agents](./agents.md) — configure agents in depth.
- [Running Agents](./running_agents.md) — `run_with_model`, `run_with_agents`, and `run_streamed`.
- [Tools](./tools.md) — the full tool system.
- [Streaming](./streaming.md) — receive events as the agent generates output.
- [Results](./results.md) — interpret `RunResult` and structured output.
- [Configuration](./config.md) — tune `ModelSettings` and `RunConfig`.
