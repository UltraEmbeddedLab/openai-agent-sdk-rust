# Quick Start

Get the OpenAI Agents Rust SDK running in 5 minutes.

## Prerequisites

- Rust 1.85+ installed ([rustup.rs](https://rustup.rs))
- An OpenAI API key

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
openai-agents = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

## Hello World

Create `src/main.rs`:

```rust
use openai_agents::{Agent, Runner};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder()
        .name("Assistant")
        .instructions("You are a helpful assistant.")
        .build();

    let result = Runner::run(&agent, "Say hello!").await?;
    println!("{}", result.final_output());

    Ok(())
}
```

Run it:

```bash
cargo run
```

## Add a tool

```rust
use openai_agents::{Agent, Runner, function_tool};

#[function_tool]
/// Get the current weather for a location.
async fn get_weather(location: String) -> String {
    format!("The weather in {location} is sunny, 22°C")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder()
        .name("WeatherBot")
        .instructions("Help users check the weather.")
        .tool(get_weather())
        .build();

    let result = Runner::run(&agent, "What's the weather in Paris?").await?;
    println!("{}", result.final_output());

    Ok(())
}
```

## Add a handoff

```rust
use openai_agents::{Agent, Runner, handoff};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let expert = Agent::builder()
        .name("Expert")
        .instructions("You are a Rust expert. Help with Rust questions.")
        .build();

    let triage = Agent::builder()
        .name("Triage")
        .instructions("Route Rust questions to the expert.")
        .handoff(handoff(&expert))
        .build();

    let result = Runner::run(&triage, "How do lifetimes work in Rust?").await?;
    println!("{}", result.final_output());

    Ok(())
}
```

## Next steps

- [Agents](./agents.md) — configure agents in depth.
- [Tools](./tools.md) — learn about the tool system.
- [Streaming](./streaming.md) — stream agent responses in real time.
- [Examples](./examples.md) — browse more code samples.
