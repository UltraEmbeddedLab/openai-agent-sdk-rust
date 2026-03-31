# OpenAI Agents SDK for Rust

[![CI](https://github.com/UltraEmbeddedLab/openai-agent-sdk-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/UltraEmbeddedLab/openai-agent-sdk-rust/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/openai-agents.svg)](https://crates.io/crates/openai-agents)
[![Documentation](https://docs.rs/openai-agents/badge.svg)](https://docs.rs/openai-agents)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, ergonomic Rust framework for building multi-agent workflows powered by OpenAI. This is a Rust port of the official [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python), bringing the same powerful abstractions to the Rust ecosystem with full type safety and async support.

## Features

- **Agents**: LLMs configured with instructions, tools, guardrails, and handoffs
- **Tools**: Function-based tools with automatic JSON Schema generation, plus hosted tools and MCP support
- **Handoffs**: Multi-agent routing and delegation with customizable history filtering
- **Guardrails**: Input and output validation that can intercept and transform agent behavior
- **Sessions**: Automatic conversation history management with pluggable storage backends
- **Streaming**: Real-time streaming of agent responses via `Stream` trait
- **Tracing**: Built-in distributed tracing for debugging and optimization
- **Human-in-the-loop**: Tool approval workflows for sensitive operations
- **Type safety**: Full compile-time type checking with generic context types

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
openai-agents = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Hello World

```rust
use openai_agents::{Agent, Runner};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder()
        .name("Greeter")
        .instructions("You are a helpful assistant. Greet the user warmly.")
        .build();

    let result = Runner::run(&agent, "Hello!").await?;
    println!("{}", result.final_output());

    Ok(())
}
```

### Using Tools

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

### Multi-Agent Handoffs

```rust
use openai_agents::{Agent, Runner, handoff};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let spanish_agent = Agent::builder()
        .name("Spanish Agent")
        .instructions("You only speak Spanish. Help the user in Spanish.")
        .build();

    let triage_agent = Agent::builder()
        .name("Triage Agent")
        .instructions("Determine the user's language and hand off accordingly.")
        .handoff(handoff(&spanish_agent))
        .build();

    let result = Runner::run(&triage_agent, "Hola, necesito ayuda").await?;
    println!("{}", result.final_output());

    Ok(())
}
```

### Streaming

```rust
use openai_agents::{Agent, Runner};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder()
        .name("Storyteller")
        .instructions("Tell a short story.")
        .build();

    let mut stream = Runner::run_streamed(&agent, "Tell me a story").await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Text(text) => print!("{text}"),
            StreamEvent::Done(result) => {
                println!("\n\nTokens used: {}", result.usage().total_tokens);
            }
            _ => {}
        }
    }

    Ok(())
}
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `mcp` | Model Context Protocol integration |
| `voice` | Voice/realtime agent support |
| `tracing-otlp` | OpenTelemetry tracing export |
| `sqlite-session` | SQLite-backed session storage |

## Documentation

- [User Guide](https://UltraEmbeddedLab.github.io/openai-agent-sdk-rust/) — comprehensive documentation
- [API Reference](https://docs.rs/openai-agents) — auto-generated from doc comments
- [Examples](./examples/) — runnable code samples

## Architecture

The SDK mirrors the architecture of the Python reference implementation:

```
Agent<C> ──→ Runner::run() ──→ RunResult<O>
  │                │
  ├── Tools        ├── Model call
  ├── Guardrails   ├── Tool execution
  ├── Handoffs     ├── Handoff resolution
  └── Hooks        └── Guardrail validation
```

### Core Abstractions

| Abstraction | Description |
|-------------|-------------|
| `Agent<C>` | An LLM configured with instructions, tools, guardrails, and handoffs. Generic over context `C`. |
| `Runner` | Entry point for executing agents. Handles the run loop, tool calls, and handoffs. |
| `Tool` | Trait for tools the agent can invoke. Includes `FunctionTool` for wrapping Rust functions. |
| `Guardrail` | Input/output validation that can intercept agent behavior. |
| `Handoff` | Defines how one agent can delegate to another. |
| `Session` | Trait for persisting conversation history across runs. |
| `Model` | Trait abstracting the LLM provider (OpenAI Responses API, Chat Completions, etc.). |
| `RunResult<O>` | The outcome of a run, containing the final output, items, and usage statistics. |

## Contributing

```bash
# Setup
git clone https://github.com/UltraEmbeddedLab/openai-agent-sdk-rust.git
cd openai-agent-sdk-rust

# Development cycle
cargo fmt --check
cargo clippy -- -W clippy::all
cargo test

# Build docs
mdbook build docs/
```

## License

MIT License. See [LICENSE](./LICENSE) for details.

## Acknowledgments

This project is a Rust port of the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python). All credit for the original architecture and design goes to the OpenAI team.
