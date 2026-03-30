# OpenAI Agents SDK for Rust

A lightweight, ergonomic Rust framework for building multi-agent workflows powered by OpenAI.

This is a Rust port of the official [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python), bringing the same powerful abstractions to the Rust ecosystem with full type safety and async support.

## Why Rust?

- **Type safety**: Catch errors at compile time, not runtime. Generic context types flow through the entire agent system.
- **Performance**: Zero-cost abstractions, no garbage collector, minimal overhead.
- **Fearless concurrency**: Rust's ownership model prevents data races in multi-agent workflows.
- **Async-native**: Built on `tokio` with first-class `Stream` support for real-time streaming.

## Key Features

| Feature | Description |
|---------|-------------|
| **Agents** | LLMs configured with instructions, tools, guardrails, and handoffs |
| **Tools** | Function-based tools with automatic JSON Schema generation |
| **Handoffs** | Multi-agent routing and delegation |
| **Guardrails** | Input and output validation |
| **Sessions** | Automatic conversation history management |
| **Streaming** | Real-time streaming via `Stream` trait |
| **Tracing** | Built-in distributed tracing |
| **Human-in-the-loop** | Tool approval workflows |

## Architecture Overview

```
Agent<C> ──→ Runner::run() ──→ RunResult<O>
  │                │
  ├── Tools        ├── Model call
  ├── Guardrails   ├── Tool execution
  ├── Handoffs     ├── Handoff resolution
  └── Hooks        └── Guardrail validation
```

The SDK follows a simple execution loop:

1. Create an `Agent` with instructions, tools, guardrails, and handoffs.
2. Call `Runner::run()` with the agent and user input.
3. The runner calls the model, executes tools, resolves handoffs, and validates outputs.
4. Returns a `RunResult` with the final output, conversation items, and usage statistics.

## Next Steps

- [Quick Start](./quickstart.md) — get up and running in 5 minutes.
- [Agents](./agents.md) — learn how to configure agents.
- [Tools](./tools.md) — add capabilities to your agents.
- [Examples](./examples.md) — browse runnable code samples.
