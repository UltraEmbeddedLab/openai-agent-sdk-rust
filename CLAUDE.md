# OpenAI Agents SDK for Rust

Read the AGENTS.md file for detailed contributor instructions.

## Quick Reference

### Project Overview
This is a Rust port of the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python). The goal is to provide an idiomatic Rust implementation that mirrors the Python SDK's architecture and features while leveraging Rust's type system, ownership model, and async ecosystem.

### Architecture Principles
- **Mirror the Python SDK**: Follow the same module structure, abstractions, and API surface as the Python reference implementation at `F:\forks\openai-agents-python`.
- **Idiomatic Rust**: Use traits instead of ABC classes, enums instead of union types, `Result<T, E>` instead of exceptions, builder patterns for complex constructors.
- **Async-first**: All I/O operations use `tokio` async runtime. Streaming uses `tokio_stream::Stream`.
- **Type safety**: Leverage Rust's type system fully — generics for context types, `PhantomData` where needed, sealed traits for internal abstractions.
- **Zero unsafe**: The `unsafe_code = "forbid"` lint is enabled globally.

### Core Module Mapping (Python → Rust)
| Python Module | Rust Module | Key Types |
|---|---|---|
| `agent.py` | `src/agent.rs` | `Agent<C>`, `AgentBuilder<C>` |
| `run.py` | `src/runner.rs` | `Runner`, `AgentRunner` |
| `tool.py` | `src/tool.rs` | `Tool` trait, `FunctionTool<C>`, `ToolOutput` |
| `items.py` | `src/items.rs` | `RunItem` enum, `MessageOutputItem`, `ToolCallItem` |
| `result.py` | `src/result.rs` | `RunResult<O>`, `RunResultStreaming<O>` |
| `guardrail.py` | `src/guardrail.rs` | `InputGuardrail<C>`, `OutputGuardrail<C>` |
| `handoffs/` | `src/handoffs/` | `Handoff<C>`, `HandoffInputData` |
| `models/interface.py` | `src/models/mod.rs` | `Model` trait, `ModelProvider` trait |
| `models/openai_responses.py` | `src/models/openai_responses.rs` | `OpenAIResponsesModel` |
| `models/openai_chatcompletions.py` | `src/models/openai_chatcompletions.rs` | `OpenAIChatCompletionsModel` |
| `memory/` | `src/memory/` | `Session` trait, `SqliteSession` |
| `run_internal/` | `src/run_internal/` | `RunLoop`, `ToolExecution`, `TurnResolution` |
| `tracing/` | `src/tracing/` | Tracing spans, exporters |
| `mcp/` | `src/mcp/` | MCP client integration |
| `stream_events.py` | `src/stream_events.rs` | `StreamEvent` enum |
| `run_config.py` | `src/config.rs` | `RunConfig`, `ModelSettings` |
| `lifecycle.py` | `src/lifecycle.rs` | `AgentHooks<C>` trait, `RunHooks<C>` trait |
| `exceptions.py` | `src/error.rs` | `AgentError` enum |
| `function_schema.py` | `src/schema.rs` | JSON Schema generation via `schemars` |

### Build & Test Commands
```bash
cargo build                    # Build the library
cargo test                     # Run all tests
cargo test -- --nocapture      # Run tests with output
cargo clippy -- -W clippy::all # Lint
cargo fmt --check              # Check formatting
cargo doc --no-deps --open     # Build and open docs
cargo bench                    # Run benchmarks
```

### Code Style
- Use `rustfmt` defaults (no custom config needed).
- Clippy with `pedantic` + `nursery` lints enabled in `Cargo.toml`.
- Write doc comments (`///`) for all public items.
- Use `#[must_use]` on functions returning values that shouldn't be ignored.
- Prefer `impl Trait` in argument position for single-use generic bounds.
- Use `thiserror` for library error types, `anyhow` only in examples/tests.
- Trait objects (`dyn Trait`) only when dynamic dispatch is genuinely needed; prefer generics.
- Comments as full sentences ending with a period.

### Testing Strategy
- Unit tests in `#[cfg(test)] mod tests` within each source file.
- Integration tests in `tests/` directory.
- Use `insta` for snapshot testing (similar to Python SDK's inline snapshots).
- Use `wiremock` for HTTP mocking in integration tests.
- Use `mockall` for trait mocking.
- Target 85%+ code coverage (matching Python SDK's threshold).

### Documentation
- Rust doc comments on all public items.
- `mdBook` for user-facing documentation (mirroring Python SDK's MkDocs structure).
- Examples in `examples/` directory matching Python SDK examples.
- Each example should be a standalone binary with `#[tokio::main]`.

### Feature Flags
- `default` — core agent functionality only.
- `mcp` — Model Context Protocol integration.
- `voice` — voice/realtime agent support.
- `tracing-otlp` — OpenTelemetry tracing export.
- `sqlite-session` — SQLite-backed session storage.

### Important Patterns
- **Builder pattern** for `Agent`, `RunConfig`, `ModelSettings` — these have many optional fields.
- **Enum dispatch** for `RunItem`, `StreamEvent`, `ToolOutput` — closed set of variants.
- **Trait objects** for `Model`, `ModelProvider`, `Session` — open for extension.
- **Generic context** `C: Send + Sync + 'static` — threads through `Agent<C>`, `Tool<C>`, `Guardrail<C>`.
- **`Pin<Box<dyn Stream>>`** for streaming responses.

### Mandatory Checks Before Completing Work
When changes affect runtime code, tests, or build configuration, run:
```bash
cargo fmt --check
cargo clippy -- -W clippy::all
cargo test
```

### Reference Implementation
The Python SDK at `F:\forks\openai-agents-python` is the authoritative reference. When in doubt about behavior, API surface, or architecture decisions, consult the Python implementation. The Rust port should be functionally equivalent but idiomatically Rust.
