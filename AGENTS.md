# Contributor Guide

This guide helps contributors get started with the OpenAI Agents Rust SDK. It covers repo structure, testing, utilities, and guidelines for commits and PRs.

**Location:** `AGENTS.md` at the repository root.

## Table of Contents

1. [Policies & Mandatory Rules](#policies--mandatory-rules)
2. [Project Structure Guide](#project-structure-guide)
3. [Operation Guide](#operation-guide)

## Policies & Mandatory Rules

### Mandatory Skill Usage

#### `$code-change-verification`

Run `$code-change-verification` before marking work complete when changes affect runtime code, tests, or build/test behavior.

Run it when you change:
- `src/` (library code) or shared utilities.
- `tests/` or add/modify snapshot tests.
- `examples/`.
- Build or test configuration such as `Cargo.toml`, `Makefile`, CI workflows.

You can skip `$code-change-verification` for docs-only or repo-meta changes (`docs/`, `.agents/`, `README.md`, `AGENTS.md`, `.github/`), unless a user explicitly asks to run the full verification stack.

#### `$openai-knowledge`

When working on OpenAI API integrations (Responses API, tools, streaming, Realtime API, auth, models, rate limits, MCP), use `$openai-knowledge` to pull authoritative docs via the OpenAI Developer Docs MCP server.

#### `$implementation-strategy`

Before changing runtime code, exported APIs, external configuration, or user-facing behavior, use `$implementation-strategy` to decide the compatibility boundary and implementation shape. Judge breaking changes against the latest release tag.

#### `$pr-draft-summary`

When a task finishes with moderate-or-larger code changes, invoke `$pr-draft-summary` in the final handoff to generate the required PR summary block. Treat this as the default close-out step after runtime code, tests, examples, or build configuration changes.

### Public API Compatibility

Treat the public API surface as a compatibility contract:
- For public constructors and builder methods, preserve existing method signatures.
- When adding new optional fields, add new builder methods rather than changing existing ones.
- Prefer builder pattern at call sites to reduce accidental breakage.
- Use `#[non_exhaustive]` on public enums and structs to allow future additions.

## Project Structure Guide

### Overview

The OpenAI Agents Rust SDK provides a framework for building multi-agent workflows. It mirrors the architecture of the [Python SDK](https://github.com/openai/openai-agents-python) using idiomatic Rust patterns.

### Repo Structure & Important Files

```
openai-agents-sdk/
├── src/                    # Core library implementation
│   ├── lib.rs              # Public API exports (crate root)
│   ├── agent.rs            # Agent struct and builder
│   ├── runner.rs           # Runner — main execution entry point
│   ├── tool.rs             # Tool trait and FunctionTool
│   ├── items.rs            # RunItem types (messages, tool calls, handoffs)
│   ├── result.rs           # RunResult, RunResultStreaming
│   ├── guardrail.rs        # Input/output guardrails
│   ├── config.rs           # RunConfig, ModelSettings
│   ├── lifecycle.rs        # AgentHooks, RunHooks traits
│   ├── error.rs            # Error types (AgentError enum)
│   ├── schema.rs           # JSON Schema generation
│   ├── stream_events.rs    # StreamEvent enum
│   ├── context.rs          # RunContext wrapper
│   ├── prompts.rs          # Prompt types and dynamic prompts
│   ├── usage.rs            # Token usage tracking
│   ├── models/             # LLM provider abstractions
│   │   ├── mod.rs          # Model and ModelProvider traits
│   │   ├── openai_responses.rs
│   │   ├── openai_chatcompletions.rs
│   │   ├── openai_provider.rs
│   │   └── multi_provider.rs
│   ├── memory/             # Session/conversation storage
│   │   ├── mod.rs          # Session trait
│   │   ├── sqlite_session.rs
│   │   └── settings.rs
│   ├── handoffs/           # Agent handoff logic
│   │   └── mod.rs
│   ├── run_internal/       # Internal orchestration
│   │   ├── mod.rs
│   │   ├── run_loop.rs
│   │   ├── tool_execution.rs
│   │   └── turn_resolution.rs
│   ├── tracing/            # Distributed tracing
│   │   └── mod.rs
│   └── mcp/                # Model Context Protocol
│       └── mod.rs
├── tests/                  # Integration tests
├── examples/               # Usage examples (each is a binary)
│   ├── hello_world.rs
│   ├── tools.rs
│   ├── handoffs.rs
│   ├── guardrails.rs
│   ├── streaming.rs
│   └── agent_patterns/
├── benches/                # Benchmarks
│   └── runner.rs
├── docs/                   # mdBook documentation source
│   ├── src/
│   │   ├── SUMMARY.md
│   │   ├── index.md
│   │   ├── quickstart.md
│   │   ├── agents.md
│   │   ├── tools.md
│   │   ├── handoffs.md
│   │   ├── guardrails.md
│   │   ├── streaming.md
│   │   ├── sessions.md
│   │   ├── tracing.md
│   │   ├── mcp.md
│   │   └── examples.md
│   └── book.toml
├── .github/                # GitHub workflows & templates
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   └── docs.yml
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE/
│   └── dependabot.yml
├── Cargo.toml              # Package manifest
├── Cargo.lock              # Dependency lockfile
├── Makefile                # Development commands
├── README.md               # Project overview
├── AGENTS.md               # This file — contributor guide
├── CLAUDE.md               # Claude Code instructions
├── LICENSE                 # MIT License
└── .gitignore
```

### Core Runtime Guidelines

- `src/runner.rs` is the runtime entry point (`Runner`, `AgentRunner`). Keep it focused on orchestration and public flow control.
- Put new runtime logic in `src/run_internal/` and import into `runner.rs`.
- Keep streaming and non-streaming paths behaviorally aligned. Changes to `run_internal/run_loop.rs` should be mirrored across both paths.
- Input guardrails run only on the first turn and only for the starting agent.
- Adding new item types requires coordinated updates across: `items.rs`, `run_internal/run_loop.rs`, `run_internal/turn_resolution.rs`, `run_internal/tool_execution.rs`, and `stream_events.rs`.
- Use `#[non_exhaustive]` on all public enums to allow future additions without breaking changes.

### Rust-Specific Patterns (vs Python SDK)

| Python Pattern | Rust Equivalent |
|---|---|
| `ABC` / abstract class | `trait` with `async_trait` |
| `dataclass` / `BaseModel` | `struct` with `#[derive(Debug, Clone, Serialize, Deserialize)]` |
| Union types `A \| B \| C` | `enum` with variants |
| `Optional[T]` | `Option<T>` |
| `list[T]` | `Vec<T>` |
| `dict[K, V]` | `HashMap<K, V>` or `BTreeMap<K, V>` |
| Exception hierarchy | `thiserror` enum with `#[error]` |
| `@dataclass` with defaults | Builder pattern via `TypedBuilder` or manual impl |
| `async def` | `async fn` with `#[async_trait]` on trait methods |
| `AsyncIterator` | `Stream<Item = T>` from `tokio-stream` |
| Generic `TContext` | Generic `C: Send + Sync + 'static` |
| `pydantic` validation | `schemars` for JSON Schema, manual validation |
| `typing.TypeVar` | Rust generics `<T>` |

## Operation Guide

### Prerequisites

- Rust 1.85+ (edition 2024).
- `cargo` for building, testing, and managing dependencies.
- `make` available to run repository tasks (optional).

### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feat/<short-description>
   ```
2. Implement changes and add/update tests alongside code updates.
3. Run verification before committing:
   ```bash
   cargo fmt --check
   cargo clippy -- -W clippy::all
   cargo test
   ```
4. Build docs when touching documentation:
   ```bash
   mdbook build docs/
   ```
5. When `$code-change-verification` applies, run it before marking work complete.
6. Commit with concise, imperative messages; keep commits small and focused.

### Testing

- Run the full test suite:
  ```bash
  cargo test
  ```
- Run a focused test:
  ```bash
  cargo test test_name
  ```
- Run tests with output:
  ```bash
  cargo test -- --nocapture
  ```
- Run benchmarks:
  ```bash
  cargo bench
  ```
- Update snapshots (insta):
  ```bash
  cargo insta review
  ```

### Formatting, Linting, and Type Checking

- Format: `cargo fmt` (applies fixes), `cargo fmt --check` (checks only).
- Lint: `cargo clippy -- -W clippy::all`.
- Rust's type system handles type checking at compile time.
- Write doc comments as full sentences ending with a period.

### Mandatory Local Run Order

When `$code-change-verification` applies:

```bash
cargo fmt --check
cargo clippy -- -W clippy::all
cargo test
```

### Pull Request & Commit Guidelines

- Use the PR template at `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`.
- Include a summary, test plan, and issue number if applicable.
- Add tests for new behavior and update documentation for user-facing changes.
- Commit messages: concise, imperative mood. Small, focused commits preferred.

### Review Checklist

- ✅ `cargo fmt --check` passes.
- ✅ `cargo clippy` passes with no warnings.
- ✅ `cargo test` passes.
- ✅ Tests cover new behavior and edge cases.
- ✅ Code is readable, maintainable, and idiomatic Rust.
- ✅ Public APIs documented with `///` doc comments.
- ✅ Examples updated if behavior changes.
- ✅ `#[non_exhaustive]` used on new public enums/structs.
