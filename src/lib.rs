//! # `OpenAI` Agents SDK for Rust
//!
//! A lightweight, ergonomic framework for building multi-agent workflows powered by `OpenAI`.
//!
//! This crate is a Rust port of the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python),
//! bringing the same powerful abstractions to the Rust ecosystem with full type safety,
//! zero unsafe code, and first-class async support via Tokio.
//!
//! ## Quick Start
//!
//! The example below uses a mock model so it compiles and runs without an API key.
//! In real usage you would supply an `Arc<dyn Model>` backed by the `OpenAI` API.
//!
//! ```no_run
//! use std::sync::Arc;
//! use openai_agents::{Agent, Runner};
//! use openai_agents::models::{Model, ModelProvider};
//!
//! // Replace `my_model` with any type that implements the `Model` trait.
//! # async fn run(my_model: Arc<dyn openai_agents::models::Model>) -> openai_agents::Result<()> {
//! let agent = Agent::<()>::builder("greeter")
//!     .instructions("You are a helpful assistant. Greet the user warmly.")
//!     .build();
//!
//! let result = Runner::run_with_model(&agent, "Hello!", (), my_model, None, None).await?;
//! println!("{}", result.final_output);
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! The SDK follows a simple execution loop:
//!
//! 1. Create an [`Agent`] with instructions, tools, guardrails, and handoffs.
//! 2. Call [`Runner::run_with_model`] with the agent, user input, context, and a model.
//! 3. The runner handles the model call, tool execution, and handoff resolution.
//! 4. Returns a [`RunResult`] with the final output, items, and usage statistics.
//!
//! ### Context
//!
//! The generic parameter `C` in `Agent<C>`, `Tool<C>`, and friends is a user-provided
//! context type that flows through every tool invocation, guardrail check, and lifecycle
//! hook. Set it to `()` if you do not need a custom context.
//!
//! ### Model trait
//!
//! Any type that implements [`models::Model`] can serve as the LLM backend.
//! Implement [`models::ModelProvider`] to resolve model names to [`models::Model`]
//! instances at runtime.
//!
//! ## Module Organization
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`agent`] | Agent definition and builder |
//! | [`config`] | Run configuration and model settings |
//! | [`context`] | Run context wrapper passed to tools and hooks |
//! | [`error`] | Error types (`AgentError`, `Result`) |
//! | [`guardrail`] | Input and output guardrails |
//! | [`handoffs`] | Agent-to-agent handoff support |
//! | [`items`] | Run items, tool output, and helper utilities |
//! | [`lifecycle`] | Run and agent lifecycle hook traits |
//! | [`mcp`] | Model Context Protocol server integration |
//! | [`memory`] | Session-based conversation history persistence |
//! | [`models`] | `Model` and `ModelProvider` traits |
//! | [`result`] | `RunResult` and `RunResultStreaming` |
//! | [`runner`] | Agent execution loop |
//! | [`schema`] | JSON Schema generation and strict-mode enforcement |
//! | [`stream_events`] | Streaming event types |
//! | [`prompts`] | System prompt construction helpers |
//! | [`retry`] | Retry policies for transient failures |
//! | [`run_state`] | Serializable checkpoint of an in-progress run |
//! | [`tool`] | Function tools and hosted tool types |
//! | [`tracing_mod`] | Distributed tracing spans and configuration |
//! | [`usage`] | Token usage tracking |
//!
//! ## Feature Flags
//!
//! | Flag | Description |
//! |------|-------------|
//! | `mcp` | Model Context Protocol integration |
//! | `voice` | Voice/realtime agent support |
//! | `tracing-otlp` | OpenTelemetry tracing export |
//! | `sqlite-session` | SQLite-backed session storage |

pub mod agent;
pub mod config;
pub mod context;
pub mod error;
pub mod guardrail;
pub mod handoffs;
pub mod items;
pub mod lifecycle;
pub mod mcp;
pub mod memory;
pub mod models;
pub mod prompts;
pub mod result;
pub mod retry;
pub(crate) mod run_internal;
pub mod run_state;
pub mod runner;
pub mod schema;
pub mod stream_events;
pub mod tool;
pub mod tracing_mod;
pub mod usage;

#[cfg(feature = "voice")]
pub mod voice;

/// Re-export of the tracing module under a friendlier name.
///
/// The module is named `tracing_mod` internally to avoid collision with
/// the `tracing` crate dependency. Use `openai_agents::tracing_support`
/// (or `openai_agents::tracing_mod`) to access span helpers and
/// configuration.
pub use tracing_mod as tracing_support;

// ---------------------------------------------------------------------------
// Convenience re-exports — the most commonly used types at the crate root.
// ---------------------------------------------------------------------------

pub use agent::{Agent, AgentBuilder, Instructions, OutputSchema, ToolUseBehavior};
pub use config::{DEFAULT_MAX_TURNS, ModelRef, ModelSettings, RunConfig, ToolChoice};
pub use context::RunContextWrapper;
pub use error::{AgentError, Result};
pub use guardrail::{
    GuardrailFunctionOutput, InputGuardrail, InputGuardrailResult, OutputGuardrail,
    OutputGuardrailResult,
};
pub use handoffs::{Handoff, HandoffInputData};
pub use items::{InputContent, ItemHelpers, ModelResponse, RunItem, ToolOutput};
pub use lifecycle::{AgentHooks, RunHooks};
pub use mcp::{MCPConfig, MCPServer};
pub use memory::{InMemorySession, Session};
pub use models::{HandoffToolSpec, Model, ModelProvider, ModelTracing, ToolSpec};
pub use result::{RunResult, RunResultStreaming};
pub use retry::RetryPolicy;
pub use run_state::{CURRENT_SCHEMA_VERSION, NextStep, PendingToolCall, RunState};
pub use runner::Runner;
pub use schema::{ensure_strict_json_schema, json_schema_for};
pub use stream_events::{RunItemEventName, StreamEvent};
pub use tool::{FunctionTool, FunctionToolResult, Tool, ToolContext, function_tool};
pub use tracing_mod::{OtlpExporterConfig, TracingConfig};
pub use usage::Usage;

// ---------------------------------------------------------------------------
// Constructor helpers for non_exhaustive types
//
// These thin wrappers allow external code (examples, integration tests, and
// downstream crate consumers) to construct types that are marked
// `#[non_exhaustive]` without needing to live inside this crate.
// ---------------------------------------------------------------------------

/// Construct a [`ModelResponse`] from its constituent parts.
///
/// `ModelResponse` is `#[non_exhaustive]` so it cannot be built with a struct
/// literal outside this crate. Use this function when implementing the
/// [`models::Model`] trait in external code such as examples, integration
/// tests, or downstream crates.
///
/// # Example
///
/// ```
/// use openai_agents::{new_model_response, usage::Usage};
///
/// let response = new_model_response(
///     vec![serde_json::json!({"type": "message"})],
///     Usage::default(),
///     Some("resp-001".to_owned()),
///     None,
/// );
///
/// assert_eq!(response.response_id.as_deref(), Some("resp-001"));
/// ```
#[must_use]
pub const fn new_model_response(
    output: Vec<serde_json::Value>,
    usage: Usage,
    response_id: Option<String>,
    request_id: Option<String>,
) -> items::ModelResponse {
    items::ModelResponse {
        output,
        usage,
        response_id,
        request_id,
    }
}
