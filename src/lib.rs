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
//! Any type that implements [`Model`] can serve as the LLM backend.
//! Implement [`ModelProvider`] to resolve model names to [`Model`]
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
pub mod computer;
pub mod config;
pub mod context;
pub mod editor;
pub mod error;
pub mod extensions;
pub mod global_config;
pub mod guardrail;
pub mod handoffs;
pub mod items;
pub mod lifecycle;
pub mod logger;
pub mod mcp;
pub mod memory;
pub mod models;
pub mod prompts;
pub mod repl;
pub mod result;
pub mod retry;
pub(crate) mod run_internal;
pub mod run_state;
pub mod runner;
pub mod schema;
pub mod stream_events;
pub mod tool;
pub mod tool_guardrails;
pub mod tracing_mod;
pub mod usage;
pub mod util;

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
pub use computer::{Button, Computer, ComputerTool, Environment};
pub use config::{DEFAULT_MAX_TURNS, ModelRef, ModelSettings, RunConfig, ToolChoice};
pub use context::RunContextWrapper;
pub use editor::{
    ApplyDiffMode, ApplyPatchEditor, ApplyPatchOperation, ApplyPatchOperationType,
    ApplyPatchResult, ApplyPatchStatus, ApplyPatchTool, apply_diff,
};
pub use error::{AgentError, Result};
pub use guardrail::{
    GuardrailFunctionOutput, InputGuardrail, InputGuardrailResult, OutputGuardrail,
    OutputGuardrailResult,
};
pub use handoffs::{
    Handoff, HandoffHistoryMapper, HandoffInputData, default_handoff_history_mapper,
    get_conversation_history_wrappers, nest_handoff_history, reset_conversation_history_wrappers,
    set_conversation_history_wrappers,
};
pub use items::{InputContent, ItemHelpers, ModelResponse, RunItem, ToolOutput};
pub use lifecycle::{AgentHooks, RunHooks};
pub use mcp::{MCPConfig, MCPServer};
pub use memory::{
    CompactingSession, CompactorFn, EncryptedSession, InMemorySession, Session,
    select_compaction_candidate_items,
};
pub use models::{
    AnyProvider, HandoffToolSpec, LiteLLMModel, LiteLLMProvider, Model, ModelProvider,
    ModelTracing, ToolSpec,
};
pub use result::{RunResult, RunResultStreaming};
pub use retry::RetryPolicy;
pub use run_state::{CURRENT_SCHEMA_VERSION, NextStep, PendingToolCall, RunState};
pub use runner::Runner;
pub use schema::{ensure_strict_json_schema, json_schema_for};
pub use stream_events::{RunItemEventName, StreamEvent};
pub use tool::{
    FunctionTool, FunctionToolResult, Tool, ToolContext, ToolSearchTool, function_tool,
};
pub use tool_guardrails::{
    GuardrailBehavior, ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailResult,
    ToolOutputGuardrail, ToolOutputGuardrailResult,
};
#[cfg(feature = "tracing-otlp")]
pub use tracing_mod::OtlpGuard;
pub use tracing_mod::{OtlpExporterConfig, TracingConfig};
pub use usage::Usage;

// Utility module re-exports.
pub use global_config::{
    OpenAiApi, ResponsesTransport, get_default_base_url, get_default_model, get_default_openai_api,
    get_default_openai_key, get_default_responses_transport, set_default_base_url,
    set_default_model, set_default_openai_api, set_default_openai_key,
    set_default_responses_transport,
};
pub use logger::enable_verbose_stdout_logging;
pub use repl::run_demo_loop;
pub use util::{
    pretty_print_result, transform_string_function_style, truncate_string, validate_json_schema,
};

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
/// [`Model`] trait in external code such as examples, integration
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
) -> ModelResponse {
    ModelResponse {
        output,
        usage,
        response_id,
        request_id,
    }
}
