//! Distributed tracing support for agent runs.
//!
//! This module provides structured spans for observing agent execution,
//! built on top of the [`tracing`] crate. Each agent run, LLM call, tool
//! invocation, handoff, and guardrail check gets its own span with relevant
//! metadata.
//!
//! # Architecture
//!
//! Rather than building a custom tracing backend (as the Python SDK does),
//! this module leverages Rust's `tracing` ecosystem directly. Span creation
//! helpers in [`spans`] return [`tracing::Span`] values that integrate with
//! any `tracing-subscriber` layer, including `tracing-opentelemetry` for
//! export to OpenTelemetry-compatible backends.
//!
//! The [`config`] sub-module provides [`TracingConfig`] for per-run
//! configuration and a global disable switch via [`set_tracing_disabled`]
//! and [`is_tracing_disabled`].
//!
//! # Example
//!
//! ```
//! use openai_agents::tracing_support::{agent_span, generation_span, TracingConfig};
//!
//! # fn main() {
//! let span = agent_span("my_agent", 1);
//! let _guard = span.enter();
//! // ... agent work happens here ...
//! let gen_span = generation_span("my_agent", "gpt-4o");
//! let _gen_guard = gen_span.enter();
//! // ... model call ...
//! # }
//! ```

pub mod config;
pub mod exporter;
pub mod spans;

pub use config::{TracingConfig, is_tracing_disabled, set_tracing_disabled};
pub use exporter::OtlpExporterConfig;
pub use spans::{agent_span, function_span, generation_span, guardrail_span, handoff_span};
