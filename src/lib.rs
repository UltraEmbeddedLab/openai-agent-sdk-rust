//! # OpenAI Agents SDK for Rust
//!
//! A lightweight, ergonomic framework for building multi-agent workflows powered by OpenAI.
//!
//! This crate is a Rust port of the [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python),
//! bringing the same powerful abstractions to the Rust ecosystem with full type safety and async support.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_agents::{Agent, Runner};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let agent = Agent::builder()
//!         .name("Greeter")
//!         .instructions("You are a helpful assistant.")
//!         .build();
//!
//!     let result = Runner::run(&agent, "Hello!").await?;
//!     println!("{}", result.final_output());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The SDK follows a simple execution loop:
//!
//! 1. Create an [`Agent`] with instructions, tools, guardrails, and handoffs.
//! 2. Call [`Runner::run()`] with the agent and user input.
//! 3. The runner handles the model call, tool execution, and handoff resolution.
//! 4. Returns a [`RunResult`] with the final output, items, and usage statistics.
//!
//! ## Feature Flags
//!
//! | Flag | Description |
//! |------|-------------|
//! | `mcp` | Model Context Protocol integration |
//! | `voice` | Voice/realtime agent support |
//! | `tracing-otlp` | OpenTelemetry tracing export |
//! | `sqlite-session` | SQLite-backed session storage |

// TODO: Implement core modules — each module below corresponds to a Python SDK module.
// See CLAUDE.md for the complete module mapping.

// pub mod agent;
// pub mod config;
// pub mod context;
// pub mod error;
// pub mod guardrail;
// pub mod handoffs;
// pub mod items;
// pub mod lifecycle;
// pub mod memory;
// pub mod mcp;
// pub mod models;
// pub mod prompts;
// pub mod result;
// pub mod run_internal;
// pub mod runner;
// pub mod schema;
// pub mod stream_events;
// pub mod tool;
// pub mod tracing_mod;
// pub mod usage;
