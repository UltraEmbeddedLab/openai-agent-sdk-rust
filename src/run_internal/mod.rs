//! Internal implementation details for the agent run loop.
//!
//! This module contains the execution logic split into focused sub-modules:
//!
//! - [`run_loop`] — Core turn loop driving agent execution.
//! - [`tool_execution`] — Finding and invoking function tools.
//! - [`turn_resolution`] — Processing model responses, extracting outputs, building specs.
//!
//! All items in this module are `pub(crate)` and not part of the public API.

pub mod run_loop;
pub mod tool_execution;
pub mod turn_resolution;
