//! Run context wrapper for agent execution.
//!
//! This module provides [`RunContextWrapper`], which wraps the user-provided context
//! value alongside run-level metadata such as accumulated token usage and turn input
//! items. It is passed to tools, guardrails, and lifecycle hooks during an agent run.
//!
//! The generic parameter `C` represents the user's custom context type, which must be
//! `Send + Sync + 'static` to support async execution across threads.
//!
//! This module mirrors the Python SDK's `run_context.py`.

use crate::items::ResponseInputItem;
use crate::usage::Usage;

/// Wrapper around the user-provided context, plus run-level metadata.
///
/// This is passed to tools, guardrails, and lifecycle hooks during an agent run.
/// The generic parameter `C` is the user's custom context type.
///
/// **NOTE:** Contexts are not passed to the LLM. They are a way to pass dependencies
/// and data to code you implement, such as tool functions, callbacks, and hooks.
#[non_exhaustive]
pub struct RunContextWrapper<C: Send + Sync + 'static> {
    /// The user-provided context value, passed by you to `Runner::run()`.
    pub context: C,
    /// Accumulated token usage across all model calls in this run.
    ///
    /// For streamed responses, the usage will be stale until the last chunk of the
    /// stream is processed.
    pub usage: Usage,
    /// The input items for the current turn.
    turn_input: Vec<ResponseInputItem>,
}

impl<C: Send + Sync + 'static> RunContextWrapper<C> {
    /// Create a new run context wrapper with the given user context.
    ///
    /// Usage starts at zero and turn input is initially empty.
    #[must_use]
    pub fn new(context: C) -> Self {
        Self {
            context,
            usage: Usage::default(),
            turn_input: Vec::new(),
        }
    }

    /// Get the current turn's input items.
    #[must_use]
    pub fn turn_input(&self) -> &[ResponseInputItem] {
        &self.turn_input
    }

    /// Set the current turn's input items (used internally by the runner).
    #[allow(dead_code)]
    pub(crate) fn set_turn_input(&mut self, input: Vec<ResponseInputItem>) {
        self.turn_input = input;
    }

    /// Add usage from a model response to the accumulated total.
    #[allow(dead_code)]
    pub(crate) fn add_usage(&mut self, other: &Usage) {
        self.usage.add(other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage::{InputTokensDetails, OutputTokensDetails};
    use serde_json::json;

    // ---- Construction with various context types ----

    #[test]
    fn new_with_unit_context() {
        let ctx = RunContextWrapper::new(());
        assert_eq!(ctx.context, ());
        assert_eq!(ctx.usage.requests, 0);
        assert!(ctx.turn_input().is_empty());
    }

    #[test]
    fn new_with_string_context() {
        let ctx = RunContextWrapper::new("my-context".to_owned());
        assert_eq!(ctx.context, "my-context");
        assert_eq!(ctx.usage.total_tokens, 0);
        assert!(ctx.turn_input().is_empty());
    }

    #[test]
    fn new_with_custom_struct_context() {
        #[derive(Debug, PartialEq)]
        struct AppContext {
            db_url: String,
            max_retries: u32,
        }

        let app_ctx = AppContext {
            db_url: "postgres://localhost/test".to_owned(),
            max_retries: 3,
        };
        let ctx = RunContextWrapper::new(app_ctx);
        assert_eq!(ctx.context.db_url, "postgres://localhost/test");
        assert_eq!(ctx.context.max_retries, 3);
    }

    // ---- Usage accumulation ----

    #[test]
    fn add_usage_accumulates() {
        let mut ctx = RunContextWrapper::new(());

        let first = Usage {
            requests: 1,
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            input_tokens_details: InputTokensDetails { cached_tokens: 10 },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 5,
            },
            request_usage_entries: Vec::new(),
        };

        let second = Usage {
            requests: 1,
            input_tokens: 200,
            output_tokens: 80,
            total_tokens: 280,
            input_tokens_details: InputTokensDetails { cached_tokens: 20 },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 15,
            },
            request_usage_entries: Vec::new(),
        };

        ctx.add_usage(&first);
        assert_eq!(ctx.usage.requests, 1);
        assert_eq!(ctx.usage.input_tokens, 100);
        assert_eq!(ctx.usage.total_tokens, 150);

        ctx.add_usage(&second);
        assert_eq!(ctx.usage.requests, 2);
        assert_eq!(ctx.usage.input_tokens, 300);
        assert_eq!(ctx.usage.output_tokens, 130);
        assert_eq!(ctx.usage.total_tokens, 430);
        assert_eq!(ctx.usage.input_tokens_details.cached_tokens, 30);
        assert_eq!(ctx.usage.output_tokens_details.reasoning_tokens, 20);
    }

    #[test]
    fn add_usage_with_zero_usage_is_noop() {
        let mut ctx = RunContextWrapper::new(());
        let empty = Usage::default();
        ctx.add_usage(&empty);
        assert_eq!(ctx.usage.requests, 0);
        assert_eq!(ctx.usage.total_tokens, 0);
    }

    // ---- Turn input get/set ----

    #[test]
    fn turn_input_initially_empty() {
        let ctx = RunContextWrapper::new(());
        assert!(ctx.turn_input().is_empty());
    }

    #[test]
    fn set_and_get_turn_input() {
        let mut ctx = RunContextWrapper::new(());
        let items = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi there"}),
        ];

        ctx.set_turn_input(items);
        assert_eq!(ctx.turn_input().len(), 2);
        assert_eq!(ctx.turn_input()[0]["role"], "user");
        assert_eq!(ctx.turn_input()[1]["role"], "assistant");
    }

    #[test]
    fn set_turn_input_replaces_previous() {
        let mut ctx = RunContextWrapper::new(());

        ctx.set_turn_input(vec![json!({"role": "user", "content": "first"})]);
        assert_eq!(ctx.turn_input().len(), 1);

        ctx.set_turn_input(vec![
            json!({"role": "user", "content": "second"}),
            json!({"role": "user", "content": "third"}),
        ]);
        assert_eq!(ctx.turn_input().len(), 2);
        assert_eq!(ctx.turn_input()[0]["content"], "second");
    }

    // ---- Send + Sync compile-time assertions ----

    #[test]
    fn run_context_wrapper_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RunContextWrapper<()>>();
        assert_send_sync::<RunContextWrapper<String>>();
        assert_send_sync::<RunContextWrapper<Vec<u8>>>();
    }
}
