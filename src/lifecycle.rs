//! Lifecycle hooks for agent runs.
//!
//! This module provides two traits — [`RunHooks`] and [`AgentHooks`] — that let you
//! observe and react to lifecycle events during an agent run. Both traits provide
//! default no-op implementations for every method, so you only need to override the
//! events you care about.
//!
//! - [`RunHooks`] receives callbacks for events across **all** agents in a run.
//! - [`AgentHooks`] receives callbacks for events on a **specific** agent.
//!
//! This module mirrors the Python SDK's `lifecycle.py`.

use async_trait::async_trait;

use crate::context::RunContextWrapper;
use crate::items::{ModelResponse, ResponseInputItem};

/// Callbacks for lifecycle events during an entire agent run.
///
/// Implement this trait to hook into events across all agents in a run.
/// All methods have default no-op implementations, so you only need to
/// override the events you care about.
#[async_trait]
pub trait RunHooks<C: Send + Sync + 'static>: Send + Sync {
    /// Called when an agent begins processing.
    async fn on_agent_start(&self, _context: &RunContextWrapper<C>, _agent_name: &str) {}

    /// Called when an agent produces its final output.
    async fn on_agent_end(
        &self,
        _context: &RunContextWrapper<C>,
        _agent_name: &str,
        _output: &serde_json::Value,
    ) {
    }

    /// Called when a handoff occurs from one agent to another.
    async fn on_handoff(
        &self,
        _context: &RunContextWrapper<C>,
        _from_agent: &str,
        _to_agent: &str,
    ) {
    }

    /// Called before a tool is executed.
    async fn on_tool_start(
        &self,
        _context: &RunContextWrapper<C>,
        _agent_name: &str,
        _tool_name: &str,
    ) {
    }

    /// Called after a tool completes execution.
    async fn on_tool_end(
        &self,
        _context: &RunContextWrapper<C>,
        _agent_name: &str,
        _tool_name: &str,
        _result: &str,
    ) {
    }

    /// Called before the LLM is invoked.
    async fn on_llm_start(
        &self,
        _context: &RunContextWrapper<C>,
        _agent_name: &str,
        _system_prompt: Option<&str>,
        _input: &[ResponseInputItem],
    ) {
    }

    /// Called after the LLM responds.
    async fn on_llm_end(
        &self,
        _context: &RunContextWrapper<C>,
        _agent_name: &str,
        _response: &ModelResponse,
    ) {
    }
}

/// Per-agent lifecycle hooks.
///
/// Unlike [`RunHooks`], these are attached to a specific agent and only fire
/// for events on that agent. All methods have default no-op implementations.
#[async_trait]
pub trait AgentHooks<C: Send + Sync + 'static>: Send + Sync {
    /// Called when this agent begins processing.
    async fn on_start(&self, _context: &RunContextWrapper<C>) {}

    /// Called when this agent produces its final output.
    async fn on_end(&self, _context: &RunContextWrapper<C>, _output: &serde_json::Value) {}

    /// Called when another agent hands off to this agent.
    async fn on_handoff(&self, _context: &RunContextWrapper<C>, _source_agent: &str) {}

    /// Called before a tool is executed by this agent.
    async fn on_tool_start(&self, _context: &RunContextWrapper<C>, _tool_name: &str) {}

    /// Called after a tool completes execution for this agent.
    async fn on_tool_end(&self, _context: &RunContextWrapper<C>, _tool_name: &str, _result: &str) {}

    /// Called before the LLM is invoked for this agent.
    async fn on_llm_start(
        &self,
        _context: &RunContextWrapper<C>,
        _system_prompt: Option<&str>,
        _input: &[ResponseInputItem],
    ) {
    }

    /// Called after the LLM responds for this agent.
    async fn on_llm_end(&self, _context: &RunContextWrapper<C>, _response: &ModelResponse) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage::Usage;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    // -- Default (no-op) RunHooks compiles and is callable --

    struct NoOpRunHooks;
    impl RunHooks<()> for NoOpRunHooks {}

    #[tokio::test]
    async fn default_run_hooks_are_callable() {
        let hooks = NoOpRunHooks;
        let ctx = RunContextWrapper::new(());

        hooks.on_agent_start(&ctx, "agent").await;
        hooks
            .on_agent_end(&ctx, "agent", &serde_json::json!("done"))
            .await;
        hooks.on_handoff(&ctx, "a", "b").await;
        hooks.on_tool_start(&ctx, "agent", "tool").await;
        hooks.on_tool_end(&ctx, "agent", "tool", "result").await;
        hooks.on_llm_start(&ctx, "agent", Some("prompt"), &[]).await;

        let response = ModelResponse {
            output: vec![],
            usage: Usage::default(),
            response_id: None,
            request_id: None,
        };
        hooks.on_llm_end(&ctx, "agent", &response).await;
    }

    // -- Custom RunHooks that tracks calls --

    struct TrackingRunHooks {
        calls: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl RunHooks<()> for TrackingRunHooks {
        async fn on_agent_start(&self, _context: &RunContextWrapper<()>, agent_name: &str) {
            self.calls
                .lock()
                .await
                .push(format!("agent_start:{agent_name}"));
        }

        async fn on_agent_end(
            &self,
            _context: &RunContextWrapper<()>,
            agent_name: &str,
            _output: &serde_json::Value,
        ) {
            self.calls
                .lock()
                .await
                .push(format!("agent_end:{agent_name}"));
        }

        async fn on_handoff(
            &self,
            _context: &RunContextWrapper<()>,
            from_agent: &str,
            to_agent: &str,
        ) {
            self.calls
                .lock()
                .await
                .push(format!("handoff:{from_agent}->{to_agent}"));
        }

        async fn on_tool_start(
            &self,
            _context: &RunContextWrapper<()>,
            agent_name: &str,
            tool_name: &str,
        ) {
            self.calls
                .lock()
                .await
                .push(format!("tool_start:{agent_name}:{tool_name}"));
        }

        async fn on_tool_end(
            &self,
            _context: &RunContextWrapper<()>,
            agent_name: &str,
            tool_name: &str,
            result: &str,
        ) {
            self.calls
                .lock()
                .await
                .push(format!("tool_end:{agent_name}:{tool_name}={result}"));
        }
    }

    #[tokio::test]
    async fn tracking_run_hooks_records_calls() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let hooks = TrackingRunHooks {
            calls: Arc::clone(&calls),
        };
        let ctx = RunContextWrapper::new(());

        hooks.on_agent_start(&ctx, "greeter").await;
        hooks.on_tool_start(&ctx, "greeter", "get_weather").await;
        hooks
            .on_tool_end(&ctx, "greeter", "get_weather", "sunny")
            .await;
        hooks.on_handoff(&ctx, "greeter", "farewell").await;
        hooks
            .on_agent_end(&ctx, "farewell", &serde_json::json!("bye"))
            .await;

        let recorded = calls.lock().await;
        assert_eq!(recorded.len(), 5);
        assert_eq!(recorded[0], "agent_start:greeter");
        assert_eq!(recorded[1], "tool_start:greeter:get_weather");
        assert_eq!(recorded[2], "tool_end:greeter:get_weather=sunny");
        assert_eq!(recorded[3], "handoff:greeter->farewell");
        assert_eq!(recorded[4], "agent_end:farewell");
        drop(recorded);
    }

    // -- Default (no-op) AgentHooks compiles and is callable --

    struct NoOpAgentHooks;
    impl AgentHooks<()> for NoOpAgentHooks {}

    #[tokio::test]
    async fn default_agent_hooks_are_callable() {
        let hooks = NoOpAgentHooks;
        let ctx = RunContextWrapper::new(());

        hooks.on_start(&ctx).await;
        hooks.on_end(&ctx, &serde_json::json!("output")).await;
        hooks.on_handoff(&ctx, "other_agent").await;
        hooks.on_tool_start(&ctx, "my_tool").await;
        hooks.on_tool_end(&ctx, "my_tool", "result").await;
        hooks.on_llm_start(&ctx, None, &[]).await;

        let response = ModelResponse {
            output: vec![],
            usage: Usage::default(),
            response_id: None,
            request_id: None,
        };
        hooks.on_llm_end(&ctx, &response).await;
    }

    // -- Custom AgentHooks with selective overrides --

    struct SelectiveAgentHooks {
        start_count: AtomicUsize,
        end_count: AtomicUsize,
    }

    #[async_trait]
    impl AgentHooks<String> for SelectiveAgentHooks {
        async fn on_start(&self, _context: &RunContextWrapper<String>) {
            self.start_count.fetch_add(1, Ordering::Relaxed);
        }

        async fn on_end(&self, _context: &RunContextWrapper<String>, _output: &serde_json::Value) {
            self.end_count.fetch_add(1, Ordering::Relaxed);
        }

        // All other methods use the default no-op implementation.
    }

    #[tokio::test]
    async fn selective_agent_hooks_only_overrides_fire() {
        let hooks = SelectiveAgentHooks {
            start_count: AtomicUsize::new(0),
            end_count: AtomicUsize::new(0),
        };
        let ctx = RunContextWrapper::new("my-context".to_owned());

        hooks.on_start(&ctx).await;
        hooks.on_start(&ctx).await;
        hooks.on_end(&ctx, &serde_json::json!(42)).await;

        // Non-overridden methods are still callable (no-op).
        hooks.on_handoff(&ctx, "source").await;
        hooks.on_tool_start(&ctx, "tool").await;
        hooks.on_tool_end(&ctx, "tool", "ok").await;

        assert_eq!(hooks.start_count.load(Ordering::Relaxed), 2);
        assert_eq!(hooks.end_count.load(Ordering::Relaxed), 1);
    }

    // -- Send + Sync compile-time assertions --

    #[test]
    fn run_hooks_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoOpRunHooks>();
        assert_send_sync::<TrackingRunHooks>();
    }

    #[test]
    fn agent_hooks_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoOpAgentHooks>();
        assert_send_sync::<SelectiveAgentHooks>();
    }

    // -- Trait objects are object-safe --

    #[test]
    fn run_hooks_is_object_safe() {
        fn accept(hooks: &dyn RunHooks<()>) {
            let _ = hooks;
        }
        let hooks = NoOpRunHooks;
        accept(&hooks);
    }

    #[test]
    fn agent_hooks_is_object_safe() {
        fn accept(hooks: &dyn AgentHooks<()>) {
            let _ = hooks;
        }
        let hooks = NoOpAgentHooks;
        accept(&hooks);
    }
}
