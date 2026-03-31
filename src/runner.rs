//! Agent execution loop (the "runner").
//!
//! This module provides [`Runner`], the main entry point for executing agents.
//! It orchestrates the core agent loop: resolving instructions, calling the model,
//! processing the response (messages, tool calls, handoffs, reasoning), executing
//! function tools, running guardrails, and firing lifecycle hooks.
//!
//! This module mirrors the Python SDK's `run.py` and `run_internal/run_loop.py`.
//!
//! # Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use openai_agents::agent::Agent;
//! use openai_agents::runner::Runner;
//!
//! # async fn example(model: Arc<dyn openai_agents::models::Model>) {
//! let agent = Agent::<()>::builder("assistant")
//!     .instructions("You are a helpful assistant.")
//!     .build();
//!
//! let result = Runner::run_with_model(&agent, "Hello!", (), model, None, None)
//!     .await
//!     .expect("run should succeed");
//! println!("{}", result.final_output);
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::agent::Agent;
use crate::config::{DEFAULT_MAX_TURNS, RunConfig};
use crate::error::Result;
use crate::items::InputContent;
use crate::lifecycle::RunHooks;
use crate::models::Model;
use crate::result::{RunResult, RunResultStreaming};
use crate::run_internal::run_loop;

/// The main entry point for running agents.
///
/// `Runner` provides static methods for executing agents in both synchronous
/// (complete) and streaming modes. It implements the core agent execution loop:
///
/// 1. Resolve agent instructions.
/// 2. Convert tools and handoffs to model-facing specifications.
/// 3. Run input guardrails (first turn only).
/// 4. Call the model.
/// 5. Process the response: extract messages, tool calls, handoffs, and reasoning.
/// 6. Execute function tools and collect their outputs.
/// 7. Determine the next step based on the agent's [`ToolUseBehavior`](crate::agent::ToolUseBehavior).
/// 8. Run output guardrails on the final output.
/// 9. Fire lifecycle hooks at each step.
/// 10. Repeat until a final output is produced or the turn limit is reached.
pub struct Runner;

impl Runner {
    /// Run an agent to completion with a provided model.
    ///
    /// This is the primary API for executing an agent. It runs the full agent loop
    /// with the given model, context, optional hooks, and optional configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::MaxTurnsExceeded`](crate::error::AgentError::MaxTurnsExceeded)
    /// if the loop exceeds the configured maximum number of turns. Also propagates
    /// errors from the model, tools, guardrails, or lifecycle hooks.
    ///
    /// # Panics
    ///
    /// Panics if the internal response list is unexpectedly empty after a model call.
    /// This should never occur in practice.
    pub async fn run_with_model<C: Send + Sync + 'static>(
        agent: &Agent<C>,
        input: impl Into<InputContent>,
        context: C,
        model: Arc<dyn Model>,
        hooks: Option<Arc<dyn RunHooks<C>>>,
        config: Option<RunConfig>,
    ) -> Result<RunResult> {
        // Delegate to the multi-agent run with no agent registry.
        run_loop::run_internal(
            agent,
            &HashMap::new(),
            input.into(),
            context,
            model,
            hooks,
            config,
        )
        .await
    }

    /// Run an agent with a registry of agents for multi-agent handoff support.
    ///
    /// When a handoff is detected, the runner looks up the target agent in `agents`,
    /// applies any handoff input filter, fires handoff hooks, and continues the loop
    /// with the new agent. The `starting_agent` does not need to be in `agents`.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`](crate::error::AgentError::UserError) if a
    /// handoff target agent is not found in the registry. Also propagates all errors
    /// from [`run_with_model`](Self::run_with_model).
    pub async fn run_with_agents<C: Send + Sync + 'static>(
        starting_agent: &Agent<C>,
        agents: &HashMap<String, &Agent<C>>,
        input: impl Into<InputContent>,
        context: C,
        model: Arc<dyn Model>,
        hooks: Option<Arc<dyn RunHooks<C>>>,
        config: Option<RunConfig>,
    ) -> Result<RunResult> {
        run_loop::run_internal(
            starting_agent,
            agents,
            input.into(),
            context,
            model,
            hooks,
            config,
        )
        .await
    }

    /// Run an agent in streaming mode, returning a [`RunResultStreaming`] immediately.
    ///
    /// The runner spawns a background task that drives the agent loop and sends
    /// [`StreamEvent`] values through a channel. Use
    /// [`RunResultStreaming::stream_events`] to consume them.
    ///
    /// The `agent` reference must be `'static` because it is moved into a spawned task.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to run.
    /// * `input` - The user input.
    /// * `context` - The user-provided context value.
    /// * `model` - The model to use for inference.
    /// * `hooks` - Optional run-level lifecycle hooks.
    /// * `config` - Optional run configuration.
    pub fn run_streamed<C: Send + Sync + 'static>(
        agent: &'static Agent<C>,
        input: impl Into<InputContent> + Send + 'static,
        context: C,
        model: Arc<dyn Model>,
        hooks: Option<Arc<dyn RunHooks<C>>>,
        config: Option<RunConfig>,
    ) -> RunResultStreaming {
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();

        let input = input.into();
        let max_turns = config.as_ref().map_or(DEFAULT_MAX_TURNS, |c| c.max_turns);

        let result = RunResultStreaming::new(
            input.clone(),
            agent.name.clone(),
            max_turns,
            event_rx,
            cancel_tx,
        );

        tokio::spawn(run_loop::streaming_loop(
            agent, input, context, model, hooks, config, event_tx, cancel_rx,
        ));

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::config::{ModelSettings, RunConfig};
    use crate::context::RunContextWrapper;
    use crate::error::AgentError;
    use crate::guardrail::{GuardrailFunctionOutput, InputGuardrail, OutputGuardrail};
    use crate::items::{ModelResponse, ResponseInputItem, RunItem};
    use crate::lifecycle::RunHooks;
    use crate::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
    use crate::usage::Usage;
    use async_trait::async_trait;
    use serde_json::json;
    use std::pin::Pin;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio_stream::Stream;

    // ---- Mock Model ----

    struct MockModel {
        responses: Mutex<Vec<ModelResponse>>,
    }

    impl MockModel {
        fn new(responses: Vec<ModelResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }

        fn single_message(text: &str) -> Self {
            Self::new(vec![ModelResponse {
                output: vec![json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}]
                })],
                usage: Usage {
                    requests: 1,
                    input_tokens: 10,
                    output_tokens: 5,
                    total_tokens: 15,
                    ..Usage::default()
                },
                response_id: Some("resp_mock".to_owned()),
                request_id: None,
            }])
        }
    }

    #[async_trait]
    impl Model for MockModel {
        async fn get_response(
            &self,
            _system_instructions: Option<&str>,
            _input: &[ResponseInputItem],
            _model_settings: &ModelSettings,
            _tools: &[ToolSpec],
            _output_schema: Option<&OutputSchemaSpec>,
            _handoffs: &[HandoffToolSpec],
            _tracing: ModelTracing,
            _previous_response_id: Option<&str>,
        ) -> crate::Result<ModelResponse> {
            let mut responses = self.responses.lock().await;
            if responses.is_empty() {
                Ok(ModelResponse {
                    output: vec![json!({
                        "type": "message",
                        "content": [{"type": "output_text", "text": "default response"}]
                    })],
                    usage: Usage::default(),
                    response_id: None,
                    request_id: None,
                })
            } else {
                Ok(responses.remove(0))
            }
        }

        fn stream_response<'a>(
            &'a self,
            _system_instructions: Option<&'a str>,
            _input: &'a [ResponseInputItem],
            _model_settings: &'a ModelSettings,
            _tools: &'a [ToolSpec],
            _output_schema: Option<&'a OutputSchemaSpec>,
            _handoffs: &'a [HandoffToolSpec],
            _tracing: ModelTracing,
            _previous_response_id: Option<&'a str>,
        ) -> Pin<Box<dyn Stream<Item = crate::Result<crate::items::ResponseStreamEvent>> + Send + 'a>>
        {
            Box::pin(tokio_stream::empty())
        }
    }

    /// A mock model that produces stream events.
    struct StreamingMockModel {
        events: Vec<serde_json::Value>,
    }

    #[async_trait]
    impl Model for StreamingMockModel {
        async fn get_response(
            &self,
            _system_instructions: Option<&str>,
            _input: &[ResponseInputItem],
            _model_settings: &ModelSettings,
            _tools: &[ToolSpec],
            _output_schema: Option<&OutputSchemaSpec>,
            _handoffs: &[HandoffToolSpec],
            _tracing: ModelTracing,
            _previous_response_id: Option<&str>,
        ) -> crate::Result<ModelResponse> {
            Ok(ModelResponse {
                output: vec![],
                usage: Usage::default(),
                response_id: None,
                request_id: None,
            })
        }

        fn stream_response<'a>(
            &'a self,
            _system_instructions: Option<&'a str>,
            _input: &'a [ResponseInputItem],
            _model_settings: &'a ModelSettings,
            _tools: &'a [ToolSpec],
            _output_schema: Option<&'a OutputSchemaSpec>,
            _handoffs: &'a [HandoffToolSpec],
            _tracing: ModelTracing,
            _previous_response_id: Option<&'a str>,
        ) -> Pin<Box<dyn Stream<Item = crate::Result<crate::items::ResponseStreamEvent>> + Send + 'a>>
        {
            let events: Vec<crate::Result<crate::items::ResponseStreamEvent>> =
                self.events.iter().map(|e| Ok(e.clone())).collect();
            Box::pin(tokio_stream::iter(events))
        }
    }

    // ---- Test: simple text response ----

    #[tokio::test]
    async fn simple_text_response() {
        let model = Arc::new(MockModel::single_message("Hello, world!"));
        let agent = Agent::<()>::builder("test-agent")
            .instructions("You are a helpful assistant.")
            .build();

        let result = Runner::run_with_model(&agent, "Hi", (), model, None, None)
            .await
            .expect("run should succeed");

        assert_eq!(result.last_agent_name, "test-agent");
        assert_eq!(result.final_output, json!("Hello, world!"));
        assert!(!result.new_items.is_empty());
        assert!(matches!(result.new_items[0], RunItem::MessageOutput(_)));
        assert_eq!(result.raw_responses.len(), 1);
    }

    // ---- Test: tool call and response cycle ----

    #[tokio::test]
    async fn tool_call_and_response_cycle() {
        use crate::items::ToolOutput;
        use crate::tool::{Tool, ToolContext, function_tool};
        use schemars::JsonSchema;
        use serde::Deserialize;

        #[derive(Deserialize, JsonSchema)]
        struct AddParams {
            a: i32,
            b: i32,
        }

        let add_tool = function_tool::<(), AddParams, _, _>(
            "add",
            "Add two numbers.",
            |_ctx: ToolContext<()>, params: AddParams| async move {
                Ok(ToolOutput::Text(format!("{}", params.a + params.b)))
            },
        )
        .expect("should create tool");

        // First response: model calls the tool.
        // Second response: model produces a final message.
        let model = Arc::new(MockModel::new(vec![
            ModelResponse {
                output: vec![json!({
                    "type": "function_call",
                    "name": "add",
                    "call_id": "call_001",
                    "arguments": r#"{"a": 3, "b": 4}"#
                })],
                usage: Usage {
                    requests: 1,
                    input_tokens: 20,
                    output_tokens: 10,
                    total_tokens: 30,
                    ..Usage::default()
                },
                response_id: Some("resp_1".to_owned()),
                request_id: None,
            },
            ModelResponse {
                output: vec![json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "The sum is 7."}]
                })],
                usage: Usage {
                    requests: 1,
                    input_tokens: 30,
                    output_tokens: 8,
                    total_tokens: 38,
                    ..Usage::default()
                },
                response_id: Some("resp_2".to_owned()),
                request_id: None,
            },
        ]));

        let agent = Agent::<()>::builder("calc-agent")
            .instructions("You are a calculator.")
            .tool(Tool::Function(add_tool))
            .build();

        let result = Runner::run_with_model(&agent, "What is 3 + 4?", (), model, None, None)
            .await
            .expect("run should succeed");

        assert_eq!(result.final_output, json!("The sum is 7."));
        assert_eq!(result.last_agent_name, "calc-agent");

        // Should have: ToolCall, ToolCallOutput (from turn 1), then MessageOutput (from turn 2).
        let has_tool_call = result
            .new_items
            .iter()
            .any(|i| matches!(i, RunItem::ToolCall(_)));
        let has_tool_output = result
            .new_items
            .iter()
            .any(|i| matches!(i, RunItem::ToolCallOutput(_)));
        let has_message = result
            .new_items
            .iter()
            .any(|i| matches!(i, RunItem::MessageOutput(_)));

        assert!(has_tool_call, "should have a ToolCall item");
        assert!(has_tool_output, "should have a ToolCallOutput item");
        assert!(has_message, "should have a MessageOutput item");

        // Usage should be accumulated from both turns.
        assert_eq!(result.usage.requests, 2);
        assert_eq!(result.usage.total_tokens, 68);
    }

    // ---- Test: max turns exceeded ----

    #[tokio::test]
    async fn max_turns_exceeded() {
        // Model always returns a tool call, so the loop never terminates.
        let responses: Vec<ModelResponse> = (0..5)
            .map(|i| ModelResponse {
                output: vec![json!({
                    "type": "function_call",
                    "name": "do_nothing",
                    "call_id": format!("call_{i}"),
                    "arguments": "{}"
                })],
                usage: Usage::default(),
                response_id: None,
                request_id: None,
            })
            .collect();

        let model = Arc::new(MockModel::new(responses));
        let agent = Agent::<()>::builder("looper").build();

        let config = RunConfig::builder().max_turns(3).build();
        let result = Runner::run_with_model(&agent, "Go", (), model, None, Some(config)).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentError::MaxTurnsExceeded { max_turns: 3 }),
            "expected MaxTurnsExceeded(3), got: {err:?}"
        );
    }

    // ---- Test: input guardrail blocks execution ----

    #[tokio::test]
    async fn input_guardrail_blocks_execution() {
        let model = Arc::new(MockModel::single_message("Should not see this."));

        let agent = Agent::<()>::builder("guarded-agent")
            .input_guardrail(InputGuardrail::new("block_all", |_ctx, _agent, _input| {
                Box::pin(async { Ok(GuardrailFunctionOutput::tripwire(json!("blocked"))) })
            }))
            .build();

        let result = Runner::run_with_model(&agent, "bad input", (), model, None, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                AgentError::InputGuardrailTripwire {
                    ref guardrail_name
                } if guardrail_name == "block_all"
            ),
            "expected InputGuardrailTripwire, got: {err:?}"
        );
    }

    // ---- Test: output guardrail blocks result ----

    #[tokio::test]
    async fn output_guardrail_blocks_result() {
        let model = Arc::new(MockModel::single_message("sensitive output"));

        let agent = Agent::<()>::builder("guarded-output")
            .output_guardrail(OutputGuardrail::new(
                "pii_check",
                |_ctx, _agent, _output| {
                    Box::pin(async { Ok(GuardrailFunctionOutput::tripwire(json!("PII detected"))) })
                },
            ))
            .build();

        let result = Runner::run_with_model(&agent, "tell me secrets", (), model, None, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                AgentError::OutputGuardrailTripwire {
                    ref guardrail_name
                } if guardrail_name == "pii_check"
            ),
            "expected OutputGuardrailTripwire, got: {err:?}"
        );
    }

    // ---- Test: lifecycle hooks are called ----

    struct TrackingHooks {
        calls: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl RunHooks<()> for TrackingHooks {
        async fn on_agent_start(&self, _ctx: &RunContextWrapper<()>, agent_name: &str) {
            self.calls
                .lock()
                .await
                .push(format!("agent_start:{agent_name}"));
        }

        async fn on_agent_end(
            &self,
            _ctx: &RunContextWrapper<()>,
            agent_name: &str,
            _output: &serde_json::Value,
        ) {
            self.calls
                .lock()
                .await
                .push(format!("agent_end:{agent_name}"));
        }

        async fn on_handoff(&self, _ctx: &RunContextWrapper<()>, from_agent: &str, to_agent: &str) {
            self.calls
                .lock()
                .await
                .push(format!("handoff:{from_agent}->{to_agent}"));
        }

        async fn on_llm_start(
            &self,
            _ctx: &RunContextWrapper<()>,
            agent_name: &str,
            _system_prompt: Option<&str>,
            _input: &[ResponseInputItem],
        ) {
            self.calls
                .lock()
                .await
                .push(format!("llm_start:{agent_name}"));
        }

        async fn on_llm_end(
            &self,
            _ctx: &RunContextWrapper<()>,
            agent_name: &str,
            _response: &ModelResponse,
        ) {
            self.calls
                .lock()
                .await
                .push(format!("llm_end:{agent_name}"));
        }
    }

    #[tokio::test]
    async fn lifecycle_hooks_are_called() {
        let calls = Arc::new(Mutex::new(Vec::<String>::new()));

        let hooks: Arc<dyn RunHooks<()>> = Arc::new(TrackingHooks {
            calls: Arc::clone(&calls),
        });

        let model = Arc::new(MockModel::single_message("Hello!"));
        let agent = Agent::<()>::builder("hooked-agent").build();

        let _result = Runner::run_with_model(&agent, "Hi", (), model, Some(hooks), None)
            .await
            .expect("run should succeed");

        let recorded = calls.lock().await;
        assert!(recorded.contains(&"agent_start:hooked-agent".to_owned()));
        assert!(recorded.contains(&"llm_start:hooked-agent".to_owned()));
        assert!(recorded.contains(&"llm_end:hooked-agent".to_owned()));
        assert!(recorded.contains(&"agent_end:hooked-agent".to_owned()));
        drop(recorded);
    }

    // ---- Test: usage accumulation ----

    #[tokio::test]
    async fn usage_accumulation() {
        let model = Arc::new(MockModel::new(vec![
            ModelResponse {
                output: vec![json!({
                    "type": "function_call",
                    "name": "noop",
                    "call_id": "c1",
                    "arguments": "{}"
                })],
                usage: Usage {
                    requests: 1,
                    input_tokens: 100,
                    output_tokens: 50,
                    total_tokens: 150,
                    ..Usage::default()
                },
                response_id: None,
                request_id: None,
            },
            ModelResponse {
                output: vec![json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}]
                })],
                usage: Usage {
                    requests: 1,
                    input_tokens: 200,
                    output_tokens: 80,
                    total_tokens: 280,
                    ..Usage::default()
                },
                response_id: None,
                request_id: None,
            },
        ]));

        let agent = Agent::<()>::builder("usage-agent").build();
        let result = Runner::run_with_model(&agent, "Go", (), model, None, None)
            .await
            .expect("run should succeed");

        assert_eq!(result.usage.requests, 2);
        assert_eq!(result.usage.input_tokens, 300);
        assert_eq!(result.usage.output_tokens, 130);
        assert_eq!(result.usage.total_tokens, 430);
    }

    // ---- Test: reasoning items are captured ----

    #[tokio::test]
    async fn reasoning_items_are_captured() {
        let model = Arc::new(MockModel::new(vec![ModelResponse {
            output: vec![
                json!({
                    "type": "reasoning",
                    "text": "Let me think about this..."
                }),
                json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Here is my answer."}]
                }),
            ],
            usage: Usage::default(),
            response_id: None,
            request_id: None,
        }]));

        let agent = Agent::<()>::builder("reasoner").build();
        let result = Runner::run_with_model(&agent, "Think", (), model, None, None)
            .await
            .expect("run should succeed");

        let reasoning_count = result
            .new_items
            .iter()
            .filter(|i| matches!(i, RunItem::Reasoning(_)))
            .count();
        assert_eq!(reasoning_count, 1, "should have one reasoning item");
        assert_eq!(result.final_output, json!("Here is my answer."));
    }

    // ---- Test: StopOnFirstTool behavior ----

    #[tokio::test]
    async fn stop_on_first_tool_behavior() {
        use crate::agent::ToolUseBehavior;
        use crate::items::ToolOutput;
        use crate::tool::{Tool, ToolContext, function_tool};
        use schemars::JsonSchema;
        use serde::Deserialize;

        #[derive(Deserialize, JsonSchema)]
        struct Empty {}

        let ft = function_tool::<(), Empty, _, _>(
            "get_data",
            "Get data.",
            |_ctx: ToolContext<()>, _params: Empty| async move {
                Ok(ToolOutput::Text("some data".to_owned()))
            },
        )
        .expect("should create tool");

        // Model returns a tool call.
        let model = Arc::new(MockModel::new(vec![ModelResponse {
            output: vec![json!({
                "type": "function_call",
                "name": "get_data",
                "call_id": "c1",
                "arguments": "{}"
            })],
            usage: Usage::default(),
            response_id: None,
            request_id: None,
        }]));

        let agent = Agent::<()>::builder("stop-tool")
            .tool(Tool::Function(ft))
            .tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .build();

        let result = Runner::run_with_model(&agent, "Go", (), model, None, None)
            .await
            .expect("run should succeed");

        // Should stop after one turn (tool was called -> StopOnFirstTool -> stop).
        assert_eq!(result.raw_responses.len(), 1);
    }

    // ---- Test: Runner is Send + Sync ----

    #[test]
    fn runner_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Runner>();
    }

    // ---- Test: input guardrail passes and result is stored ----

    #[tokio::test]
    async fn input_guardrail_passes_and_stored() {
        let model = Arc::new(MockModel::single_message("OK"));

        let agent = Agent::<()>::builder("guarded")
            .input_guardrail(InputGuardrail::new("safe_check", |_ctx, _agent, _input| {
                Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!("safe"))) })
            }))
            .build();

        let result = Runner::run_with_model(&agent, "safe input", (), model, None, None)
            .await
            .expect("run should succeed");

        assert_eq!(result.input_guardrail_results.len(), 1);
        assert_eq!(
            result.input_guardrail_results[0].guardrail_name,
            "safe_check"
        );
        assert!(!result.input_guardrail_results[0].output.tripwire_triggered);
    }

    // ---- Test: output guardrail passes and result is stored ----

    #[tokio::test]
    async fn output_guardrail_passes_and_stored() {
        let model = Arc::new(MockModel::single_message("clean output"));

        let agent = Agent::<()>::builder("guarded")
            .output_guardrail(OutputGuardrail::new(
                "clean_check",
                |_ctx, _agent, _output| {
                    Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!("clean"))) })
                },
            ))
            .build();

        let result = Runner::run_with_model(&agent, "check output", (), model, None, None)
            .await
            .expect("run should succeed");

        assert_eq!(result.output_guardrail_results.len(), 1);
        assert_eq!(
            result.output_guardrail_results[0].guardrail_name,
            "clean_check"
        );
    }

    // ---- Test: multi-agent handoff (agent A hands off to agent B, B produces output) ----

    #[tokio::test]
    async fn multi_agent_handoff() {
        // Agent A receives input, returns a handoff call to B.
        // Agent B receives the continued conversation and returns a message.
        let model = Arc::new(MockModel::new(vec![
            // Turn 1: Agent A calls handoff to B.
            ModelResponse {
                output: vec![json!({
                    "type": "function_call",
                    "name": "transfer_to_agent_b",
                    "call_id": "handoff_call_1",
                    "arguments": "{}"
                })],
                usage: Usage {
                    requests: 1,
                    input_tokens: 10,
                    output_tokens: 5,
                    total_tokens: 15,
                    ..Usage::default()
                },
                response_id: Some("resp_a".to_owned()),
                request_id: None,
            },
            // Turn 2: Agent B produces final output.
            ModelResponse {
                output: vec![json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello from Agent B!"}]
                })],
                usage: Usage {
                    requests: 1,
                    input_tokens: 20,
                    output_tokens: 10,
                    total_tokens: 30,
                    ..Usage::default()
                },
                response_id: Some("resp_b".to_owned()),
                request_id: None,
            },
        ]));

        let agent_a = Agent::<()>::builder("agent_a")
            .instructions("You are Agent A.")
            .handoff(crate::handoffs::Handoff::to_agent("agent_b").build())
            .build();

        let agent_b = Agent::<()>::builder("agent_b")
            .instructions("You are Agent B.")
            .build();

        let mut agents: HashMap<String, &Agent<()>> = HashMap::new();
        agents.insert("agent_b".to_owned(), &agent_b);

        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let hooks: Arc<dyn RunHooks<()>> = Arc::new(TrackingHooks {
            calls: Arc::clone(&calls),
        });

        let result =
            Runner::run_with_agents(&agent_a, &agents, "Hello", (), model, Some(hooks), None)
                .await
                .expect("run should succeed");

        assert_eq!(result.final_output, json!("Hello from Agent B!"));
        assert_eq!(result.last_agent_name, "agent_b");

        // Should have handoff items.
        let has_handoff_call = result
            .new_items
            .iter()
            .any(|i| matches!(i, RunItem::HandoffCall(_)));
        let has_handoff_output = result
            .new_items
            .iter()
            .any(|i| matches!(i, RunItem::HandoffOutput(_)));
        assert!(has_handoff_call, "should have a HandoffCall item");
        assert!(has_handoff_output, "should have a HandoffOutput item");

        // Verify handoff hook was fired.
        let recorded = calls.lock().await;
        assert!(
            recorded.contains(&"handoff:agent_a->agent_b".to_owned()),
            "handoff hook should have fired, got: {recorded:?}"
        );
        drop(recorded);

        // Usage should be accumulated from both agents.
        assert_eq!(result.usage.requests, 2);
        assert_eq!(result.usage.total_tokens, 45);
    }

    // ---- Test: handoff to unknown agent returns error ----

    #[tokio::test]
    async fn handoff_to_unknown_agent_returns_error() {
        let model = Arc::new(MockModel::new(vec![ModelResponse {
            output: vec![json!({
                "type": "function_call",
                "name": "transfer_to_nonexistent",
                "call_id": "handoff_call_1",
                "arguments": "{}"
            })],
            usage: Usage::default(),
            response_id: None,
            request_id: None,
        }]));

        let agent = Agent::<()>::builder("agent_a")
            .handoff(crate::handoffs::Handoff::to_agent("nonexistent").build())
            .build();

        // Registry has no "nonexistent" agent.
        let agents: HashMap<String, &Agent<()>> = HashMap::new();
        // We need at least one entry to indicate multi-agent mode.
        let mut agents_with_dummy = agents;
        let dummy = Agent::<()>::builder("dummy").build();
        agents_with_dummy.insert("dummy".to_owned(), &dummy);

        let result =
            Runner::run_with_agents(&agent, &agents_with_dummy, "Hello", (), model, None, None)
                .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(&err, AgentError::UserError { message } if message.contains("nonexistent")),
            "expected UserError about missing agent, got: {err:?}"
        );
    }

    // ---- Test: streaming produces events ----

    #[tokio::test]
    async fn streaming_produces_events() {
        use crate::stream_events::StreamEvent;
        use tokio_stream::StreamExt;

        // We need a 'static agent for run_streamed.
        static STREAMING_AGENT: std::sync::LazyLock<Agent<()>> = std::sync::LazyLock::new(|| {
            Agent::<()>::builder("stream-agent")
                .instructions("You are a streaming assistant.")
                .build()
        });

        let model: Arc<dyn Model> = Arc::new(StreamingMockModel {
            events: vec![
                json!({"type": "response.output_text.delta", "delta": "Hello"}),
                json!({"type": "response.output_text.delta", "delta": " world"}),
                json!({"type": "message", "content": [{"type": "output_text", "text": "Hello world"}]}),
            ],
        });

        let mut result = Runner::run_streamed(&STREAMING_AGENT, "Hi", (), model, None, None);

        let mut stream = result.stream_events();
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event);
        }

        // Should have at least the raw response events.
        assert!(
            events.len() >= 3,
            "expected at least 3 events, got {}",
            events.len()
        );

        // First events should be raw responses.
        assert!(matches!(&events[0], StreamEvent::RawResponse(_)));
        assert!(matches!(&events[1], StreamEvent::RawResponse(_)));
        assert!(matches!(&events[2], StreamEvent::RawResponse(_)));
    }

    // ---- Test: streaming cancel works ----

    #[tokio::test]
    async fn streaming_cancel_works() {
        use tokio_stream::StreamExt;

        // Slow streaming model that yields events slowly.
        struct SlowStreamModel;

        #[async_trait]
        impl Model for SlowStreamModel {
            async fn get_response(
                &self,
                _system_instructions: Option<&str>,
                _input: &[ResponseInputItem],
                _model_settings: &ModelSettings,
                _tools: &[ToolSpec],
                _output_schema: Option<&OutputSchemaSpec>,
                _handoffs: &[HandoffToolSpec],
                _tracing: ModelTracing,
                _previous_response_id: Option<&str>,
            ) -> crate::Result<ModelResponse> {
                Ok(ModelResponse {
                    output: vec![],
                    usage: Usage::default(),
                    response_id: None,
                    request_id: None,
                })
            }

            fn stream_response<'a>(
                &'a self,
                _system_instructions: Option<&'a str>,
                _input: &'a [ResponseInputItem],
                _model_settings: &'a ModelSettings,
                _tools: &'a [ToolSpec],
                _output_schema: Option<&'a OutputSchemaSpec>,
                _handoffs: &'a [HandoffToolSpec],
                _tracing: ModelTracing,
                _previous_response_id: Option<&'a str>,
            ) -> Pin<
                Box<
                    dyn Stream<Item = crate::Result<crate::items::ResponseStreamEvent>> + Send + 'a,
                >,
            > {
                // Yield events with delays.
                Box::pin(async_stream::stream! {
                    for i in 0..100 {
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                        yield Ok(json!({"type": "delta", "index": i}));
                    }
                })
            }
        }

        static CANCEL_AGENT: std::sync::LazyLock<Agent<()>> =
            std::sync::LazyLock::new(|| Agent::<()>::builder("cancel-agent").build());

        let model: Arc<dyn Model> = Arc::new(SlowStreamModel);

        let mut result = Runner::run_streamed(&CANCEL_AGENT, "Hi", (), model, None, None);

        let mut stream = result.stream_events();

        // Read a few events.
        let first_event = stream.next().await;
        assert!(first_event.is_some(), "should get at least one event");

        // Cancel the stream.
        drop(stream);
        result.cancel();

        // Give the background task time to notice cancellation.
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // The test passes if it completes without hanging.
    }
}
