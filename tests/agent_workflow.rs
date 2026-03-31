//! Integration tests for complete agent workflows using mock models.
//!
//! These tests verify that the runner correctly orchestrates the full agent loop:
//! model calls, tool execution, guardrails, lifecycle hooks, and turn limits.

#![allow(clippy::significant_drop_tightening)]

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tokio_stream::Stream;

use openai_agents::agent::Agent;
use openai_agents::config::{ModelSettings, RunConfig};
use openai_agents::context::RunContextWrapper;
use openai_agents::error::AgentError;
use openai_agents::guardrail::{GuardrailFunctionOutput, InputGuardrail};
use openai_agents::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent, ToolOutput};
use openai_agents::lifecycle::RunHooks;
use openai_agents::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use openai_agents::result::RunResult;
use openai_agents::runner::Runner;
use openai_agents::tool::{ToolContext, function_tool};
use openai_agents::usage::Usage;

// ---------------------------------------------------------------------------
// Mock model infrastructure
// ---------------------------------------------------------------------------

/// A mock model that returns pre-configured responses in sequence.
///
/// Each call to `get_response` pops the next response from the queue.
/// If the queue is empty, returns an empty message response.
struct MockModel {
    responses: Mutex<Vec<ModelResponse>>,
}

impl MockModel {
    fn new(responses: Vec<ModelResponse>) -> Self {
        // Reverse so we can pop from the end efficiently.
        let mut reversed = responses;
        reversed.reverse();
        Self {
            responses: Mutex::new(reversed),
        }
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
    ) -> openai_agents::error::Result<ModelResponse> {
        let response = self.responses.lock().unwrap().pop().unwrap_or_else(|| {
            ModelResponse::new(
                vec![json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "(no response configured)"}]
                })],
                Usage::default(),
                None,
                None,
            )
        });
        Ok(response)
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
    ) -> Pin<Box<dyn Stream<Item = openai_agents::error::Result<ResponseStreamEvent>> + Send + 'a>>
    {
        Box::pin(tokio_stream::empty())
    }
}

/// Convenience constructor: build a model response that returns a plain text message.
fn text_response(text: &str) -> ModelResponse {
    let mut usage = Usage::default();
    usage.requests = 1;
    usage.input_tokens = 10;
    usage.output_tokens = 5;
    usage.total_tokens = 15;
    ModelResponse::new(
        vec![json!({
            "type": "message",
            "content": [{"type": "output_text", "text": text}]
        })],
        usage,
        Some("resp_001".to_owned()),
        None,
    )
}

/// Convenience constructor: build a model response that calls a function tool.
fn function_call_response(tool_name: &str, call_id: &str, args_json: &str) -> ModelResponse {
    let mut usage = Usage::default();
    usage.requests = 1;
    usage.input_tokens = 20;
    usage.output_tokens = 10;
    usage.total_tokens = 30;
    ModelResponse::new(
        vec![json!({
            "type": "function_call",
            "name": tool_name,
            "call_id": call_id,
            "arguments": args_json,
        })],
        usage,
        Some("resp_tool".to_owned()),
        None,
    )
}

// ---------------------------------------------------------------------------
// Helper to run an agent against a MockModel
// ---------------------------------------------------------------------------

async fn run_agent(
    agent: &Agent<()>,
    input: &str,
    responses: Vec<ModelResponse>,
    config: Option<RunConfig>,
) -> openai_agents::error::Result<RunResult> {
    let model = Arc::new(MockModel::new(responses));
    Runner::run_with_model(agent, input, (), model, None, config).await
}

// ---------------------------------------------------------------------------
// Test 1: Simple Q&A agent
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_simple_qa_agent_returns_expected_output() {
    let agent = Agent::<()>::builder("qa-bot")
        .instructions("You are a helpful assistant.")
        .build();

    let result = run_agent(
        &agent,
        "What is 2+2?",
        vec![text_response("The answer is 4.")],
        None,
    )
    .await
    .expect("run should succeed");

    assert_eq!(result.last_agent_name, "qa-bot");
    // The final output should contain text from the message.
    let output_str = result.final_output.to_string();
    assert!(
        output_str.contains("The answer is 4."),
        "expected output to contain the answer, got: {output_str}"
    );
}

#[tokio::test]
async fn test_simple_qa_agent_populates_new_items() {
    let agent = Agent::<()>::builder("assistant").build();

    let result = run_agent(&agent, "Hello", vec![text_response("Hi there!")], None)
        .await
        .expect("run should succeed");

    // At minimum we should have one MessageOutput item.
    assert!(
        !result.new_items.is_empty(),
        "expected at least one run item"
    );

    // Verify it is a MessageOutput.
    assert!(
        result
            .new_items
            .iter()
            .any(|item| matches!(item, openai_agents::items::RunItem::MessageOutput(_))),
        "expected a MessageOutput item"
    );
}

#[tokio::test]
async fn test_simple_qa_agent_accumulates_usage() {
    let agent = Agent::<()>::builder("assistant").build();

    let result = run_agent(&agent, "Hello", vec![text_response("World!")], None)
        .await
        .expect("run should succeed");

    // Usage should be non-zero since we configured token counts on the response.
    assert!(
        result.usage.total_tokens > 0,
        "expected non-zero token usage, got: {}",
        result.usage.total_tokens
    );
    assert_eq!(result.usage.requests, 1);
}

#[tokio::test]
async fn test_simple_qa_agent_records_raw_responses() {
    let agent = Agent::<()>::builder("assistant").build();

    let result = run_agent(&agent, "Ping", vec![text_response("Pong")], None)
        .await
        .expect("run should succeed");

    assert_eq!(result.raw_responses.len(), 1);
    assert_eq!(
        result.raw_responses[0].response_id.as_deref(),
        Some("resp_001")
    );
}

#[tokio::test]
async fn test_simple_qa_agent_preserves_input() {
    let agent = Agent::<()>::builder("assistant").build();

    let result = run_agent(
        &agent,
        "My original question",
        vec![text_response("My answer")],
        None,
    )
    .await
    .expect("run should succeed");

    match &result.input {
        openai_agents::items::InputContent::Text(s) => {
            assert_eq!(s, "My original question");
        }
        other => panic!("expected Text input, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 2: Agent with tools — two-turn cycle
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct AddParams {
    a: i64,
    b: i64,
}

#[tokio::test]
async fn test_agent_with_tool_executes_two_turn_cycle() {
    let calc_tool = function_tool::<(), AddParams, _, _>(
        "add",
        "Add two numbers.",
        |_ctx: ToolContext<()>, params: AddParams| async move {
            Ok(ToolOutput::Text(format!("{}", params.a + params.b)))
        },
    )
    .expect("tool creation should succeed");

    let agent = Agent::<()>::builder("calculator")
        .instructions("You are a calculator.")
        .tool(openai_agents::tool::Tool::Function(calc_tool))
        .build();

    // Turn 1: model calls the "add" tool.
    // Turn 2: model uses the tool result and produces a final message.
    let result = run_agent(
        &agent,
        "What is 3 + 4?",
        vec![
            function_call_response("add", "call_abc", r#"{"a": 3, "b": 4}"#),
            text_response("The answer is 7."),
        ],
        None,
    )
    .await
    .expect("run should succeed");

    // Verify the final output contains the expected text.
    let output_str = result.final_output.to_string();
    assert!(
        output_str.contains('7'),
        "expected output to contain 7, got: {output_str}"
    );

    // Verify we have a ToolCallOutput item.
    let has_tool_output = result
        .new_items
        .iter()
        .any(|item| matches!(item, openai_agents::items::RunItem::ToolCallOutput(_)));
    assert!(
        has_tool_output,
        "expected a ToolCallOutput item in new_items"
    );

    // Verify tool call item is also present.
    let has_tool_call = result
        .new_items
        .iter()
        .any(|item| matches!(item, openai_agents::items::RunItem::ToolCall(_)));
    assert!(has_tool_call, "expected a ToolCall item in new_items");

    // Two turns means two raw responses.
    assert_eq!(
        result.raw_responses.len(),
        2,
        "expected two model responses"
    );
}

#[tokio::test]
async fn test_agent_with_tool_accumulates_multi_turn_usage() {
    let calc_tool = function_tool::<(), AddParams, _, _>(
        "add",
        "Add two numbers.",
        |_ctx: ToolContext<()>, params: AddParams| async move {
            Ok(ToolOutput::Text(format!("{}", params.a + params.b)))
        },
    )
    .expect("tool creation should succeed");

    let agent = Agent::<()>::builder("calculator")
        .tool(openai_agents::tool::Tool::Function(calc_tool))
        .build();

    let result = run_agent(
        &agent,
        "5 + 6?",
        vec![
            function_call_response("add", "call_1", r#"{"a": 5, "b": 6}"#),
            text_response("11"),
        ],
        None,
    )
    .await
    .expect("run should succeed");

    // Both turns should have their usage accumulated.
    assert_eq!(result.usage.requests, 2, "expected 2 requests");
    assert!(
        result.usage.total_tokens > 0,
        "expected non-zero total tokens"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Agent with input guardrail
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_input_guardrail_blocks_forbidden_input() {
    let guardrail = InputGuardrail::<()>::new("block_all", |_ctx, _agent, _input| {
        Box::pin(async {
            Ok(GuardrailFunctionOutput::tripwire(json!({
                "reason": "all input blocked"
            })))
        })
    });

    let agent = Agent::<()>::builder("guarded-agent")
        .instructions("You are helpful.")
        .input_guardrail(guardrail)
        .build();

    let err = run_agent(&agent, "hello", vec![text_response("hi")], None)
        .await
        .expect_err("should fail with guardrail tripwire");

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

#[tokio::test]
async fn test_input_guardrail_allows_safe_input() {
    let guardrail = InputGuardrail::<()>::new("safe_check", |_ctx, _agent, _input| {
        Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
    });

    let agent = Agent::<()>::builder("guarded-agent")
        .instructions("You are helpful.")
        .input_guardrail(guardrail)
        .build();

    let result = run_agent(&agent, "safe input", vec![text_response("OK")], None)
        .await
        .expect("safe input should pass guardrail");

    assert_eq!(result.input_guardrail_results.len(), 1);
    assert!(!result.input_guardrail_results[0].output.tripwire_triggered);
}

#[tokio::test]
async fn test_input_guardrail_result_stored_on_success() {
    let guardrail = InputGuardrail::<()>::new("my_guardrail", |_ctx, _agent, _input| {
        Box::pin(async {
            Ok(GuardrailFunctionOutput::passed(json!({
                "checked": true
            })))
        })
    });

    let agent = Agent::<()>::builder("agent")
        .input_guardrail(guardrail)
        .build();

    let result = run_agent(&agent, "test", vec![text_response("ok")], None)
        .await
        .expect("should succeed");

    assert_eq!(result.input_guardrail_results.len(), 1);
    assert_eq!(
        result.input_guardrail_results[0].guardrail_name,
        "my_guardrail"
    );
    assert_eq!(
        result.input_guardrail_results[0].output.output_info,
        json!({"checked": true})
    );
}

#[tokio::test]
async fn test_multiple_guardrails_all_pass() {
    let g1 = InputGuardrail::<()>::new("check_1", |_ctx, _agent, _input| {
        Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!("pass"))) })
    });
    let g2 = InputGuardrail::<()>::new("check_2", |_ctx, _agent, _input| {
        Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!("pass"))) })
    });

    let agent = Agent::<()>::builder("agent")
        .input_guardrail(g1)
        .input_guardrail(g2)
        .build();

    let result = run_agent(&agent, "test", vec![text_response("done")], None)
        .await
        .expect("both guardrails pass");

    assert_eq!(result.input_guardrail_results.len(), 2);
}

// ---------------------------------------------------------------------------
// Test 4: Agent with lifecycle hooks
// ---------------------------------------------------------------------------

/// Tracks which lifecycle hooks were called and in what order.
struct TrackingHooks {
    calls: Arc<Mutex<Vec<String>>>,
}

impl TrackingHooks {
    fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

#[async_trait]
impl RunHooks<()> for TrackingHooks {
    async fn on_agent_start(&self, _ctx: &RunContextWrapper<()>, agent_name: &str) {
        self.calls
            .lock()
            .unwrap()
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
            .unwrap()
            .push(format!("agent_end:{agent_name}"));
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
            .unwrap()
            .push(format!("llm_start:{agent_name}"));
    }

    async fn on_llm_end(
        &self,
        _ctx: &RunContextWrapper<()>,
        agent_name: &str,
        _response: &openai_agents::items::ModelResponse,
    ) {
        self.calls
            .lock()
            .unwrap()
            .push(format!("llm_end:{agent_name}"));
    }
}

#[tokio::test]
async fn test_lifecycle_hooks_fire_in_correct_order() {
    let (hooks, calls) = TrackingHooks::new();

    let agent = Agent::<()>::builder("hook-agent")
        .instructions("Be helpful.")
        .build();

    let model = Arc::new(MockModel::new(vec![text_response("Hello from the model")]));
    Runner::run_with_model(&agent, "Hi", (), model, Some(Arc::new(hooks)), None)
        .await
        .expect("run should succeed");

    let recorded = calls.lock().unwrap();
    assert!(
        recorded.len() >= 4,
        "expected at least 4 hook calls, got: {recorded:?}"
    );

    // The order must be: agent_start, llm_start, llm_end, agent_end.
    assert_eq!(recorded[0], "agent_start:hook-agent");
    assert_eq!(recorded[1], "llm_start:hook-agent");
    assert_eq!(recorded[2], "llm_end:hook-agent");
    assert_eq!(recorded[3], "agent_end:hook-agent");
}

/// Tracks tool-related lifecycle hooks.
struct ToolTrackingHooks {
    calls: Arc<Mutex<Vec<String>>>,
}

impl ToolTrackingHooks {
    fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

#[async_trait]
impl RunHooks<()> for ToolTrackingHooks {
    async fn on_tool_start(&self, _ctx: &RunContextWrapper<()>, agent_name: &str, tool_name: &str) {
        self.calls
            .lock()
            .unwrap()
            .push(format!("tool_start:{agent_name}:{tool_name}"));
    }

    async fn on_tool_end(
        &self,
        _ctx: &RunContextWrapper<()>,
        agent_name: &str,
        tool_name: &str,
        result: &str,
    ) {
        self.calls
            .lock()
            .unwrap()
            .push(format!("tool_end:{agent_name}:{tool_name}={result}"));
    }
}

#[tokio::test]
async fn test_tool_lifecycle_hooks_fire() {
    let (hooks, calls) = ToolTrackingHooks::new();

    let tool = function_tool::<(), AddParams, _, _>(
        "add",
        "Add numbers.",
        |_ctx: ToolContext<()>, params: AddParams| async move {
            Ok(ToolOutput::Text(format!("{}", params.a + params.b)))
        },
    )
    .expect("tool should be created");

    let agent = Agent::<()>::builder("tool-agent")
        .tool(openai_agents::tool::Tool::Function(tool))
        .build();

    let model = Arc::new(MockModel::new(vec![
        function_call_response("add", "call_1", r#"{"a": 2, "b": 3}"#),
        text_response("5"),
    ]));

    Runner::run_with_model(&agent, "Calculate", (), model, Some(Arc::new(hooks)), None)
        .await
        .expect("run should succeed");

    let recorded = calls.lock().unwrap();
    let tool_start = recorded
        .iter()
        .find(|s| s.starts_with("tool_start:"))
        .cloned();
    let tool_end = recorded
        .iter()
        .find(|s| s.starts_with("tool_end:"))
        .cloned();

    assert!(
        tool_start.is_some(),
        "expected tool_start hook to fire, calls: {recorded:?}"
    );
    assert!(
        tool_end.is_some(),
        "expected tool_end hook to fire, calls: {recorded:?}"
    );
    assert!(
        tool_start.unwrap().contains("add"),
        "tool_start should mention 'add'"
    );
    assert!(
        tool_end.unwrap().contains('5'),
        "tool_end should include tool result '5'"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Max turns exceeded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_max_turns_exceeded_returns_error() {
    // A model that always calls a tool (never produces a final message).
    // With max_turns=1, the runner should return MaxTurnsExceeded after 1 turn
    // because no final message is produced and there are tool calls to continue.
    // Actually the runner stops when no tool calls AND no handoffs — since we
    // always return a tool call, it will try turn 2 but max_turns=1 ends first.
    let config = RunConfig::builder().max_turns(1).build();

    // The tool that is "called" — doesn't need to exist in the agent since we
    // will not reach turn 2.  But to ensure the runner stays in the loop we
    // configure a tool response for the only turn allowed.
    let agent = Agent::<()>::builder("looping-agent").build();

    // Always respond with a function call, never a final message.
    // With max_turns=1, after processing turn 0 (which has a tool call but no
    // stop condition), the loop ends and MaxTurnsExceeded is returned.
    let err = run_agent(
        &agent,
        "Do something",
        vec![
            // If the runner somehow executes more than one turn it will pop this
            // but we expect MaxTurnsExceeded after the first turn's tool call
            // is processed without producing a final output.
            function_call_response("nonexistent_tool", "c1", "{}"),
        ],
        Some(config),
    )
    .await
    .expect_err("should fail with MaxTurnsExceeded");

    assert!(
        matches!(err, AgentError::MaxTurnsExceeded { max_turns: 1 }),
        "expected MaxTurnsExceeded with max_turns=1, got: {err:?}"
    );
}

#[tokio::test]
async fn test_max_turns_exceeded_error_message() {
    let config = RunConfig::builder().max_turns(2).build();
    let agent = Agent::<()>::builder("looping-agent").build();

    // Provide tool call responses for both turns so the loop keeps going.
    let err = run_agent(
        &agent,
        "Do something",
        vec![
            function_call_response("no_tool", "c1", "{}"),
            function_call_response("no_tool", "c2", "{}"),
        ],
        Some(config),
    )
    .await
    .expect_err("should fail with MaxTurnsExceeded");

    assert_eq!(err.to_string(), "max turns (2) exceeded");
}

// ---------------------------------------------------------------------------
// Test 6: Multi-turn tool conversation — verify all RunItems are present
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_multi_turn_tool_conversation_all_items_present() {
    let greet_tool = function_tool::<(), GreetParams, _, _>(
        "greet",
        "Greet a person.",
        |_ctx: ToolContext<()>, params: GreetParams| async move {
            Ok(ToolOutput::Text(format!("Hello, {}!", params.name)))
        },
    )
    .expect("tool creation should succeed");

    let agent = Agent::<()>::builder("greeter")
        .tool(openai_agents::tool::Tool::Function(greet_tool))
        .build();

    // Turn 1: model calls greet("Alice").
    // Turn 2: model produces final message after seeing the tool result.
    let result = run_agent(
        &agent,
        "Greet Alice",
        vec![
            function_call_response("greet", "call_g1", r#"{"name": "Alice"}"#),
            text_response("I've greeted Alice for you."),
        ],
        None,
    )
    .await
    .expect("run should succeed");

    // Verify item types present: ToolCall, ToolCallOutput, MessageOutput.
    let has_tool_call = result
        .new_items
        .iter()
        .any(|item| matches!(item, openai_agents::items::RunItem::ToolCall(_)));
    let has_tool_output = result
        .new_items
        .iter()
        .any(|item| matches!(item, openai_agents::items::RunItem::ToolCallOutput(_)));
    let has_message = result
        .new_items
        .iter()
        .any(|item| matches!(item, openai_agents::items::RunItem::MessageOutput(_)));

    assert!(has_tool_call, "expected ToolCall item");
    assert!(has_tool_output, "expected ToolCallOutput item");
    assert!(has_message, "expected MessageOutput item");

    // Two model responses (one per turn).
    assert_eq!(result.raw_responses.len(), 2);

    // Final output reflects the last message.
    let output_str = result.final_output.to_string();
    assert!(
        output_str.contains("Alice"),
        "final output should mention Alice, got: {output_str}"
    );
}

#[tokio::test]
async fn test_multi_turn_tool_result_passed_back_to_model() {
    // We verify the tool result is correct by inspecting the ToolCallOutput item.
    let tool = function_tool::<(), AddParams, _, _>(
        "add",
        "Add numbers.",
        |_ctx: ToolContext<()>, params: AddParams| async move {
            Ok(ToolOutput::Text(format!("{}", params.a + params.b)))
        },
    )
    .expect("tool creation should succeed");

    let agent = Agent::<()>::builder("math-agent")
        .tool(openai_agents::tool::Tool::Function(tool))
        .build();

    let result = run_agent(
        &agent,
        "What is 10 + 20?",
        vec![
            function_call_response("add", "call_sum", r#"{"a": 10, "b": 20}"#),
            text_response("The sum is 30."),
        ],
        None,
    )
    .await
    .expect("run should succeed");

    // Find the ToolCallOutput and verify the output value.
    let tool_output_item = result
        .new_items
        .iter()
        .find(|item| matches!(item, openai_agents::items::RunItem::ToolCallOutput(_)));

    if let Some(openai_agents::items::RunItem::ToolCallOutput(tco)) = tool_output_item {
        // The output field should contain the computed sum.
        let output_str = tco.output.to_string();
        assert!(
            output_str.contains("30"),
            "tool output should contain 30, got: {output_str}"
        );
    } else {
        panic!("expected a ToolCallOutput item");
    }
}

// ---------------------------------------------------------------------------
// Test 7: Agent with no instructions still runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_agent_without_instructions_runs_normally() {
    let agent = Agent::<()>::builder("bare-agent").build();

    let result = run_agent(&agent, "Hello", vec![text_response("World")], None)
        .await
        .expect("agent without instructions should run");

    assert_eq!(result.last_agent_name, "bare-agent");
}

// ---------------------------------------------------------------------------
// Test 8: RunConfig workflow_name and tracing_disabled
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_custom_run_config_is_accepted() {
    let config = RunConfig::builder()
        .max_turns(5)
        .workflow_name("test_workflow")
        .tracing_disabled(true)
        .build();

    let agent = Agent::<()>::builder("config-agent").build();

    let result = run_agent(&agent, "test", vec![text_response("ok")], Some(config))
        .await
        .expect("run should succeed");

    assert_eq!(result.last_agent_name, "config-agent");
}

// ---------------------------------------------------------------------------
// Test 9: to_input_list can be used for follow-up runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_to_input_list_contains_all_response_items() {
    let agent = Agent::<()>::builder("agent").build();

    let result = run_agent(&agent, "Q1", vec![text_response("A1")], None)
        .await
        .expect("run should succeed");

    let input_list = result.to_input_list();
    // The response contains one message item, so we should have at least one entry.
    assert!(
        !input_list.is_empty(),
        "to_input_list should not be empty after a successful run"
    );
}

// ---------------------------------------------------------------------------
// Test 10: Multiple sequential tool calls in one turn
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_multiple_tool_calls_in_one_turn() {
    let tool = function_tool::<(), AddParams, _, _>(
        "add",
        "Add numbers.",
        |_ctx: ToolContext<()>, params: AddParams| async move {
            Ok(ToolOutput::Text(format!("{}", params.a + params.b)))
        },
    )
    .expect("tool creation should succeed");

    let agent = Agent::<()>::builder("multi-call-agent")
        .tool(openai_agents::tool::Tool::Function(tool))
        .build();

    // Provide a response with two tool calls in one turn.
    let two_calls_response = ModelResponse::new(
        vec![
            json!({
                "type": "function_call",
                "name": "add",
                "call_id": "c1",
                "arguments": r#"{"a": 1, "b": 2}"#,
            }),
            json!({
                "type": "function_call",
                "name": "add",
                "call_id": "c2",
                "arguments": r#"{"a": 10, "b": 20}"#,
            }),
        ],
        Usage::default(),
        None,
        None,
    );

    let result = run_agent(
        &agent,
        "compute two sums",
        vec![two_calls_response, text_response("Done: 3 and 30")],
        None,
    )
    .await
    .expect("run should succeed");

    // Should have two ToolCall items and two ToolCallOutput items.
    let tool_call_count = result
        .new_items
        .iter()
        .filter(|item| matches!(item, openai_agents::items::RunItem::ToolCall(_)))
        .count();
    let tool_output_count = result
        .new_items
        .iter()
        .filter(|item| matches!(item, openai_agents::items::RunItem::ToolCallOutput(_)))
        .count();

    assert_eq!(tool_call_count, 2, "expected 2 tool call items");
    assert_eq!(tool_output_count, 2, "expected 2 tool output items");
}

// ---------------------------------------------------------------------------
// Supporting types for tests above
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct GreetParams {
    name: String,
}
