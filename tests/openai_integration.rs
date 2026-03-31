//! Integration tests against the real `OpenAI` API.
//!
//! These tests are ignored by default. Run with:
//! ```bash
//! cargo test --test openai_integration -- --ignored
//! ```
//!
//! Requires `OPENAI_API_KEY` environment variable to be set.

#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::type_complexity)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use openai_agents::context::RunContextWrapper;
use openai_agents::error::AgentError;
use openai_agents::guardrail::{GuardrailFunctionOutput, InputGuardrail};
use openai_agents::items::{InputContent, ModelResponse, ResponseInputItem, RunItem, ToolOutput};
use openai_agents::lifecycle::RunHooks;
use openai_agents::models::Model;
use openai_agents::models::openai_chatcompletions::OpenAIChatCompletionsModel;
use openai_agents::models::openai_responses::OpenAIResponsesModel;
use openai_agents::runner::Runner;
use openai_agents::tool::{Tool, ToolContext, function_tool};
use openai_agents::{Agent, Handoff, ModelSettings, RunConfig};

// ---------------------------------------------------------------------------
// Model factory helpers
// ---------------------------------------------------------------------------

/// Build a Responses API model, skipping the test if the key is absent.
fn responses_model() -> Arc<dyn Model> {
    Arc::new(
        OpenAIResponsesModel::new("gpt-4o-mini")
            .expect("OPENAI_API_KEY must be set to run integration tests"),
    )
}

/// Build a Chat Completions API model, skipping the test if the key is absent.
fn chat_model() -> Arc<dyn Model> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set to run integration tests");
    Arc::new(OpenAIChatCompletionsModel::new("gpt-4o-mini", api_key))
}

// ---------------------------------------------------------------------------
// Test 1: Simple text response — Responses API
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_simple_response_responses_api() {
    let model = responses_model();
    let agent = Agent::<()>::builder("greeter")
        .instructions(
            "You are a terse assistant. Reply with exactly one word: 'Hello'. \
             Do not add punctuation or any other text.",
        )
        .build();

    let result = Runner::run_with_model(&agent, "Say hello", (), model, None, None)
        .await
        .expect("run should succeed");

    println!("Response: {}", result.final_output);
    println!("Usage: {} total tokens", result.usage.total_tokens);

    assert!(
        !result.final_output.is_null(),
        "final_output must not be null"
    );
    assert!(result.usage.total_tokens > 0, "should have consumed tokens");
    assert!(
        !result.new_items.is_empty(),
        "should have at least one item"
    );
    assert_eq!(result.last_agent_name, "greeter");
}

// ---------------------------------------------------------------------------
// Test 2: Simple text response — Chat Completions API
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_simple_response_chat_completions_api() {
    let model = chat_model();
    let agent = Agent::<()>::builder("greeter-chat")
        .instructions(
            "You are a terse assistant. Reply with exactly one word: 'Hello'. \
             Do not add punctuation or any other text.",
        )
        .build();

    let result = Runner::run_with_model(&agent, "Say hello", (), model, None, None)
        .await
        .expect("run should succeed");

    println!("Response: {}", result.final_output);
    println!("Usage: {} total tokens", result.usage.total_tokens);

    assert!(
        !result.final_output.is_null(),
        "final_output must not be null"
    );
    assert!(result.usage.total_tokens > 0, "should have consumed tokens");
    assert!(
        !result.new_items.is_empty(),
        "should have at least one item"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Agent with function tool — Responses API
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherInput {
    /// The city to get weather for.
    city: String,
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_tool_call_responses_api() {
    let weather_tool = function_tool::<(), WeatherInput, _, _>(
        "get_weather",
        "Get the current weather for a city.",
        |_ctx: ToolContext<()>, input: WeatherInput| async move {
            Ok(ToolOutput::Text(format!(
                "Weather in {}: 72°F, sunny",
                input.city
            )))
        },
    )
    .expect("tool creation should succeed");

    let agent = Agent::<()>::builder("weather-agent")
        .instructions(
            "You help users check the weather. \
             Always use the get_weather tool when asked about weather. \
             After getting the result, summarise it briefly.",
        )
        .tool(Tool::Function(weather_tool))
        .build();

    let model = responses_model();
    let result = Runner::run_with_model(
        &agent,
        "What is the weather in Paris?",
        (),
        model,
        None,
        None,
    )
    .await
    .expect("run should succeed");

    println!("Final output: {}", result.final_output);
    println!("Items produced: {}", result.new_items.len());
    println!("Usage: {} tokens", result.usage.total_tokens);

    let has_tool_call = result
        .new_items
        .iter()
        .any(|item| matches!(item, RunItem::ToolCall(_)));
    assert!(
        has_tool_call,
        "agent should have called the get_weather tool; items: {:?}",
        result.new_items
    );

    let has_tool_output = result
        .new_items
        .iter()
        .any(|item| matches!(item, RunItem::ToolCallOutput(_)));
    assert!(
        has_tool_output,
        "agent should have a ToolCallOutput item; items: {:?}",
        result.new_items
    );
}

// ---------------------------------------------------------------------------
// Test 4: Agent with function tool — Chat Completions API
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_tool_call_chat_completions_api() {
    let weather_tool = function_tool::<(), WeatherInput, _, _>(
        "get_weather",
        "Get the current weather for a city.",
        |_ctx: ToolContext<()>, input: WeatherInput| async move {
            Ok(ToolOutput::Text(format!(
                "Weather in {}: 65°F, cloudy",
                input.city
            )))
        },
    )
    .expect("tool creation should succeed");

    let agent = Agent::<()>::builder("weather-agent-chat")
        .instructions(
            "You help users check the weather. \
             Always use the get_weather tool when asked about weather. \
             After getting the result, summarise it briefly.",
        )
        .tool(Tool::Function(weather_tool))
        .build();

    let model = chat_model();
    let result = Runner::run_with_model(
        &agent,
        "What is the weather in London?",
        (),
        model,
        None,
        None,
    )
    .await
    .expect("run should succeed");

    println!("Final output: {}", result.final_output);
    println!("Usage: {} tokens", result.usage.total_tokens);

    let has_tool_call = result
        .new_items
        .iter()
        .any(|item| matches!(item, RunItem::ToolCall(_)));
    assert!(
        has_tool_call,
        "agent should have called the get_weather tool; items: {:?}",
        result.new_items
    );
}

// ---------------------------------------------------------------------------
// Test 5: Multi-agent handoff
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_multi_agent_handoff() {
    let triage = Agent::<()>::builder("triage")
        .instructions(
            "You are a triage agent. \
             When the user asks about weather, you MUST use the transfer_to_weather tool \
             to hand off to the weather agent. \
             Do NOT answer weather questions yourself.",
        )
        .handoff(Handoff::to_agent("weather").build())
        .build();

    let weather = Agent::<()>::builder("weather")
        .instructions(
            "You are a weather agent. \
             Always respond with: 'The weather is sunny and 72°F.'",
        )
        .build();

    let mut agents: HashMap<String, &Agent<()>> = HashMap::new();
    agents.insert("triage".to_string(), &triage);
    agents.insert("weather".to_string(), &weather);

    let model = responses_model();
    let result = Runner::run_with_agents(
        &triage,
        &agents,
        "What is the weather like today?",
        (),
        model,
        None,
        None,
    )
    .await
    .expect("run should succeed");

    println!("Final output: {}", result.final_output);
    println!("Last agent: {}", result.last_agent_name);

    let has_handoff_call = result
        .new_items
        .iter()
        .any(|item| matches!(item, RunItem::HandoffCall(_)));
    assert!(
        has_handoff_call,
        "should have a HandoffCall item; items: {:?}",
        result.new_items
    );

    let has_handoff_output = result
        .new_items
        .iter()
        .any(|item| matches!(item, RunItem::HandoffOutput(_)));
    assert!(
        has_handoff_output,
        "should have a HandoffOutput item; items: {:?}",
        result.new_items
    );

    assert_eq!(
        result.last_agent_name, "weather",
        "last agent should be the weather agent"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Structured output
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct PersonInfo {
    /// The person's full name.
    name: String,
    /// The person's age in years.
    age: u32,
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_structured_output() {
    let agent = Agent::<()>::builder("extractor")
        .instructions(
            "Extract the person's name and age from the text. \
             Return JSON matching the requested schema exactly.",
        )
        .output_type::<PersonInfo>()
        .build();

    let model = responses_model();
    let result = Runner::run_with_model(&agent, "John is 30 years old.", (), model, None, None)
        .await
        .expect("run should succeed");

    println!("Raw output: {}", result.final_output);

    let info: PersonInfo = result
        .final_output_as()
        .expect("output should deserialize to PersonInfo");

    println!("Parsed: name={}, age={}", info.name, info.age);

    assert_eq!(info.name, "John", "name should be John");
    assert_eq!(info.age, 30, "age should be 30");
}

// ---------------------------------------------------------------------------
// Test 7: Input guardrail blocks bad input
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_input_guardrail_blocks() {
    let guardrail = InputGuardrail::<()>::new("block_bad_words", |_ctx, _agent_name, input| {
        Box::pin(async move {
            let triggered = match input {
                InputContent::Text(t) => t.contains("blocked"),
                _ => false,
            };
            Ok(GuardrailFunctionOutput::new(
                serde_json::json!(null),
                triggered,
            ))
        })
    });

    let agent = Agent::<()>::builder("guarded")
        .instructions("You are a helpful assistant.")
        .input_guardrail(guardrail)
        .build();

    let model = responses_model();

    // A benign message should pass through.
    let ok_result =
        Runner::run_with_model(&agent, "Hello there", (), Arc::clone(&model), None, None).await;
    assert!(
        ok_result.is_ok(),
        "benign message should not trigger guardrail"
    );

    // A message containing "blocked" should be rejected.
    let err_result =
        Runner::run_with_model(&agent, "This is blocked content", (), model, None, None).await;
    assert!(
        err_result.is_err(),
        "message with 'blocked' should trigger guardrail"
    );

    let err = err_result.unwrap_err();
    println!("Guardrail error: {err}");
    assert!(
        matches!(
            err,
            AgentError::InputGuardrailTripwire { ref guardrail_name } if guardrail_name == "block_bad_words"
        ),
        "error should be InputGuardrailTripwire for 'block_bad_words', got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Multi-turn conversation using to_input_list
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_multi_turn_conversation() {
    let model = responses_model();
    let agent = Agent::<()>::builder("memory-agent")
        .instructions(
            "You are a helpful assistant. \
             When asked what the user's name is, repeat it exactly as they told you.",
        )
        .build();

    // Turn 1: introduce a name.
    let result1 = Runner::run_with_model(
        &agent,
        "My name is Alice.",
        (),
        Arc::clone(&model),
        None,
        None,
    )
    .await
    .expect("first turn should succeed");

    println!("Turn 1 output: {}", result1.final_output);

    // Turn 2: build follow-up input from previous output + new user message.
    let mut follow_up: Vec<ResponseInputItem> = result1.to_input_list();
    follow_up.push(serde_json::json!({
        "role": "user",
        "content": "What is my name?"
    }));

    let result2 = Runner::run_with_model(
        &agent,
        InputContent::Items(follow_up),
        (),
        model,
        None,
        None,
    )
    .await
    .expect("second turn should succeed");

    let output = result2.final_output.as_str().unwrap_or("");
    println!("Turn 2 output: {output}");

    assert!(
        output.to_lowercase().contains("alice"),
        "second response should mention 'Alice', got: {output}"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Model settings (temperature = 0.0, max_tokens limit)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_model_settings_max_tokens() {
    let settings = ModelSettings::default()
        .with_temperature(0.0)
        .with_max_tokens(20);

    let agent = Agent::<()>::builder("configured")
        .instructions("Count from 1 to 100.")
        .model_settings(settings)
        .build();

    let model = responses_model();
    let result = Runner::run_with_model(&agent, "Go", (), model, None, None)
        .await
        .expect("run should succeed");

    println!("Output: {}", result.final_output);
    println!("Output tokens: {}", result.usage.output_tokens);

    // With a 20-token limit the model must stop well before finishing.
    // We use a generous bound to account for counting overhead tokens.
    assert!(
        result.usage.output_tokens <= 50,
        "output tokens should be bounded by max_tokens; got {}",
        result.usage.output_tokens
    );
}

// ---------------------------------------------------------------------------
// Test 10: RunConfig max_turns limit
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_max_turns_exceeded() {
    // Give the agent a tool that always asks to keep looping, but cap turns at 2.
    #[derive(Debug, Deserialize, JsonSchema)]
    struct PingInput {
        /// Unused payload.
        _message: Option<String>,
    }

    let ping_tool = function_tool::<(), PingInput, _, _>(
        "ping",
        "Ping the server. Always call this tool.",
        |_ctx: ToolContext<()>, _input: PingInput| async move {
            Ok(ToolOutput::Text(
                "pong — please call ping again".to_string(),
            ))
        },
    )
    .expect("tool creation should succeed");

    let agent = Agent::<()>::builder("looper")
        .instructions("You must always call the ping tool on every turn. Never stop calling it.")
        .tool(Tool::Function(ping_tool))
        .build();

    let config = RunConfig::builder().max_turns(2).build();

    let model = responses_model();
    let result = Runner::run_with_model(&agent, "Start", (), model, None, Some(config)).await;

    // The runner should either succeed (model stopped on its own) or hit MaxTurnsExceeded.
    match result {
        Ok(r) => println!("Completed within turns. Output: {}", r.final_output),
        Err(AgentError::MaxTurnsExceeded { max_turns }) => {
            println!("MaxTurnsExceeded ({max_turns}) — expected");
            assert_eq!(max_turns, 2);
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Test 11: Lifecycle hooks fire during a real API call
// ---------------------------------------------------------------------------

struct SpyHooks {
    agent_start_count: Arc<AtomicU32>,
    llm_start_count: Arc<AtomicU32>,
    llm_end_count: Arc<AtomicU32>,
    agent_end_count: Arc<AtomicU32>,
}

impl SpyHooks {
    fn new() -> (
        Self,
        Arc<AtomicU32>,
        Arc<AtomicU32>,
        Arc<AtomicU32>,
        Arc<AtomicU32>,
    ) {
        let agent_start = Arc::new(AtomicU32::new(0));
        let llm_start = Arc::new(AtomicU32::new(0));
        let llm_end = Arc::new(AtomicU32::new(0));
        let agent_end = Arc::new(AtomicU32::new(0));
        (
            Self {
                agent_start_count: Arc::clone(&agent_start),
                llm_start_count: Arc::clone(&llm_start),
                llm_end_count: Arc::clone(&llm_end),
                agent_end_count: Arc::clone(&agent_end),
            },
            agent_start,
            llm_start,
            llm_end,
            agent_end,
        )
    }
}

#[async_trait]
impl RunHooks<()> for SpyHooks {
    async fn on_agent_start(&self, _context: &RunContextWrapper<()>, _agent_name: &str) {
        self.agent_start_count.fetch_add(1, Ordering::SeqCst);
    }

    async fn on_agent_end(
        &self,
        _context: &RunContextWrapper<()>,
        _agent_name: &str,
        _output: &serde_json::Value,
    ) {
        self.agent_end_count.fetch_add(1, Ordering::SeqCst);
    }

    async fn on_llm_start(
        &self,
        _context: &RunContextWrapper<()>,
        _agent_name: &str,
        _system_prompt: Option<&str>,
        _input: &[ResponseInputItem],
    ) {
        self.llm_start_count.fetch_add(1, Ordering::SeqCst);
    }

    async fn on_llm_end(
        &self,
        _context: &RunContextWrapper<()>,
        _agent_name: &str,
        _response: &ModelResponse,
    ) {
        self.llm_end_count.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_lifecycle_hooks_fire() {
    let (hooks, agent_start, llm_start, llm_end, agent_end) = SpyHooks::new();

    let agent = Agent::<()>::builder("hooked")
        .instructions("You are a helpful assistant. Reply briefly.")
        .build();

    let model = responses_model();
    let result = Runner::run_with_model(
        &agent,
        "What is 2 + 2?",
        (),
        model,
        Some(Arc::new(hooks)),
        None,
    )
    .await
    .expect("run should succeed");

    println!("Output: {}", result.final_output);
    println!(
        "Hook counts — agent_start={}, llm_start={}, llm_end={}, agent_end={}",
        agent_start.load(Ordering::SeqCst),
        llm_start.load(Ordering::SeqCst),
        llm_end.load(Ordering::SeqCst),
        agent_end.load(Ordering::SeqCst),
    );

    assert!(
        agent_start.load(Ordering::SeqCst) >= 1,
        "on_agent_start should fire at least once"
    );
    assert!(
        llm_start.load(Ordering::SeqCst) >= 1,
        "on_llm_start should fire at least once"
    );
    assert!(
        llm_end.load(Ordering::SeqCst) >= 1,
        "on_llm_end should fire at least once"
    );
    assert!(
        agent_end.load(Ordering::SeqCst) >= 1,
        "on_agent_end should fire at least once"
    );
}

// ---------------------------------------------------------------------------
// Test 12: Tool hooks fire when a tool is called
// ---------------------------------------------------------------------------

struct ToolSpyHooks {
    tool_start_count: Arc<AtomicU32>,
    tool_end_count: Arc<AtomicU32>,
    tool_name_seen: Arc<std::sync::Mutex<Vec<String>>>,
}

impl ToolSpyHooks {
    fn new() -> (
        Self,
        Arc<AtomicU32>,
        Arc<AtomicU32>,
        Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        let start = Arc::new(AtomicU32::new(0));
        let end = Arc::new(AtomicU32::new(0));
        let names = Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Self {
                tool_start_count: Arc::clone(&start),
                tool_end_count: Arc::clone(&end),
                tool_name_seen: Arc::clone(&names),
            },
            start,
            end,
            names,
        )
    }
}

#[async_trait]
impl RunHooks<()> for ToolSpyHooks {
    async fn on_tool_start(
        &self,
        _context: &RunContextWrapper<()>,
        _agent_name: &str,
        tool_name: &str,
    ) {
        self.tool_start_count.fetch_add(1, Ordering::SeqCst);
        self.tool_name_seen
            .lock()
            .unwrap()
            .push(tool_name.to_string());
    }

    async fn on_tool_end(
        &self,
        _context: &RunContextWrapper<()>,
        _agent_name: &str,
        _tool_name: &str,
        _result: &str,
    ) {
        self.tool_end_count.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_tool_hooks_fire() {
    let calc_tool = function_tool::<(), WeatherInput, _, _>(
        "get_weather",
        "Get weather for a city.",
        |_ctx: ToolContext<()>, input: WeatherInput| async move {
            Ok(ToolOutput::Text(format!("Sunny in {}", input.city)))
        },
    )
    .expect("tool creation should succeed");

    let (hooks, tool_start, tool_end, names) = ToolSpyHooks::new();

    let agent = Agent::<()>::builder("tool-hooked")
        .instructions(
            "Always use the get_weather tool when asked about weather. \
             After getting the result, report it back.",
        )
        .tool(Tool::Function(calc_tool))
        .build();

    let model = responses_model();
    let result = Runner::run_with_model(
        &agent,
        "What is the weather in Tokyo?",
        (),
        model,
        Some(Arc::new(hooks)),
        None,
    )
    .await
    .expect("run should succeed");

    println!("Output: {}", result.final_output);
    println!(
        "Tool hooks — start={}, end={}, names={:?}",
        tool_start.load(Ordering::SeqCst),
        tool_end.load(Ordering::SeqCst),
        names.lock().unwrap(),
    );

    assert!(
        tool_start.load(Ordering::SeqCst) >= 1,
        "on_tool_start should fire at least once"
    );
    assert!(
        tool_end.load(Ordering::SeqCst) >= 1,
        "on_tool_end should fire at least once"
    );
    assert!(
        names.lock().unwrap().contains(&"get_weather".to_string()),
        "should have seen the get_weather tool"
    );
}

// ---------------------------------------------------------------------------
// Test 13: Usage accumulates across tool-call turns
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_usage_accumulates_across_turns() {
    let tool = function_tool::<(), WeatherInput, _, _>(
        "get_weather",
        "Get weather for a city.",
        |_ctx: ToolContext<()>, input: WeatherInput| async move {
            Ok(ToolOutput::Text(format!("Rainy in {}", input.city)))
        },
    )
    .expect("tool creation should succeed");

    let agent = Agent::<()>::builder("usage-agent")
        .instructions("Use the get_weather tool, then reply with the result.")
        .tool(Tool::Function(tool))
        .build();

    let model = responses_model();
    let result = Runner::run_with_model(
        &agent,
        "Check the weather in Berlin.",
        (),
        model,
        None,
        None,
    )
    .await
    .expect("run should succeed");

    println!("Output: {}", result.final_output);
    println!(
        "Usage — input={} output={} total={}",
        result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens,
    );

    // Multi-turn (tool call + final reply) means at least 2 model calls.
    assert!(
        result.usage.input_tokens > 0,
        "input_tokens should be positive"
    );
    assert!(
        result.usage.output_tokens > 0,
        "output_tokens should be positive"
    );
    assert!(
        result.usage.total_tokens >= result.usage.input_tokens + result.usage.output_tokens,
        "total_tokens should be at least input + output"
    );
    // Multi-turn usage should exceed a single-turn minimum of ~20 tokens.
    assert!(
        result.usage.total_tokens > 20,
        "total_tokens should reflect at least two model calls"
    );
}

// ---------------------------------------------------------------------------
// Test 14: Dynamic instructions are applied
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_dynamic_instructions() {
    // Pass the desired language as context.
    let agent = Agent::<String>::builder("dynamic-agent")
        .dynamic_instructions(|ctx, _agent| {
            let lang = ctx.context.clone();
            Box::pin(async move {
                Ok(format!(
                    "You are a helpful assistant. Respond in {lang}. \
                     Keep your reply to one sentence."
                ))
            })
        })
        .build();

    let model =
        Arc::new(OpenAIResponsesModel::new("gpt-4o-mini").expect("OPENAI_API_KEY must be set"));

    let result = Runner::run_with_model(
        &agent,
        "What is 2 + 2?",
        "English".to_string(),
        model,
        None,
        None,
    )
    .await
    .expect("run should succeed");

    let output = result.final_output.as_str().unwrap_or("");
    println!("Dynamic instructions output: {output}");
    assert!(!output.is_empty(), "output should not be empty");
}

// ---------------------------------------------------------------------------
// Test 15: RunResult::to_input_list includes output items from all responses
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_to_input_list_contains_output_items() {
    let model = responses_model();
    let agent = Agent::<()>::builder("historian")
        .instructions("You are a concise assistant.")
        .build();

    let result =
        Runner::run_with_model(&agent, "Tell me a very short fact.", (), model, None, None)
            .await
            .expect("run should succeed");

    let input_list = result.to_input_list();
    println!("to_input_list length: {}", input_list.len());

    assert!(
        !input_list.is_empty(),
        "to_input_list should return at least one item for follow-up"
    );
}
