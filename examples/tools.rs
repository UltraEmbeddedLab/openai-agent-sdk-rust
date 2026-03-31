#![allow(clippy::option_if_let_else, clippy::missing_docs_in_private_items)]
//! Tool Usage -- demonstrate function tools with an agent.
//!
//! This example shows how to define custom function tools, attach them to an agent,
//! and run the agent so the LLM can call those tools during execution.
//!
//! Two modes:
//! - **Real API**: Uses `gpt-4o-mini` via the `OpenAI` Responses API.
//! - **Mock fallback**: If no API key is set, uses a mock model that simulates a
//!   tool call followed by a final message.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example tools
//! ```

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio_stream::Stream;

use openai_agents::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent, ToolOutput};
use openai_agents::models::openai_responses::OpenAIResponsesModel;
use openai_agents::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use openai_agents::tool::{ToolContext, function_tool};
use openai_agents::usage::Usage;
use openai_agents::{Agent, ModelSettings, Result, Runner, Tool, new_model_response};

// ---------------------------------------------------------------------------
// Tool input types
// ---------------------------------------------------------------------------

/// Input schema for the `get_weather` tool.
#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherInput {
    /// The city to get weather for.
    city: String,
}

/// Input schema for the "calculate" tool.
#[derive(Debug, Deserialize, JsonSchema)]
struct CalculateInput {
    /// Mathematical expression to evaluate (e.g. "6 * 7").
    expression: String,
}

// ---------------------------------------------------------------------------
// Mock model (simulates a tool call when no API key is available)
// ---------------------------------------------------------------------------

/// A mock model that issues a `get_weather` tool call on the first request and
/// a final message on the second request (after receiving the tool output).
struct MockToolModel {
    /// Tracks whether we have already issued the tool call.
    called: std::sync::Mutex<bool>,
}

#[async_trait]
impl Model for MockToolModel {
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
    ) -> Result<ModelResponse> {
        let mut called = self.called.lock().unwrap();
        if *called {
            // Second call: final assistant message incorporating the tool result.
            Ok(new_model_response(
                vec![serde_json::json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "The weather in Paris is sunny and 22 degrees Celsius -- a beautiful day!"
                    }]
                })],
                Usage::default(),
                Some("mock-resp-tool-2".to_owned()),
                None,
            ))
        } else {
            // First call: simulate the model requesting the get_weather tool.
            *called = true;
            drop(called);
            Ok(new_model_response(
                vec![serde_json::json!({
                    "type": "function_call",
                    "id": "fc_mock_001",
                    "call_id": "fc_mock_001",
                    "name": "get_weather",
                    "arguments": "{\"city\": \"Paris\"}"
                })],
                Usage::default(),
                Some("mock-resp-tool-1".to_owned()),
                None,
            ))
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
    ) -> Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent>> + Send + 'a>> {
        Box::pin(tokio_stream::empty())
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create the get_weather tool.
    let weather_tool = function_tool::<(), WeatherInput, _, _>(
        "get_weather",
        "Get the current weather for a city.",
        |_ctx: ToolContext<()>, input: WeatherInput| async move {
            // In a real application you would call a weather API here.
            Ok(ToolOutput::Text(format!(
                "Weather in {}: Sunny, 22 degrees Celsius",
                input.city
            )))
        },
    )?;

    // Create the calculate tool.
    let calc_tool = function_tool::<(), CalculateInput, _, _>(
        "calculate",
        "Evaluate a simple math expression.",
        |_ctx: ToolContext<()>, input: CalculateInput| async move {
            // Stub: always returns 42 for demo purposes.
            Ok(ToolOutput::Text(format!(
                "Result of '{}' = 42",
                input.expression
            )))
        },
    )?;

    // Build the agent with both tools.
    let agent = Agent::<()>::builder("math-weather-helper")
        .instructions(
            "You help users with math and weather questions. \
             Use the get_weather tool for weather queries and the calculate tool for math. \
             After receiving tool results, explain them clearly to the user.",
        )
        .tool(Tool::Function(weather_tool))
        .tool(Tool::Function(calc_tool))
        .build();

    println!("Agent: {} (tools: {})", agent.name, agent.tools.len());

    // Choose model: real API or mock.
    let model: Arc<dyn Model> = OpenAIResponsesModel::new("gpt-4o-mini").map_or_else(
        |_| {
            println!("OPENAI_API_KEY not set -- using mock model");
            Arc::new(MockToolModel {
                called: std::sync::Mutex::new(false),
            }) as Arc<dyn Model>
        },
        |real_model| {
            println!("Using OpenAI API (gpt-4o-mini)");
            Arc::new(real_model)
        },
    );

    // Run the agent.
    let result = Runner::run_with_model(
        &agent,
        "What is the weather like in Paris?",
        (),
        model,
        None,
        None,
    )
    .await?;

    // Display results.
    println!("\nFinal answer: {}", result.final_output);
    println!("Items generated: {}", result.new_items.len());
    for (i, item) in result.new_items.iter().enumerate() {
        println!("  Item {i}: {item:?}");
    }
    println!(
        "Tokens used: input={}, output={}, total={}",
        result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens
    );

    Ok(())
}
