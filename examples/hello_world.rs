//! Hello World -- the simplest agent example.
//!
//! This example demonstrates two modes:
//!
//! 1. **Real API mode** (default): Uses the `OpenAI` Responses API with `gpt-4o-mini`.
//!    Requires the `OPENAI_API_KEY` environment variable.
//!
//! 2. **Mock mode** (fallback): If no API key is set, falls back to a mock model
//!    that returns a fixed greeting. This is useful for testing without an API key.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example hello_world
//! ```

#![allow(clippy::missing_docs_in_private_items)]

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_stream::Stream;

use openai_agents::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use openai_agents::models::openai_responses::OpenAIResponsesModel;
use openai_agents::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use openai_agents::usage::Usage;
use openai_agents::{Agent, ModelSettings, Result, Runner, new_model_response};

// ---------------------------------------------------------------------------
// Mock model (used when OPENAI_API_KEY is not set)
// ---------------------------------------------------------------------------

/// A minimal `Model` implementation that always returns a fixed message.
struct MockModel;

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
    ) -> Result<ModelResponse> {
        Ok(new_model_response(
            vec![serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello! It's wonderful to meet you -- I'm your friendly assistant, ready to help with whatever you need today!"
                    }
                ]
            })],
            Usage::default(),
            Some("mock-resp-001".to_owned()),
            None,
        ))
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
// Helper
// ---------------------------------------------------------------------------

/// Try to create a real `OpenAI` model; fall back to a mock.
fn create_model() -> Arc<dyn Model> {
    OpenAIResponsesModel::new("gpt-4o-mini").map_or_else(
        |_| {
            println!("OPENAI_API_KEY not set -- using mock model");
            Arc::new(MockModel) as Arc<dyn Model>
        },
        |real_model| {
            println!("Using OpenAI API (gpt-4o-mini)");
            Arc::new(real_model)
        },
    )
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build an agent with static instructions.
    let agent = Agent::<()>::builder("greeter")
        .instructions("You are a cheerful assistant. Greet the user warmly in one sentence.")
        .build();

    println!("Agent name: {}", agent.name);

    let model = create_model();

    // Run the agent to completion.
    let result = Runner::run_with_model(
        &agent,
        "Hello!",
        (), // context -- () because this agent needs no custom context
        model,
        None, // hooks -- no lifecycle hooks for this example
        None, // config -- use default RunConfig (10 max turns, tracing enabled)
    )
    .await?;

    // Display the result.
    println!("Final output: {}", result.final_output);
    println!(
        "Tokens used: input={}, output={}, total={}",
        result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens
    );

    Ok(())
}
