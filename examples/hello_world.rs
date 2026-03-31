//! Basic hello world example demonstrating the `OpenAI` Agents SDK for Rust.
//!
//! This example creates an agent with the builder pattern and runs it against a
//! mock model implementation that returns a fixed greeting. In a real application
//! you would replace `MockModel` with an implementation backed by the `OpenAI` API.
//!
//! Run with: `cargo run --example hello_world`

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_stream::Stream;

use openai_agents::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use openai_agents::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use openai_agents::usage::Usage;
use openai_agents::{Agent, ModelSettings, Result, Runner, new_model_response};

// ---------------------------------------------------------------------------
// Mock model
// ---------------------------------------------------------------------------

/// A minimal `Model` implementation that always returns a fixed message.
///
/// Replace this with a real `OpenAI`-backed model once the HTTP integration is
/// available.
struct MockModel {
    /// The text that the mock model will always return as its response.
    response_text: String,
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
    ) -> Result<ModelResponse> {
        Ok(new_model_response(
            vec![serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": self.response_text
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
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build an agent with static instructions.
    let agent = Agent::<()>::builder("greeter")
        .instructions("You are a friendly assistant. Greet the user warmly.")
        .build();

    println!("Agent name: {}", agent.name);

    // Provide a mock model that returns a fixed greeting.
    let model = Arc::new(MockModel {
        response_text: "Hello! It is wonderful to meet you. How can I help you today?".to_owned(),
    });

    // Run the agent and capture the result.
    let result = Runner::run_with_model(
        &agent,
        "Hello!",
        (), // context — () because this agent needs no custom context
        model,
        None, // hooks — no lifecycle hooks for this example
        None, // config — use default RunConfig (10 max turns, tracing enabled)
    )
    .await?;

    println!("Final output: {}", result.final_output);
    println!(
        "Tokens used: input={}, output={}, total={}",
        result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens
    );

    Ok(())
}
