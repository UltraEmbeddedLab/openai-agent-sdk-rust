#![allow(clippy::option_if_let_else, clippy::missing_docs_in_private_items)]
//! Streaming -- demonstrate streaming events during an agent run.
//!
//! This example shows how to use `Runner::run_streamed` to receive real-time
//! events as the agent processes. Events include raw model deltas, run item
//! creation notifications, and agent change notifications.
//!
//! **Note:** `run_streamed` requires a `'static` agent reference. In this
//! example we use `Box::leak` to create a static reference. In a real
//! application you would typically store agents in a long-lived data structure.
//!
//! Two modes:
//! - **Real API**: Uses `gpt-4o-mini` via the `OpenAI` Responses API.
//! - **Mock fallback**: A mock model that emits a single response.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example streaming
//! ```

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_stream::Stream;
use tokio_stream::StreamExt;

use openai_agents::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use openai_agents::models::openai_responses::OpenAIResponsesModel;
use openai_agents::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use openai_agents::stream_events::StreamEvent;
use openai_agents::usage::Usage;
use openai_agents::{Agent, ModelSettings, Result, Runner, new_model_response};

// ---------------------------------------------------------------------------
// Mock model
// ---------------------------------------------------------------------------

/// A mock model that returns a single message response.
struct MockStreamModel;

#[async_trait]
impl Model for MockStreamModel {
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
                "content": [{
                    "type": "output_text",
                    "text": "Rust is a systems programming language focused on safety, speed, and concurrency. It achieves memory safety without a garbage collector."
                }]
            })],
            Usage::default(),
            Some("mock-stream-resp".to_owned()),
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
    // Build the agent. We leak it to get a 'static reference, which
    // `Runner::run_streamed` requires.
    let agent: &'static Agent<()> = Box::leak(Box::new(
        Agent::<()>::builder("streaming-demo")
            .instructions("You are a concise assistant. Answer in 2-3 sentences.")
            .build(),
    ));

    println!("Agent: {}", agent.name);

    // Choose model: real API or mock.
    let model: Arc<dyn Model> = if let Ok(real_model) = OpenAIResponsesModel::new("gpt-4o-mini") {
        println!("Using OpenAI API (gpt-4o-mini)");
        Arc::new(real_model)
    } else {
        println!("OPENAI_API_KEY not set -- using mock model");
        Arc::new(MockStreamModel)
    };

    // Start a streaming run. This returns immediately while the agent processes
    // in the background.
    let mut streaming_result = Runner::run_streamed(
        agent,
        "What is Rust (the programming language)?",
        (),
        model,
        None, // no hooks
        None, // default config
    );

    println!("\n--- Streaming events ---");

    // Consume events as they arrive.
    let mut event_count = 0;
    let mut stream = streaming_result.stream_events();
    while let Some(event) = stream.next().await {
        event_count += 1;
        match &event {
            StreamEvent::RawResponse(raw) => {
                // Raw model delta -- in a real streaming scenario these arrive
                // incrementally as the model generates tokens.
                let event_type = raw
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                println!("[Event {event_count}] RawResponse: type={event_type}");
            }
            StreamEvent::RunItemCreated { name, item } => {
                println!("[Event {event_count}] RunItemCreated: {name} -> {item:?}");
            }
            StreamEvent::AgentUpdated { new_agent_name } => {
                println!("[Event {event_count}] AgentUpdated: {new_agent_name}");
            }
            _ => {
                println!("[Event {event_count}] Other event");
            }
        }
    }
    // Drop the stream so we can access streaming_result again.
    drop(stream);

    println!("--- End of stream ({event_count} events) ---\n");

    // After the stream is exhausted the result fields are populated.
    println!("Final output: {}", streaming_result.final_output);
    println!("Agent: {}", streaming_result.current_agent_name);
    println!("Complete: {}", streaming_result.is_complete);

    Ok(())
}
