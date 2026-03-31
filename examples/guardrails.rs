#![allow(clippy::option_if_let_else, clippy::missing_docs_in_private_items)]
//! Guardrails -- demonstrate input and output guardrail validation.
//!
//! This example shows how to attach guardrails to an agent to validate both
//! user input and model output. Guardrails can reject requests that violate
//! your policies by triggering a "tripwire".
//!
//! Two modes:
//! - **Real API**: Uses `gpt-4o-mini` via the `OpenAI` Responses API.
//! - **Mock fallback**: A mock model that returns a fixed response.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example guardrails
//! ```

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio_stream::Stream;

use openai_agents::guardrail::{GuardrailFunctionOutput, InputGuardrail, OutputGuardrail};
use openai_agents::items::{InputContent, ModelResponse, ResponseInputItem, ResponseStreamEvent};
use openai_agents::models::openai_responses::OpenAIResponsesModel;
use openai_agents::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use openai_agents::usage::Usage;
use openai_agents::{Agent, AgentError, ModelSettings, Result, Runner, new_model_response};

// ---------------------------------------------------------------------------
// Mock model
// ---------------------------------------------------------------------------

/// A mock model that returns a fixed response.
struct MockGuardedModel;

#[async_trait]
impl Model for MockGuardedModel {
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
                    "text": "The capital of France is Paris, a beautiful city known for the Eiffel Tower."
                }]
            })],
            Usage::default(),
            Some("mock-guarded-resp".to_owned()),
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
    // --- Input guardrail: reject off-topic questions ---
    //
    // This guardrail checks if the user input contains certain forbidden topics.
    // If triggered, the agent run is aborted before the LLM is ever called.
    let topic_guardrail = InputGuardrail::<()>::new("topic_filter", |_ctx, _agent_name, input| {
        let input = input.clone();
        Box::pin(async move {
            let text = match &input {
                InputContent::Text(t) => t.to_lowercase(),
                _ => String::new(),
            };

            // Block questions about hacking or illegal activities.
            if text.contains("hack") || text.contains("illegal") {
                Ok(GuardrailFunctionOutput::tripwire(json!({
                    "reason": "off-topic: potentially harmful content detected"
                })))
            } else {
                Ok(GuardrailFunctionOutput::passed(json!({
                    "check": "topic_filter",
                    "status": "passed"
                })))
            }
        })
    });

    // --- Output guardrail: check for personally identifiable information (PII) ---
    //
    // This guardrail checks the model's output for patterns that look like PII
    // (e.g. social security numbers). If found, the run is aborted.
    let pii_guardrail = OutputGuardrail::<()>::new("pii_check", |_ctx, _agent_name, output| {
        let output = output.clone();
        Box::pin(async move {
            let text = output.as_str().unwrap_or_default();

            // Very simple SSN pattern check for demonstration.
            if text.contains("SSN") || text.contains("social security") {
                Ok(GuardrailFunctionOutput::tripwire(json!({
                    "reason": "PII detected in output"
                })))
            } else {
                Ok(GuardrailFunctionOutput::passed(json!({
                    "check": "pii_check",
                    "status": "clean"
                })))
            }
        })
    });

    // Build the agent with both guardrails.
    let agent = Agent::<()>::builder("guarded-assistant")
        .instructions("You are a helpful assistant that answers geography questions.")
        .input_guardrail(topic_guardrail)
        .output_guardrail(pii_guardrail)
        .build();

    println!("Agent: {}", agent.name);
    println!("  Input guardrails: {}", agent.input_guardrails.len());
    println!("  Output guardrails: {}", agent.output_guardrails.len());

    // Choose model: real API or mock.
    let model: Arc<dyn Model> = if let Ok(real_model) = OpenAIResponsesModel::new("gpt-4o-mini") {
        println!("Using OpenAI API (gpt-4o-mini)\n");
        Arc::new(real_model)
    } else {
        println!("OPENAI_API_KEY not set -- using mock model\n");
        Arc::new(MockGuardedModel)
    };

    // --- Test 1: A legitimate question (should pass both guardrails) ---
    println!("=== Test 1: Legitimate question ===");
    let result = Runner::run_with_model(
        &agent,
        "What is the capital of France?",
        (),
        Arc::clone(&model),
        None,
        None,
    )
    .await?;

    println!("Output: {}", result.final_output);
    println!(
        "Input guardrail results: {}",
        result.input_guardrail_results.len()
    );
    for gr in &result.input_guardrail_results {
        println!(
            "  {} -- tripwire: {}",
            gr.guardrail_name, gr.output.tripwire_triggered
        );
    }

    // --- Test 2: An off-topic question (should trigger the input guardrail) ---
    println!("\n=== Test 2: Off-topic question (should be blocked) ===");
    let blocked_result = Runner::run_with_model(
        &agent,
        "How do I hack into a computer?",
        (),
        Arc::clone(&model),
        None,
        None,
    )
    .await;

    match blocked_result {
        Err(AgentError::InputGuardrailTripwire { guardrail_name }) => {
            println!("Blocked! Input guardrail '{guardrail_name}' triggered tripwire.");
        }
        Ok(_) => println!("Unexpectedly passed (guardrail did not trigger)."),
        Err(e) => println!("Other error: {e}"),
    }

    Ok(())
}
