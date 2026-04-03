#![allow(clippy::option_if_let_else, clippy::missing_docs_in_private_items)]
//! Multi-Agent Handoffs -- demonstrate agent-to-agent delegation.
//!
//! This example builds a triage agent that can hand off to either a billing agent
//! or a support agent. The triage agent decides which specialist to call based on
//! the user's question.
//!
//! Two modes:
//! - **Real API**: Uses `gpt-4o-mini` via the `OpenAI` Responses API.
//! - **Mock fallback**: A mock model that simulates a handoff to the billing agent.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example handoffs
//! ```

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_stream::Stream;

use openai_agents::handoffs::Handoff;
use openai_agents::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use openai_agents::models::openai_responses::OpenAIResponsesModel;
use openai_agents::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use openai_agents::usage::Usage;
use openai_agents::{Agent, ModelSettings, Result, Runner, new_model_response};

// ---------------------------------------------------------------------------
// Mock model (simulates a handoff when no API key is available)
// ---------------------------------------------------------------------------

/// Mock model for the triage agent: issues a handoff tool call to the billing agent.
struct MockTriageModel;

#[async_trait]
impl Model for MockTriageModel {
    async fn get_response(
        &self,
        _system_instructions: Option<&str>,
        input: &[ResponseInputItem],
        _model_settings: &ModelSettings,
        _tools: &[ToolSpec],
        _output_schema: Option<&OutputSchemaSpec>,
        handoffs: &[HandoffToolSpec],
        _tracing: ModelTracing,
        _previous_response_id: Option<&str>,
    ) -> Result<ModelResponse> {
        // Check if this is a handoff-ed call (input contains a function_call_output).
        let is_post_handoff = input
            .iter()
            .any(|item| item.get("type").and_then(|v| v.as_str()) == Some("function_call_output"));

        if is_post_handoff {
            // We are in the target agent now. Return a final message.
            return Ok(new_model_response(
                vec![serde_json::json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "I'm the billing agent. Your last invoice was $42.00, paid on March 15th. Is there anything else I can help with?"
                    }]
                })],
                Usage::default(),
                Some("mock-billing-resp".to_owned()),
                None,
            ));
        }

        // First call from the triage agent: hand off to billing.
        // Find the billing handoff tool name.
        let billing_tool = handoffs
            .iter()
            .find(|h| h.tool_name.contains("billing"))
            .map_or("transfer_to_billing", |h| h.tool_name.as_str());

        Ok(new_model_response(
            vec![serde_json::json!({
                "type": "function_call",
                "id": "fc_handoff_001",
                "call_id": "fc_handoff_001",
                "name": billing_tool,
                "arguments": "{}"
            })],
            Usage::default(),
            Some("mock-triage-resp".to_owned()),
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
    // Define handoffs to specialist agents.
    let billing_handoff: Handoff<()> = Handoff::to_agent("billing")
        .tool_description("Transfer to the billing agent for invoice and payment questions.")
        .build();

    let support_handoff: Handoff<()> = Handoff::to_agent("support")
        .tool_description("Transfer to the support agent for technical issues.")
        .build();

    // Build the triage agent with both handoffs.
    let triage_agent = Agent::<()>::builder("triage")
        .instructions(
            "You are a triage agent. Based on the user's question, hand off to the \
             appropriate specialist:\n\
             - For billing, invoices, or payment questions: use transfer_to_billing\n\
             - For technical issues or bugs: use transfer_to_support\n\
             If the question is general, answer it yourself.",
        )
        .handoff(billing_handoff)
        .handoff(support_handoff)
        .build();

    println!(
        "\nAgent: {} (handoffs: {})",
        triage_agent.name,
        triage_agent.handoffs.len()
    );

    // Choose model: real API or mock.
    let model: Arc<dyn Model> = if let Ok(real_model) = OpenAIResponsesModel::new("gpt-4o-mini") {
        println!("Using OpenAI API (gpt-4o-mini)");
        Arc::new(real_model)
    } else {
        println!("OPENAI_API_KEY not set -- using mock model");
        Arc::new(MockTriageModel) as Arc<dyn Model>
    };

    // Run the triage agent with a billing-related question.
    let result = Runner::run_with_model(
        &triage_agent,
        "Can you check the status of my last invoice?",
        (),
        model,
        None,
        None,
    )
    .await?;

    // Display results.
    println!("\nFinal output: {}", result.final_output);
    println!("Last agent: {}", result.last_agent_name);
    println!("Items generated: {}", result.new_items.len());
    for (i, item) in result.new_items.iter().enumerate() {
        println!("  Item {i}: {item:?}");
    }

    Ok(())
}
