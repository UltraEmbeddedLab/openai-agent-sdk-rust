//! Interactive REPL for testing agents.
//!
//! Provides [`run_demo_loop`], a convenience function that starts an
//! interactive conversation loop with an agent in the terminal.
//! This mirrors the Python SDK's `repl.py` module.

use std::io::{self, Write};
use std::sync::Arc;

use tokio_stream::StreamExt;

use crate::agent::Agent;
use crate::config::RunConfig;
use crate::error::Result;
use crate::items::InputContent;
use crate::models::Model;
use crate::runner::Runner;
use crate::stream_events::StreamEvent;

/// Run an interactive demo loop with the given agent.
///
/// Reads user input from stdin, sends it to the agent, and prints the
/// response. Type "exit" or "quit" to stop the loop. Conversation history
/// is preserved across turns.
///
/// When `stream` is true, events are printed incrementally as they arrive.
/// When false, the full response is printed after the run completes.
///
/// # Arguments
///
/// * `agent` - The agent to converse with (must be `'static` for streaming).
/// * `model` - The model backend to use.
/// * `stream` - Whether to stream the agent output.
/// * `config` - Optional run configuration (controls max turns, etc.).
///
/// # Errors
///
/// Returns an error if any non-streaming run invocation fails.
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use openai_agents::agent::Agent;
/// use openai_agents::repl::run_demo_loop;
///
/// # async fn example(model: Arc<dyn openai_agents::models::Model>) -> openai_agents::Result<()> {
/// let agent: &'static Agent<()> = Box::leak(Box::new(
///     Agent::<()>::builder("assistant")
///         .instructions("You are a helpful assistant.")
///         .build()
/// ));
///
/// run_demo_loop(agent, model, true, None).await?;
/// # Ok(())
/// # }
/// ```
pub async fn run_demo_loop(
    agent: &'static Agent<()>,
    model: Arc<dyn Model>,
    stream: bool,
    config: Option<RunConfig>,
) -> Result<()> {
    println!("Agent '{}' ready. Type 'exit' to quit.\n", agent.name);

    let mut input_items: Vec<serde_json::Value> = Vec::new();

    loop {
        print!(" > ");
        io::stdout().flush().ok();

        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).is_err() {
            break;
        }

        let trimmed = user_input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            break;
        }

        input_items.push(serde_json::json!({
            "role": "user",
            "content": trimmed,
        }));

        if stream {
            let mut result = Runner::run_streamed(
                agent,
                InputContent::Items(input_items.clone()),
                (),
                Arc::clone(&model),
                None,
                config.clone(),
            );

            {
                let mut events = result.stream_events();
                while let Some(event) = events.next().await {
                    match &event {
                        StreamEvent::RawResponse(data) => {
                            // Print text deltas inline.
                            if let Some(delta) = data.get("delta").and_then(|d| d.as_str()) {
                                print!("{delta}");
                                io::stdout().flush().ok();
                            }
                        }
                        StreamEvent::RunItemCreated { item, .. } => {
                            if matches!(item, crate::items::RunItem::ToolCall(_)) {
                                println!("\n[tool called]");
                            } else if let crate::items::RunItem::ToolCallOutput(tco) = item {
                                println!("\n[tool output: {}]", tco.output);
                            }
                        }
                        StreamEvent::AgentUpdated { new_agent_name } => {
                            println!("\n[Agent updated: {new_agent_name}]");
                        }
                    }
                }
            }
            println!();

            // Update history for multi-turn.
            input_items = result
                .raw_responses
                .iter()
                .flat_map(|r| r.output.clone())
                .collect();
        } else {
            let run_result = Runner::run_with_model(
                agent,
                InputContent::Items(input_items.clone()),
                (),
                Arc::clone(&model),
                None,
                config.clone(),
            )
            .await?;

            if let Some(s) = run_result.final_output.as_str() {
                println!("{s}");
            } else if !run_result.final_output.is_null() {
                println!("{}", run_result.final_output);
            }

            // Update history for multi-turn.
            input_items = run_result.to_input_list();
        }
    }

    Ok(())
}
