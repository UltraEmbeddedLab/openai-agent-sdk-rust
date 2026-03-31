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

use serde_json::json;
use tokio::sync::RwLock;

use crate::agent::{Agent, ToolUseBehavior};
use crate::config::{DEFAULT_MAX_TURNS, RunConfig};
use crate::context::RunContextWrapper;
use crate::error::{AgentError, Result};
use crate::guardrail::{InputGuardrailResult, OutputGuardrailResult};
use crate::items::{
    HandoffCallItem, HandoffOutputItem, InputContent, ItemHelpers, MessageOutputItem,
    ModelResponse, ReasoningItem, ResponseInputItem, RunItem, ToolCallItem, ToolCallOutputItem,
    ToolOutput,
};
use crate::lifecycle::RunHooks;
use crate::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use crate::prompts;
use crate::result::{RunResult, RunResultStreaming};
use crate::retry::RetryPolicy;
use crate::stream_events::{RunItemEventName, StreamEvent};
use crate::tool::{FunctionTool, Tool, ToolContext};
use crate::usage::Usage;

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
/// 7. Determine the next step based on the agent's [`ToolUseBehavior`].
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
    /// Returns [`AgentError::MaxTurnsExceeded`] if the loop exceeds the configured
    /// maximum number of turns. Also propagates errors from the model, tools,
    /// guardrails, or lifecycle hooks.
    ///
    /// # Panics
    ///
    /// Panics if the internal response list is unexpectedly empty after a model call.
    /// This should never occur in practice.
    #[allow(clippy::too_many_lines, clippy::significant_drop_tightening)]
    pub async fn run_with_model<C: Send + Sync + 'static>(
        agent: &Agent<C>,
        input: impl Into<InputContent>,
        context: C,
        model: Arc<dyn Model>,
        hooks: Option<Arc<dyn RunHooks<C>>>,
        config: Option<RunConfig>,
    ) -> Result<RunResult> {
        // Delegate to the multi-agent run with no agent registry.
        Self::run_internal(
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
    /// Returns [`AgentError::UserError`] if a handoff target agent is not found in
    /// the registry. Also propagates all errors from [`run_with_model`](Self::run_with_model).
    #[allow(clippy::too_many_lines, clippy::significant_drop_tightening)]
    pub async fn run_with_agents<C: Send + Sync + 'static>(
        starting_agent: &Agent<C>,
        agents: &HashMap<String, &Agent<C>>,
        input: impl Into<InputContent>,
        context: C,
        model: Arc<dyn Model>,
        hooks: Option<Arc<dyn RunHooks<C>>>,
        config: Option<RunConfig>,
    ) -> Result<RunResult> {
        Self::run_internal(
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

        tokio::spawn(Self::streaming_loop(
            agent, input, context, model, hooks, config, event_tx, cancel_rx,
        ));

        result
    }

    /// Internal method that implements the full agent loop with optional multi-agent support.
    #[allow(clippy::too_many_lines, clippy::significant_drop_tightening)]
    async fn run_internal<C: Send + Sync + 'static>(
        starting_agent: &Agent<C>,
        agents: &HashMap<String, &Agent<C>>,
        input: InputContent,
        context: C,
        model: Arc<dyn Model>,
        hooks: Option<Arc<dyn RunHooks<C>>>,
        config: Option<RunConfig>,
    ) -> Result<RunResult> {
        let config = config.unwrap_or_default();
        let ctx = Arc::new(RwLock::new(RunContextWrapper::new(context)));
        let max_turns = config.max_turns;

        let mut current_agent = starting_agent;
        let mut current_input = ItemHelpers::input_to_new_input_list(&input);
        let mut all_items: Vec<RunItem> = Vec::new();
        let mut all_responses: Vec<ModelResponse> = Vec::new();
        let mut input_guardrail_results: Vec<InputGuardrailResult> = Vec::new();
        let mut output_guardrail_results: Vec<OutputGuardrailResult> = Vec::new();
        let mut total_usage = Usage::default();
        let mut previous_response_id: Option<String> = None;

        let tracing = if config.tracing_disabled {
            ModelTracing::Disabled
        } else {
            ModelTracing::Enabled
        };

        let retry_policy = RetryPolicy::none();

        for turn in 0..max_turns {
            // 1. Fire agent start hooks.
            {
                let ctx_read = ctx.read().await;
                if let Some(ref hooks) = hooks {
                    hooks.on_agent_start(&ctx_read, &current_agent.name).await;
                }
                if let Some(ref agent_hooks) = current_agent.hooks {
                    agent_hooks.on_start(&ctx_read).await;
                }
            }

            // 2. Get agent instructions and build system prompt.
            let system_prompt = {
                let ctx_read = ctx.read().await;
                let raw_instructions = current_agent.get_instructions(&ctx_read).await?;
                let tool_specs = build_tool_specs(current_agent);
                let handoff_specs_for_prompt = &current_agent.handoffs;
                prompts::build_system_prompt(
                    raw_instructions.as_deref(),
                    &tool_specs,
                    handoff_specs_for_prompt,
                )
            };

            // 3. Set the turn input on the context.
            {
                let mut ctx_write = ctx.write().await;
                ctx_write.set_turn_input(current_input.clone());
            }

            // 4. Build tool specs and handoff specs.
            let tool_specs = build_tool_specs(current_agent);
            let handoff_specs = build_handoff_specs(current_agent);
            let output_schema_spec = current_agent
                .output_type
                .as_ref()
                .map(|s| OutputSchemaSpec {
                    json_schema: s.json_schema.clone(),
                    strict: s.strict,
                });

            // 5. Run input guardrails (first turn only).
            if turn == 0 {
                let ctx_read = ctx.read().await;
                for guardrail in &current_agent.input_guardrails {
                    let result = guardrail
                        .run(&ctx_read, &current_agent.name, &input)
                        .await?;
                    input_guardrail_results.push(result);
                }
            }

            // 6. Fire LLM start hooks.
            {
                let ctx_read = ctx.read().await;
                if let Some(ref hooks) = hooks {
                    hooks
                        .on_llm_start(
                            &ctx_read,
                            &current_agent.name,
                            system_prompt.as_deref(),
                            &current_input,
                        )
                        .await;
                }
                if let Some(ref agent_hooks) = current_agent.hooks {
                    agent_hooks
                        .on_llm_start(&ctx_read, system_prompt.as_deref(), &current_input)
                        .await;
                }
            }

            // Resolve model settings.
            let model_settings = current_agent
                .model_settings
                .resolve(config.model_settings.as_ref());

            // 7. Call the model with retry support.
            let sys_prompt_ref = system_prompt.as_deref();
            let input_ref = &current_input;
            let tool_specs_ref = &tool_specs;
            let output_schema_ref = output_schema_spec.as_ref();
            let handoff_specs_ref = &handoff_specs;
            let prev_resp_id = previous_response_id.as_deref();

            let response = retry_policy
                .execute(|| {
                    model.get_response(
                        sys_prompt_ref,
                        input_ref,
                        &model_settings,
                        tool_specs_ref,
                        output_schema_ref,
                        handoff_specs_ref,
                        tracing,
                        prev_resp_id,
                    )
                })
                .await?;

            // 8. Accumulate usage.
            total_usage.add(&response.usage);
            {
                let mut ctx_write = ctx.write().await;
                ctx_write.add_usage(&response.usage);
            }

            // 9. Fire LLM end hooks.
            {
                let ctx_read = ctx.read().await;
                if let Some(ref hooks) = hooks {
                    hooks
                        .on_llm_end(&ctx_read, &current_agent.name, &response)
                        .await;
                }
                if let Some(ref agent_hooks) = current_agent.hooks {
                    agent_hooks.on_llm_end(&ctx_read, &response).await;
                }
            }

            // Track previous response ID.
            previous_response_id.clone_from(&response.response_id);

            // 10. Process the model response.
            let processed = process_model_response(current_agent, &response);
            let mut turn_items = processed.new_items;
            let function_calls = processed.function_calls;
            let handoff_calls = processed.handoff_calls;

            // 11. Execute function tool calls.
            let mut tool_outputs: Vec<ResponseInputItem> = Vec::new();
            let mut had_tool_calls = false;
            let mut stop_tool_names: Vec<String> = Vec::new();

            for fc in &function_calls {
                had_tool_calls = true;
                let tool_name = &fc.name;
                let call_id = &fc.call_id;
                let arguments = &fc.arguments;

                // Fire tool start hooks.
                {
                    let ctx_read = ctx.read().await;
                    if let Some(ref hooks) = hooks {
                        hooks
                            .on_tool_start(&ctx_read, &current_agent.name, tool_name)
                            .await;
                    }
                    if let Some(ref agent_hooks) = current_agent.hooks {
                        agent_hooks.on_tool_start(&ctx_read, tool_name).await;
                    }
                }

                // Find the matching function tool.
                let function_tool = find_function_tool(current_agent, tool_name);

                let output = if let Some(ft) = function_tool {
                    let tool_ctx = ToolContext {
                        context: Arc::clone(&ctx),
                        tool_name: tool_name.clone(),
                        tool_call_id: call_id.clone(),
                    };
                    match ft.invoke(tool_ctx, arguments.clone()).await {
                        Ok(out) => out,
                        Err(e) => ToolOutput::Text(format!("Error: {e}")),
                    }
                } else {
                    ToolOutput::Text(format!("Error: tool '{tool_name}' not found"))
                };

                let output_text = match &output {
                    ToolOutput::Text(s) => s.clone(),
                    _ => String::from("<non-text output>"),
                };

                // Fire tool end hooks.
                {
                    let ctx_read = ctx.read().await;
                    if let Some(ref hooks) = hooks {
                        hooks
                            .on_tool_end(&ctx_read, &current_agent.name, tool_name, &output_text)
                            .await;
                    }
                    if let Some(ref agent_hooks) = current_agent.hooks {
                        agent_hooks
                            .on_tool_end(&ctx_read, tool_name, &output_text)
                            .await;
                    }
                }

                // Build the tool call output item.
                let output_item = ItemHelpers::tool_call_output_item(call_id, &output);
                tool_outputs.push(output_item.clone());

                turn_items.push(RunItem::ToolCallOutput(ToolCallOutputItem {
                    agent_name: current_agent.name.clone(),
                    raw_item: output_item,
                    output: serde_json::to_value(&output).unwrap_or(json!(null)),
                }));

                stop_tool_names.push(tool_name.clone());
            }

            // 12. Handle handoffs.
            if let Some(handoff_call) = handoff_calls.first() {
                // Find the matching handoff.
                let handoff = current_agent
                    .handoffs
                    .iter()
                    .find(|h| h.tool_name == handoff_call.tool_name);

                if let Some(h) = handoff {
                    let ctx_read = ctx.read().await;
                    let target_agent_name =
                        h.invoke(&ctx_read, handoff_call.arguments.clone()).await?;

                    let transfer_msg = h.get_transfer_message(&current_agent.name);
                    let handoff_output_raw = json!({
                        "type": "function_call_output",
                        "call_id": handoff_call.call_id,
                        "output": transfer_msg,
                    });

                    turn_items.push(RunItem::HandoffOutput(HandoffOutputItem {
                        agent_name: current_agent.name.clone(),
                        raw_item: handoff_output_raw.clone(),
                        source_agent_name: current_agent.name.clone(),
                        target_agent_name: target_agent_name.clone(),
                    }));

                    // Fire handoff hooks.
                    if let Some(ref hooks) = hooks {
                        hooks
                            .on_handoff(&ctx_read, &current_agent.name, &target_agent_name)
                            .await;
                    }

                    // Collect items before switching agent.
                    all_items.extend(turn_items);
                    all_responses.push(response);

                    // Look up the target agent in the registry.
                    if let Some(target_agent) = agents.get(target_agent_name.as_str()) {
                        current_agent = target_agent;

                        // Build the next input: current input + response output + handoff output.
                        let last_response =
                            all_responses.last().ok_or_else(|| AgentError::UserError {
                                message: "internal error: no model responses after model call"
                                    .to_string(),
                            })?;
                        let mut next_input = current_input.clone();
                        next_input.extend(last_response.to_input_items());
                        next_input.push(handoff_output_raw);
                        current_input = next_input;

                        continue;
                    }

                    // If the registry is empty (single-agent mode), return as before.
                    if agents.is_empty() {
                        let final_output = extract_final_output(&all_items, &all_responses);

                        // Run output guardrails.
                        {
                            let ctx_read2 = ctx.read().await;
                            for guardrail in &current_agent.output_guardrails {
                                let result = guardrail
                                    .run(&ctx_read2, &current_agent.name, &final_output)
                                    .await?;
                                output_guardrail_results.push(result);
                            }
                        }

                        // Fire agent end hooks.
                        {
                            let ctx_read2 = ctx.read().await;
                            if let Some(ref hooks) = hooks {
                                hooks
                                    .on_agent_end(&ctx_read2, &current_agent.name, &final_output)
                                    .await;
                            }
                            if let Some(ref agent_hooks) = current_agent.hooks {
                                agent_hooks.on_end(&ctx_read2, &final_output).await;
                            }
                        }

                        return Ok(RunResult {
                            input: input.clone(),
                            new_items: all_items,
                            raw_responses: all_responses,
                            final_output,
                            last_agent_name: current_agent.name.clone(),
                            input_guardrail_results,
                            output_guardrail_results,
                            usage: total_usage,
                        });
                    }

                    // Agent not found in registry.
                    return Err(AgentError::UserError {
                        message: format!(
                            "Handoff target agent '{target_agent_name}' not found in registry"
                        ),
                    });
                }
            }

            // 13. Collect items and response.
            all_items.extend(turn_items);
            all_responses.push(response);

            // 14. Determine next step based on ToolUseBehavior.
            let has_handoff = !handoff_calls.is_empty();

            let should_stop = match &current_agent.tool_use_behavior {
                ToolUseBehavior::RunLlmAgain => {
                    // Stop if there were no tool calls and no handoffs.
                    !had_tool_calls && !has_handoff
                }
                ToolUseBehavior::StopOnFirstTool => {
                    // Always stop after the first turn (tool call or not).
                    true
                }
                ToolUseBehavior::StopAtTools(stop_tools) => {
                    // Stop if any of the called tools are in the stop list.
                    let should_stop_at_tool =
                        stop_tool_names.iter().any(|name| stop_tools.contains(name));
                    if should_stop_at_tool {
                        true
                    } else {
                        // If no tool calls and no handoff, also stop (final message).
                        !had_tool_calls && !has_handoff
                    }
                }
            };

            if should_stop || has_handoff {
                // Build the final output.
                let final_output = extract_final_output(&all_items, &all_responses);

                // Run output guardrails.
                {
                    let ctx_read = ctx.read().await;
                    for guardrail in &current_agent.output_guardrails {
                        let result = guardrail
                            .run(&ctx_read, &current_agent.name, &final_output)
                            .await?;
                        output_guardrail_results.push(result);
                    }
                }

                // Fire agent end hooks.
                {
                    let ctx_read = ctx.read().await;
                    if let Some(ref hooks) = hooks {
                        hooks
                            .on_agent_end(&ctx_read, &current_agent.name, &final_output)
                            .await;
                    }
                    if let Some(ref agent_hooks) = current_agent.hooks {
                        agent_hooks.on_end(&ctx_read, &final_output).await;
                    }
                }

                return Ok(RunResult {
                    input: input.clone(),
                    new_items: all_items,
                    raw_responses: all_responses,
                    final_output,
                    last_agent_name: current_agent.name.clone(),
                    input_guardrail_results,
                    output_guardrail_results,
                    usage: total_usage,
                });
            }

            // 15. If we have tool calls and should continue, append tool outputs to input.
            let last_response = all_responses.last().ok_or_else(|| AgentError::UserError {
                message: "internal error: no model responses recorded after model call".to_string(),
            })?;
            let mut next_input = current_input.clone();
            next_input.extend(last_response.to_input_items());
            next_input.extend(tool_outputs);
            current_input = next_input;
        }

        Err(AgentError::MaxTurnsExceeded { max_turns })
    }

    /// Background task that drives the streaming agent loop.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    async fn streaming_loop<C: Send + Sync + 'static>(
        agent: &'static Agent<C>,
        input: InputContent,
        context: C,
        model: Arc<dyn Model>,
        hooks: Option<Arc<dyn RunHooks<C>>>,
        config: Option<RunConfig>,
        event_tx: tokio::sync::mpsc::Sender<StreamEvent>,
        mut cancel_rx: tokio::sync::oneshot::Receiver<()>,
    ) {
        let config = config.unwrap_or_default();
        let ctx = Arc::new(RwLock::new(RunContextWrapper::new(context)));
        let max_turns = config.max_turns;

        let mut current_input = ItemHelpers::input_to_new_input_list(&input);

        let tracing = if config.tracing_disabled {
            ModelTracing::Disabled
        } else {
            ModelTracing::Enabled
        };

        for _turn in 0..max_turns {
            // Check for cancellation.
            if cancel_rx.try_recv().is_ok() {
                return;
            }

            // Get system prompt via prompts module.
            let raw_instructions = {
                let ctx_read = ctx.read().await;
                let Ok(instr) = agent.get_instructions(&ctx_read).await else {
                    return;
                };
                instr
            };
            let tool_specs_for_prompt = build_tool_specs(agent);
            let system_prompt = prompts::build_system_prompt(
                raw_instructions.as_deref(),
                &tool_specs_for_prompt,
                &agent.handoffs,
            );

            let tool_specs = build_tool_specs(agent);
            let handoff_specs = build_handoff_specs(agent);
            let output_schema_spec = agent.output_type.as_ref().map(|s| OutputSchemaSpec {
                json_schema: s.json_schema.clone(),
                strict: s.strict,
            });

            let model_settings = agent.model_settings.resolve(config.model_settings.as_ref());

            // Fire agent start hooks.
            {
                let ctx_read = ctx.read().await;
                if let Some(ref hooks) = hooks {
                    hooks.on_agent_start(&ctx_read, &agent.name).await;
                }
            }

            // Stream model response in its own scope so borrows are released.
            let collected_output = {
                use tokio_stream::StreamExt;

                let mut stream = model.stream_response(
                    system_prompt.as_deref(),
                    &current_input,
                    &model_settings,
                    &tool_specs,
                    output_schema_spec.as_ref(),
                    &handoff_specs,
                    tracing,
                    None,
                );

                let mut collected: Vec<serde_json::Value> = Vec::new();

                loop {
                    tokio::select! {
                        biased;
                        _ = &mut cancel_rx => {
                            return;
                        }
                        maybe_event = stream.next() => {
                            match maybe_event {
                                Some(Ok(event)) => {
                                    collected.push(event.clone());
                                    // Send raw event.
                                    if event_tx.send(StreamEvent::RawResponse(event)).await.is_err() {
                                        return;
                                    }
                                }
                                Some(Err(_)) => {
                                    return;
                                }
                                None => {
                                    break;
                                }
                            }
                        }
                    }
                }

                collected
            };

            // Process collected events into run items.
            let model_response = ModelResponse {
                output: collected_output,
                usage: Usage::default(),
                response_id: None,
                request_id: None,
            };

            let processed = process_model_response(agent, &model_response);

            for item in &processed.new_items {
                let event_name = match item {
                    RunItem::MessageOutput(_) => RunItemEventName::MessageOutputCreated,
                    RunItem::HandoffCall(_) => RunItemEventName::HandoffRequested,
                    RunItem::HandoffOutput(_) => RunItemEventName::HandoffOccurred,
                    RunItem::ToolCall(_) => RunItemEventName::ToolCalled,
                    RunItem::ToolCallOutput(_) => RunItemEventName::ToolOutput,
                    RunItem::Reasoning(_) => RunItemEventName::ReasoningItemCreated,
                };

                if event_tx
                    .send(StreamEvent::RunItemCreated {
                        name: event_name,
                        item: item.clone(),
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }

            // If there are no tool calls and no handoff calls, this is the final turn.
            if processed.function_calls.is_empty() && processed.handoff_calls.is_empty() {
                return;
            }

            // Execute function tool calls and append to input.
            let mut tool_outputs_for_input: Vec<ResponseInputItem> = Vec::new();
            for fc in &processed.function_calls {
                let function_tool = find_function_tool(agent, &fc.name);
                let output = if let Some(ft) = function_tool {
                    let tool_ctx = ToolContext {
                        context: Arc::clone(&ctx),
                        tool_name: fc.name.clone(),
                        tool_call_id: fc.call_id.clone(),
                    };
                    match ft.invoke(tool_ctx, fc.arguments.clone()).await {
                        Ok(out) => out,
                        Err(e) => ToolOutput::Text(format!("Error: {e}")),
                    }
                } else {
                    ToolOutput::Text(format!("Error: tool '{}' not found", fc.name))
                };
                let output_item = ItemHelpers::tool_call_output_item(&fc.call_id, &output);
                tool_outputs_for_input.push(output_item);
            }

            let mut next_input = current_input;
            next_input.extend(model_response.to_input_items());
            next_input.extend(tool_outputs_for_input);
            current_input = next_input;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helper types and functions
// ---------------------------------------------------------------------------

/// A parsed function call from the model response.
struct ParsedFunctionCall {
    /// The tool name.
    name: String,
    /// The call ID.
    call_id: String,
    /// The raw JSON arguments string.
    arguments: String,
}

/// A parsed handoff call from the model response.
struct ParsedHandoffCall {
    /// The handoff tool name.
    tool_name: String,
    /// The call ID.
    call_id: String,
    /// The raw JSON arguments string.
    arguments: String,
}

/// Result of processing a model response.
struct ProcessedResponse {
    /// New run items extracted from the response.
    new_items: Vec<RunItem>,
    /// Function tool calls to execute.
    function_calls: Vec<ParsedFunctionCall>,
    /// Handoff calls to process.
    handoff_calls: Vec<ParsedHandoffCall>,
}

/// Build tool specifications from an agent's tools.
fn build_tool_specs<C: Send + Sync + 'static>(agent: &Agent<C>) -> Vec<ToolSpec> {
    agent
        .tools
        .iter()
        .map(|tool| match tool {
            Tool::Function(f) => ToolSpec {
                name: f.name.clone(),
                description: f.description.clone(),
                params_json_schema: f.params_json_schema.clone(),
                strict: f.strict_json_schema,
            },
            Tool::WebSearch(_) => ToolSpec {
                name: "web_search".into(),
                description: "Search the web for information.".into(),
                params_json_schema: json!({}),
                strict: false,
            },
            Tool::FileSearch(_) => ToolSpec {
                name: "file_search".into(),
                description: "Search over files in vector stores.".into(),
                params_json_schema: json!({}),
                strict: false,
            },
            Tool::CodeInterpreter(_) => ToolSpec {
                name: "code_interpreter".into(),
                description: "Execute code in a sandboxed environment.".into(),
                params_json_schema: json!({}),
                strict: false,
            },
        })
        .collect()
}

/// Build handoff tool specifications from an agent's handoffs.
fn build_handoff_specs<C: Send + Sync + 'static>(agent: &Agent<C>) -> Vec<HandoffToolSpec> {
    agent
        .handoffs
        .iter()
        .map(|h| HandoffToolSpec {
            tool_name: h.tool_name.clone(),
            tool_description: h.tool_description.clone(),
            input_json_schema: h.input_json_schema.clone(),
            strict: h.strict_json_schema,
        })
        .collect()
}

/// Process a model response, extracting run items, function calls, and handoff calls.
fn process_model_response<C: Send + Sync + 'static>(
    agent: &Agent<C>,
    response: &ModelResponse,
) -> ProcessedResponse {
    let mut new_items = Vec::new();
    let mut function_calls = Vec::new();
    let mut handoff_calls = Vec::new();

    let handoff_names: Vec<&str> = agent
        .handoffs
        .iter()
        .map(|h| h.tool_name.as_str())
        .collect();

    for output in &response.output {
        let output_type = output
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");

        match output_type {
            "message" => {
                new_items.push(RunItem::MessageOutput(MessageOutputItem {
                    agent_name: agent.name.clone(),
                    raw_item: output.clone(),
                }));
            }
            "function_call" => {
                let name = output
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let call_id = output
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let arguments = output
                    .get("arguments")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("{}")
                    .to_owned();

                if handoff_names.contains(&name.as_str()) {
                    // This is a handoff call.
                    new_items.push(RunItem::HandoffCall(HandoffCallItem {
                        agent_name: agent.name.clone(),
                        raw_item: output.clone(),
                    }));
                    handoff_calls.push(ParsedHandoffCall {
                        tool_name: name,
                        call_id,
                        arguments,
                    });
                } else {
                    // This is a function tool call.
                    new_items.push(RunItem::ToolCall(ToolCallItem {
                        agent_name: agent.name.clone(),
                        raw_item: output.clone(),
                    }));
                    function_calls.push(ParsedFunctionCall {
                        name,
                        call_id,
                        arguments,
                    });
                }
            }
            "reasoning" => {
                new_items.push(RunItem::Reasoning(ReasoningItem {
                    agent_name: agent.name.clone(),
                    raw_item: output.clone(),
                }));
            }
            _ => {
                // Unknown output type; skip silently.
            }
        }
    }

    ProcessedResponse {
        new_items,
        function_calls,
        handoff_calls,
    }
}

/// Find a function tool by name in the agent's tool list.
fn find_function_tool<'a, C: Send + Sync + 'static>(
    agent: &'a Agent<C>,
    tool_name: &str,
) -> Option<&'a FunctionTool<C>> {
    agent.tools.iter().find_map(|t| {
        if let Tool::Function(f) = t {
            if f.name == tool_name {
                return Some(f);
            }
        }
        None
    })
}

/// Extract the final output from the run items and model responses.
///
/// Looks for the last message output text. If no text is found, returns `json!(null)`.
fn extract_final_output(items: &[RunItem], _responses: &[ModelResponse]) -> serde_json::Value {
    // Walk backwards to find the last message output.
    for item in items.iter().rev() {
        if let RunItem::MessageOutput(msg) = item {
            if let Some(text) = ItemHelpers::extract_text(&msg.raw_item) {
                return serde_json::Value::String(text);
            }
        }
    }
    json!(null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::config::{ModelSettings, RunConfig};
    use crate::context::RunContextWrapper;
    use crate::guardrail::{GuardrailFunctionOutput, InputGuardrail, OutputGuardrail};
    use crate::items::ModelResponse;
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
        ) -> Result<ModelResponse> {
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
        ) -> Pin<Box<dyn Stream<Item = Result<crate::items::ResponseStreamEvent>> + Send + 'a>>
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
        ) -> Result<ModelResponse> {
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
        ) -> Pin<Box<dyn Stream<Item = Result<crate::items::ResponseStreamEvent>> + Send + 'a>>
        {
            let events: Vec<Result<crate::items::ResponseStreamEvent>> =
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
        use crate::tool::{ToolContext, function_tool};
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

    // ---- Test: process_model_response correctly categorizes output items ----

    #[test]
    fn process_model_response_categorizes_items() {
        let agent = Agent::<()>::builder("test")
            .handoff(crate::handoffs::Handoff::to_agent("other").build())
            .build();

        let response = ModelResponse {
            output: vec![
                json!({"type": "message", "content": [{"type": "output_text", "text": "hi"}]}),
                json!({"type": "function_call", "name": "add", "call_id": "c1", "arguments": "{}"}),
                json!({"type": "function_call", "name": "transfer_to_other", "call_id": "c2", "arguments": "{}"}),
                json!({"type": "reasoning", "text": "thinking"}),
            ],
            usage: Usage::default(),
            response_id: None,
            request_id: None,
        };

        let processed = process_model_response(&agent, &response);

        // Should have 4 items: message, tool call, handoff call, reasoning.
        assert_eq!(processed.new_items.len(), 4);
        assert!(matches!(processed.new_items[0], RunItem::MessageOutput(_)));
        assert!(matches!(processed.new_items[1], RunItem::ToolCall(_)));
        assert!(matches!(processed.new_items[2], RunItem::HandoffCall(_)));
        assert!(matches!(processed.new_items[3], RunItem::Reasoning(_)));

        assert_eq!(processed.function_calls.len(), 1);
        assert_eq!(processed.function_calls[0].name, "add");

        assert_eq!(processed.handoff_calls.len(), 1);
        assert_eq!(processed.handoff_calls[0].tool_name, "transfer_to_other");
    }

    // ---- Test: build_tool_specs produces correct specs ----

    #[test]
    fn build_tool_specs_from_agent() {
        use crate::tool::{ToolContext, WebSearchTool, function_tool};
        use schemars::JsonSchema;
        use serde::Deserialize;

        #[derive(Deserialize, JsonSchema)]
        #[allow(dead_code)]
        struct Params {
            x: i32,
        }

        let ft = function_tool::<(), Params, _, _>(
            "my_tool",
            "Does stuff.",
            |_ctx: ToolContext<()>, _params: Params| async move {
                Ok(ToolOutput::Text("ok".to_owned()))
            },
        )
        .expect("should create tool");

        let agent = Agent::<()>::builder("test")
            .tool(Tool::Function(ft))
            .tool(Tool::WebSearch(WebSearchTool::default()))
            .build();

        let specs = build_tool_specs(&agent);
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "my_tool");
        assert_eq!(specs[0].description, "Does stuff.");
        assert!(specs[0].strict);
        assert_eq!(specs[1].name, "web_search");
        assert!(!specs[1].strict);
    }

    // ---- Test: extract_final_output returns last message text ----

    #[test]
    fn extract_final_output_returns_last_message() {
        let items = vec![
            RunItem::MessageOutput(MessageOutputItem {
                agent_name: "a".to_owned(),
                raw_item: json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "First"}]
                }),
            }),
            RunItem::ToolCall(ToolCallItem {
                agent_name: "a".to_owned(),
                raw_item: json!({"type": "function_call"}),
            }),
            RunItem::MessageOutput(MessageOutputItem {
                agent_name: "a".to_owned(),
                raw_item: json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Second"}]
                }),
            }),
        ];

        let output = extract_final_output(&items, &[]);
        assert_eq!(output, json!("Second"));
    }

    // ---- Test: extract_final_output with no messages returns null ----

    #[test]
    fn extract_final_output_no_messages_returns_null() {
        let items = vec![RunItem::Reasoning(ReasoningItem {
            agent_name: "a".to_owned(),
            raw_item: json!({"type": "reasoning"}),
        })];
        let output = extract_final_output(&items, &[]);
        assert_eq!(output, json!(null));
    }

    // ---- Test: StopOnFirstTool behavior ----

    #[tokio::test]
    async fn stop_on_first_tool_behavior() {
        use crate::tool::{ToolContext, function_tool};
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
            ) -> Result<ModelResponse> {
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
            ) -> Pin<Box<dyn Stream<Item = Result<crate::items::ResponseStreamEvent>> + Send + 'a>>
            {
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

    // ---- Test: process_model_response correctly categorizes output items ----

    #[test]
    fn build_tool_specs_from_agent_with_handoffs() {
        let agent = Agent::<()>::builder("test")
            .handoff(crate::handoffs::Handoff::to_agent("billing").build())
            .build();

        let specs = build_handoff_specs(&agent);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].tool_name, "transfer_to_billing");
    }
}
