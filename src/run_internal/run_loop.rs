//! Core agent turn loop.
//!
//! This module contains the internal run loop that drives agent execution.
//! It orchestrates instruction resolution, model calling, tool execution,
//! handoff handling, guardrail checking, and lifecycle hook invocation.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::RwLock;

use crate::agent::{Agent, ToolUseBehavior};
use crate::config::RunConfig;
use crate::context::RunContextWrapper;
use crate::error::{AgentError, Result};
use crate::guardrail::{InputGuardrailResult, OutputGuardrailResult};
use crate::items::{
    HandoffOutputItem, InputContent, ItemHelpers, ModelResponse, ResponseInputItem, RunItem,
};
use crate::lifecycle::RunHooks;
use crate::models::{Model, ModelTracing, OutputSchemaSpec};
use crate::prompts;
use crate::result::RunResult;
use crate::retry::RetryPolicy;
use crate::stream_events::{RunItemEventName, StreamEvent};
use crate::usage::Usage;

use super::tool_execution::{execute_tool_calls, find_function_tool};
use super::turn_resolution::{
    build_handoff_specs, build_tool_specs, extract_final_output, process_model_response,
};

/// Internal method that implements the full agent loop with optional multi-agent support.
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)]
pub async fn run_internal<C: Send + Sync + 'static>(
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
        let prev_resp_id: Option<&str> = None;

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
        //
        // Note: `previous_response_id` is tracked here but currently passed as
        // `None` to the model because the runner operates in stateless mode —
        // it always sends the full conversation history as `input`. Using
        // `previous_response_id` with the Responses API in stateful mode requires
        // sending only the *delta* items (tool outputs) rather than the full
        // history, which is not yet implemented.
        let _ = &response.response_id; // suppress unused-variable warning

        // 10. Process the model response.
        let processed = process_model_response(current_agent, &response);
        let mut turn_items = processed.new_items;
        let function_calls = processed.function_calls;
        let handoff_calls = processed.handoff_calls;

        // 11. Execute function tool calls.
        let tool_result =
            execute_tool_calls(current_agent, &function_calls, &ctx, hooks.as_ref()).await;
        turn_items.extend(tool_result.new_items);
        let tool_outputs = tool_result.tool_outputs;
        let had_tool_calls = tool_result.had_tool_calls;
        let stop_tool_names = tool_result.stop_tool_names;

        // 12. Handle handoffs.
        if let Some(handoff_call) = handoff_calls.first() {
            // Find the matching handoff.
            let handoff = current_agent
                .handoffs
                .iter()
                .find(|h| h.tool_name == handoff_call.tool_name);

            if let Some(h) = handoff {
                let ctx_read = ctx.read().await;
                let target_agent_name = h.invoke(&ctx_read, handoff_call.arguments.clone()).await?;

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
pub async fn streaming_loop<C: Send + Sync + 'static>(
    agent: &'static Agent<C>,
    input: InputContent,
    context: C,
    model: Arc<dyn Model>,
    hooks: Option<Arc<dyn RunHooks<C>>>,
    config: Option<RunConfig>,
    event_tx: tokio::sync::mpsc::Sender<StreamEvent>,
    mut cancel_rx: tokio::sync::oneshot::Receiver<()>,
    run_loop_exception: std::sync::Arc<std::sync::Mutex<Option<AgentError>>>,
) {
    // Helper closure that stashes an error for `RunResultStreaming::run_loop_exception`.
    let record_error = |err: AgentError| {
        if let Ok(mut slot) = run_loop_exception.lock() {
            if slot.is_none() {
                *slot = Some(err);
            }
        }
    };
    let config = config.unwrap_or_default();
    let ctx = Arc::new(RwLock::new(RunContextWrapper::new(context)));
    let max_turns = config.max_turns;

    let mut current_input = ItemHelpers::input_to_new_input_list(&input);

    let tracing = if config.tracing_disabled {
        ModelTracing::Disabled
    } else {
        ModelTracing::Enabled
    };

    for turn in 0..max_turns {
        // Check for cancellation.
        if cancel_rx.try_recv().is_ok() {
            return;
        }

        // Run input guardrails (first turn only). A tripwire trigger halts the
        // streamed run before any tool execution or further model calls, matching
        // the Python SDK's behaviour (see commit `fa049a26` / issue #2688).
        if turn == 0 {
            let ctx_read = ctx.read().await;
            for guardrail in &agent.input_guardrails {
                match guardrail.run(&ctx_read, &agent.name, &input).await {
                    Ok(result) => {
                        if result.output.tripwire_triggered {
                            record_error(AgentError::InputGuardrailTripwire {
                                guardrail_name: result.guardrail_name,
                            });
                            return;
                        }
                    }
                    Err(err) => {
                        record_error(err);
                        return;
                    }
                }
            }
        }

        // Get system prompt via prompts module.
        let raw_instructions = {
            let ctx_read = ctx.read().await;
            match agent.get_instructions(&ctx_read).await {
                Ok(instr) => instr,
                Err(err) => {
                    record_error(err);
                    return;
                }
            }
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
                            Some(Err(err)) => {
                                record_error(err);
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
                let tool_ctx = crate::tool::ToolContext {
                    context: Arc::clone(&ctx),
                    tool_name: fc.name.clone(),
                    tool_call_id: fc.call_id.clone(),
                };
                match ft.invoke(tool_ctx, fc.arguments.clone()).await {
                    Ok(out) => out,
                    Err(e) => crate::items::ToolOutput::Text(format!("Error: {e}")),
                }
            } else {
                crate::items::ToolOutput::Text(format!("Error: tool '{}' not found", fc.name))
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
