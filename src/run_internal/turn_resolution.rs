//! Turn resolution: processing model responses, extracting outputs, and building specs.
//!
//! This module contains the logic for determining what to do after a model
//! response is received: categorise output items, detect function calls vs
//! handoff calls, extract the final output text, and build tool / handoff
//! specifications for the model.

use serde_json::json;

use crate::agent::Agent;
use crate::items::{
    HandoffCallItem, ItemHelpers, MessageOutputItem, ModelResponse, ReasoningItem, RunItem,
    ToolCallItem,
};
use crate::models::{HandoffToolSpec, ToolSpec};
use crate::tool::Tool;

// ---------------------------------------------------------------------------
// Internal helper types
// ---------------------------------------------------------------------------

/// A parsed function call from the model response.
pub struct ParsedFunctionCall {
    /// The tool name.
    pub name: String,
    /// The call ID.
    pub call_id: String,
    /// The raw JSON arguments string.
    pub arguments: String,
}

/// A parsed handoff call from the model response.
pub struct ParsedHandoffCall {
    /// The handoff tool name.
    pub tool_name: String,
    /// The call ID.
    pub call_id: String,
    /// The raw JSON arguments string.
    pub arguments: String,
}

/// Result of processing a model response.
pub struct ProcessedResponse {
    /// New run items extracted from the response.
    pub new_items: Vec<RunItem>,
    /// Function tool calls to execute.
    pub function_calls: Vec<ParsedFunctionCall>,
    /// Handoff calls to process.
    pub handoff_calls: Vec<ParsedHandoffCall>,
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// Process a model response, extracting run items, function calls, and handoff calls.
pub fn process_model_response<C: Send + Sync + 'static>(
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

/// Build tool specifications from an agent's tools.
pub fn build_tool_specs<C: Send + Sync + 'static>(agent: &Agent<C>) -> Vec<ToolSpec> {
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
            Tool::Computer(_) => ToolSpec {
                name: "computer".into(),
                description: "Control a computer via screenshots and actions.".into(),
                params_json_schema: json!({}),
                strict: false,
            },
            Tool::ApplyPatch(_) => ToolSpec {
                name: "apply_patch".into(),
                description: "Apply a patch to edit files.".into(),
                params_json_schema: json!({}),
                strict: false,
            },
            Tool::ToolSearch(_) => ToolSpec {
                name: "tool_search".into(),
                description: "Search for deferred tools by namespace.".into(),
                params_json_schema: json!({}),
                strict: false,
            },
        })
        .collect()
}

/// Build handoff tool specifications from an agent's handoffs.
pub fn build_handoff_specs<C: Send + Sync + 'static>(agent: &Agent<C>) -> Vec<HandoffToolSpec> {
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

/// Extract the final output from the run items and model responses.
///
/// Looks for the last message output text. If no text is found, returns `json!(null)`.
/// If the extracted text parses as valid JSON, the parsed value is returned so that
/// structured-output responses (which the model emits as a JSON string in the text
/// field) are not double-encoded.
pub fn extract_final_output(items: &[RunItem], _responses: &[ModelResponse]) -> serde_json::Value {
    // Walk backwards to find the last message output.
    for item in items.iter().rev() {
        if let RunItem::MessageOutput(msg) = item {
            if let Some(text) = ItemHelpers::extract_text(&msg.raw_item) {
                // Attempt to parse as JSON to handle structured-output responses.
                // If the text is valid JSON (object or array), return the parsed value.
                // Otherwise wrap it as a JSON string, which is the normal plain-text case.
                let trimmed = text.trim();
                if trimmed.starts_with('{') || trimmed.starts_with('[') {
                    if let Ok(parsed) = serde_json::from_str(trimmed) {
                        return parsed;
                    }
                }
                return serde_json::Value::String(text);
            }
        }
    }
    json!(null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::{MessageOutputItem, ReasoningItem, ToolCallItem};
    use crate::usage::Usage;
    use serde_json::json;

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

    #[test]
    fn build_tool_specs_from_agent() {
        use crate::items::ToolOutput;
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

    #[test]
    fn build_handoff_specs_from_agent() {
        let agent = Agent::<()>::builder("test")
            .handoff(crate::handoffs::Handoff::to_agent("billing").build())
            .build();

        let specs = build_handoff_specs(&agent);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].tool_name, "transfer_to_billing");
    }

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

    #[test]
    fn extract_final_output_no_messages_returns_null() {
        let items = vec![RunItem::Reasoning(ReasoningItem {
            agent_name: "a".to_owned(),
            raw_item: json!({"type": "reasoning"}),
        })];
        let output = extract_final_output(&items, &[]);
        assert_eq!(output, json!(null));
    }
}
