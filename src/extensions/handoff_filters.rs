//! Handoff input filters for transforming data before agent-to-agent transfers.
//!
//! These utilities create [`HandoffInputFilter`]
//! instances that strip tool calls, reasoning items, and MCP metadata from the
//! conversation history before handing off to the next agent.
//!
//! This mirrors the Python SDK's `extensions/handoff_filters.py`.

use std::sync::Arc;

use crate::handoffs::{HandoffInputData, HandoffInputFilter};
use crate::items::RunItem;

/// Item types in the input history that represent tool-related items and should
/// be removed by [`remove_all_tools`].
const TOOL_INPUT_TYPES: &[&str] = &[
    "function_call",
    "function_call_output",
    "reasoning",
    "mcp_list_tools",
    "mcp_approval_request",
    "mcp_approval_response",
    "tool_search_call",
    "tool_search_output",
    "code_interpreter_call",
    "image_generation_call",
    "local_shell_call",
    "local_shell_call_output",
    "shell_call",
    "shell_call_output",
    "apply_patch_call",
    "apply_patch_call_output",
];

/// Create a [`HandoffInputFilter`] that removes all tool-related items from
/// the handoff input data.
///
/// This filter removes:
/// - `ToolCall` and `ToolCallOutput` run items
/// - `HandoffCall` and `HandoffOutput` run items
/// - `Reasoning` run items
/// - Input history items with types: `function_call`, `function_call_output`,
///   `reasoning`, and MCP-specific types
///
/// The resulting input contains only message items, giving the next agent a
/// clean conversational context.
///
/// # Example
///
/// ```
/// use openai_agents::extensions::handoff_filters::remove_all_tools;
/// use openai_agents::handoffs::Handoff;
///
/// let filter = remove_all_tools();
/// let handoff: Handoff<()> = Handoff::to_agent("support")
///     .input_filter(filter)
///     .build();
/// ```
#[must_use]
pub fn remove_all_tools() -> HandoffInputFilter {
    Arc::new(|mut data: HandoffInputData| {
        Box::pin(async move {
            // Filter input history items.
            data.input_history = match data.input_history {
                crate::items::InputContent::Text(t) => crate::items::InputContent::Text(t),
                crate::items::InputContent::Items(items) => {
                    let filtered = items
                        .into_iter()
                        .filter(|item| {
                            let item_type = item
                                .get("type")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("");
                            !TOOL_INPUT_TYPES.contains(&item_type)
                        })
                        .collect();
                    crate::items::InputContent::Items(filtered)
                }
            };

            // Filter pre-handoff run items.
            data.pre_handoff_items = filter_run_items(data.pre_handoff_items);

            // Filter new items from the current turn.
            data.new_items = filter_run_items(data.new_items);

            data
        })
    })
}

/// Filter run items, keeping only message outputs.
fn filter_run_items(items: Vec<RunItem>) -> Vec<RunItem> {
    items
        .into_iter()
        .filter(|item| matches!(item, RunItem::MessageOutput(_)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handoffs::HandoffInputData;
    use crate::items::*;
    use serde_json::json;

    fn make_test_data() -> HandoffInputData {
        HandoffInputData {
            input_history: InputContent::Items(vec![
                json!({"role": "user", "content": "hello"}),
                json!({"type": "function_call", "name": "get_weather", "call_id": "c1"}),
                json!({"type": "function_call_output", "call_id": "c1", "output": "sunny"}),
                json!({"type": "reasoning", "text": "thinking..."}),
                json!({"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}),
                json!({"type": "mcp_approval_request", "tool": "test"}),
            ]),
            pre_handoff_items: vec![
                RunItem::MessageOutput(MessageOutputItem {
                    agent_name: "a".to_owned(),
                    raw_item: json!({"type": "message"}),
                }),
                RunItem::ToolCall(ToolCallItem {
                    agent_name: "a".to_owned(),
                    raw_item: json!({"type": "function_call"}),
                    tool_origin: None,
                }),
                RunItem::Reasoning(ReasoningItem {
                    agent_name: "a".to_owned(),
                    raw_item: json!({"type": "reasoning"}),
                }),
            ],
            new_items: vec![
                RunItem::HandoffCall(HandoffCallItem {
                    agent_name: "a".to_owned(),
                    raw_item: json!({"type": "function_call"}),
                }),
                RunItem::ToolCallOutput(ToolCallOutputItem {
                    agent_name: "a".to_owned(),
                    raw_item: json!({"type": "function_call_output"}),
                    output: json!("result"),
                    tool_origin: None,
                }),
                RunItem::MessageOutput(MessageOutputItem {
                    agent_name: "a".to_owned(),
                    raw_item: json!({"type": "message"}),
                }),
            ],
        }
    }

    #[tokio::test]
    async fn remove_all_tools_filters_input_history() {
        let filter = remove_all_tools();
        let data = make_test_data();
        let filtered = filter(data).await;

        if let InputContent::Items(items) = &filtered.input_history {
            // Only the user message and assistant message should remain.
            assert_eq!(items.len(), 2);
            assert_eq!(items[0]["role"], "user");
            assert_eq!(items[1]["role"], "assistant");
        } else {
            panic!("expected Items variant");
        }
    }

    #[tokio::test]
    async fn remove_all_tools_filters_pre_handoff_items() {
        let filter = remove_all_tools();
        let data = make_test_data();
        let filtered = filter(data).await;

        assert_eq!(filtered.pre_handoff_items.len(), 1);
        assert!(matches!(
            filtered.pre_handoff_items[0],
            RunItem::MessageOutput(_)
        ));
    }

    #[tokio::test]
    async fn remove_all_tools_filters_new_items() {
        let filter = remove_all_tools();
        let data = make_test_data();
        let filtered = filter(data).await;

        assert_eq!(filtered.new_items.len(), 1);
        assert!(matches!(filtered.new_items[0], RunItem::MessageOutput(_)));
    }

    #[tokio::test]
    async fn remove_all_tools_filters_hosted_tool_types() {
        let filter = remove_all_tools();
        let hosted_types = [
            "code_interpreter_call",
            "image_generation_call",
            "local_shell_call",
            "local_shell_call_output",
            "shell_call",
            "shell_call_output",
            "apply_patch_call",
            "apply_patch_call_output",
        ];
        let mut items = vec![json!({"role": "user", "content": "Hello"})];
        for t in hosted_types {
            items.push(json!({"id": "ht1", "type": t}));
        }
        items.push(json!({"role": "user", "content": "World"}));

        let data = HandoffInputData {
            input_history: InputContent::Items(items),
            pre_handoff_items: vec![],
            new_items: vec![],
        };
        let filtered = filter(data).await;
        if let InputContent::Items(items) = &filtered.input_history {
            assert_eq!(items.len(), 2);
            for item in items {
                let ty = item.get("type").and_then(serde_json::Value::as_str);
                assert!(!hosted_types.contains(&ty.unwrap_or("")));
            }
        } else {
            panic!("expected Items variant");
        }
    }

    #[tokio::test]
    async fn remove_all_tools_preserves_text_input() {
        let filter = remove_all_tools();
        let data = HandoffInputData {
            input_history: InputContent::Text("hello".to_owned()),
            pre_handoff_items: vec![],
            new_items: vec![],
        };
        let filtered = filter(data).await;
        assert_eq!(
            filtered.input_history,
            InputContent::Text("hello".to_owned())
        );
    }
}
