//! Run items and helper utilities for agent execution.
//!
//! This module defines the data types produced during an agent run, including message
//! outputs, tool calls, handoff items, and reasoning items. It also provides
//! [`ItemHelpers`] with utility methods for extracting text content and converting
//! between input/output representations.
//!
//! The types here mirror the Python SDK's `items.py` module, using `serde_json::Value`
//! as the representation for raw API items (since we do not have typed `OpenAI` response
//! models in Rust).

use crate::usage::Usage;
use serde::{Deserialize, Serialize};

/// Type alias for response input items (JSON values from the Responses API).
pub type ResponseInputItem = serde_json::Value;

/// Type alias for response output items.
pub type ResponseOutputItem = serde_json::Value;

/// Type alias for response stream events.
pub type ResponseStreamEvent = serde_json::Value;

/// Input content provided to an agent: either a plain text string or a list of input items.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum InputContent {
    /// A plain text input string.
    Text(String),
    /// A list of structured input items.
    Items(Vec<ResponseInputItem>),
}

impl From<&str> for InputContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl From<String> for InputContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<Vec<ResponseInputItem>> for InputContent {
    fn from(items: Vec<ResponseInputItem>) -> Self {
        Self::Items(items)
    }
}

/// Output variants from a tool invocation.
///
/// Mirrors the Python SDK's `ToolOutputText`, `ToolOutputImage`, and `ToolOutputFileContent`
/// types, unified into a single enum for ergonomic pattern matching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum ToolOutput {
    /// Plain text output from a tool.
    Text(String),
    /// Image output from a tool, identified by URL or file ID.
    Image {
        /// The URL of the image, if available.
        image_url: Option<String>,
        /// The file ID of the image, if available.
        file_id: Option<String>,
    },
    /// File output from a tool, identified by data, URL, or file ID.
    File {
        /// Base64-encoded file data, if available.
        file_data: Option<String>,
        /// The URL of the file, if available.
        file_url: Option<String>,
        /// The file ID, if available.
        file_id: Option<String>,
        /// The filename, if available.
        filename: Option<String>,
    },
}

/// An item generated during an agent run.
///
/// Each variant wraps a specific item struct that carries the agent name and raw API item.
/// This enum is the Rust equivalent of the Python SDK's `RunItem` type alias.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RunItem {
    /// A message output from the model.
    MessageOutput(MessageOutputItem),
    /// A handoff call initiated by the model.
    HandoffCall(HandoffCallItem),
    /// The output/result of a handoff between agents.
    HandoffOutput(HandoffOutputItem),
    /// A tool call made by the model.
    ToolCall(ToolCallItem),
    /// The output of a tool call execution.
    ToolCallOutput(ToolCallOutputItem),
    /// A reasoning step from the model.
    Reasoning(ReasoningItem),
}

/// A message output item from the model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MessageOutputItem {
    /// The name of the agent that generated this message.
    pub agent_name: String,
    /// The raw response output message from the API.
    pub raw_item: ResponseOutputItem,
}

/// A handoff call item representing a request to transfer to another agent.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HandoffCallItem {
    /// The name of the agent that initiated the handoff.
    pub agent_name: String,
    /// The raw response function tool call representing the handoff.
    pub raw_item: ResponseOutputItem,
}

/// The output of a handoff, recording the transfer from one agent to another.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HandoffOutputItem {
    /// The name of the agent associated with this handoff output.
    pub agent_name: String,
    /// The raw input item representing the handoff.
    pub raw_item: ResponseInputItem,
    /// The name of the agent that initiated the handoff.
    pub source_agent_name: String,
    /// The name of the agent that is being handed off to.
    pub target_agent_name: String,
}

/// A tool call item from the model (function call, computer action, etc.).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolCallItem {
    /// The name of the agent that made the tool call.
    pub agent_name: String,
    /// The raw tool call item from the API.
    pub raw_item: serde_json::Value,
}

/// The output of a tool call execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolCallOutputItem {
    /// The name of the agent that executed the tool.
    pub agent_name: String,
    /// The raw tool call output item from the API.
    pub raw_item: serde_json::Value,
    /// The output value produced by the tool.
    pub output: serde_json::Value,
}

/// A reasoning item produced by the model during chain-of-thought.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ReasoningItem {
    /// The name of the agent that produced the reasoning.
    pub agent_name: String,
    /// The raw reasoning item from the API.
    pub raw_item: serde_json::Value,
}

/// Response from a model call, containing outputs and metadata.
///
/// This corresponds to the Python SDK's `ModelResponse` dataclass.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ModelResponse {
    /// A list of output items (messages, tool calls, etc.) generated by the model.
    pub output: Vec<ResponseOutputItem>,
    /// Token usage information for this model call.
    pub usage: Usage,
    /// An optional response ID that can be used to reference this response in subsequent calls.
    pub response_id: Option<String>,
    /// The transport request ID for this model call, if provided by the model SDK.
    pub request_id: Option<String>,
}

impl ModelResponse {
    /// Convert the output items into a list of input items suitable for passing back to the model.
    #[must_use]
    pub fn to_input_items(&self) -> Vec<ResponseInputItem> {
        self.output.clone()
    }
}

/// Helper methods for extracting content from items and converting between representations.
///
/// This is a collection of associated functions (no instance state) that mirror the
/// Python SDK's `ItemHelpers` class.
pub struct ItemHelpers;

impl ItemHelpers {
    /// Extract the last text content from a message output item.
    ///
    /// Looks for the last element in the `"content"` array with `"type": "output_text"`
    /// and returns its `"text"` field. Returns `None` if the item is not a message or
    /// has no text content.
    #[must_use]
    pub fn extract_last_text(message: &ResponseOutputItem) -> Option<String> {
        // Only process items with type "message".
        let msg_type = message.get("type")?.as_str()?;
        if msg_type != "message" {
            return None;
        }

        let content = message.get("content")?.as_array()?;
        // Iterate in reverse to find the last output_text.
        for item in content.iter().rev() {
            if item.get("type").and_then(serde_json::Value::as_str) == Some("output_text") {
                return item
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .map(String::from);
            }
        }
        None
    }

    /// Extract all text content from a message output item, concatenated.
    ///
    /// Looks through the `"content"` array for all elements with `"type": "output_text"`
    /// and concatenates their `"text"` fields. Returns `None` if no text content is found.
    #[must_use]
    pub fn extract_text(message: &ResponseOutputItem) -> Option<String> {
        let msg_type = message.get("type")?.as_str()?;
        if msg_type != "message" {
            return None;
        }

        let content = message.get("content")?.as_array()?;
        let mut text = String::new();
        for item in content {
            if item.get("type").and_then(serde_json::Value::as_str) == Some("output_text") {
                if let Some(t) = item.get("text").and_then(serde_json::Value::as_str) {
                    text.push_str(t);
                }
            }
        }

        if text.is_empty() { None } else { Some(text) }
    }

    /// Convert input content to a list of input items.
    ///
    /// If the input is text, it is wrapped as a user message item. If it is already a
    /// list of items, the items are returned as-is.
    #[must_use]
    pub fn input_to_new_input_list(input: &InputContent) -> Vec<ResponseInputItem> {
        match input {
            InputContent::Text(s) => {
                vec![serde_json::json!({
                    "content": s,
                    "role": "user"
                })]
            }
            InputContent::Items(items) => items.clone(),
        }
    }

    /// Concatenate all text message outputs from a list of run items.
    ///
    /// Iterates through the items, extracts text from each `MessageOutput` variant,
    /// and concatenates them into a single string.
    #[must_use]
    pub fn text_message_outputs(items: &[RunItem]) -> String {
        let mut text = String::new();
        for item in items {
            if let RunItem::MessageOutput(msg) = item {
                if let Some(extracted) = Self::extract_text(&msg.raw_item) {
                    text.push_str(&extracted);
                }
            }
        }
        text
    }

    /// Create a tool call output item as a `ResponseInputItem`.
    ///
    /// Builds a JSON object with `"type": "function_call_output"`, the given `call_id`,
    /// and the output converted according to the [`ToolOutput`] variant.
    #[must_use]
    pub fn tool_call_output_item(call_id: &str, output: &ToolOutput) -> ResponseInputItem {
        let output_value = Self::convert_tool_output(output);
        serde_json::json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_value
        })
    }

    /// Convert a [`ToolOutput`] into a JSON value suitable for the Responses API.
    ///
    /// Text outputs become a plain string. Image and file outputs become a single-element
    /// array containing the structured content item.
    #[must_use]
    fn convert_tool_output(output: &ToolOutput) -> serde_json::Value {
        match output {
            ToolOutput::Text(s) => serde_json::Value::String(s.clone()),
            ToolOutput::Image { image_url, file_id } => {
                let mut item = serde_json::json!({"type": "input_image"});
                if let Some(url) = image_url {
                    item["image_url"] = serde_json::Value::String(url.clone());
                }
                if let Some(id) = file_id {
                    item["file_id"] = serde_json::Value::String(id.clone());
                }
                serde_json::json!([item])
            }
            ToolOutput::File {
                file_data,
                file_url,
                file_id,
                filename,
            } => {
                let mut item = serde_json::json!({"type": "input_file"});
                if let Some(data) = file_data {
                    item["file_data"] = serde_json::Value::String(data.clone());
                }
                if let Some(url) = file_url {
                    item["file_url"] = serde_json::Value::String(url.clone());
                }
                if let Some(id) = file_id {
                    item["file_id"] = serde_json::Value::String(id.clone());
                }
                if let Some(name) = filename {
                    item["filename"] = serde_json::Value::String(name.clone());
                }
                serde_json::json!([item])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- InputContent From conversions ----

    #[test]
    fn input_content_from_str_ref() {
        let content: InputContent = "hello".into();
        assert_eq!(content, InputContent::Text("hello".to_owned()));
    }

    #[test]
    fn input_content_from_string() {
        let content: InputContent = String::from("world").into();
        assert_eq!(content, InputContent::Text("world".to_owned()));
    }

    #[test]
    fn input_content_from_vec() {
        let items = vec![json!({"role": "user", "content": "hi"})];
        let content: InputContent = items.clone().into();
        assert_eq!(content, InputContent::Items(items));
    }

    #[test]
    fn input_content_serde_round_trip_text() {
        let content = InputContent::Text("hello".to_owned());
        let json_str = serde_json::to_string(&content).expect("serialize");
        let deserialized: InputContent = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(content, deserialized);
    }

    #[test]
    fn input_content_serde_round_trip_items() {
        let content = InputContent::Items(vec![json!({"role": "user", "content": "hi"})]);
        let json_str = serde_json::to_string(&content).expect("serialize");
        let deserialized: InputContent = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(content, deserialized);
    }

    // ---- RunItem construction for each variant ----

    #[test]
    fn run_item_message_output() {
        let item = RunItem::MessageOutput(MessageOutputItem {
            agent_name: "test_agent".to_owned(),
            raw_item: json!({"type": "message", "content": []}),
        });
        assert!(matches!(item, RunItem::MessageOutput(_)));
    }

    #[test]
    fn run_item_handoff_call() {
        let item = RunItem::HandoffCall(HandoffCallItem {
            agent_name: "agent_a".to_owned(),
            raw_item: json!({"type": "function_call", "name": "transfer_to_b"}),
        });
        assert!(matches!(item, RunItem::HandoffCall(_)));
    }

    #[test]
    fn run_item_handoff_output() {
        let item = RunItem::HandoffOutput(HandoffOutputItem {
            agent_name: "agent_a".to_owned(),
            raw_item: json!({"type": "function_call_output"}),
            source_agent_name: "agent_a".to_owned(),
            target_agent_name: "agent_b".to_owned(),
        });
        if let RunItem::HandoffOutput(ref ho) = item {
            assert_eq!(ho.source_agent_name, "agent_a");
            assert_eq!(ho.target_agent_name, "agent_b");
        } else {
            panic!("expected HandoffOutput variant");
        }
    }

    #[test]
    fn run_item_tool_call() {
        let item = RunItem::ToolCall(ToolCallItem {
            agent_name: "agent".to_owned(),
            raw_item: json!({"type": "function_call", "name": "get_weather"}),
        });
        assert!(matches!(item, RunItem::ToolCall(_)));
    }

    #[test]
    fn run_item_tool_call_output() {
        let item = RunItem::ToolCallOutput(ToolCallOutputItem {
            agent_name: "agent".to_owned(),
            raw_item: json!({"type": "function_call_output"}),
            output: json!("sunny"),
        });
        if let RunItem::ToolCallOutput(ref tco) = item {
            assert_eq!(tco.output, json!("sunny"));
        } else {
            panic!("expected ToolCallOutput variant");
        }
    }

    #[test]
    fn run_item_reasoning() {
        let item = RunItem::Reasoning(ReasoningItem {
            agent_name: "agent".to_owned(),
            raw_item: json!({"type": "reasoning", "text": "thinking..."}),
        });
        assert!(matches!(item, RunItem::Reasoning(_)));
    }

    // ---- ItemHelpers::extract_last_text ----

    #[test]
    fn extract_last_text_with_single_output_text() {
        let msg = json!({
            "type": "message",
            "content": [
                {"type": "output_text", "text": "Hello, world!"}
            ]
        });
        assert_eq!(
            ItemHelpers::extract_last_text(&msg),
            Some("Hello, world!".to_owned())
        );
    }

    #[test]
    fn extract_last_text_with_multiple_content_items() {
        let msg = json!({
            "type": "message",
            "content": [
                {"type": "output_text", "text": "First"},
                {"type": "output_text", "text": "Second"}
            ]
        });
        // Should return the last output_text.
        assert_eq!(
            ItemHelpers::extract_last_text(&msg),
            Some("Second".to_owned())
        );
    }

    #[test]
    fn extract_last_text_with_refusal_last() {
        let msg = json!({
            "type": "message",
            "content": [
                {"type": "output_text", "text": "Hello"},
                {"type": "refusal", "refusal": "I cannot do that"}
            ]
        });
        // Should return the last output_text, ignoring refusal.
        assert_eq!(
            ItemHelpers::extract_last_text(&msg),
            Some("Hello".to_owned())
        );
    }

    #[test]
    fn extract_last_text_empty_content() {
        let msg = json!({
            "type": "message",
            "content": []
        });
        assert_eq!(ItemHelpers::extract_last_text(&msg), None);
    }

    #[test]
    fn extract_last_text_not_a_message() {
        let item = json!({"type": "function_call", "name": "foo"});
        assert_eq!(ItemHelpers::extract_last_text(&item), None);
    }

    #[test]
    fn extract_last_text_no_content_field() {
        let msg = json!({"type": "message"});
        assert_eq!(ItemHelpers::extract_last_text(&msg), None);
    }

    #[test]
    fn extract_last_text_content_not_array() {
        let msg = json!({"type": "message", "content": "just a string"});
        assert_eq!(ItemHelpers::extract_last_text(&msg), None);
    }

    // ---- ItemHelpers::extract_text ----

    #[test]
    fn extract_text_concatenates_all() {
        let msg = json!({
            "type": "message",
            "content": [
                {"type": "output_text", "text": "Hello, "},
                {"type": "refusal", "refusal": "nope"},
                {"type": "output_text", "text": "world!"}
            ]
        });
        assert_eq!(
            ItemHelpers::extract_text(&msg),
            Some("Hello, world!".to_owned())
        );
    }

    #[test]
    fn extract_text_returns_none_for_non_message() {
        let item = json!({"type": "function_call"});
        assert_eq!(ItemHelpers::extract_text(&item), None);
    }

    #[test]
    fn extract_text_returns_none_for_empty_content() {
        let msg = json!({"type": "message", "content": []});
        assert_eq!(ItemHelpers::extract_text(&msg), None);
    }

    // ---- ItemHelpers::input_to_new_input_list ----

    #[test]
    fn input_to_new_input_list_from_text() {
        let input = InputContent::Text("Hello".to_owned());
        let result = ItemHelpers::input_to_new_input_list(&input);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[0]["content"], "Hello");
    }

    #[test]
    fn input_to_new_input_list_from_items() {
        let items = vec![
            json!({"role": "user", "content": "first"}),
            json!({"role": "assistant", "content": "second"}),
        ];
        let input = InputContent::Items(items.clone());
        let result = ItemHelpers::input_to_new_input_list(&input);
        assert_eq!(result, items);
    }

    // ---- ItemHelpers::text_message_outputs ----

    #[test]
    fn text_message_outputs_concatenates_messages() {
        let items = vec![
            RunItem::MessageOutput(MessageOutputItem {
                agent_name: "agent".to_owned(),
                raw_item: json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello "}]
                }),
            }),
            RunItem::ToolCall(ToolCallItem {
                agent_name: "agent".to_owned(),
                raw_item: json!({"type": "function_call"}),
            }),
            RunItem::MessageOutput(MessageOutputItem {
                agent_name: "agent".to_owned(),
                raw_item: json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "world!"}]
                }),
            }),
        ];
        let result = ItemHelpers::text_message_outputs(&items);
        assert_eq!(result, "Hello world!");
    }

    #[test]
    fn text_message_outputs_empty_items() {
        let result = ItemHelpers::text_message_outputs(&[]);
        assert_eq!(result, "");
    }

    #[test]
    fn text_message_outputs_no_messages() {
        let items = vec![RunItem::ToolCall(ToolCallItem {
            agent_name: "agent".to_owned(),
            raw_item: json!({"type": "function_call"}),
        })];
        let result = ItemHelpers::text_message_outputs(&items);
        assert_eq!(result, "");
    }

    // ---- ItemHelpers::tool_call_output_item ----

    #[test]
    fn tool_call_output_item_text() {
        let output = ToolOutput::Text("result".to_owned());
        let item = ItemHelpers::tool_call_output_item("call_123", &output);

        assert_eq!(item["type"], "function_call_output");
        assert_eq!(item["call_id"], "call_123");
        assert_eq!(item["output"], "result");
    }

    #[test]
    fn tool_call_output_item_image() {
        let output = ToolOutput::Image {
            image_url: Some("https://example.com/img.png".to_owned()),
            file_id: None,
        };
        let item = ItemHelpers::tool_call_output_item("call_456", &output);

        assert_eq!(item["type"], "function_call_output");
        assert_eq!(item["call_id"], "call_456");
        let output_arr = item["output"].as_array().expect("output should be array");
        assert_eq!(output_arr.len(), 1);
        assert_eq!(output_arr[0]["type"], "input_image");
        assert_eq!(output_arr[0]["image_url"], "https://example.com/img.png");
        assert!(output_arr[0].get("file_id").is_none());
    }

    #[test]
    fn tool_call_output_item_image_with_file_id() {
        let output = ToolOutput::Image {
            image_url: None,
            file_id: Some("file_abc".to_owned()),
        };
        let item = ItemHelpers::tool_call_output_item("call_789", &output);

        let output_arr = item["output"].as_array().expect("output should be array");
        assert_eq!(output_arr[0]["type"], "input_image");
        assert_eq!(output_arr[0]["file_id"], "file_abc");
        assert!(output_arr[0].get("image_url").is_none());
    }

    #[test]
    fn tool_call_output_item_file() {
        let output = ToolOutput::File {
            file_data: Some("base64data".to_owned()),
            file_url: None,
            file_id: Some("file_xyz".to_owned()),
            filename: Some("report.pdf".to_owned()),
        };
        let item = ItemHelpers::tool_call_output_item("call_file", &output);

        assert_eq!(item["type"], "function_call_output");
        let output_arr = item["output"].as_array().expect("output should be array");
        assert_eq!(output_arr.len(), 1);
        assert_eq!(output_arr[0]["type"], "input_file");
        assert_eq!(output_arr[0]["file_data"], "base64data");
        assert_eq!(output_arr[0]["file_id"], "file_xyz");
        assert_eq!(output_arr[0]["filename"], "report.pdf");
        assert!(output_arr[0].get("file_url").is_none());
    }

    // ---- ToolOutput serialization ----

    #[test]
    fn tool_output_text_serde_round_trip() {
        let output = ToolOutput::Text("hello".to_owned());
        let json_str = serde_json::to_string(&output).expect("serialize");
        let deserialized: ToolOutput = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(output, deserialized);
    }

    #[test]
    fn tool_output_image_serde_round_trip() {
        let output = ToolOutput::Image {
            image_url: Some("https://example.com/img.png".to_owned()),
            file_id: None,
        };
        let json_str = serde_json::to_string(&output).expect("serialize");
        let deserialized: ToolOutput = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(output, deserialized);
    }

    #[test]
    fn tool_output_file_serde_round_trip() {
        let output = ToolOutput::File {
            file_data: None,
            file_url: Some("https://example.com/file.pdf".to_owned()),
            file_id: None,
            filename: Some("file.pdf".to_owned()),
        };
        let json_str = serde_json::to_string(&output).expect("serialize");
        let deserialized: ToolOutput = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(output, deserialized);
    }

    #[test]
    fn tool_output_file_all_fields() {
        let output = ToolOutput::File {
            file_data: Some("data".to_owned()),
            file_url: Some("url".to_owned()),
            file_id: Some("id".to_owned()),
            filename: Some("name".to_owned()),
        };
        let json_str = serde_json::to_string(&output).expect("serialize");
        let deserialized: ToolOutput = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(output, deserialized);
    }

    // ---- ModelResponse ----

    #[test]
    fn model_response_to_input_items() {
        let response = ModelResponse {
            output: vec![
                json!({"type": "message", "content": []}),
                json!({"type": "function_call", "name": "foo"}),
            ],
            usage: Usage::default(),
            response_id: Some("resp_123".to_owned()),
            request_id: None,
        };
        let input_items = response.to_input_items();
        assert_eq!(input_items.len(), 2);
        assert_eq!(input_items[0]["type"], "message");
        assert_eq!(input_items[1]["type"], "function_call");
    }

    #[test]
    fn model_response_empty_output() {
        let response = ModelResponse {
            output: vec![],
            usage: Usage::default(),
            response_id: None,
            request_id: None,
        };
        assert!(response.to_input_items().is_empty());
    }
}
