//! Utility functions and helpers for the Agents SDK.
//!
//! This module provides convenience functions for pretty-printing run results,
//! basic JSON schema validation, string manipulation, and name transformations.
//! It mirrors the Python SDK's `util/` package.

use std::fmt::Write;

use crate::error::{AgentError, Result};
use crate::items::RunItem;
use crate::result::RunResult;

/// Pretty-print a [`RunResult`] for debugging.
///
/// Produces a human-readable multi-line summary including the last agent name,
/// final output, item count, raw response count, guardrail results, and a
/// per-item breakdown by variant kind.
#[must_use]
pub fn pretty_print_result(result: &RunResult) -> String {
    let mut output = String::from("RunResult:");
    let _ = write!(
        output,
        "\n- Last agent: Agent(name=\"{}\", ...)",
        result.last_agent_name
    );

    let final_output_str = final_output_display(&result.final_output);
    let _ = write!(
        output,
        "\n- Final output:\n{}",
        indent(&final_output_str, 2)
    );

    let _ = write!(output, "\n- {} new item(s)", result.new_items.len());
    let _ = write!(output, "\n- {} raw response(s)", result.raw_responses.len());
    let _ = write!(
        output,
        "\n- {} input guardrail result(s)",
        result.input_guardrail_results.len()
    );
    let _ = write!(
        output,
        "\n- {} output guardrail result(s)",
        result.output_guardrail_results.len()
    );

    if !result.new_items.is_empty() {
        output.push_str("\n- Items:");
        for (i, item) in result.new_items.iter().enumerate() {
            let _ = write!(output, "\n  [{i}] {}", run_item_kind(item));
        }
    }

    let _ = write!(
        output,
        "\n- Usage: {} total ({} input, {} output)",
        result.usage.total_tokens, result.usage.input_tokens, result.usage.output_tokens,
    );
    output.push_str("\n(See `RunResult` for more details)");
    output
}

/// Validate that a JSON value conforms to a JSON schema (basic validation).
///
/// Performs type checking against the `"type"` field and validates that all
/// properties listed in `"required"` are present in the value. This is a
/// lightweight validator intended for quick sanity checks, not a full
/// JSON Schema implementation.
///
/// # Errors
///
/// Returns [`AgentError::UserError`] if the value does not match the expected
/// type or is missing a required property.
pub fn validate_json_schema(value: &serde_json::Value, schema: &serde_json::Value) -> Result<()> {
    // Basic type checking.
    if let Some(schema_type) = schema.get("type").and_then(|t| t.as_str()) {
        let type_ok = match schema_type {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "number" | "integer" => value.is_number(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => true,
        };
        if !type_ok {
            return Err(AgentError::UserError {
                message: format!(
                    "Expected type '{schema_type}', got {}",
                    value_type_name(value)
                ),
            });
        }
    }

    // Check required properties.
    if let (Some(required), Some(obj)) = (
        schema.get("required").and_then(|r| r.as_array()),
        value.as_object(),
    ) {
        for req in required {
            if let Some(prop_name) = req.as_str() {
                if !obj.contains_key(prop_name) {
                    return Err(AgentError::UserError {
                        message: format!("Missing required property: '{prop_name}'"),
                    });
                }
            }
        }
    }

    Ok(())
}

/// Truncate a string to a maximum length, appending "..." if truncated.
///
/// If the string is already within `max_len`, it is returned as-is. When
/// `max_len` is greater than 3, the returned string ends with "..." to indicate
/// truncation. For very small `max_len` values (<= 3), the string is simply
/// sliced without an ellipsis.
#[must_use]
pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        let boundary = floor_char_boundary(s, max_len - 3);
        format!("{}...", &s[..boundary])
    } else {
        let boundary = floor_char_boundary(s, max_len);
        s[..boundary].to_string()
    }
}

/// Transform a string into a valid function-style name.
///
/// Replaces spaces and non-alphanumeric characters (except underscores) with
/// underscores, then lowercases the result. This mirrors the Python SDK's
/// `transform_string_function_style` utility.
#[must_use]
pub fn transform_string_function_style(name: &str) -> String {
    let transformed: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    transformed.to_lowercase()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return a human-readable name for a `serde_json::Value` variant.
const fn value_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Return the kind name of a [`RunItem`] variant.
const fn run_item_kind(item: &RunItem) -> &'static str {
    match item {
        RunItem::MessageOutput(_) => "MessageOutput",
        RunItem::ToolCall(_) => "ToolCall",
        RunItem::ToolCallOutput(_) => "ToolCallOutput",
        RunItem::HandoffCall(_) => "HandoffCall",
        RunItem::HandoffOutput(_) => "HandoffOutput",
        RunItem::Reasoning(_) => "Reasoning",
    }
}

/// Display a final output value as a string.
fn final_output_display(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    }
}

/// Indent every line of `text` by `level * 2` spaces.
fn indent(text: &str, level: usize) -> String {
    let prefix = "  ".repeat(level);
    text.lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Find the largest byte index at or before `index` that is a char boundary.
///
/// This is a simplified version of `str::floor_char_boundary` which is
/// currently nightly-only.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::{MessageOutputItem, ModelResponse, ToolCallItem};
    use crate::usage::Usage;
    use serde_json::json;

    // ---- pretty_print_result ----

    #[test]
    fn pretty_print_result_basic() {
        let result = RunResult {
            input: crate::items::InputContent::Text("hello".to_owned()),
            new_items: vec![
                RunItem::MessageOutput(MessageOutputItem {
                    agent_name: "agent".to_owned(),
                    raw_item: json!({"type": "message", "content": []}),
                }),
                RunItem::ToolCall(ToolCallItem {
                    agent_name: "agent".to_owned(),
                    raw_item: json!({"type": "function_call"}),
                }),
            ],
            raw_responses: vec![ModelResponse::new(
                vec![json!({"type": "message"})],
                Usage::default(),
                Some("resp_1".to_owned()),
                None,
            )],
            final_output: json!("The answer is 42"),
            last_agent_name: "test_agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage {
                requests: 1,
                input_tokens: 100,
                output_tokens: 50,
                total_tokens: 150,
                ..Usage::default()
            },
        };

        let output = pretty_print_result(&result);
        assert!(output.contains("RunResult:"));
        assert!(output.contains("test_agent"));
        assert!(output.contains("2 new item(s)"));
        assert!(output.contains("1 raw response(s)"));
        assert!(output.contains("[0] MessageOutput"));
        assert!(output.contains("[1] ToolCall"));
        assert!(output.contains("150 total"));
        assert!(output.contains("100 input"));
        assert!(output.contains("50 output"));
        assert!(output.contains("The answer is 42"));
    }

    #[test]
    fn pretty_print_result_no_items() {
        let result = RunResult {
            input: crate::items::InputContent::Text("q".to_owned()),
            new_items: vec![],
            raw_responses: vec![],
            final_output: json!(null),
            last_agent_name: "agent".to_owned(),
            input_guardrail_results: vec![],
            output_guardrail_results: vec![],
            usage: Usage::default(),
        };

        let output = pretty_print_result(&result);
        assert!(output.contains("0 new item(s)"));
        assert!(output.contains("None"));
        // Should not contain the Items section.
        assert!(!output.contains("[0]"));
    }

    // ---- validate_json_schema ----

    #[test]
    fn validate_json_schema_correct_type() {
        let value = json!({"name": "Alice", "age": 30});
        let schema = json!({"type": "object", "required": ["name"]});
        assert!(validate_json_schema(&value, &schema).is_ok());
    }

    #[test]
    fn validate_json_schema_wrong_type() {
        let value = json!("a string");
        let schema = json!({"type": "object"});
        let err = validate_json_schema(&value, &schema).unwrap_err();
        assert!(err.to_string().contains("Expected type 'object'"));
        assert!(err.to_string().contains("string"));
    }

    #[test]
    fn validate_json_schema_missing_required() {
        let value = json!({"name": "Alice"});
        let schema = json!({"type": "object", "required": ["name", "age"]});
        let err = validate_json_schema(&value, &schema).unwrap_err();
        assert!(err.to_string().contains("Missing required property: 'age'"));
    }

    #[test]
    fn validate_json_schema_all_required_present() {
        let value = json!({"name": "Alice", "age": 30});
        let schema = json!({"type": "object", "required": ["name", "age"]});
        assert!(validate_json_schema(&value, &schema).is_ok());
    }

    #[test]
    fn validate_json_schema_no_type_in_schema() {
        let value = json!(42);
        let schema = json!({});
        assert!(validate_json_schema(&value, &schema).is_ok());
    }

    #[test]
    fn validate_json_schema_array_type() {
        let value = json!([1, 2, 3]);
        let schema = json!({"type": "array"});
        assert!(validate_json_schema(&value, &schema).is_ok());
    }

    #[test]
    fn validate_json_schema_boolean_type() {
        let value = json!(true);
        let schema = json!({"type": "boolean"});
        assert!(validate_json_schema(&value, &schema).is_ok());
    }

    #[test]
    fn validate_json_schema_null_type() {
        let value = json!(null);
        let schema = json!({"type": "null"});
        assert!(validate_json_schema(&value, &schema).is_ok());
    }

    #[test]
    fn validate_json_schema_number_accepts_integer() {
        let value = json!(42);
        let schema = json!({"type": "number"});
        assert!(validate_json_schema(&value, &schema).is_ok());
    }

    #[test]
    fn validate_json_schema_integer_accepts_number() {
        let value = json!(42);
        let schema = json!({"type": "integer"});
        assert!(validate_json_schema(&value, &schema).is_ok());
    }

    // ---- truncate_string ----

    #[test]
    fn truncate_string_no_truncation() {
        assert_eq!(truncate_string("hello", 10), "hello");
    }

    #[test]
    fn truncate_string_exact_length() {
        assert_eq!(truncate_string("hello", 5), "hello");
    }

    #[test]
    fn truncate_string_with_ellipsis() {
        assert_eq!(truncate_string("hello world", 8), "hello...");
    }

    #[test]
    fn truncate_string_very_short_max() {
        assert_eq!(truncate_string("hello", 3), "hel");
    }

    #[test]
    fn truncate_string_max_zero() {
        assert_eq!(truncate_string("hello", 0), "");
    }

    #[test]
    fn truncate_string_max_one() {
        assert_eq!(truncate_string("hello", 1), "h");
    }

    #[test]
    fn truncate_string_empty_input() {
        assert_eq!(truncate_string("", 5), "");
    }

    #[test]
    fn truncate_string_unicode_safe() {
        // Ensure we do not panic on multi-byte characters.
        let s = "hello \u{1F600} world";
        let result = truncate_string(s, 8);
        // Should not panic and should produce a valid string.
        assert!(result.len() <= 11); // 8 bytes max + "..."
    }

    // ---- transform_string_function_style ----

    #[test]
    fn transform_simple_name() {
        assert_eq!(
            transform_string_function_style("get_weather"),
            "get_weather"
        );
    }

    #[test]
    fn transform_with_spaces() {
        assert_eq!(
            transform_string_function_style("get weather"),
            "get_weather"
        );
    }

    #[test]
    fn transform_with_special_chars() {
        assert_eq!(transform_string_function_style("my-tool.v2"), "my_tool_v2");
    }

    #[test]
    fn transform_uppercase() {
        assert_eq!(transform_string_function_style("GetWeather"), "getweather");
    }

    #[test]
    fn transform_empty() {
        assert_eq!(transform_string_function_style(""), "");
    }

    // ---- internal helpers ----

    #[test]
    fn value_type_name_coverage() {
        assert_eq!(value_type_name(&json!(null)), "null");
        assert_eq!(value_type_name(&json!(true)), "boolean");
        assert_eq!(value_type_name(&json!(42)), "number");
        assert_eq!(value_type_name(&json!("hi")), "string");
        assert_eq!(value_type_name(&json!([1])), "array");
        assert_eq!(value_type_name(&json!({"a": 1})), "object");
    }

    #[test]
    fn indent_helper() {
        assert_eq!(indent("line1\nline2", 1), "  line1\n  line2");
        assert_eq!(indent("single", 2), "    single");
    }

    #[test]
    fn final_output_display_variants() {
        assert_eq!(final_output_display(&json!(null)), "None");
        assert_eq!(final_output_display(&json!("hello")), "hello");
        // Object should be pretty-printed JSON.
        let obj_display = final_output_display(&json!({"a": 1}));
        assert!(obj_display.contains("\"a\""));
    }

    #[test]
    fn floor_char_boundary_ascii() {
        assert_eq!(floor_char_boundary("hello", 3), 3);
        assert_eq!(floor_char_boundary("hello", 10), 5);
        assert_eq!(floor_char_boundary("hello", 0), 0);
    }

    #[test]
    fn floor_char_boundary_multibyte() {
        let s = "\u{1F600}abc"; // 4-byte emoji + 3 ascii
        // Index 2 falls within the emoji, should snap back to 0.
        assert_eq!(floor_char_boundary(s, 2), 0);
        // Index 4 is right after the emoji.
        assert_eq!(floor_char_boundary(s, 4), 4);
    }
}
