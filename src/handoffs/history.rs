//! Conversation history management for agent handoffs.
//!
//! When one agent hands off to another, this module controls how the
//! conversation history is transformed. By default, the previous transcript is
//! summarized into a single assistant message wrapped with configurable markers.
//! Custom mappers can replace this behaviour entirely.
//!
//! The implementation mirrors the Python SDK's `handoffs/history.py`.

use std::sync::{Mutex, OnceLock};

use crate::items::{InputContent, ItemHelpers, ResponseInputItem, RunItem};

use super::HandoffInputData;

/// Default start marker for the nested conversation summary.
const DEFAULT_CONVERSATION_HISTORY_START: &str = "<CONVERSATION HISTORY>";

/// Default end marker for the nested conversation summary.
const DEFAULT_CONVERSATION_HISTORY_END: &str = "</CONVERSATION HISTORY>";

/// Input types that are represented in the summary text only.
///
/// These items must not be forwarded verbatim to the next agent because their
/// content is already captured in the generated summary message.
const SUMMARY_ONLY_INPUT_TYPES: &[&str] = &["function_call", "function_call_output", "reasoning"];

/// A function that maps a transcript of conversation items to the history the
/// next agent will see.
pub type HandoffHistoryMapper =
    Box<dyn Fn(&[ResponseInputItem]) -> Vec<ResponseInputItem> + Send + Sync>;

// ---------------------------------------------------------------------------
// Global conversation-history wrappers (mirrors the Python module-level state)
// ---------------------------------------------------------------------------

/// Holder for the mutable start/end wrapper strings.
struct Wrappers {
    start: String,
    end: String,
}

/// Global singleton for the conversation-history wrappers.
fn wrappers() -> &'static Mutex<Wrappers> {
    static INSTANCE: OnceLock<Mutex<Wrappers>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        Mutex::new(Wrappers {
            start: DEFAULT_CONVERSATION_HISTORY_START.to_owned(),
            end: DEFAULT_CONVERSATION_HISTORY_END.to_owned(),
        })
    })
}

/// Override the markers that wrap the generated conversation summary.
///
/// Pass `None` to leave either side unchanged.
///
/// # Panics
///
/// Panics if the internal wrappers mutex has been poisoned.
pub fn set_conversation_history_wrappers(start: Option<&str>, end: Option<&str>) {
    let mut w = wrappers().lock().expect("wrappers lock poisoned");
    if let Some(s) = start {
        s.clone_into(&mut w.start);
    }
    if let Some(e) = end {
        e.clone_into(&mut w.end);
    }
}

/// Restore the default `<CONVERSATION HISTORY>` markers.
///
/// # Panics
///
/// Panics if the internal wrappers mutex has been poisoned.
pub fn reset_conversation_history_wrappers() {
    let mut w = wrappers().lock().expect("wrappers lock poisoned");
    DEFAULT_CONVERSATION_HISTORY_START.clone_into(&mut w.start);
    DEFAULT_CONVERSATION_HISTORY_END.clone_into(&mut w.end);
}

/// Return the current start/end markers used for the nested conversation summary.
///
/// # Panics
///
/// Panics if the internal wrappers mutex has been poisoned.
#[must_use]
pub fn get_conversation_history_wrappers() -> (String, String) {
    let w = wrappers().lock().expect("wrappers lock poisoned");
    (w.start.clone(), w.end.clone())
}

// ---------------------------------------------------------------------------
// Core history transformation
// ---------------------------------------------------------------------------

/// Summarize the previous transcript for the next agent.
///
/// This function normalizes the input history, flattens any previously nested
/// summaries, converts run items to plain inputs, builds a unified transcript,
/// and applies the `history_mapper` (defaulting to [`default_handoff_history_mapper`]).
///
/// The returned [`HandoffInputData`] has its `input_history` replaced with the
/// mapped summary, and its `pre_handoff_items` and `new_items` filtered to
/// exclude items already represented in the summary.
#[must_use]
pub fn nest_handoff_history(
    handoff_input_data: &HandoffInputData,
    history_mapper: Option<&HandoffHistoryMapper>,
) -> HandoffInputData {
    let normalized_history = normalize_input_history(&handoff_input_data.input_history);
    let flattened_history = flatten_nested_history_messages(&normalized_history);

    // Convert pre-handoff items to plain inputs for the transcript.
    let mut pre_items_as_inputs: Vec<ResponseInputItem> = Vec::new();
    let mut filtered_pre_items: Vec<RunItem> = Vec::new();
    for run_item in &handoff_input_data.pre_handoff_items {
        let plain_input = run_item_to_plain_input(run_item);
        pre_items_as_inputs.push(plain_input.clone());
        if should_forward_pre_item(&plain_input) {
            filtered_pre_items.push(run_item.clone());
        }
    }

    // Convert new items to plain inputs for the transcript.
    let mut new_items_as_inputs: Vec<ResponseInputItem> = Vec::new();
    let mut filtered_input_items: Vec<RunItem> = Vec::new();
    for run_item in &handoff_input_data.new_items {
        let plain_input = run_item_to_plain_input(run_item);
        new_items_as_inputs.push(plain_input.clone());
        if should_forward_new_item(&plain_input) {
            filtered_input_items.push(run_item.clone());
        }
    }

    // Build the full transcript.
    let mut transcript = flattened_history;
    transcript.extend(pre_items_as_inputs);
    transcript.extend(new_items_as_inputs);

    // Apply the mapper (default or custom).
    let history_items = history_mapper.map_or_else(
        || default_handoff_history_mapper(&transcript),
        |mapper| mapper(&transcript),
    );

    HandoffInputData {
        input_history: InputContent::Items(history_items),
        pre_handoff_items: filtered_pre_items,
        new_items: filtered_input_items,
    }
}

/// Return a single assistant message summarizing the transcript.
///
/// This is the default history mapper used when no custom mapper is supplied.
#[must_use]
pub fn default_handoff_history_mapper(transcript: &[ResponseInputItem]) -> Vec<ResponseInputItem> {
    let summary_message = build_summary_message(transcript);
    vec![summary_message]
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Normalize input history to a flat list of input items.
fn normalize_input_history(input_history: &InputContent) -> Vec<ResponseInputItem> {
    ItemHelpers::input_to_new_input_list(input_history)
}

/// Convert a `RunItem` to a plain `ResponseInputItem` (its `raw_item`).
fn run_item_to_plain_input(run_item: &RunItem) -> ResponseInputItem {
    match run_item {
        RunItem::MessageOutput(item) => item.raw_item.clone(),
        RunItem::HandoffCall(item) => item.raw_item.clone(),
        RunItem::HandoffOutput(item) => item.raw_item.clone(),
        RunItem::ToolCall(item) => item.raw_item.clone(),
        RunItem::ToolCallOutput(item) => item.raw_item.clone(),
        RunItem::Reasoning(item) => item.raw_item.clone(),
    }
}

/// Build a single assistant message summarizing the transcript.
fn build_summary_message(transcript: &[ResponseInputItem]) -> ResponseInputItem {
    let summary_lines: Vec<String> = if transcript.is_empty() {
        vec!["(no previous turns recorded)".to_owned()]
    } else {
        transcript
            .iter()
            .enumerate()
            .map(|(idx, item)| format!("{}. {}", idx + 1, format_transcript_item(item)))
            .collect()
    };

    let (start_marker, end_marker) = get_conversation_history_wrappers();
    let mut content_lines = vec![
        "For context, here is the conversation so far between the user and the previous agent:"
            .to_owned(),
        start_marker,
    ];
    content_lines.extend(summary_lines);
    content_lines.push(end_marker);
    let content = content_lines.join("\n");

    serde_json::json!({
        "role": "assistant",
        "content": content,
    })
}

/// Format a single transcript item as a human-readable summary line.
fn format_transcript_item(item: &ResponseInputItem) -> String {
    // If the item has a "role", format as "role: content".
    if let Some(role) = item.get("role").and_then(serde_json::Value::as_str) {
        let mut prefix = role.to_owned();
        if let Some(name) = item.get("name").and_then(serde_json::Value::as_str) {
            if !name.is_empty() {
                prefix = format!("{prefix} ({name})");
            }
        }
        let content_str = stringify_content(item.get("content"));
        if content_str.is_empty() {
            return prefix;
        }
        return format!("{prefix}: {content_str}");
    }

    // Otherwise, format by type.
    let item_type = item
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("item");

    let rest: serde_json::Map<String, serde_json::Value> = item
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter(|(k, _)| k.as_str() != "type" && k.as_str() != "provider_data")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default();

    let serialized = serde_json::to_string(&rest).unwrap_or_else(|_| format!("{rest:?}"));
    if serialized.is_empty() {
        item_type.to_owned()
    } else {
        format!("{item_type}: {serialized}")
    }
}

/// Convert content to a string representation.
fn stringify_content(content: Option<&serde_json::Value>) -> String {
    match content {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_else(|_| format!("{other}")),
    }
}

/// Flatten previously nested history messages by extracting their transcript.
fn flatten_nested_history_messages(items: &[ResponseInputItem]) -> Vec<ResponseInputItem> {
    let mut flattened = Vec::new();
    for item in items {
        if let Some(nested_transcript) = extract_nested_history_transcript(item) {
            flattened.extend(nested_transcript);
        } else {
            flattened.push(item.clone());
        }
    }
    flattened
}

/// Extract the transcript from a previously nested history message, if any.
fn extract_nested_history_transcript(item: &ResponseInputItem) -> Option<Vec<ResponseInputItem>> {
    let content = item.get("content")?.as_str()?;
    let (start_marker, end_marker) = get_conversation_history_wrappers();
    let start_idx = content.find(&start_marker)?;
    let end_idx = content.find(&end_marker)?;
    if end_idx <= start_idx {
        return None;
    }
    let body = &content[start_idx + start_marker.len()..end_idx];
    let lines: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let mut parsed = Vec::new();
    for line in lines {
        if let Some(parsed_item) = parse_summary_line(line) {
            parsed.push(parsed_item);
        }
    }
    Some(parsed)
}

/// Parse a single numbered summary line back into a `ResponseInputItem`.
fn parse_summary_line(line: &str) -> Option<ResponseInputItem> {
    let stripped = line.trim();
    if stripped.is_empty() {
        return None;
    }

    // Strip leading number prefix like "1. ".
    let stripped = stripped.find('.').map_or(stripped, |dot_index| {
        if stripped[..dot_index].chars().all(|c| c.is_ascii_digit()) {
            stripped[dot_index + 1..].trim_start()
        } else {
            stripped
        }
    });

    // Split on the first colon.
    let (role_part, remainder) = stripped.split_once(':')?;
    let role_text = role_part.trim();
    if role_text.is_empty() {
        return None;
    }

    let (role, name) = split_role_and_name(role_text);
    let mut reconstructed = serde_json::json!({ "role": role });
    if let Some(n) = name {
        reconstructed["name"] = serde_json::Value::String(n);
    }
    let content = remainder.trim();
    if !content.is_empty() {
        reconstructed["content"] = serde_json::Value::String(content.to_owned());
    }
    Some(reconstructed)
}

/// Split a role string like `"assistant (agent_name)"` into `("assistant", Some("agent_name"))`.
fn split_role_and_name(role_text: &str) -> (String, Option<String>) {
    if role_text.ends_with(')') {
        if let Some(open_idx) = role_text.rfind('(') {
            let possible_name = role_text[open_idx + 1..role_text.len() - 1].trim();
            let role_candidate = role_text[..open_idx].trim();
            if !possible_name.is_empty() {
                let role = if role_candidate.is_empty() {
                    "developer".to_owned()
                } else {
                    role_candidate.to_owned()
                };
                return (role, Some(possible_name.to_owned()));
            }
        }
    }
    let role = if role_text.is_empty() {
        "developer".to_owned()
    } else {
        role_text.to_owned()
    };
    (role, None)
}

/// Return `false` when the pre-handoff item is represented in the summary.
fn should_forward_pre_item(input_item: &ResponseInputItem) -> bool {
    // Assistant messages from the previous agent are summarized.
    if input_item.get("role").and_then(serde_json::Value::as_str) == Some("assistant") {
        return false;
    }
    // Summary-only types are already captured in the summary text.
    if let Some(item_type) = input_item.get("type").and_then(serde_json::Value::as_str) {
        return !SUMMARY_ONLY_INPUT_TYPES.contains(&item_type);
    }
    true
}

/// Return `false` for tool or side-effect items that the summary already covers.
fn should_forward_new_item(input_item: &ResponseInputItem) -> bool {
    // Items with a role should always be forwarded.
    if let Some(role) = input_item.get("role").and_then(serde_json::Value::as_str) {
        if !role.is_empty() {
            return true;
        }
    }
    // Summary-only types are already captured in the summary text.
    if let Some(item_type) = input_item.get("type").and_then(serde_json::Value::as_str) {
        return !SUMMARY_ONLY_INPUT_TYPES.contains(&item_type);
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::items::{
        HandoffCallItem, HandoffOutputItem, MessageOutputItem, ReasoningItem, RunItem,
        ToolCallItem, ToolCallOutputItem,
    };

    use super::*;

    // Serialize tests that touch the global wrappers mutex to prevent races
    // under `cargo tarpaulin` which may run tests in a single process.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    // Ensure we reset global state between tests to avoid ordering issues.
    fn with_default_wrappers<F: FnOnce()>(f: F) {
        let _guard = TEST_LOCK.lock().expect("test lock poisoned");
        reset_conversation_history_wrappers();
        f();
        reset_conversation_history_wrappers();
    }

    // ---- get/set/reset conversation history wrappers ----

    #[test]
    fn default_wrappers() {
        with_default_wrappers(|| {
            let (start, end) = get_conversation_history_wrappers();
            assert_eq!(start, "<CONVERSATION HISTORY>");
            assert_eq!(end, "</CONVERSATION HISTORY>");
        });
    }

    #[test]
    fn set_wrappers_both() {
        with_default_wrappers(|| {
            set_conversation_history_wrappers(Some("[START]"), Some("[END]"));
            let (start, end) = get_conversation_history_wrappers();
            assert_eq!(start, "[START]");
            assert_eq!(end, "[END]");
        });
    }

    #[test]
    fn set_wrappers_partial() {
        with_default_wrappers(|| {
            set_conversation_history_wrappers(Some("[START]"), None);
            let (start, end) = get_conversation_history_wrappers();
            assert_eq!(start, "[START]");
            assert_eq!(end, "</CONVERSATION HISTORY>");
        });
    }

    #[test]
    fn reset_wrappers() {
        with_default_wrappers(|| {
            set_conversation_history_wrappers(Some("X"), Some("Y"));
            reset_conversation_history_wrappers();
            let (start, end) = get_conversation_history_wrappers();
            assert_eq!(start, "<CONVERSATION HISTORY>");
            assert_eq!(end, "</CONVERSATION HISTORY>");
        });
    }

    // ---- default_handoff_history_mapper ----

    #[test]
    fn default_mapper_empty_transcript() {
        with_default_wrappers(|| {
            let result = default_handoff_history_mapper(&[]);
            assert_eq!(result.len(), 1);
            let content = result[0]["content"].as_str().unwrap();
            assert!(content.contains("(no previous turns recorded)"));
            assert!(content.contains("<CONVERSATION HISTORY>"));
            assert!(content.contains("</CONVERSATION HISTORY>"));
        });
    }

    #[test]
    fn default_mapper_with_items() {
        with_default_wrappers(|| {
            let transcript = vec![
                json!({"role": "user", "content": "Hello"}),
                json!({"role": "assistant", "content": "Hi there!"}),
            ];
            let result = default_handoff_history_mapper(&transcript);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0]["role"], "assistant");
            let content = result[0]["content"].as_str().unwrap();
            assert!(content.contains("1. user: Hello"));
            assert!(content.contains("2. assistant: Hi there!"));
        });
    }

    // ---- format_transcript_item ----

    #[test]
    fn format_item_with_role_and_content() {
        let item = json!({"role": "user", "content": "Hello"});
        assert_eq!(format_transcript_item(&item), "user: Hello");
    }

    #[test]
    fn format_item_with_role_name_and_content() {
        let item = json!({"role": "assistant", "name": "helper", "content": "Hi"});
        assert_eq!(format_transcript_item(&item), "assistant (helper): Hi");
    }

    #[test]
    fn format_item_with_role_no_content() {
        let item = json!({"role": "user"});
        assert_eq!(format_transcript_item(&item), "user");
    }

    #[test]
    fn format_item_without_role() {
        let item = json!({"type": "function_call", "name": "get_weather", "arguments": "{}"});
        let result = format_transcript_item(&item);
        assert!(result.starts_with("function_call:"));
        assert!(result.contains("get_weather"));
    }

    // ---- stringify_content ----

    #[test]
    fn stringify_none() {
        assert_eq!(stringify_content(None), "");
    }

    #[test]
    fn stringify_null() {
        let v = serde_json::Value::Null;
        assert_eq!(stringify_content(Some(&v)), "");
    }

    #[test]
    fn stringify_string() {
        let v = json!("hello");
        assert_eq!(stringify_content(Some(&v)), "hello");
    }

    #[test]
    fn stringify_array() {
        let v = json!([{"type": "text", "text": "hi"}]);
        let result = stringify_content(Some(&v));
        assert!(result.contains("text"));
    }

    // ---- split_role_and_name ----

    #[test]
    fn split_simple_role() {
        let (role, name) = split_role_and_name("user");
        assert_eq!(role, "user");
        assert!(name.is_none());
    }

    #[test]
    fn split_role_with_name() {
        let (role, name) = split_role_and_name("assistant (helper)");
        assert_eq!(role, "assistant");
        assert_eq!(name.as_deref(), Some("helper"));
    }

    #[test]
    fn split_empty_parens() {
        let (role, name) = split_role_and_name("assistant ()");
        // Empty name inside parens => no split.
        assert_eq!(role, "assistant ()");
        assert!(name.is_none());
    }

    #[test]
    fn split_empty_role() {
        let (role, name) = split_role_and_name("");
        assert_eq!(role, "developer");
        assert!(name.is_none());
    }

    #[test]
    fn split_name_only_in_parens() {
        let (role, name) = split_role_and_name("(agent_x)");
        assert_eq!(role, "developer");
        assert_eq!(name.as_deref(), Some("agent_x"));
    }

    // ---- parse_summary_line ----

    #[test]
    fn parse_numbered_line() {
        let item = parse_summary_line("1. user: Hello world").unwrap();
        assert_eq!(item["role"], "user");
        assert_eq!(item["content"], "Hello world");
    }

    #[test]
    fn parse_unnumbered_line() {
        let item = parse_summary_line("assistant: Hi there").unwrap();
        assert_eq!(item["role"], "assistant");
        assert_eq!(item["content"], "Hi there");
    }

    #[test]
    fn parse_line_with_name() {
        let item = parse_summary_line("2. assistant (helper): content here").unwrap();
        assert_eq!(item["role"], "assistant");
        assert_eq!(item["name"], "helper");
        assert_eq!(item["content"], "content here");
    }

    #[test]
    fn parse_empty_line() {
        assert!(parse_summary_line("").is_none());
    }

    #[test]
    fn parse_line_no_colon() {
        assert!(parse_summary_line("just some text").is_none());
    }

    // ---- flatten_nested_history_messages ----

    #[test]
    fn flatten_non_nested_passes_through() {
        with_default_wrappers(|| {
            let items = vec![json!({"role": "user", "content": "Hello"})];
            let result = flatten_nested_history_messages(&items);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0]["role"], "user");
        });
    }

    #[test]
    fn flatten_extracts_nested_history() {
        with_default_wrappers(|| {
            let nested_content = format!(
                "For context:\n{}\n1. user: Hi\n2. assistant: Hello\n{}\n",
                "<CONVERSATION HISTORY>", "</CONVERSATION HISTORY>"
            );
            let items = vec![json!({"role": "assistant", "content": nested_content})];
            let result = flatten_nested_history_messages(&items);
            assert_eq!(result.len(), 2);
            assert_eq!(result[0]["role"], "user");
            assert_eq!(result[0]["content"], "Hi");
            assert_eq!(result[1]["role"], "assistant");
            assert_eq!(result[1]["content"], "Hello");
        });
    }

    // ---- should_forward_pre_item ----

    #[test]
    fn pre_item_filters_assistant() {
        let item = json!({"role": "assistant", "content": "Hello"});
        assert!(!should_forward_pre_item(&item));
    }

    #[test]
    fn pre_item_forwards_user() {
        let item = json!({"role": "user", "content": "Hello"});
        assert!(should_forward_pre_item(&item));
    }

    #[test]
    fn pre_item_filters_function_call() {
        let item = json!({"type": "function_call", "name": "foo"});
        assert!(!should_forward_pre_item(&item));
    }

    #[test]
    fn pre_item_filters_function_call_output() {
        let item = json!({"type": "function_call_output", "output": "bar"});
        assert!(!should_forward_pre_item(&item));
    }

    #[test]
    fn pre_item_filters_reasoning() {
        let item = json!({"type": "reasoning", "text": "thinking..."});
        assert!(!should_forward_pre_item(&item));
    }

    // ---- should_forward_new_item ----

    #[test]
    fn new_item_forwards_user() {
        let item = json!({"role": "user", "content": "Hi"});
        assert!(should_forward_new_item(&item));
    }

    #[test]
    fn new_item_forwards_assistant() {
        let item = json!({"role": "assistant", "content": "Hello"});
        assert!(should_forward_new_item(&item));
    }

    #[test]
    fn new_item_filters_function_call() {
        let item = json!({"type": "function_call", "name": "foo"});
        assert!(!should_forward_new_item(&item));
    }

    #[test]
    fn new_item_filters_reasoning() {
        let item = json!({"type": "reasoning"});
        assert!(!should_forward_new_item(&item));
    }

    // ---- nest_handoff_history integration ----

    #[test]
    fn nest_handoff_history_empty_data() {
        with_default_wrappers(|| {
            let data = HandoffInputData {
                input_history: InputContent::Items(vec![]),
                pre_handoff_items: vec![],
                new_items: vec![],
            };
            let result = nest_handoff_history(&data, None);
            if let InputContent::Items(items) = &result.input_history {
                assert_eq!(items.len(), 1);
                let content = items[0]["content"].as_str().unwrap();
                assert!(content.contains("(no previous turns recorded)"));
            } else {
                panic!("expected Items variant");
            }
        });
    }

    #[test]
    fn nest_handoff_history_with_conversation() {
        with_default_wrappers(|| {
            let data = HandoffInputData {
                input_history: InputContent::Items(vec![
                    json!({"role": "user", "content": "What is the weather?"}),
                    json!({"role": "assistant", "content": "Let me check."}),
                ]),
                pre_handoff_items: vec![RunItem::ToolCall(ToolCallItem {
                    agent_name: "agent_a".to_owned(),
                    raw_item: json!({"type": "function_call", "name": "get_weather", "arguments": "{}"}),
                })],
                new_items: vec![RunItem::HandoffCall(HandoffCallItem {
                    agent_name: "agent_a".to_owned(),
                    raw_item: json!({"type": "function_call", "name": "transfer_to_b"}),
                })],
            };
            let result = nest_handoff_history(&data, None);
            if let InputContent::Items(items) = &result.input_history {
                assert_eq!(items.len(), 1);
                let content = items[0]["content"].as_str().unwrap();
                assert!(content.contains("What is the weather?"));
                assert!(content.contains("Let me check."));
                assert!(content.contains("get_weather"));
            } else {
                panic!("expected Items variant");
            }
        });
    }

    #[test]
    fn nest_handoff_history_filters_pre_items() {
        with_default_wrappers(|| {
            let data = HandoffInputData {
                input_history: InputContent::Items(vec![]),
                pre_handoff_items: vec![
                    RunItem::MessageOutput(MessageOutputItem {
                        agent_name: "a".to_owned(),
                        raw_item: json!({"role": "assistant", "content": "Hi"}),
                    }),
                    RunItem::ToolCallOutput(ToolCallOutputItem {
                        agent_name: "a".to_owned(),
                        raw_item: json!({"type": "function_call_output", "output": "result"}),
                        output: json!("result"),
                    }),
                ],
                new_items: vec![],
            };
            let result = nest_handoff_history(&data, None);
            // Assistant messages and function_call_output are filtered from pre_handoff_items.
            assert!(result.pre_handoff_items.is_empty());
        });
    }

    #[test]
    fn nest_handoff_history_keeps_user_pre_items() {
        with_default_wrappers(|| {
            let data = HandoffInputData {
                input_history: InputContent::Items(vec![]),
                pre_handoff_items: vec![RunItem::HandoffOutput(HandoffOutputItem {
                    agent_name: "a".to_owned(),
                    raw_item: json!({"role": "user", "content": "handoff context"}),
                    source_agent_name: "a".to_owned(),
                    target_agent_name: "b".to_owned(),
                })],
                new_items: vec![],
            };
            let result = nest_handoff_history(&data, None);
            assert_eq!(result.pre_handoff_items.len(), 1);
        });
    }

    #[test]
    fn nest_handoff_history_filters_reasoning_new_items() {
        with_default_wrappers(|| {
            let data = HandoffInputData {
                input_history: InputContent::Items(vec![]),
                pre_handoff_items: vec![],
                new_items: vec![RunItem::Reasoning(ReasoningItem {
                    agent_name: "a".to_owned(),
                    raw_item: json!({"type": "reasoning", "text": "thinking"}),
                })],
            };
            let result = nest_handoff_history(&data, None);
            assert!(result.new_items.is_empty());
        });
    }

    #[test]
    fn nest_handoff_history_with_custom_mapper() {
        with_default_wrappers(|| {
            let mapper: HandoffHistoryMapper = Box::new(|transcript| {
                // Custom mapper that just returns a single item with the count.
                vec![json!({
                    "role": "developer",
                    "content": format!("Transcript had {} items.", transcript.len()),
                })]
            });
            let data = HandoffInputData {
                input_history: InputContent::Items(vec![
                    json!({"role": "user", "content": "a"}),
                    json!({"role": "assistant", "content": "b"}),
                ]),
                pre_handoff_items: vec![],
                new_items: vec![],
            };
            let result = nest_handoff_history(&data, Some(&mapper));
            if let InputContent::Items(items) = &result.input_history {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0]["content"], "Transcript had 2 items.");
            } else {
                panic!("expected Items variant");
            }
        });
    }

    #[test]
    fn nest_handoff_history_with_text_input() {
        with_default_wrappers(|| {
            let data = HandoffInputData {
                input_history: InputContent::Text("Hello from user".to_owned()),
                pre_handoff_items: vec![],
                new_items: vec![],
            };
            let result = nest_handoff_history(&data, None);
            if let InputContent::Items(items) = &result.input_history {
                assert_eq!(items.len(), 1);
                let content = items[0]["content"].as_str().unwrap();
                assert!(content.contains("Hello from user"));
            } else {
                panic!("expected Items variant");
            }
        });
    }

    // ---- HandoffHistoryMapper is Send + Sync ----

    #[test]
    fn history_mapper_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HandoffHistoryMapper>();
    }
}
