//! Prompt utilities for building system prompts and formatting tool/handoff information.
//!
//! This module provides helper functions for constructing the system prompt that is
//! sent to the LLM. It combines the agent's instructions with descriptions of
//! available tools and handoffs so the model understands what capabilities it has.
//!
//! The primary entry point is [`build_system_prompt`], which resolves an agent's
//! instructions and optionally appends tool and handoff descriptions. Lower-level
//! functions like [`format_tool_descriptions`], [`format_handoff_descriptions`], and
//! [`combine_prompt_parts`] are also public for custom prompt construction workflows.
//!
//! This module mirrors the prompt-building utilities found across the Python SDK's
//! `prompts.py` and `run_internal/` modules.

use std::fmt::Write;

use crate::handoffs::Handoff;
use crate::models::ToolSpec;

/// Build the system prompt for an agent by combining instructions with tool and
/// handoff descriptions.
///
/// When `instructions` is `None`, the function returns `None` only if there are
/// also no tools or handoffs to describe. Otherwise, it assembles all non-empty
/// parts separated by double newlines.
///
/// # Arguments
///
/// * `instructions` - The resolved instructions string for the agent (may be `None`).
/// * `tools` - Tool specifications visible to the model.
/// * `handoffs` - Handoffs available to the agent.
#[must_use]
pub fn build_system_prompt<C: Send + Sync + 'static>(
    instructions: Option<&str>,
    tools: &[ToolSpec],
    handoffs: &[Handoff<C>],
) -> Option<String> {
    let tool_desc = format_tool_descriptions(tools);
    let handoff_desc = format_handoff_descriptions(handoffs);
    combine_prompt_parts(instructions, &tool_desc, &handoff_desc)
}

/// Format tool descriptions into a human-readable string for inclusion in the
/// system prompt.
///
/// Returns an empty string when the slice is empty. Otherwise, produces a
/// Markdown section listing each tool with its name and description.
#[must_use]
pub fn format_tool_descriptions(tools: &[ToolSpec]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut result = String::from("# Tools\n\nYou have access to the following tools:\n");
    for tool in tools {
        let _ = write!(result, "\n## {}\n\n{}\n", tool.name, tool.description);
    }
    result
}

/// Format handoff descriptions into a human-readable string for inclusion in the
/// system prompt.
///
/// Returns an empty string when the slice is empty. Otherwise, produces a
/// Markdown section listing each handoff with the tool name to call and its
/// description.
#[must_use]
pub fn format_handoff_descriptions<C: Send + Sync + 'static>(handoffs: &[Handoff<C>]) -> String {
    if handoffs.is_empty() {
        return String::new();
    }
    let mut result = String::from("# Handoffs\n\nYou can hand off to the following agents:\n");
    for handoff in handoffs {
        let _ = write!(
            result,
            "\n## {}\n\nCall the `{}` tool to transfer.\n{}\n",
            handoff.agent_name, handoff.tool_name, handoff.tool_description,
        );
    }
    result
}

/// Combine all prompt parts into a final system prompt string.
///
/// Each non-empty part is joined with double newlines. Returns `None` when all
/// parts are empty.
///
/// # Arguments
///
/// * `instructions` - The agent's resolved instruction text.
/// * `tool_descriptions` - Formatted tool descriptions (from [`format_tool_descriptions`]).
/// * `handoff_descriptions` - Formatted handoff descriptions (from [`format_handoff_descriptions`]).
#[must_use]
pub fn combine_prompt_parts(
    instructions: Option<&str>,
    tool_descriptions: &str,
    handoff_descriptions: &str,
) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    if let Some(inst) = instructions {
        if !inst.is_empty() {
            parts.push(inst);
        }
    }
    if !tool_descriptions.is_empty() {
        parts.push(tool_descriptions);
    }
    if !handoff_descriptions.is_empty() {
        parts.push(handoff_descriptions);
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handoffs::Handoff;
    use crate::models::ToolSpec;

    // ---- Helper constructors ----

    fn make_tool(name: &str, description: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_owned(),
            description: description.to_owned(),
            params_json_schema: serde_json::json!({}),
            strict: true,
        }
    }

    fn make_handoff(agent_name: &str) -> Handoff<()> {
        Handoff::to_agent(agent_name).build()
    }

    // ---- format_tool_descriptions ----

    #[test]
    fn format_tool_descriptions_empty() {
        let result = format_tool_descriptions(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn format_tool_descriptions_single() {
        let tools = [make_tool("get_weather", "Fetch the current weather.")];
        let result = format_tool_descriptions(&tools);
        assert!(result.contains("# Tools"));
        assert!(result.contains("## get_weather"));
        assert!(result.contains("Fetch the current weather."));
    }

    #[test]
    fn format_tool_descriptions_multiple() {
        let tools = [
            make_tool("get_weather", "Fetch weather."),
            make_tool("search", "Search the web."),
        ];
        let result = format_tool_descriptions(&tools);
        assert!(result.contains("## get_weather"));
        assert!(result.contains("Fetch weather."));
        assert!(result.contains("## search"));
        assert!(result.contains("Search the web."));
    }

    // ---- format_handoff_descriptions ----

    #[test]
    fn format_handoff_descriptions_empty() {
        let result = format_handoff_descriptions::<()>(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn format_handoff_descriptions_single() {
        let handoffs = [make_handoff("billing")];
        let result = format_handoff_descriptions(&handoffs);
        assert!(result.contains("# Handoffs"));
        assert!(result.contains("## billing"));
        assert!(result.contains("transfer_to_billing"));
    }

    #[test]
    fn format_handoff_descriptions_multiple() {
        let handoffs = [make_handoff("billing"), make_handoff("support")];
        let result = format_handoff_descriptions(&handoffs);
        assert!(result.contains("## billing"));
        assert!(result.contains("transfer_to_billing"));
        assert!(result.contains("## support"));
        assert!(result.contains("transfer_to_support"));
    }

    #[test]
    fn format_handoff_descriptions_includes_tool_description() {
        let handoff: Handoff<()> = Handoff::to_agent("billing")
            .tool_description("Route to the billing department.")
            .build();
        let result = format_handoff_descriptions(&[handoff]);
        assert!(result.contains("Route to the billing department."));
    }

    // ---- combine_prompt_parts ----

    #[test]
    fn combine_all_none_or_empty() {
        assert!(combine_prompt_parts(None, "", "").is_none());
    }

    #[test]
    fn combine_instructions_only() {
        let result = combine_prompt_parts(Some("Be helpful."), "", "");
        assert_eq!(result.as_deref(), Some("Be helpful."));
    }

    #[test]
    fn combine_tools_only() {
        let result = combine_prompt_parts(None, "# Tools\n\nSome tools.", "");
        assert_eq!(result.as_deref(), Some("# Tools\n\nSome tools."));
    }

    #[test]
    fn combine_handoffs_only() {
        let result = combine_prompt_parts(None, "", "# Handoffs\n\nSome handoffs.");
        assert_eq!(result.as_deref(), Some("# Handoffs\n\nSome handoffs."));
    }

    #[test]
    fn combine_instructions_and_tools() {
        let result = combine_prompt_parts(Some("Be helpful."), "# Tools\n\ntools", "");
        let text = result.unwrap();
        assert!(text.starts_with("Be helpful."));
        assert!(text.contains("# Tools"));
        // Parts are separated by double newline.
        assert!(text.contains("\n\n# Tools"));
    }

    #[test]
    fn combine_all_three() {
        let result =
            combine_prompt_parts(Some("Instructions."), "Tool section.", "Handoff section.");
        let text = result.unwrap();
        assert!(text.starts_with("Instructions."));
        assert!(text.contains("\n\nTool section."));
        assert!(text.contains("\n\nHandoff section."));
    }

    #[test]
    fn combine_empty_instructions_string_is_skipped() {
        let result = combine_prompt_parts(Some(""), "Tool section.", "");
        let text = result.unwrap();
        assert_eq!(text, "Tool section.");
    }

    // ---- build_system_prompt ----

    #[test]
    fn build_system_prompt_no_instructions_no_tools_no_handoffs() {
        let result = build_system_prompt::<()>(None, &[], &[]);
        assert!(result.is_none());
    }

    #[test]
    fn build_system_prompt_instructions_only() {
        let result = build_system_prompt::<()>(Some("Be helpful."), &[], &[]);
        assert_eq!(result.as_deref(), Some("Be helpful."));
    }

    #[test]
    fn build_system_prompt_with_tools() {
        let tools = [make_tool("search", "Search the web.")];
        let result = build_system_prompt::<()>(Some("Be helpful."), &tools, &[]);
        let text = result.unwrap();
        assert!(text.contains("Be helpful."));
        assert!(text.contains("# Tools"));
        assert!(text.contains("## search"));
    }

    #[test]
    fn build_system_prompt_with_handoffs() {
        let handoffs = [make_handoff("billing")];
        let result = build_system_prompt(Some("Be helpful."), &[], &handoffs);
        let text = result.unwrap();
        assert!(text.contains("Be helpful."));
        assert!(text.contains("# Handoffs"));
        assert!(text.contains("## billing"));
    }

    #[test]
    fn build_system_prompt_with_tools_and_handoffs() {
        let tools = [make_tool("search", "Search.")];
        let handoffs = [make_handoff("support")];
        let result = build_system_prompt(Some("Help."), &tools, &handoffs);
        let text = result.unwrap();
        assert!(text.contains("Help."));
        assert!(text.contains("# Tools"));
        assert!(text.contains("# Handoffs"));
    }

    #[test]
    fn build_system_prompt_tools_only_no_instructions() {
        let tools = [make_tool("calc", "Calculate.")];
        let result = build_system_prompt::<()>(None, &tools, &[]);
        let text = result.unwrap();
        assert!(text.contains("# Tools"));
        assert!(text.contains("## calc"));
    }
}
