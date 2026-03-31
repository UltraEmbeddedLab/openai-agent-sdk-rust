//! `CodexTool` -- a tool wrapper that delegates work to a Codex thread.
//!
//! [`CodexTool`] allows a standard agent to delegate complex tasks to a Codex
//! thread, combining the simplicity of the agent framework with the power of
//! thread-based execution.
//!
//! Mirrors the Python SDK's `codex_tool.py` in the Codex extension.

use serde::{Deserialize, Serialize};

use super::events::CodexUsage;
use super::options::CodexOptions;

// ---------------------------------------------------------------------------
// CodexToolResult
// ---------------------------------------------------------------------------

/// Result from a Codex tool invocation.
///
/// Captures the thread ID (for resuming), the response text, and token usage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CodexToolResult {
    /// The thread ID, if assigned during execution.
    pub thread_id: Option<String>,
    /// The final response text from the Codex thread.
    pub response: String,
    /// Token usage for the turn, if available.
    pub usage: Option<CodexUsage>,
}

impl CodexToolResult {
    /// Create a new `CodexToolResult`.
    #[must_use]
    pub fn new(
        thread_id: Option<String>,
        response: impl Into<String>,
        usage: Option<CodexUsage>,
    ) -> Self {
        Self {
            thread_id,
            response: response.into(),
            usage,
        }
    }
}

impl std::fmt::Display for CodexToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match serde_json::to_string(self) {
            Ok(json) => f.write_str(&json),
            Err(_) => write!(f, "CodexToolResult {{ response: {} }}", self.response),
        }
    }
}

// ---------------------------------------------------------------------------
// CodexToolOptions
// ---------------------------------------------------------------------------

/// Options for configuring a [`CodexTool`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CodexToolOptions {
    /// The name of the tool as exposed to the LLM.
    pub name: Option<String>,
    /// Description of the tool.
    pub description: Option<String>,
    /// Codex-level options (API key, base URL, etc.).
    pub codex_options: Option<CodexOptions>,
}

impl CodexToolOptions {
    /// Create a new `CodexToolOptions` with all fields set to `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the tool name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the tool description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the Codex options.
    #[must_use]
    pub fn with_codex_options(mut self, options: CodexOptions) -> Self {
        self.codex_options = Some(options);
        self
    }
}

// ---------------------------------------------------------------------------
// CodexTool
// ---------------------------------------------------------------------------

/// Default name for a Codex tool.
pub const DEFAULT_CODEX_TOOL_NAME: &str = "codex";

/// A tool that wraps Codex execution as an agent tool.
///
/// This allows a standard agent to delegate complex tasks to a Codex
/// thread, combining the simplicity of the agent framework with the
/// power of thread-based execution.
///
/// # Example
///
/// ```
/// use openai_agents::extensions::codex::CodexTool;
///
/// let tool = CodexTool::new("codex_worker", "Delegate complex coding tasks to Codex.");
/// assert_eq!(tool.name(), "codex_worker");
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CodexTool {
    name: String,
    description: String,
    codex_options: CodexOptions,
}

impl CodexTool {
    /// Create a new `CodexTool` with the given name and description.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            codex_options: CodexOptions::default(),
        }
    }

    /// Create a new `CodexTool` from [`CodexToolOptions`].
    #[must_use]
    pub fn from_options(options: CodexToolOptions) -> Self {
        Self {
            name: options
                .name
                .unwrap_or_else(|| DEFAULT_CODEX_TOOL_NAME.to_owned()),
            description: options.description.unwrap_or_else(|| {
                "Delegate complex tasks to Codex for thread-based execution.".to_owned()
            }),
            codex_options: options.codex_options.unwrap_or_default(),
        }
    }

    /// Set the Codex options for this tool.
    #[must_use]
    pub fn with_codex_options(mut self, options: CodexOptions) -> Self {
        self.codex_options = options;
        self
    }

    /// Get the tool name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the tool description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get a reference to the Codex options.
    #[must_use]
    pub const fn codex_options(&self) -> &CodexOptions {
        &self.codex_options
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_tool_new() {
        let tool = CodexTool::new("my_codex", "A Codex tool.");
        assert_eq!(tool.name(), "my_codex");
        assert_eq!(tool.description(), "A Codex tool.");
    }

    #[test]
    fn codex_tool_from_options_defaults() {
        let tool = CodexTool::from_options(CodexToolOptions::default());
        assert_eq!(tool.name(), DEFAULT_CODEX_TOOL_NAME);
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn codex_tool_from_options_custom() {
        let opts = CodexToolOptions::new()
            .with_name("custom_codex")
            .with_description("Custom description.")
            .with_codex_options(CodexOptions::new().with_api_key("sk-test"));
        let tool = CodexTool::from_options(opts);
        assert_eq!(tool.name(), "custom_codex");
        assert_eq!(tool.description(), "Custom description.");
        assert_eq!(tool.codex_options().api_key.as_deref(), Some("sk-test"));
    }

    #[test]
    fn codex_tool_with_codex_options() {
        let tool = CodexTool::new("test", "test tool")
            .with_codex_options(CodexOptions::new().with_base_url("https://api.example.com"));
        assert_eq!(
            tool.codex_options().base_url.as_deref(),
            Some("https://api.example.com")
        );
    }

    // ---- CodexToolResult ----

    #[test]
    fn codex_tool_result_new() {
        let result = CodexToolResult::new(Some("t-1".to_owned()), "done", None);
        assert_eq!(result.thread_id.as_deref(), Some("t-1"));
        assert_eq!(result.response, "done");
        assert!(result.usage.is_none());
    }

    #[test]
    fn codex_tool_result_display() {
        let result = CodexToolResult::new(None, "hello", None);
        let display = result.to_string();
        assert!(display.contains("hello"));
    }

    #[test]
    fn codex_tool_result_serde_round_trip() {
        let result = CodexToolResult::new(
            Some("t-1".to_owned()),
            "response text",
            Some(CodexUsage::new(10, 5, 20)),
        );
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: CodexToolResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);
    }

    // ---- CodexToolOptions ----

    #[test]
    fn codex_tool_options_default() {
        let opts = CodexToolOptions::default();
        assert!(opts.name.is_none());
        assert!(opts.description.is_none());
        assert!(opts.codex_options.is_none());
    }

    #[test]
    fn codex_tool_options_builder() {
        let opts = CodexToolOptions::new()
            .with_name("tool")
            .with_description("desc");
        assert_eq!(opts.name.as_deref(), Some("tool"));
        assert_eq!(opts.description.as_deref(), Some("desc"));
    }

    // ---- Send + Sync ----

    #[test]
    fn codex_tool_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CodexTool>();
        assert_send_sync::<CodexToolResult>();
        assert_send_sync::<CodexToolOptions>();
    }

    // ---- Debug ----

    #[test]
    fn codex_tool_debug() {
        let tool = CodexTool::new("test", "A test tool.");
        let debug = format!("{tool:?}");
        assert!(debug.contains("CodexTool"));
        assert!(debug.contains("test"));
    }
}
