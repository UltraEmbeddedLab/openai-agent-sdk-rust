//! Per-agent MCP configuration.
//!
//! [`MCPConfig`] controls how MCP tool errors are surfaced to the model and
//! how long tool calls are allowed to run before timing out.

/// Configuration for MCP integration within an agent.
///
/// Controls error handling and timeout behaviour for MCP tool calls made on
/// behalf of the agent. Use the builder-style methods to customise the
/// configuration.
///
/// # Example
///
/// ```
/// use openai_agents::mcp::MCPConfig;
///
/// let cfg = MCPConfig::new()
///     .with_convert_errors(true)
///     .with_timeout(30.0);
///
/// assert!(cfg.convert_errors_to_messages);
/// assert_eq!(cfg.tool_timeout_seconds, Some(30.0));
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct MCPConfig {
    /// Whether to convert MCP tool errors into model-visible error messages
    /// instead of propagating them as [`AgentError`](crate::error::AgentError).
    ///
    /// When `true`, failures from MCP tool invocations are returned to the
    /// model as a textual error so it can recover or retry. When `false`,
    /// the error propagates to the caller.
    pub convert_errors_to_messages: bool,

    /// Optional timeout for MCP tool calls, in seconds.
    ///
    /// If a tool call exceeds this duration, it is cancelled and an error is
    /// returned (or converted to a message, depending on
    /// [`convert_errors_to_messages`](Self::convert_errors_to_messages)).
    pub tool_timeout_seconds: Option<f64>,
}

impl MCPConfig {
    /// Create a new MCP configuration with default values.
    ///
    /// Defaults: errors are **not** converted to messages and no timeout is
    /// set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to convert tool errors into model-visible messages.
    #[must_use]
    pub const fn with_convert_errors(mut self, convert: bool) -> Self {
        self.convert_errors_to_messages = convert;
        self
    }

    /// Set the tool call timeout in seconds.
    #[must_use]
    pub const fn with_timeout(mut self, seconds: f64) -> Self {
        self.tool_timeout_seconds = Some(seconds);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = MCPConfig::default();
        assert!(!cfg.convert_errors_to_messages);
        assert!(cfg.tool_timeout_seconds.is_none());
    }

    #[test]
    fn new_returns_defaults() {
        let cfg = MCPConfig::new();
        assert!(!cfg.convert_errors_to_messages);
        assert!(cfg.tool_timeout_seconds.is_none());
    }

    #[test]
    fn builder_convert_errors() {
        let cfg = MCPConfig::new().with_convert_errors(true);
        assert!(cfg.convert_errors_to_messages);
    }

    #[test]
    fn builder_timeout() {
        let cfg = MCPConfig::new().with_timeout(15.5);
        assert_eq!(cfg.tool_timeout_seconds, Some(15.5));
    }

    #[test]
    fn builder_chaining() {
        let cfg = MCPConfig::new()
            .with_convert_errors(true)
            .with_timeout(60.0);
        assert!(cfg.convert_errors_to_messages);
        assert_eq!(cfg.tool_timeout_seconds, Some(60.0));
    }

    #[test]
    fn debug_impl() {
        let cfg = MCPConfig::new().with_timeout(10.0);
        let debug = format!("{cfg:?}");
        assert!(debug.contains("MCPConfig"));
        assert!(debug.contains("10.0"));
    }

    #[test]
    fn clone_impl() {
        let cfg = MCPConfig::new().with_convert_errors(true).with_timeout(5.0);
        let cloned = cfg.clone();
        assert_eq!(
            cloned.convert_errors_to_messages,
            cfg.convert_errors_to_messages
        );
        assert_eq!(cloned.tool_timeout_seconds, cfg.tool_timeout_seconds);
    }

    #[test]
    fn send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MCPConfig>();
    }
}
