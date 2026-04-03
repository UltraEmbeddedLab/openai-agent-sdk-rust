//! Per-agent MCP configuration.
//!
//! [`MCPConfig`] controls how MCP tool errors are surfaced to the model and
//! how long tool calls are allowed to run before timing out.
//!
//! Approval policies control whether MCP tool invocations require user
//! approval before execution. See [`ApprovalPolicy`] and
//! [`ApprovalPolicySetting`].

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// A simple approval policy for an MCP tool.
///
/// Maps to the Python SDK's `RequireApprovalPolicy = Literal["always", "never"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ApprovalPolicy {
    /// Always require approval before invoking the tool.
    Always,
    /// Never require approval.
    Never,
}

/// A callable approval policy that receives the tool name and returns
/// whether approval is required.
///
/// This mirrors the Python SDK's `LocalMCPApprovalCallable`. The callback
/// receives the tool name and should return `true` if the tool requires
/// approval.
pub type ApprovalCallable =
    Arc<dyn Fn(&str) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync>;

/// The approval policy setting for MCP tool invocations.
///
/// This mirrors the Python SDK's `RequireApprovalSetting`, which can be a
/// simple policy, a per-tool mapping, or a callable.
#[non_exhaustive]
pub enum ApprovalPolicySetting {
    /// A single policy applied to all tools.
    Policy(ApprovalPolicy),
    /// A per-tool mapping. Tools not in the map default to [`ApprovalPolicy::Never`].
    PerTool(HashMap<String, ApprovalPolicy>),
    /// A callable that determines the approval policy per tool at runtime.
    Callable(ApprovalCallable),
}

impl Clone for ApprovalPolicySetting {
    fn clone(&self) -> Self {
        match self {
            Self::Policy(p) => Self::Policy(*p),
            Self::PerTool(m) => Self::PerTool(m.clone()),
            Self::Callable(f) => Self::Callable(Arc::clone(f)),
        }
    }
}

impl std::fmt::Debug for ApprovalPolicySetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Policy(p) => f.debug_tuple("Policy").field(p).finish(),
            Self::PerTool(m) => f.debug_tuple("PerTool").field(m).finish(),
            Self::Callable(_) => f.debug_tuple("Callable").field(&"<fn>").finish(),
        }
    }
}

impl ApprovalPolicySetting {
    /// Check whether a specific tool requires approval.
    ///
    /// For [`Policy`](Self::Policy), returns the global setting.
    /// For [`PerTool`](Self::PerTool), looks up the tool name, defaulting to
    /// `Never`. For [`Callable`](Self::Callable), invokes the callback.
    pub async fn needs_approval(&self, tool_name: &str) -> bool {
        match self {
            Self::Policy(ApprovalPolicy::Always) => true,
            Self::Policy(ApprovalPolicy::Never) => false,
            Self::PerTool(map) => {
                matches!(map.get(tool_name), Some(ApprovalPolicy::Always))
            }
            Self::Callable(f) => f(tool_name).await,
        }
    }
}

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

    /// Approval policy for MCP tool invocations.
    ///
    /// When set, tools matching the policy will require user approval before
    /// execution. Defaults to `None` (no approval required).
    pub approval_policy: Option<ApprovalPolicySetting>,
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

    /// Set the approval policy for MCP tool invocations.
    #[must_use]
    pub fn with_approval_policy(mut self, policy: ApprovalPolicySetting) -> Self {
        self.approval_policy = Some(policy);
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

    // ---- Approval policies ----

    #[test]
    fn default_no_approval_policy() {
        let cfg = MCPConfig::default();
        assert!(cfg.approval_policy.is_none());
    }

    #[tokio::test]
    async fn approval_policy_always() {
        let setting = ApprovalPolicySetting::Policy(ApprovalPolicy::Always);
        assert!(setting.needs_approval("any_tool").await);
    }

    #[tokio::test]
    async fn approval_policy_never() {
        let setting = ApprovalPolicySetting::Policy(ApprovalPolicy::Never);
        assert!(!setting.needs_approval("any_tool").await);
    }

    #[tokio::test]
    async fn approval_policy_per_tool() {
        let mut map = std::collections::HashMap::new();
        map.insert("dangerous".to_owned(), ApprovalPolicy::Always);
        map.insert("safe".to_owned(), ApprovalPolicy::Never);
        let setting = ApprovalPolicySetting::PerTool(map);

        assert!(setting.needs_approval("dangerous").await);
        assert!(!setting.needs_approval("safe").await);
        assert!(!setting.needs_approval("unknown").await);
    }

    #[tokio::test]
    async fn approval_policy_callable() {
        let setting = ApprovalPolicySetting::Callable(Arc::new(|name: &str| {
            let requires = name.starts_with("admin_");
            Box::pin(async move { requires })
        }));

        assert!(setting.needs_approval("admin_delete").await);
        assert!(!setting.needs_approval("get_weather").await);
    }

    #[test]
    fn approval_policy_debug() {
        let setting = ApprovalPolicySetting::Policy(ApprovalPolicy::Always);
        let debug = format!("{setting:?}");
        assert!(debug.contains("Always"));

        let setting =
            ApprovalPolicySetting::Callable(Arc::new(|_: &str| Box::pin(async { false })));
        let debug = format!("{setting:?}");
        assert!(debug.contains("Callable"));
    }

    #[test]
    fn with_approval_policy_builder() {
        let cfg = MCPConfig::new()
            .with_approval_policy(ApprovalPolicySetting::Policy(ApprovalPolicy::Always));
        assert!(cfg.approval_policy.is_some());
    }
}
