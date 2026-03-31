//! MCP utility types for tool filtering and metadata resolution.
//!
//! This module provides types that control which tools from an MCP server are
//! exposed to an agent and how tool call metadata is resolved.

use std::collections::HashSet;

/// Context information available to tool filter functions.
///
/// Passed to dynamic [`ToolFilter`] callbacks so they can make per-invocation
/// decisions about which MCP tools to expose.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolFilterContext {
    /// The name of the MCP server being filtered.
    pub server_name: String,
    /// The name of the agent requesting the tool list.
    pub agent_name: String,
}

impl ToolFilterContext {
    /// Create a new tool filter context.
    #[must_use]
    pub fn new(server_name: impl Into<String>, agent_name: impl Into<String>) -> Self {
        Self {
            server_name: server_name.into(),
            agent_name: agent_name.into(),
        }
    }
}

/// Static tool filter configuration using allowlists and blocklists.
///
/// When both `allowed_tool_names` and `blocked_tool_names` are set, the
/// allowlist is applied first, then the blocklist removes tools from the
/// result.
///
/// # Example
///
/// ```
/// use openai_agents::mcp::ToolFilterStatic;
///
/// let filter = ToolFilterStatic::allow(vec!["search".to_owned(), "fetch".to_owned()]);
/// assert!(filter.is_allowed("search"));
/// assert!(!filter.is_allowed("delete"));
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ToolFilterStatic {
    /// Optional set of tool names to allow (allowlist).
    ///
    /// When `Some`, only tools whose names appear in this set are available.
    /// When `None`, all tools are allowed (subject to the blocklist).
    pub allowed_tool_names: Option<HashSet<String>>,

    /// Optional set of tool names to exclude (blocklist).
    ///
    /// Tools whose names appear in this set are filtered out, even if they
    /// also appear in the allowlist.
    pub blocked_tool_names: Option<HashSet<String>>,
}

impl ToolFilterStatic {
    /// Create a filter that only allows the specified tool names.
    #[must_use]
    pub fn allow(names: Vec<String>) -> Self {
        Self {
            allowed_tool_names: Some(names.into_iter().collect()),
            blocked_tool_names: None,
        }
    }

    /// Create a filter that blocks the specified tool names.
    #[must_use]
    pub fn block(names: Vec<String>) -> Self {
        Self {
            allowed_tool_names: None,
            blocked_tool_names: Some(names.into_iter().collect()),
        }
    }

    /// Check whether a tool name passes this filter.
    #[must_use]
    pub fn is_allowed(&self, tool_name: &str) -> bool {
        if let Some(allowed) = &self.allowed_tool_names {
            if !allowed.contains(tool_name) {
                return false;
            }
        }
        if let Some(blocked) = &self.blocked_tool_names {
            if blocked.contains(tool_name) {
                return false;
            }
        }
        true
    }
}

/// A tool filter that controls which MCP tools are exposed to an agent.
///
/// Tool filters can be either a static configuration (allowlist/blocklist)
/// or absent (no filtering). Dynamic callback-based filters can be added
/// in the future behind the `mcp` feature flag.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub enum ToolFilter {
    /// No filtering; all tools are exposed.
    #[default]
    None,
    /// Static allowlist/blocklist filter.
    Static(ToolFilterStatic),
}

impl ToolFilter {
    /// Create a static tool filter from allowlist and blocklist parameters.
    ///
    /// Returns [`ToolFilter::None`] if both lists are `None`.
    #[must_use]
    pub fn from_lists(allowed: Option<Vec<String>>, blocked: Option<Vec<String>>) -> Self {
        if allowed.is_none() && blocked.is_none() {
            return Self::None;
        }
        Self::Static(ToolFilterStatic {
            allowed_tool_names: allowed.map(|v| v.into_iter().collect()),
            blocked_tool_names: blocked.map(|v| v.into_iter().collect()),
        })
    }

    /// Check whether a tool name passes this filter.
    #[must_use]
    pub fn is_allowed(&self, tool_name: &str) -> bool {
        match self {
            Self::None => true,
            Self::Static(f) => f.is_allowed(tool_name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ToolFilterStatic ----

    #[test]
    fn static_filter_allow() {
        let filter = ToolFilterStatic::allow(vec!["a".to_owned(), "b".to_owned()]);
        assert!(filter.is_allowed("a"));
        assert!(filter.is_allowed("b"));
        assert!(!filter.is_allowed("c"));
    }

    #[test]
    fn static_filter_block() {
        let filter = ToolFilterStatic::block(vec!["bad".to_owned()]);
        assert!(filter.is_allowed("good"));
        assert!(!filter.is_allowed("bad"));
    }

    #[test]
    fn static_filter_allow_and_block() {
        let filter = ToolFilterStatic {
            allowed_tool_names: Some(["a", "b", "c"].iter().map(|s| (*s).to_owned()).collect()),
            blocked_tool_names: Some(std::iter::once("b".to_owned()).collect()),
        };
        assert!(filter.is_allowed("a"));
        assert!(!filter.is_allowed("b")); // blocked overrides allowed
        assert!(filter.is_allowed("c"));
        assert!(!filter.is_allowed("d")); // not in allowlist
    }

    #[test]
    fn static_filter_default_allows_all() {
        let filter = ToolFilterStatic::default();
        assert!(filter.is_allowed("anything"));
    }

    // ---- ToolFilter enum ----

    #[test]
    fn tool_filter_none_allows_all() {
        let filter = ToolFilter::None;
        assert!(filter.is_allowed("anything"));
    }

    #[test]
    fn tool_filter_static_delegates() {
        let filter = ToolFilter::Static(ToolFilterStatic::allow(vec!["x".to_owned()]));
        assert!(filter.is_allowed("x"));
        assert!(!filter.is_allowed("y"));
    }

    #[test]
    fn tool_filter_from_lists_none() {
        let filter = ToolFilter::from_lists(None, None);
        assert!(matches!(filter, ToolFilter::None));
    }

    #[test]
    fn tool_filter_from_lists_allowed() {
        let filter = ToolFilter::from_lists(Some(vec!["a".to_owned()]), None);
        assert!(filter.is_allowed("a"));
        assert!(!filter.is_allowed("b"));
    }

    #[test]
    fn tool_filter_from_lists_blocked() {
        let filter = ToolFilter::from_lists(None, Some(vec!["bad".to_owned()]));
        assert!(filter.is_allowed("good"));
        assert!(!filter.is_allowed("bad"));
    }

    #[test]
    fn tool_filter_default() {
        let filter = ToolFilter::default();
        assert!(matches!(filter, ToolFilter::None));
    }

    // ---- ToolFilterContext ----

    #[test]
    fn filter_context_creation() {
        let ctx = ToolFilterContext::new("server-1", "agent-a");
        assert_eq!(ctx.server_name, "server-1");
        assert_eq!(ctx.agent_name, "agent-a");
    }

    #[test]
    fn filter_context_debug() {
        let ctx = ToolFilterContext::new("srv", "agt");
        let debug = format!("{ctx:?}");
        assert!(debug.contains("srv"));
        assert!(debug.contains("agt"));
    }

    #[test]
    fn filter_context_clone() {
        let ctx = ToolFilterContext::new("srv", "agt");
        let cloned = ctx.clone();
        assert_eq!(cloned.server_name, ctx.server_name);
        assert_eq!(cloned.agent_name, ctx.agent_name);
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ToolFilter>();
        assert_send_sync::<ToolFilterStatic>();
        assert_send_sync::<ToolFilterContext>();
    }
}
