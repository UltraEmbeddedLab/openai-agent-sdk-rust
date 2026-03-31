//! Configuration types for the Codex extension.
//!
//! This module provides [`CodexOptions`], [`ThreadOptions`], and [`TurnOptions`]
//! which mirror the Python SDK's `codex_options.py`, `thread_options.py`, and
//! `turn_options.py` modules respectively.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ApprovalMode
// ---------------------------------------------------------------------------

/// Approval policy for tool invocations within Codex.
///
/// Maps to the Python SDK's `ApprovalMode` literal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ApprovalMode {
    /// Never require approval.
    #[default]
    #[serde(rename = "never")]
    Never,
    /// Require approval on request.
    #[serde(rename = "on-request")]
    OnRequest,
    /// Require approval on failure.
    #[serde(rename = "on-failure")]
    OnFailure,
    /// Require approval for untrusted operations.
    #[serde(rename = "untrusted")]
    Untrusted,
}

// ---------------------------------------------------------------------------
// SandboxMode
// ---------------------------------------------------------------------------

/// Sandbox permissions for filesystem and network access.
///
/// Maps to the Python SDK's `SandboxMode` literal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum SandboxMode {
    /// Read-only sandbox (default).
    #[default]
    #[serde(rename = "read-only")]
    ReadOnly,
    /// Allow writes within the workspace directory.
    #[serde(rename = "workspace-write")]
    WorkspaceWrite,
    /// Full filesystem access (dangerous).
    #[serde(rename = "danger-full-access")]
    DangerFullAccess,
}

// ---------------------------------------------------------------------------
// ModelReasoningEffort
// ---------------------------------------------------------------------------

/// Model reasoning effort level.
///
/// Maps to the Python SDK's `ModelReasoningEffort` literal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ModelReasoningEffort {
    /// Minimal reasoning effort.
    #[serde(rename = "minimal")]
    Minimal,
    /// Low reasoning effort.
    #[serde(rename = "low")]
    Low,
    /// Medium reasoning effort (default).
    #[default]
    #[serde(rename = "medium")]
    Medium,
    /// High reasoning effort.
    #[serde(rename = "high")]
    High,
    /// Extra-high reasoning effort.
    #[serde(rename = "xhigh")]
    XHigh,
}

// ---------------------------------------------------------------------------
// WebSearchMode
// ---------------------------------------------------------------------------

/// Web search mode for the Codex CLI.
///
/// Maps to the Python SDK's `WebSearchMode` literal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum WebSearchMode {
    /// Web search disabled (default).
    #[default]
    #[serde(rename = "disabled")]
    Disabled,
    /// Use cached web search results.
    #[serde(rename = "cached")]
    Cached,
    /// Use live web search.
    #[serde(rename = "live")]
    Live,
}

// ---------------------------------------------------------------------------
// CodexOptions
// ---------------------------------------------------------------------------

/// Options for configuring a [`Codex`](super::Codex) instance.
///
/// Mirrors the Python SDK's `CodexOptions` dataclass. These options control
/// the Codex CLI process environment, API credentials, and subprocess limits.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CodexOptions {
    /// Optional absolute path to the Codex CLI binary.
    pub codex_path_override: Option<String>,
    /// Override `OpenAI` base URL for the Codex CLI process.
    pub base_url: Option<String>,
    /// API key passed to the Codex CLI (`CODEX_API_KEY`).
    pub api_key: Option<String>,
    /// Environment variables for the Codex CLI process.
    pub env: Option<std::collections::HashMap<String, String>>,
    /// `StreamReader` byte limit used for Codex subprocess stdout/stderr pipes.
    pub codex_subprocess_stream_limit_bytes: Option<usize>,
}

impl CodexOptions {
    /// Create a new `CodexOptions` with all fields set to `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the path override for the Codex CLI binary.
    #[must_use]
    pub fn with_codex_path_override(mut self, path: impl Into<String>) -> Self {
        self.codex_path_override = Some(path.into());
        self
    }

    /// Set the base URL for the `OpenAI` API.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the API key.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the environment variables for the Codex CLI process.
    #[must_use]
    pub fn with_env(mut self, env: std::collections::HashMap<String, String>) -> Self {
        self.env = Some(env);
        self
    }

    /// Set the subprocess stream limit in bytes.
    #[must_use]
    pub const fn with_subprocess_stream_limit_bytes(mut self, limit: usize) -> Self {
        self.codex_subprocess_stream_limit_bytes = Some(limit);
        self
    }
}

// ---------------------------------------------------------------------------
// ThreadOptions
// ---------------------------------------------------------------------------

/// Options for configuring a Codex thread.
///
/// Mirrors the Python SDK's `ThreadOptions` dataclass. These options control
/// model selection, sandbox permissions, working directory, and other
/// per-thread settings passed to the Codex CLI.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ThreadOptions {
    /// Model identifier passed to the Codex CLI (`--model`).
    pub model: Option<String>,
    /// Sandbox permissions for filesystem/network access.
    pub sandbox_mode: Option<SandboxMode>,
    /// Working directory for the Codex CLI process.
    pub working_directory: Option<String>,
    /// Allow running outside a Git repository.
    pub skip_git_repo_check: Option<bool>,
    /// Configure model reasoning effort.
    pub model_reasoning_effort: Option<ModelReasoningEffort>,
    /// Toggle network access in sandboxed workspace writes.
    pub network_access_enabled: Option<bool>,
    /// Configure web search mode via codex config.
    pub web_search_mode: Option<WebSearchMode>,
    /// Legacy toggle for web search behavior.
    pub web_search_enabled: Option<bool>,
    /// Approval policy for tool invocations within Codex.
    pub approval_policy: Option<ApprovalMode>,
    /// Additional filesystem roots available to Codex.
    pub additional_directories: Option<Vec<String>>,
}

impl ThreadOptions {
    /// Create a new `ThreadOptions` with all fields set to `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model identifier.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the sandbox mode.
    #[must_use]
    pub const fn with_sandbox_mode(mut self, mode: SandboxMode) -> Self {
        self.sandbox_mode = Some(mode);
        self
    }

    /// Set the working directory.
    #[must_use]
    pub fn with_working_directory(mut self, dir: impl Into<String>) -> Self {
        self.working_directory = Some(dir.into());
        self
    }

    /// Set whether to skip the Git repository check.
    #[must_use]
    pub const fn with_skip_git_repo_check(mut self, skip: bool) -> Self {
        self.skip_git_repo_check = Some(skip);
        self
    }

    /// Set the model reasoning effort.
    #[must_use]
    pub const fn with_model_reasoning_effort(mut self, effort: ModelReasoningEffort) -> Self {
        self.model_reasoning_effort = Some(effort);
        self
    }

    /// Set the approval policy.
    #[must_use]
    pub const fn with_approval_policy(mut self, policy: ApprovalMode) -> Self {
        self.approval_policy = Some(policy);
        self
    }

    /// Set the web search mode.
    #[must_use]
    pub const fn with_web_search_mode(mut self, mode: WebSearchMode) -> Self {
        self.web_search_mode = Some(mode);
        self
    }

    /// Set additional directories available to Codex.
    #[must_use]
    pub fn with_additional_directories(mut self, dirs: Vec<String>) -> Self {
        self.additional_directories = Some(dirs);
        self
    }
}

// ---------------------------------------------------------------------------
// TurnOptions
// ---------------------------------------------------------------------------

/// Options for a single turn within a Codex thread.
///
/// Mirrors the Python SDK's `TurnOptions` dataclass.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct TurnOptions {
    /// JSON schema used by Codex for structured output.
    pub output_schema: Option<serde_json::Value>,
    /// Abort the Codex CLI if no events arrive within this many seconds.
    pub idle_timeout_seconds: Option<f64>,
}

impl TurnOptions {
    /// Create a new `TurnOptions` with all fields set to `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the output schema for structured output.
    #[must_use]
    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Set the idle timeout in seconds.
    #[must_use]
    pub const fn with_idle_timeout_seconds(mut self, timeout: f64) -> Self {
        self.idle_timeout_seconds = Some(timeout);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ApprovalMode ----

    #[test]
    fn approval_mode_default_is_never() {
        assert_eq!(ApprovalMode::default(), ApprovalMode::Never);
    }

    #[test]
    fn approval_mode_serde_round_trip() {
        let modes = [
            ApprovalMode::Never,
            ApprovalMode::OnRequest,
            ApprovalMode::OnFailure,
            ApprovalMode::Untrusted,
        ];
        for mode in &modes {
            let json = serde_json::to_string(mode).expect("serialize");
            let deserialized: ApprovalMode = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*mode, deserialized);
        }
    }

    #[test]
    fn approval_mode_serde_values() {
        assert_eq!(
            serde_json::to_string(&ApprovalMode::Never).unwrap(),
            "\"never\""
        );
        assert_eq!(
            serde_json::to_string(&ApprovalMode::OnRequest).unwrap(),
            "\"on-request\""
        );
        assert_eq!(
            serde_json::to_string(&ApprovalMode::OnFailure).unwrap(),
            "\"on-failure\""
        );
        assert_eq!(
            serde_json::to_string(&ApprovalMode::Untrusted).unwrap(),
            "\"untrusted\""
        );
    }

    // ---- SandboxMode ----

    #[test]
    fn sandbox_mode_default_is_read_only() {
        assert_eq!(SandboxMode::default(), SandboxMode::ReadOnly);
    }

    #[test]
    fn sandbox_mode_serde_round_trip() {
        let modes = [
            SandboxMode::ReadOnly,
            SandboxMode::WorkspaceWrite,
            SandboxMode::DangerFullAccess,
        ];
        for mode in &modes {
            let json = serde_json::to_string(mode).expect("serialize");
            let deserialized: SandboxMode = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*mode, deserialized);
        }
    }

    #[test]
    fn sandbox_mode_serde_values() {
        assert_eq!(
            serde_json::to_string(&SandboxMode::ReadOnly).unwrap(),
            "\"read-only\""
        );
        assert_eq!(
            serde_json::to_string(&SandboxMode::WorkspaceWrite).unwrap(),
            "\"workspace-write\""
        );
        assert_eq!(
            serde_json::to_string(&SandboxMode::DangerFullAccess).unwrap(),
            "\"danger-full-access\""
        );
    }

    // ---- ModelReasoningEffort ----

    #[test]
    fn model_reasoning_effort_default_is_medium() {
        assert_eq!(
            ModelReasoningEffort::default(),
            ModelReasoningEffort::Medium
        );
    }

    #[test]
    fn model_reasoning_effort_serde_round_trip() {
        let efforts = [
            ModelReasoningEffort::Minimal,
            ModelReasoningEffort::Low,
            ModelReasoningEffort::Medium,
            ModelReasoningEffort::High,
            ModelReasoningEffort::XHigh,
        ];
        for effort in &efforts {
            let json = serde_json::to_string(effort).expect("serialize");
            let deserialized: ModelReasoningEffort =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*effort, deserialized);
        }
    }

    // ---- WebSearchMode ----

    #[test]
    fn web_search_mode_default_is_disabled() {
        assert_eq!(WebSearchMode::default(), WebSearchMode::Disabled);
    }

    #[test]
    fn web_search_mode_serde_round_trip() {
        let modes = [
            WebSearchMode::Disabled,
            WebSearchMode::Cached,
            WebSearchMode::Live,
        ];
        for mode in &modes {
            let json = serde_json::to_string(mode).expect("serialize");
            let deserialized: WebSearchMode = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*mode, deserialized);
        }
    }

    // ---- CodexOptions ----

    #[test]
    fn codex_options_default() {
        let opts = CodexOptions::default();
        assert!(opts.codex_path_override.is_none());
        assert!(opts.base_url.is_none());
        assert!(opts.api_key.is_none());
        assert!(opts.env.is_none());
        assert!(opts.codex_subprocess_stream_limit_bytes.is_none());
    }

    #[test]
    fn codex_options_builder_chain() {
        let opts = CodexOptions::new()
            .with_codex_path_override("/usr/local/bin/codex")
            .with_base_url("https://api.example.com")
            .with_api_key("sk-test-key")
            .with_subprocess_stream_limit_bytes(1024 * 1024);

        assert_eq!(
            opts.codex_path_override.as_deref(),
            Some("/usr/local/bin/codex")
        );
        assert_eq!(opts.base_url.as_deref(), Some("https://api.example.com"));
        assert_eq!(opts.api_key.as_deref(), Some("sk-test-key"));
        assert_eq!(opts.codex_subprocess_stream_limit_bytes, Some(1024 * 1024));
    }

    #[test]
    fn codex_options_with_env() {
        let mut env = std::collections::HashMap::new();
        env.insert("KEY".to_owned(), "VALUE".to_owned());
        let opts = CodexOptions::new().with_env(env);
        assert!(opts.env.is_some());
        assert_eq!(opts.env.as_ref().unwrap().get("KEY").unwrap(), "VALUE");
    }

    // ---- ThreadOptions ----

    #[test]
    fn thread_options_default() {
        let opts = ThreadOptions::default();
        assert!(opts.model.is_none());
        assert!(opts.sandbox_mode.is_none());
        assert!(opts.working_directory.is_none());
        assert!(opts.skip_git_repo_check.is_none());
        assert!(opts.model_reasoning_effort.is_none());
        assert!(opts.network_access_enabled.is_none());
        assert!(opts.web_search_mode.is_none());
        assert!(opts.web_search_enabled.is_none());
        assert!(opts.approval_policy.is_none());
        assert!(opts.additional_directories.is_none());
    }

    #[test]
    fn thread_options_builder_chain() {
        let opts = ThreadOptions::new()
            .with_model("o3")
            .with_sandbox_mode(SandboxMode::WorkspaceWrite)
            .with_working_directory("/home/user/project")
            .with_skip_git_repo_check(true)
            .with_model_reasoning_effort(ModelReasoningEffort::High)
            .with_approval_policy(ApprovalMode::OnRequest)
            .with_web_search_mode(WebSearchMode::Live)
            .with_additional_directories(vec!["/tmp".to_owned()]);

        assert_eq!(opts.model.as_deref(), Some("o3"));
        assert_eq!(opts.sandbox_mode, Some(SandboxMode::WorkspaceWrite));
        assert_eq!(
            opts.working_directory.as_deref(),
            Some("/home/user/project")
        );
        assert_eq!(opts.skip_git_repo_check, Some(true));
        assert_eq!(
            opts.model_reasoning_effort,
            Some(ModelReasoningEffort::High)
        );
        assert_eq!(opts.approval_policy, Some(ApprovalMode::OnRequest));
        assert_eq!(opts.web_search_mode, Some(WebSearchMode::Live));
        assert_eq!(opts.additional_directories, Some(vec!["/tmp".to_owned()]));
    }

    // ---- TurnOptions ----

    #[test]
    fn turn_options_default() {
        let opts = TurnOptions::default();
        assert!(opts.output_schema.is_none());
        assert!(opts.idle_timeout_seconds.is_none());
    }

    #[test]
    fn turn_options_builder_chain() {
        let schema = serde_json::json!({"type": "object"});
        let opts = TurnOptions::new()
            .with_output_schema(schema.clone())
            .with_idle_timeout_seconds(30.0);

        assert_eq!(opts.output_schema, Some(schema));
        assert_eq!(opts.idle_timeout_seconds, Some(30.0));
    }

    // ---- Debug impls ----

    #[test]
    fn options_debug_impls() {
        let codex = CodexOptions::new().with_base_url("https://api.example.com");
        let debug = format!("{codex:?}");
        assert!(debug.contains("CodexOptions"));
        assert!(debug.contains("api.example.com"));

        let thread = ThreadOptions::new().with_model("o3");
        let debug = format!("{thread:?}");
        assert!(debug.contains("ThreadOptions"));
        assert!(debug.contains("o3"));

        let turn = TurnOptions::new().with_idle_timeout_seconds(10.0);
        let debug = format!("{turn:?}");
        assert!(debug.contains("TurnOptions"));
        assert!(debug.contains("10.0"));
    }

    // ---- Send + Sync assertions ----

    #[test]
    fn options_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CodexOptions>();
        assert_send_sync::<ThreadOptions>();
        assert_send_sync::<TurnOptions>();
        assert_send_sync::<ApprovalMode>();
        assert_send_sync::<SandboxMode>();
        assert_send_sync::<ModelReasoningEffort>();
        assert_send_sync::<WebSearchMode>();
    }
}
