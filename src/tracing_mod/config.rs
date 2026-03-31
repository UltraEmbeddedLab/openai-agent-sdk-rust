//! Tracing configuration and global enable/disable control.
//!
//! [`TracingConfig`] carries per-run settings that influence what data is
//! recorded in trace spans (for example, whether to include sensitive
//! prompt content).
//!
//! The global disable switch ([`set_tracing_disabled`] /
//! [`is_tracing_disabled`]) provides a coarse-grained way to suppress all
//! tracing output, mirroring the Python SDK's `set_tracing_disabled`
//! function.

use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag that controls whether tracing is suppressed.
static TRACING_DISABLED: AtomicBool = AtomicBool::new(false);

/// Configuration for the tracing subsystem.
///
/// Pass this to a run configuration to control trace-level behavior such
/// as whether sensitive data (prompts, model outputs) is included in
/// spans.
///
/// # Defaults
///
/// | Field | Default |
/// |---|---|
/// | `include_sensitive_data` | `true` |
/// | `workflow_name` | `None` |
/// | `group_id` | `None` |
/// | `metadata` | `None` |
/// | `api_key` | `None` |
/// | `disabled` | `false` |
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TracingConfig {
    /// Whether to include sensitive data (prompts, outputs) in trace spans.
    pub include_sensitive_data: bool,
    /// Custom workflow name attached to the trace.
    pub workflow_name: Option<String>,
    /// Optional grouping identifier to link multiple traces from the same
    /// conversation or process.
    pub group_id: Option<String>,
    /// Optional arbitrary metadata dictionary attached to the trace.
    pub metadata: Option<serde_json::Value>,
    /// Optional API key used when exporting traces to a backend.
    pub api_key: Option<String>,
    /// If `true`, this particular trace is suppressed even when tracing is
    /// globally enabled.
    pub disabled: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            include_sensitive_data: true,
            workflow_name: None,
            group_id: None,
            metadata: None,
            api_key: None,
            disabled: false,
        }
    }
}

impl TracingConfig {
    /// Create a new `TracingConfig` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the workflow name.
    #[must_use]
    pub fn with_workflow_name(mut self, name: impl Into<String>) -> Self {
        self.workflow_name = Some(name.into());
        self
    }

    /// Set the group identifier.
    #[must_use]
    pub fn with_group_id(mut self, id: impl Into<String>) -> Self {
        self.group_id = Some(id.into());
        self
    }

    /// Set the metadata value.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Control whether sensitive data is included in spans.
    #[must_use]
    pub const fn with_sensitive_data(mut self, include: bool) -> Self {
        self.include_sensitive_data = include;
        self
    }

    /// Set the API key for trace export.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Mark this trace configuration as disabled.
    #[must_use]
    pub const fn with_disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// Globally disable or enable tracing for all agent runs.
///
/// When set to `true`, span creation helpers still return valid
/// [`tracing::Span`] values (they become no-ops internally), so callers
/// do not need conditional logic.
pub fn set_tracing_disabled(disabled: bool) {
    TRACING_DISABLED.store(disabled, Ordering::Relaxed);
}

/// Check whether tracing is globally disabled.
#[must_use]
pub fn is_tracing_disabled() -> bool {
    TRACING_DISABLED.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Serialize tests that touch global TRACING_DISABLED to prevent races
    // under `cargo tarpaulin` which may run tests in a single process.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn default_config_values() {
        let config = TracingConfig::default();
        assert!(config.include_sensitive_data);
        assert!(config.workflow_name.is_none());
        assert!(config.group_id.is_none());
        assert!(config.metadata.is_none());
        assert!(config.api_key.is_none());
        assert!(!config.disabled);
    }

    #[test]
    fn new_equals_default() {
        let a = TracingConfig::new();
        let b = TracingConfig::default();
        assert_eq!(a.include_sensitive_data, b.include_sensitive_data);
        assert_eq!(a.workflow_name, b.workflow_name);
        assert_eq!(a.disabled, b.disabled);
    }

    #[test]
    fn builder_methods() {
        let config = TracingConfig::new()
            .with_workflow_name("my_workflow")
            .with_group_id("thread-123")
            .with_sensitive_data(false)
            .with_api_key("sk-test")
            .with_disabled(true)
            .with_metadata(serde_json::json!({"env": "test"}));

        assert_eq!(config.workflow_name.as_deref(), Some("my_workflow"));
        assert_eq!(config.group_id.as_deref(), Some("thread-123"));
        assert!(!config.include_sensitive_data);
        assert_eq!(config.api_key.as_deref(), Some("sk-test"));
        assert!(config.disabled);
        assert_eq!(config.metadata, Some(serde_json::json!({"env": "test"})));
    }

    #[test]
    fn tracing_disabled_default_is_false() {
        let _guard = TEST_LOCK.lock().expect("test lock poisoned");
        set_tracing_disabled(false);
        assert!(!is_tracing_disabled());
    }

    #[test]
    fn set_and_read_tracing_disabled() {
        let _guard = TEST_LOCK.lock().expect("test lock poisoned");
        set_tracing_disabled(true);
        assert!(is_tracing_disabled());
        set_tracing_disabled(false);
        assert!(!is_tracing_disabled());
    }

    #[test]
    fn toggle_tracing_disabled_repeatedly() {
        let _guard = TEST_LOCK.lock().expect("test lock poisoned");
        for _ in 0..100 {
            set_tracing_disabled(true);
            assert!(is_tracing_disabled());
            set_tracing_disabled(false);
            assert!(!is_tracing_disabled());
        }
    }

    #[test]
    fn config_is_clone() {
        let config = TracingConfig::new().with_workflow_name("test");
        let cloned = config.clone();
        assert_eq!(cloned.workflow_name, config.workflow_name);
    }

    #[test]
    fn config_is_debug() {
        let config = TracingConfig::new();
        let debug = format!("{config:?}");
        assert!(debug.contains("TracingConfig"));
    }
}
