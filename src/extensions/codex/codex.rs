//! The main Codex executor.
//!
//! [`Codex`] is the entry point for thread-based agent execution. It creates
//! and resumes threads, managing the Codex CLI options and environment.
//!
//! Mirrors the Python SDK's `codex.py` in the Codex extension.

use std::sync::Arc;

use crate::models::Model;

use super::options::{CodexOptions, ThreadOptions};
use super::thread::Thread;

// ---------------------------------------------------------------------------
// Codex
// ---------------------------------------------------------------------------

/// The Codex executor -- a high-level orchestrator for thread-based agent execution.
///
/// Codex manages threads of work, each containing multiple turns, with
/// support for tool approval workflows, sandboxed execution, and event streaming.
///
/// # Example
///
/// ```
/// use openai_agents::extensions::codex::{Codex, CodexOptions, ThreadOptions};
///
/// let codex = Codex::new(CodexOptions::new().with_api_key("sk-test"));
/// let thread = codex.start_thread(None);
/// assert!(thread.id().is_none()); // ID is assigned after first run.
/// ```
#[non_exhaustive]
pub struct Codex {
    options: CodexOptions,
    thread_options_default: ThreadOptions,
    model: Option<Arc<dyn Model>>,
}

impl std::fmt::Debug for Codex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Codex")
            .field("options", &self.options)
            .field("thread_options_default", &self.thread_options_default)
            .field("model", &self.model.as_ref().map(|_| "<model>"))
            .finish()
    }
}

impl Codex {
    /// Create a new Codex instance with the given options.
    #[must_use]
    pub fn new(options: CodexOptions) -> Self {
        Self {
            options,
            thread_options_default: ThreadOptions::default(),
            model: None,
        }
    }

    /// Create a new Codex instance with default options.
    #[must_use]
    pub fn default_instance() -> Self {
        Self::new(CodexOptions::default())
    }

    /// Set the model to use for thread execution.
    ///
    /// The model is propagated to every thread created by this Codex instance.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Get a reference to the current options.
    #[must_use]
    pub const fn options(&self) -> &CodexOptions {
        &self.options
    }

    /// Start a new thread with optional thread-specific options.
    ///
    /// If `options` is `None`, the default thread options are used.
    /// The thread inherits the model from this Codex instance.
    #[must_use]
    pub fn start_thread(&self, options: Option<ThreadOptions>) -> Thread {
        let thread_options = options.unwrap_or_else(|| self.thread_options_default.clone());
        let mut thread = Thread::new().with_options(thread_options);
        if let Some(ref model) = self.model {
            thread = thread.with_model(Arc::clone(model));
        }
        thread
    }

    /// Resume an existing thread by its ID with optional thread-specific options.
    ///
    /// If `options` is `None`, the default thread options are used.
    /// The thread inherits the model from this Codex instance.
    #[must_use]
    pub fn resume_thread(
        &self,
        thread_id: impl Into<String>,
        options: Option<ThreadOptions>,
    ) -> Thread {
        let thread_options = options.unwrap_or_else(|| self.thread_options_default.clone());
        let mut thread = Thread::with_id(thread_id).with_options(thread_options);
        if let Some(ref model) = self.model {
            thread = thread.with_model(Arc::clone(model));
        }
        thread
    }
}

impl Default for Codex {
    fn default() -> Self {
        Self::default_instance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_new_with_default_options() {
        let codex = Codex::new(CodexOptions::default());
        assert!(codex.options().base_url.is_none());
        assert!(codex.options().api_key.is_none());
    }

    #[test]
    fn codex_new_with_custom_options() {
        let opts = CodexOptions::new()
            .with_base_url("https://api.example.com")
            .with_api_key("sk-test");
        let codex = Codex::new(opts);
        assert_eq!(
            codex.options().base_url.as_deref(),
            Some("https://api.example.com")
        );
        assert_eq!(codex.options().api_key.as_deref(), Some("sk-test"));
    }

    #[test]
    fn codex_default_instance() {
        let codex = Codex::default_instance();
        assert!(codex.options().base_url.is_none());
    }

    #[test]
    fn codex_default_trait() {
        let codex = Codex::default();
        assert!(codex.options().base_url.is_none());
    }

    #[test]
    fn codex_start_thread_returns_new_thread() {
        let codex = Codex::default();
        let thread = codex.start_thread(None);
        assert!(thread.id().is_none());
    }

    #[test]
    fn codex_start_thread_with_options() {
        let codex = Codex::default();
        let opts = ThreadOptions::new().with_model("o3");
        let thread = codex.start_thread(Some(opts));
        assert!(thread.id().is_none());
    }

    #[test]
    fn codex_resume_thread() {
        let codex = Codex::default();
        let thread = codex.resume_thread("t-123", None);
        assert_eq!(thread.id(), Some("t-123"));
    }

    #[test]
    fn codex_resume_thread_with_options() {
        let codex = Codex::default();
        let opts = ThreadOptions::new().with_model("o3");
        let thread = codex.resume_thread("t-456", Some(opts));
        assert_eq!(thread.id(), Some("t-456"));
    }

    // ---- Send + Sync ----

    #[test]
    fn codex_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Codex>();
    }

    // ---- Debug ----

    #[test]
    fn codex_debug() {
        let codex = Codex::new(CodexOptions::new().with_base_url("https://api.example.com"));
        let debug = format!("{codex:?}");
        assert!(debug.contains("Codex"));
        assert!(debug.contains("api.example.com"));
    }
}
