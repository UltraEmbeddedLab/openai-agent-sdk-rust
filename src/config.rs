//! Configuration types for agent runs and model settings.
//!
//! This module provides [`ModelSettings`] for configuring individual LLM calls
//! and [`RunConfig`] for configuring an entire agent run. Both types use the
//! builder pattern for ergonomic construction.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Default maximum number of turns for an agent run.
pub const DEFAULT_MAX_TURNS: u32 = 10;

/// Tool choice strategy for an LLM call.
///
/// Controls how the model selects tools during a conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides whether to use a tool.
    Auto,
    /// The model must use at least one tool.
    Required,
    /// The model must not use any tools.
    None,
    /// The model must use the tool with the given name.
    #[serde(untagged)]
    Named(String),
}

/// Truncation strategy for model input.
///
/// Controls how the model handles input that exceeds its context window.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Truncation {
    /// Automatically truncate input to fit the context window.
    Auto,
    /// Do not truncate; return an error if input exceeds the context window.
    Disabled,
}

/// Settings for an individual LLM call.
///
/// This struct holds optional model configuration parameters such as temperature,
/// top-p, penalties, truncation, and more. Not all models or providers support
/// every parameter; consult the API documentation for the model you are using.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelSettings {
    /// The sampling temperature. Higher values produce more random output.
    pub temperature: Option<f64>,
    /// Nucleus sampling parameter. The model considers tokens with top-p probability mass.
    pub top_p: Option<f64>,
    /// Penalizes tokens based on their frequency in the output so far.
    pub frequency_penalty: Option<f64>,
    /// Penalizes tokens based on whether they have appeared in the output so far.
    pub presence_penalty: Option<f64>,
    /// Controls how the model selects tools.
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may issue multiple tool calls in parallel.
    pub parallel_tool_calls: Option<bool>,
    /// The truncation strategy for model input.
    pub truncation: Option<Truncation>,
    /// The maximum number of output tokens to generate.
    pub max_tokens: Option<u32>,
    /// Metadata to include with the model response call.
    pub metadata: Option<HashMap<String, String>>,
    /// Whether to store the generated model response for later retrieval.
    pub store: Option<bool>,
    /// Additional body fields to include in the request.
    pub extra_body: Option<serde_json::Value>,
    /// Additional headers to include in the request.
    pub extra_headers: Option<HashMap<String, String>>,
    /// Arbitrary extra arguments passed directly to the model API call.
    pub extra_args: Option<HashMap<String, serde_json::Value>>,
}

impl ModelSettings {
    /// Creates a new `ModelSettings` with all fields set to `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the temperature and return `self` (fluent builder).
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the top-p and return `self` (fluent builder).
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set the max tokens and return `self` (fluent builder).
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the tool choice and return `self` (fluent builder).
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set the truncation strategy and return `self` (fluent builder).
    #[must_use]
    pub const fn with_truncation(mut self, truncation: Truncation) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Set whether to store the response and return `self` (fluent builder).
    #[must_use]
    pub const fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Produces a new `ModelSettings` by overlaying any non-`None` values from
    /// `override_settings` on top of this instance.
    ///
    /// If `override_settings` is `None`, returns a clone of `self`.
    /// The `extra_args` field is merged (union of both maps) rather than replaced.
    #[must_use]
    pub fn resolve(&self, override_settings: Option<&Self>) -> Self {
        let Some(ov) = override_settings else {
            return self.clone();
        };

        Self {
            temperature: ov.temperature.or(self.temperature),
            top_p: ov.top_p.or(self.top_p),
            frequency_penalty: ov.frequency_penalty.or(self.frequency_penalty),
            presence_penalty: ov.presence_penalty.or(self.presence_penalty),
            tool_choice: ov.tool_choice.clone().or_else(|| self.tool_choice.clone()),
            parallel_tool_calls: ov.parallel_tool_calls.or(self.parallel_tool_calls),
            truncation: ov.truncation.clone().or_else(|| self.truncation.clone()),
            max_tokens: ov.max_tokens.or(self.max_tokens),
            metadata: ov.metadata.clone().or_else(|| self.metadata.clone()),
            store: ov.store.or(self.store),
            extra_body: ov.extra_body.clone().or_else(|| self.extra_body.clone()),
            extra_headers: ov
                .extra_headers
                .clone()
                .or_else(|| self.extra_headers.clone()),
            extra_args: merge_extra_args(self.extra_args.as_ref(), ov.extra_args.as_ref()),
        }
    }
}

/// Merges two optional `extra_args` maps. The override values take precedence.
fn merge_extra_args(
    base: Option<&HashMap<String, serde_json::Value>>,
    overrides: Option<&HashMap<String, serde_json::Value>>,
) -> Option<HashMap<String, serde_json::Value>> {
    match (base, overrides) {
        (None, None) => None,
        (Some(b), None) => Some(b.clone()),
        (None, Some(o)) => Some(o.clone()),
        (Some(b), Some(o)) => {
            let mut merged = b.clone();
            merged.extend(o.iter().map(|(k, v)| (k.clone(), v.clone())));
            Some(merged)
        }
    }
}

/// A reference to a model, either by name or (in the future) by trait object.
///
/// The `Instance` variant will be added when the `models` module is implemented.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ModelRef {
    /// A model identified by its string name (e.g. `"gpt-4o"`).
    Name(String),
}

impl From<&str> for ModelRef {
    fn from(s: &str) -> Self {
        Self::Name(s.to_owned())
    }
}

impl From<String> for ModelRef {
    fn from(s: String) -> Self {
        Self::Name(s)
    }
}

/// Configuration for an entire agent run.
///
/// Controls the model, model settings, turn limits, tracing, and other
/// run-level parameters. Use [`RunConfigBuilder`] for ergonomic construction.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RunConfig {
    /// The model to use for the entire run, overriding per-agent models.
    pub model: Option<ModelRef>,
    /// Global model settings; non-`None` values override agent-specific settings.
    pub model_settings: Option<ModelSettings>,
    /// The maximum number of turns before the run is stopped.
    pub max_turns: u32,
    /// Whether tracing is disabled for this run.
    pub tracing_disabled: bool,
    /// A logical name for the run, used in tracing.
    pub workflow_name: String,
    /// A custom trace ID. If not set, one is generated automatically.
    pub trace_id: Option<String>,
    /// A grouping identifier to link related traces (e.g. a conversation thread ID).
    pub group_id: Option<String>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            model: None,
            model_settings: None,
            max_turns: DEFAULT_MAX_TURNS,
            tracing_disabled: false,
            workflow_name: String::from("agent_workflow"),
            trace_id: None,
            group_id: None,
        }
    }
}

impl RunConfig {
    /// Creates a new [`RunConfigBuilder`] for constructing a `RunConfig`.
    #[must_use]
    pub fn builder() -> RunConfigBuilder {
        RunConfigBuilder::default()
    }
}

/// Builder for [`RunConfig`].
///
/// All fields are optional and default to the values defined in [`RunConfig::default`].
#[derive(Debug, Clone, Default)]
pub struct RunConfigBuilder {
    model: Option<ModelRef>,
    model_settings: Option<ModelSettings>,
    max_turns: Option<u32>,
    tracing_disabled: Option<bool>,
    workflow_name: Option<String>,
    trace_id: Option<String>,
    group_id: Option<String>,
}

impl RunConfigBuilder {
    /// Sets the model for the run.
    #[must_use]
    pub fn model(mut self, model: impl Into<ModelRef>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the global model settings for the run.
    #[must_use]
    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = Some(settings);
        self
    }

    /// Sets the maximum number of turns for the run.
    #[must_use]
    pub const fn max_turns(mut self, max_turns: u32) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// Disables tracing for the run.
    #[must_use]
    pub const fn tracing_disabled(mut self, disabled: bool) -> Self {
        self.tracing_disabled = Some(disabled);
        self
    }

    /// Sets the workflow name for tracing.
    #[must_use]
    pub fn workflow_name(mut self, name: impl Into<String>) -> Self {
        self.workflow_name = Some(name.into());
        self
    }

    /// Sets a custom trace ID for the run.
    #[must_use]
    pub fn trace_id(mut self, id: impl Into<String>) -> Self {
        self.trace_id = Some(id.into());
        self
    }

    /// Sets a group ID for linking related traces.
    #[must_use]
    pub fn group_id(mut self, id: impl Into<String>) -> Self {
        self.group_id = Some(id.into());
        self
    }

    /// Builds the [`RunConfig`], applying defaults for any unset fields.
    #[must_use]
    pub fn build(self) -> RunConfig {
        let defaults = RunConfig::default();
        RunConfig {
            model: self.model,
            model_settings: self.model_settings,
            max_turns: self.max_turns.unwrap_or(defaults.max_turns),
            tracing_disabled: self.tracing_disabled.unwrap_or(defaults.tracing_disabled),
            workflow_name: self.workflow_name.unwrap_or(defaults.workflow_name),
            trace_id: self.trace_id,
            group_id: self.group_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_settings_default_is_all_none() {
        let s = ModelSettings::default();
        assert!(s.temperature.is_none());
        assert!(s.top_p.is_none());
        assert!(s.frequency_penalty.is_none());
        assert!(s.presence_penalty.is_none());
        assert!(s.tool_choice.is_none());
        assert!(s.parallel_tool_calls.is_none());
        assert!(s.truncation.is_none());
        assert!(s.max_tokens.is_none());
        assert!(s.metadata.is_none());
        assert!(s.store.is_none());
        assert!(s.extra_body.is_none());
        assert!(s.extra_headers.is_none());
        assert!(s.extra_args.is_none());
    }

    #[test]
    fn resolve_with_none_returns_clone() {
        let base = ModelSettings {
            temperature: Some(0.7),
            max_tokens: Some(100),
            ..Default::default()
        };
        let resolved = base.resolve(None);
        assert_eq!(resolved.temperature, Some(0.7));
        assert_eq!(resolved.max_tokens, Some(100));
    }

    #[test]
    fn resolve_overrides_non_none_fields() {
        let base = ModelSettings {
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(100),
            ..Default::default()
        };
        let overrides = ModelSettings {
            temperature: Some(0.3),
            max_tokens: Some(200),
            ..Default::default()
        };
        let resolved = base.resolve(Some(&overrides));
        // Overridden.
        assert_eq!(resolved.temperature, Some(0.3));
        assert_eq!(resolved.max_tokens, Some(200));
        // Kept from base.
        assert_eq!(resolved.top_p, Some(0.9));
    }

    #[test]
    fn resolve_keeps_base_when_override_is_none() {
        let base = ModelSettings {
            store: Some(true),
            truncation: Some(Truncation::Auto),
            ..Default::default()
        };
        let overrides = ModelSettings::default();
        let resolved = base.resolve(Some(&overrides));
        assert_eq!(resolved.store, Some(true));
        assert_eq!(resolved.truncation, Some(Truncation::Auto));
    }

    #[test]
    fn resolve_merges_extra_args() {
        let mut base_args = HashMap::new();
        base_args.insert("a".to_string(), serde_json::json!(1));
        base_args.insert("b".to_string(), serde_json::json!(2));

        let mut override_args = HashMap::new();
        override_args.insert("b".to_string(), serde_json::json!(99));
        override_args.insert("c".to_string(), serde_json::json!(3));

        let base = ModelSettings {
            extra_args: Some(base_args),
            ..Default::default()
        };
        let overrides = ModelSettings {
            extra_args: Some(override_args),
            ..Default::default()
        };

        let resolved = base.resolve(Some(&overrides));
        let args = resolved.extra_args.unwrap();
        assert_eq!(args.get("a"), Some(&serde_json::json!(1)));
        assert_eq!(args.get("b"), Some(&serde_json::json!(99))); // Override wins.
        assert_eq!(args.get("c"), Some(&serde_json::json!(3)));
    }

    #[test]
    fn run_config_defaults() {
        let config = RunConfig::default();
        assert!(config.model.is_none());
        assert!(config.model_settings.is_none());
        assert_eq!(config.max_turns, DEFAULT_MAX_TURNS);
        assert!(!config.tracing_disabled);
        assert_eq!(config.workflow_name, "agent_workflow");
        assert!(config.trace_id.is_none());
        assert!(config.group_id.is_none());
    }

    #[test]
    fn run_config_builder_defaults() {
        let config = RunConfig::builder().build();
        assert!(config.model.is_none());
        assert_eq!(config.max_turns, DEFAULT_MAX_TURNS);
        assert!(!config.tracing_disabled);
        assert_eq!(config.workflow_name, "agent_workflow");
    }

    #[test]
    fn run_config_builder_all_fields() {
        let config = RunConfig::builder()
            .model("gpt-4o")
            .model_settings(ModelSettings {
                temperature: Some(0.5),
                ..Default::default()
            })
            .max_turns(5)
            .tracing_disabled(true)
            .workflow_name("test_workflow")
            .trace_id("trace-123")
            .group_id("group-456")
            .build();

        assert!(matches!(config.model, Some(ModelRef::Name(ref n)) if n == "gpt-4o"));
        assert_eq!(config.model_settings.unwrap().temperature, Some(0.5));
        assert_eq!(config.max_turns, 5);
        assert!(config.tracing_disabled);
        assert_eq!(config.workflow_name, "test_workflow");
        assert_eq!(config.trace_id.as_deref(), Some("trace-123"));
        assert_eq!(config.group_id.as_deref(), Some("group-456"));
    }

    #[test]
    fn tool_choice_serialization() {
        let auto = serde_json::to_string(&ToolChoice::Auto).unwrap();
        assert_eq!(auto, "\"auto\"");

        let required = serde_json::to_string(&ToolChoice::Required).unwrap();
        assert_eq!(required, "\"required\"");

        let none = serde_json::to_string(&ToolChoice::None).unwrap();
        assert_eq!(none, "\"none\"");

        let named = serde_json::to_string(&ToolChoice::Named("my_tool".into())).unwrap();
        assert_eq!(named, "\"my_tool\"");
    }

    #[test]
    fn tool_choice_deserialization() {
        let auto: ToolChoice = serde_json::from_str("\"auto\"").unwrap();
        assert_eq!(auto, ToolChoice::Auto);

        let required: ToolChoice = serde_json::from_str("\"required\"").unwrap();
        assert_eq!(required, ToolChoice::Required);

        let none: ToolChoice = serde_json::from_str("\"none\"").unwrap();
        assert_eq!(none, ToolChoice::None);
    }

    #[test]
    fn truncation_serialization() {
        let auto = serde_json::to_string(&Truncation::Auto).unwrap();
        assert_eq!(auto, "\"auto\"");

        let disabled = serde_json::to_string(&Truncation::Disabled).unwrap();
        assert_eq!(disabled, "\"disabled\"");
    }

    #[test]
    fn truncation_deserialization() {
        let auto: Truncation = serde_json::from_str("\"auto\"").unwrap();
        assert_eq!(auto, Truncation::Auto);

        let disabled: Truncation = serde_json::from_str("\"disabled\"").unwrap();
        assert_eq!(disabled, Truncation::Disabled);
    }

    #[test]
    fn model_ref_from_str() {
        let r: ModelRef = "gpt-4o".into();
        assert!(matches!(r, ModelRef::Name(ref n) if n == "gpt-4o"));
    }

    #[test]
    fn model_ref_from_string() {
        let r: ModelRef = String::from("gpt-4o-mini").into();
        assert!(matches!(r, ModelRef::Name(ref n) if n == "gpt-4o-mini"));
    }

    #[test]
    fn model_settings_round_trip_json() {
        let settings = ModelSettings {
            temperature: Some(0.8),
            top_p: Some(0.95),
            tool_choice: Some(ToolChoice::Required),
            truncation: Some(Truncation::Auto),
            max_tokens: Some(4096),
            store: Some(true),
            ..Default::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: ModelSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.temperature, Some(0.8));
        assert_eq!(deserialized.top_p, Some(0.95));
        assert_eq!(deserialized.tool_choice, Some(ToolChoice::Required));
        assert_eq!(deserialized.truncation, Some(Truncation::Auto));
        assert_eq!(deserialized.max_tokens, Some(4096));
        assert_eq!(deserialized.store, Some(true));
    }
}
