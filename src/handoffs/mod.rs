//! Handoff support for delegating execution between agents.
//!
//! A handoff allows one agent to transfer control to another agent during a run.
//! Handoffs are exposed to the LLM as tools; when the model calls the handoff tool,
//! execution transfers to the target agent.
//!
//! This module mirrors the Python SDK's `handoffs/__init__.py`.
//!
//! # Example
//!
//! ```
//! use openai_agents::handoffs::Handoff;
//!
//! let handoff: Handoff<()> = Handoff::to_agent("billing_agent").build();
//! assert_eq!(handoff.tool_name, "transfer_to_billing_agent");
//! ```

use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use crate::context::RunContextWrapper;
use crate::error::Result;
use crate::items::{InputContent, RunItem};
use crate::schema::{ensure_strict_json_schema, json_schema_for};

/// A boxed future that is `Send`.
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The type of the on-invoke callback for a handoff.
type OnInvokeCallback<C> = Arc<
    dyn for<'a> Fn(&'a RunContextWrapper<C>, String) -> BoxFuture<'a, Result<String>> + Send + Sync,
>;

/// Input data provided to the handoff input filter.
///
/// Contains the conversation history and items generated before and during the
/// current turn, allowing filters to transform what the next agent sees.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HandoffInputData {
    /// The conversation history up to this point.
    pub input_history: InputContent,
    /// Items generated before the current turn.
    pub pre_handoff_items: Vec<RunItem>,
    /// Items generated in the current turn, including the handoff call.
    pub new_items: Vec<RunItem>,
}

/// A filter that transforms input data before passing it to the next agent.
///
/// The filter receives a [`HandoffInputData`] and returns a potentially modified
/// version. By default, the new agent sees the entire conversation history; an
/// input filter can be used to remove older inputs, strip tool calls, or otherwise
/// reshape the context.
pub type HandoffInputFilter =
    Arc<dyn Fn(HandoffInputData) -> BoxFuture<'static, HandoffInputData> + Send + Sync>;

/// A handoff delegates execution from one agent to another.
///
/// Handoffs are exposed to the LLM as tools. When the LLM calls the handoff tool,
/// execution transfers to the target agent. Use [`Handoff::to_agent`] to create a
/// builder for constructing handoffs.
pub struct Handoff<C: Send + Sync + 'static> {
    /// The tool name exposed to the LLM (e.g., `"transfer_to_triage_agent"`).
    pub tool_name: String,
    /// Description of when and why to use this handoff.
    pub tool_description: String,
    /// JSON schema for structured handoff arguments, if any.
    pub input_json_schema: serde_json::Value,
    /// Name of the target agent.
    pub agent_name: String,
    /// Whether to enforce strict JSON schema on the handoff arguments.
    pub strict_json_schema: bool,
    /// Optional filter to transform input before the next agent sees it.
    pub input_filter: Option<HandoffInputFilter>,
    /// Whether this handoff is currently enabled.
    pub is_enabled: bool,
    /// The invocation callback. Receives the run context and raw JSON args, returns
    /// the target agent name.
    on_invoke: OnInvokeCallback<C>,
}

impl<C: Send + Sync + 'static> Handoff<C> {
    /// Generate the default tool name for a handoff to the given agent.
    ///
    /// Converts the agent name to lowercase and replaces spaces with underscores,
    /// producing names like `"transfer_to_billing_agent"`.
    #[must_use]
    pub fn default_tool_name(agent_name: &str) -> String {
        format!(
            "transfer_to_{}",
            agent_name.to_lowercase().replace(' ', "_")
        )
    }

    /// Generate the default tool description for a handoff.
    #[must_use]
    pub fn default_tool_description(agent_name: &str) -> String {
        format!("Handoff to the {agent_name} agent to handle the request.")
    }

    /// Get the transfer message for this handoff, recording the source agent.
    #[must_use]
    pub fn get_transfer_message(&self, source_agent: &str) -> String {
        format!(
            "Transferred from '{source_agent}' to '{}'.",
            self.agent_name
        )
    }

    /// Invoke the handoff. Returns the name of the target agent.
    ///
    /// # Errors
    ///
    /// Returns an error if the `on_invoke` callback fails.
    pub async fn invoke(&self, ctx: &RunContextWrapper<C>, args_json: String) -> Result<String> {
        (self.on_invoke)(ctx, args_json).await
    }

    /// Create a builder for a handoff to the given agent.
    #[must_use]
    pub fn to_agent(agent_name: impl Into<String>) -> HandoffBuilder<C> {
        HandoffBuilder::new(agent_name.into())
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for Handoff<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handoff")
            .field("tool_name", &self.tool_name)
            .field("tool_description", &self.tool_description)
            .field("input_json_schema", &self.input_json_schema)
            .field("agent_name", &self.agent_name)
            .field("strict_json_schema", &self.strict_json_schema)
            .field("input_filter", &self.input_filter.as_ref().map(|_| "..."))
            .field("is_enabled", &self.is_enabled)
            .field("on_invoke", &"<callback>")
            .finish()
    }
}

/// A builder for constructing [`Handoff`] instances.
///
/// Created via [`Handoff::to_agent`]. All fields have sensible defaults; call
/// [`build`](HandoffBuilder::build) to produce the final [`Handoff`].
pub struct HandoffBuilder<C: Send + Sync + 'static> {
    agent_name: String,
    tool_name: Option<String>,
    tool_description: Option<String>,
    input_json_schema: Option<serde_json::Value>,
    strict_json_schema: bool,
    input_filter: Option<HandoffInputFilter>,
    is_enabled: bool,
    on_invoke: Option<OnInvokeCallback<C>>,
    _phantom: PhantomData<C>,
}

impl<C: Send + Sync + 'static> HandoffBuilder<C> {
    fn new(agent_name: String) -> Self {
        Self {
            agent_name,
            tool_name: None,
            tool_description: None,
            input_json_schema: None,
            strict_json_schema: true,
            input_filter: None,
            is_enabled: true,
            on_invoke: None,
            _phantom: PhantomData,
        }
    }

    /// Override the tool name exposed to the LLM.
    #[must_use]
    pub fn tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self
    }

    /// Override the tool description exposed to the LLM.
    #[must_use]
    pub fn tool_description(mut self, desc: impl Into<String>) -> Self {
        self.tool_description = Some(desc.into());
        self
    }

    /// Set the input type for structured handoff arguments.
    ///
    /// Generates a JSON schema from `T` using `schemars`. The schema is passed to
    /// the model as the handoff tool's `parameters`.
    #[must_use]
    pub fn input_type<T: schemars::JsonSchema>(mut self) -> Self {
        let schema = json_schema_for::<T>();
        self.input_json_schema = Some(schema);
        self
    }

    /// Set the raw JSON schema for handoff arguments directly.
    #[must_use]
    pub fn input_json_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_json_schema = Some(schema);
        self
    }

    /// Set whether to enforce strict JSON schema on the handoff arguments.
    #[must_use]
    pub const fn strict_json_schema(mut self, strict: bool) -> Self {
        self.strict_json_schema = strict;
        self
    }

    /// Set an input filter to transform data before the next agent sees it.
    #[must_use]
    pub fn input_filter(mut self, filter: HandoffInputFilter) -> Self {
        self.input_filter = Some(filter);
        self
    }

    /// Set whether this handoff is enabled.
    #[must_use]
    pub const fn is_enabled(mut self, enabled: bool) -> Self {
        self.is_enabled = enabled;
        self
    }

    /// Set a custom invocation callback.
    ///
    /// The callback receives the run context and the raw JSON arguments string,
    /// and must return the name of the target agent.
    #[must_use]
    pub fn on_invoke<F>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&'a RunContextWrapper<C>, String) -> BoxFuture<'a, Result<String>>
            + Send
            + Sync
            + 'static,
    {
        self.on_invoke = Some(Arc::new(f));
        self
    }

    /// Build the [`Handoff`].
    ///
    /// Uses default values for any fields not explicitly set:
    /// - `tool_name`: `"transfer_to_{agent_name}"` (lowercased, spaces to underscores).
    /// - `tool_description`: `"Handoff to the {agent_name} agent to handle the request."`.
    /// - `input_json_schema`: empty object `{}`.
    /// - `on_invoke`: returns the `agent_name` unchanged.
    ///
    /// When `strict_json_schema` is `true` (the default), the input schema is
    /// transformed via [`ensure_strict_json_schema`].
    #[must_use]
    pub fn build(self) -> Handoff<C> {
        let agent_name = self.agent_name;
        let tool_name = self
            .tool_name
            .unwrap_or_else(|| Handoff::<C>::default_tool_name(&agent_name));
        let tool_description = self
            .tool_description
            .unwrap_or_else(|| Handoff::<C>::default_tool_description(&agent_name));

        let raw_schema = self
            .input_json_schema
            .unwrap_or_else(|| serde_json::json!({}));

        let input_json_schema = if self.strict_json_schema {
            // Best-effort strict enforcement; fall back to the raw schema on error.
            ensure_strict_json_schema(raw_schema.clone()).unwrap_or(raw_schema)
        } else {
            raw_schema
        };

        let agent_name_for_invoke = agent_name.clone();
        let on_invoke = self.on_invoke.unwrap_or_else(|| {
            Arc::new(move |_ctx: &RunContextWrapper<C>, _args: String| {
                let name = agent_name_for_invoke.clone();
                Box::pin(async move { Ok(name) })
            })
        });

        Handoff {
            tool_name,
            tool_description,
            input_json_schema,
            agent_name,
            strict_json_schema: self.strict_json_schema,
            input_filter: self.input_filter,
            is_enabled: self.is_enabled,
            on_invoke,
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for HandoffBuilder<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandoffBuilder")
            .field("agent_name", &self.agent_name)
            .field("tool_name", &self.tool_name)
            .field("tool_description", &self.tool_description)
            .field("input_json_schema", &self.input_json_schema)
            .field("strict_json_schema", &self.strict_json_schema)
            .field("input_filter", &self.input_filter.as_ref().map(|_| "..."))
            .field("is_enabled", &self.is_enabled)
            .field("on_invoke", &self.on_invoke.as_ref().map(|_| "..."))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Default tool name generation ----

    #[test]
    fn default_tool_name_simple() {
        assert_eq!(
            Handoff::<()>::default_tool_name("billing"),
            "transfer_to_billing"
        );
    }

    #[test]
    fn default_tool_name_with_spaces() {
        assert_eq!(
            Handoff::<()>::default_tool_name("Triage Agent"),
            "transfer_to_triage_agent"
        );
    }

    #[test]
    fn default_tool_name_uppercase() {
        assert_eq!(
            Handoff::<()>::default_tool_name("BILLING"),
            "transfer_to_billing"
        );
    }

    #[test]
    fn default_tool_name_empty() {
        assert_eq!(Handoff::<()>::default_tool_name(""), "transfer_to_");
    }

    // ---- Default tool description ----

    #[test]
    fn default_tool_description_simple() {
        assert_eq!(
            Handoff::<()>::default_tool_description("billing"),
            "Handoff to the billing agent to handle the request."
        );
    }

    #[test]
    fn default_tool_description_with_name() {
        assert_eq!(
            Handoff::<()>::default_tool_description("Triage Agent"),
            "Handoff to the Triage Agent agent to handle the request."
        );
    }

    // ---- Builder with defaults ----

    #[test]
    fn builder_defaults() {
        let handoff: Handoff<()> = Handoff::to_agent("billing").build();
        assert_eq!(handoff.tool_name, "transfer_to_billing");
        assert_eq!(
            handoff.tool_description,
            "Handoff to the billing agent to handle the request."
        );
        assert_eq!(handoff.agent_name, "billing");
        assert!(handoff.strict_json_schema);
        assert!(handoff.input_filter.is_none());
        assert!(handoff.is_enabled);
    }

    // ---- Builder with custom tool name and description ----

    #[test]
    fn builder_custom_tool_name() {
        let handoff: Handoff<()> = Handoff::to_agent("billing")
            .tool_name("route_to_billing")
            .build();
        assert_eq!(handoff.tool_name, "route_to_billing");
    }

    #[test]
    fn builder_custom_tool_description() {
        let handoff: Handoff<()> = Handoff::to_agent("billing")
            .tool_description("Send to billing department.")
            .build();
        assert_eq!(handoff.tool_description, "Send to billing department.");
    }

    #[test]
    fn builder_custom_name_and_description() {
        let handoff: Handoff<()> = Handoff::to_agent("billing")
            .tool_name("route_billing")
            .tool_description("Route to billing.")
            .build();
        assert_eq!(handoff.tool_name, "route_billing");
        assert_eq!(handoff.tool_description, "Route to billing.");
    }

    // ---- Builder with input_type schema ----

    #[test]
    fn builder_with_input_type_schema() {
        use schemars::JsonSchema;
        use serde::Deserialize;

        #[derive(Deserialize, JsonSchema)]
        #[allow(dead_code)]
        struct HandoffArgs {
            reason: String,
            priority: u32,
        }

        let handoff: Handoff<()> = Handoff::to_agent("billing")
            .input_type::<HandoffArgs>()
            .build();

        // The schema should contain properties for `reason` and `priority`.
        let props = handoff.input_json_schema["properties"]
            .as_object()
            .expect("schema should have properties");
        assert!(props.contains_key("reason"));
        assert!(props.contains_key("priority"));

        // Strict mode should set additionalProperties: false.
        assert_eq!(
            handoff.input_json_schema["additionalProperties"],
            serde_json::Value::Bool(false)
        );
    }

    // ---- Invoke returns agent name ----

    #[tokio::test]
    async fn invoke_returns_agent_name() {
        let handoff: Handoff<()> = Handoff::to_agent("billing").build();
        let ctx = RunContextWrapper::new(());
        let result = handoff.invoke(&ctx, String::new()).await;
        assert_eq!(result.unwrap(), "billing");
    }

    #[tokio::test]
    async fn invoke_returns_agent_name_with_args() {
        let handoff: Handoff<()> = Handoff::to_agent("support").build();
        let ctx = RunContextWrapper::new(());
        let result = handoff
            .invoke(&ctx, r#"{"reason": "help"}"#.to_string())
            .await;
        assert_eq!(result.unwrap(), "support");
    }

    // ---- Custom on_invoke callback ----

    #[tokio::test]
    async fn custom_on_invoke() {
        let handoff: Handoff<String> = Handoff::to_agent("billing")
            .on_invoke(|ctx, _args| {
                let target = format!("{}_{}", ctx.context, "billing");
                Box::pin(async move { Ok(target) })
            })
            .build();

        let ctx = RunContextWrapper::new("prefix".to_string());
        let result = handoff.invoke(&ctx, String::new()).await;
        assert_eq!(result.unwrap(), "prefix_billing");
    }

    #[tokio::test]
    async fn custom_on_invoke_uses_args() {
        let handoff: Handoff<()> = Handoff::to_agent("router")
            .on_invoke(|_ctx, args| Box::pin(async move { Ok(args) }))
            .build();

        let ctx = RunContextWrapper::new(());
        let result = handoff.invoke(&ctx, "target_agent".to_string()).await;
        assert_eq!(result.unwrap(), "target_agent");
    }

    // ---- get_transfer_message ----

    #[test]
    fn get_transfer_message_format() {
        let handoff: Handoff<()> = Handoff::to_agent("billing").build();
        let msg = handoff.get_transfer_message("triage");
        assert_eq!(msg, "Transferred from 'triage' to 'billing'.");
    }

    #[test]
    fn get_transfer_message_with_spaces() {
        let handoff: Handoff<()> = Handoff::to_agent("Billing Agent").build();
        let msg = handoff.get_transfer_message("Triage Agent");
        assert_eq!(msg, "Transferred from 'Triage Agent' to 'Billing Agent'.");
    }

    // ---- Debug impl ----

    #[test]
    fn debug_impl_handoff() {
        let handoff: Handoff<()> = Handoff::to_agent("billing").build();
        let debug = format!("{handoff:?}");
        assert!(debug.contains("Handoff"));
        assert!(debug.contains("billing"));
        assert!(debug.contains("transfer_to_billing"));
        assert!(debug.contains("<callback>"));
    }

    #[test]
    fn debug_impl_builder() {
        let builder: HandoffBuilder<()> = Handoff::to_agent("billing");
        let debug = format!("{builder:?}");
        assert!(debug.contains("HandoffBuilder"));
        assert!(debug.contains("billing"));
    }

    // ---- Builder disabled ----

    #[test]
    fn builder_disabled() {
        let handoff: Handoff<()> = Handoff::to_agent("billing").is_enabled(false).build();
        assert!(!handoff.is_enabled);
    }

    // ---- Builder strict_json_schema false ----

    #[test]
    fn builder_non_strict_schema() {
        let handoff: Handoff<()> = Handoff::to_agent("billing")
            .strict_json_schema(false)
            .build();
        assert!(!handoff.strict_json_schema);
        // With non-strict, the empty schema stays as-is.
        assert_eq!(handoff.input_json_schema, serde_json::json!({}));
    }

    // ---- Send + Sync assertions ----

    #[test]
    fn handoff_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Handoff<()>>();
        assert_send_sync::<Handoff<String>>();
    }

    #[test]
    fn handoff_input_data_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HandoffInputData>();
    }
}
