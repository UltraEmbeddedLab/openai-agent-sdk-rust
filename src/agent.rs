//! Agent definition and builder for configuring AI agents.
//!
//! This module provides [`Agent`], the core type representing an AI model configured
//! with instructions, tools, guardrails, handoffs, and more. Agents are constructed
//! using the [`AgentBuilder`], which provides an ergonomic fluent API.
//!
//! The generic parameter `C` is the user-provided context type that flows through
//! tools, guardrails, and hooks during execution. It defaults to `()` for agents
//! that do not need a custom context.
//!
//! This module mirrors the Python SDK's `agent.py`.
//!
//! # Example
//!
//! ```
//! use openai_agents::agent::Agent;
//!
//! let agent = Agent::<()>::builder("my-assistant")
//!     .instructions("You are a helpful assistant.")
//!     .build();
//!
//! assert_eq!(agent.name, "my-assistant");
//! ```

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::config::{ModelRef, ModelSettings};
use crate::context::RunContextWrapper;
use crate::error::Result;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::handoffs::Handoff;
use crate::lifecycle::AgentHooks;
use crate::tool::Tool;

/// A boxed, pinned, `Send` future used for async closures in agent instructions.
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Type alias for the dynamic instructions callback.
type DynamicInstructionsFn<C> = dyn for<'a> Fn(&'a RunContextWrapper<C>, &'a Agent<C>) -> BoxFuture<'a, Result<String>>
    + Send
    + Sync;

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

/// How instructions are provided to the agent.
///
/// Instructions can be a static string known at build time, or a dynamic async
/// function that generates instructions at runtime from the current context and agent.
#[non_exhaustive]
pub enum Instructions<C: Send + Sync + 'static> {
    /// Static string instructions (system prompt).
    Static(String),
    /// Dynamic instructions generated at runtime.
    ///
    /// The closure receives the run context wrapper and a reference to the agent,
    /// and returns a future resolving to the instruction string.
    Dynamic(Arc<DynamicInstructionsFn<C>>),
}

impl<C: Send + Sync + 'static> Clone for Instructions<C> {
    fn clone(&self) -> Self {
        match self {
            Self::Static(s) => Self::Static(s.clone()),
            Self::Dynamic(f) => Self::Dynamic(Arc::clone(f)),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for Instructions<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(s) => f.debug_tuple("Static").field(s).finish(),
            Self::Dynamic(_) => f.debug_tuple("Dynamic").field(&"<closure>").finish(),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolUseBehavior
// ---------------------------------------------------------------------------

/// Controls how tool call results are handled by the agent.
///
/// By default, after tool calls complete the LLM is called again with the results.
/// Alternative behaviors allow stopping after the first tool call or stopping only
/// when specific tools are called.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolUseBehavior {
    /// After tool calls complete, call the LLM again with the results (default).
    #[default]
    RunLlmAgain,
    /// Stop after the first tool call and use the result as the final output.
    StopOnFirstTool,
    /// Stop only if specific tools are called; otherwise run the LLM again.
    ///
    /// The vector contains the names of tools that should cause the agent to stop.
    StopAtTools(Vec<String>),
}

// ---------------------------------------------------------------------------
// OutputSchema
// ---------------------------------------------------------------------------

/// Schema for the agent's structured output.
///
/// When an agent has an output schema, the model is instructed to produce output
/// conforming to the given JSON schema. This enables deterministic structured output.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OutputSchema {
    /// The JSON schema for the output format.
    pub json_schema: serde_json::Value,
    /// Whether to enforce strict schema validation.
    pub strict: bool,
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// An agent is an AI model configured with instructions, tools, guardrails, handoffs,
/// and more.
///
/// The generic parameter `C` is the user-provided context type that flows through
/// tools, guardrails, and hooks. Defaults to `()` for agents that don't need context.
///
/// Agents are constructed via the [`AgentBuilder`] returned by [`Agent::builder`].
///
/// # Example
///
/// ```
/// use openai_agents::agent::Agent;
///
/// let agent = Agent::<()>::builder("assistant")
///     .instructions("You are a helpful assistant.")
///     .model("gpt-4o")
///     .build();
///
/// assert_eq!(agent.name, "assistant");
/// ```
pub struct Agent<C: Send + Sync + 'static = ()> {
    /// The agent's name, used for identification in logs and handoffs.
    pub name: String,
    /// Instructions (system prompt) for the agent.
    pub instructions: Option<Instructions<C>>,
    /// Description used when this agent is a handoff target.
    pub handoff_description: Option<String>,
    /// Model override for this agent.
    pub model: Option<ModelRef>,
    /// Model settings (temperature, etc.) for this agent.
    pub model_settings: ModelSettings,
    /// Tools available to this agent.
    pub tools: Vec<Tool<C>>,
    /// Handoffs this agent can delegate to.
    pub handoffs: Vec<Handoff<C>>,
    /// Input guardrails run before or during agent execution.
    pub input_guardrails: Vec<InputGuardrail<C>>,
    /// Output guardrails run after the agent produces output.
    pub output_guardrails: Vec<OutputGuardrail<C>>,
    /// Optional structured output schema.
    pub output_type: Option<OutputSchema>,
    /// Per-agent lifecycle hooks.
    pub hooks: Option<Box<dyn AgentHooks<C>>>,
    /// How to handle tool call results.
    pub tool_use_behavior: ToolUseBehavior,
    /// Whether to reset `tool_choice` to the default after the first LLM call.
    pub reset_tool_choice: bool,
}

impl<C: Send + Sync + 'static> Agent<C> {
    /// Create a new [`AgentBuilder`] with the given name.
    ///
    /// This is the primary API for constructing agents. All fields except `name`
    /// have sensible defaults.
    ///
    /// # Example
    ///
    /// ```
    /// use openai_agents::agent::Agent;
    ///
    /// let agent = Agent::<()>::builder("greeter")
    ///     .instructions("Greet the user warmly.")
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder(name: impl Into<String>) -> AgentBuilder<C> {
        AgentBuilder::new(name.into())
    }

    /// Resolve the instructions for this agent, evaluating dynamic instructions if needed.
    ///
    /// Returns `Ok(None)` if no instructions are set, `Ok(Some(string))` for static
    /// instructions, or the result of evaluating the dynamic closure.
    ///
    /// # Errors
    ///
    /// Returns an error if the dynamic instructions closure fails.
    pub async fn get_instructions(&self, ctx: &RunContextWrapper<C>) -> Result<Option<String>> {
        match &self.instructions {
            None => Ok(None),
            Some(Instructions::Static(s)) => Ok(Some(s.clone())),
            Some(Instructions::Dynamic(f)) => Ok(Some(f(ctx, self).await?)),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for Agent<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.name)
            .field("instructions", &self.instructions)
            .field("handoff_description", &self.handoff_description)
            .field("model", &self.model)
            .field("model_settings", &self.model_settings)
            .field("tools", &self.tools)
            .field("handoffs", &self.handoffs)
            .field("input_guardrails", &self.input_guardrails)
            .field("output_guardrails", &self.output_guardrails)
            .field("output_type", &self.output_type)
            .field("hooks", &self.hooks.as_ref().map(|_| "<hooks>"))
            .field("tool_use_behavior", &self.tool_use_behavior)
            .field("reset_tool_choice", &self.reset_tool_choice)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// AgentBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing [`Agent<C>`] instances.
///
/// Created via [`Agent::builder`]. All fields except `name` have sensible defaults.
/// Call [`build`](AgentBuilder::build) to produce the final [`Agent`].
pub struct AgentBuilder<C: Send + Sync + 'static> {
    name: String,
    instructions: Option<Instructions<C>>,
    handoff_description: Option<String>,
    model: Option<ModelRef>,
    model_settings: ModelSettings,
    tools: Vec<Tool<C>>,
    handoffs: Vec<Handoff<C>>,
    input_guardrails: Vec<InputGuardrail<C>>,
    output_guardrails: Vec<OutputGuardrail<C>>,
    output_type: Option<OutputSchema>,
    hooks: Option<Box<dyn AgentHooks<C>>>,
    tool_use_behavior: ToolUseBehavior,
    reset_tool_choice: bool,
}

impl<C: Send + Sync + 'static> AgentBuilder<C> {
    fn new(name: String) -> Self {
        Self {
            name,
            instructions: None,
            handoff_description: None,
            model: None,
            model_settings: ModelSettings::default(),
            tools: Vec::new(),
            handoffs: Vec::new(),
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            output_type: None,
            hooks: None,
            tool_use_behavior: ToolUseBehavior::default(),
            reset_tool_choice: true,
        }
    }

    /// Set the agent's name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set static instructions (system prompt) for the agent.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(Instructions::Static(instructions.into()));
        self
    }

    /// Set dynamic instructions that are generated at runtime.
    ///
    /// The closure receives the run context and a reference to the agent, and must
    /// return a boxed future resolving to the instruction string.
    ///
    /// # Example
    ///
    /// ```
    /// use openai_agents::agent::Agent;
    ///
    /// let agent = Agent::<String>::builder("dynamic-agent")
    ///     .dynamic_instructions(|ctx, _agent| {
    ///         let lang = ctx.context.clone();
    ///         Box::pin(async move { Ok(format!("Respond in {lang}.")) })
    ///     })
    ///     .build();
    /// ```
    #[must_use]
    pub fn dynamic_instructions<F>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&'a RunContextWrapper<C>, &'a Agent<C>) -> BoxFuture<'a, Result<String>>
            + Send
            + Sync
            + 'static,
    {
        self.instructions = Some(Instructions::Dynamic(Arc::new(f)));
        self
    }

    /// Set the handoff description used when this agent is a handoff target.
    #[must_use]
    pub fn handoff_description(mut self, desc: impl Into<String>) -> Self {
        self.handoff_description = Some(desc.into());
        self
    }

    /// Set the model override for this agent.
    #[must_use]
    pub fn model(mut self, model: impl Into<ModelRef>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the model settings (temperature, etc.) for this agent.
    #[must_use]
    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = settings;
        self
    }

    /// Add a single tool to the agent.
    #[must_use]
    pub fn tool(mut self, tool: Tool<C>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set all tools for the agent, replacing any previously added tools.
    #[must_use]
    pub fn tools(mut self, tools: Vec<Tool<C>>) -> Self {
        self.tools = tools;
        self
    }

    /// Add a single handoff to the agent.
    #[must_use]
    pub fn handoff(mut self, handoff: Handoff<C>) -> Self {
        self.handoffs.push(handoff);
        self
    }

    /// Set all handoffs for the agent, replacing any previously added handoffs.
    #[must_use]
    pub fn handoffs(mut self, handoffs: Vec<Handoff<C>>) -> Self {
        self.handoffs = handoffs;
        self
    }

    /// Add an input guardrail to the agent.
    #[must_use]
    pub fn input_guardrail(mut self, guardrail: InputGuardrail<C>) -> Self {
        self.input_guardrails.push(guardrail);
        self
    }

    /// Add an output guardrail to the agent.
    #[must_use]
    pub fn output_guardrail(mut self, guardrail: OutputGuardrail<C>) -> Self {
        self.output_guardrails.push(guardrail);
        self
    }

    /// Set the output type by generating a JSON schema from a type that implements
    /// [`schemars::JsonSchema`].
    ///
    /// This enables structured output mode with strict schema validation.
    ///
    /// # Example
    ///
    /// ```
    /// use openai_agents::agent::Agent;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct MyOutput {
    ///     answer: String,
    ///     confidence: f64,
    /// }
    ///
    /// let agent = Agent::<()>::builder("structured-agent")
    ///     .output_type::<MyOutput>()
    ///     .build();
    ///
    /// assert!(agent.output_type.is_some());
    /// ```
    #[must_use]
    pub fn output_type<T: schemars::JsonSchema>(mut self) -> Self {
        let schema = crate::schema::json_schema_for::<T>();
        // Best-effort strict enforcement; fall back to the raw schema on error.
        let strict_schema =
            crate::schema::ensure_strict_json_schema(schema.clone()).unwrap_or(schema);
        self.output_type = Some(OutputSchema {
            json_schema: strict_schema,
            strict: true,
        });
        self
    }

    /// Set the output schema directly.
    #[must_use]
    pub fn output_schema(mut self, schema: OutputSchema) -> Self {
        self.output_type = Some(schema);
        self
    }

    /// Set per-agent lifecycle hooks.
    #[must_use]
    pub fn hooks(mut self, hooks: impl AgentHooks<C> + 'static) -> Self {
        self.hooks = Some(Box::new(hooks));
        self
    }

    /// Set the tool use behavior for this agent.
    #[must_use]
    pub fn tool_use_behavior(mut self, behavior: ToolUseBehavior) -> Self {
        self.tool_use_behavior = behavior;
        self
    }

    /// Set whether to reset `tool_choice` to the default after the first LLM call.
    ///
    /// Defaults to `true`.
    #[must_use]
    pub const fn reset_tool_choice(mut self, reset: bool) -> Self {
        self.reset_tool_choice = reset;
        self
    }

    /// Build the [`Agent`], consuming this builder.
    #[must_use]
    pub fn build(self) -> Agent<C> {
        Agent {
            name: self.name,
            instructions: self.instructions,
            handoff_description: self.handoff_description,
            model: self.model,
            model_settings: self.model_settings,
            tools: self.tools,
            handoffs: self.handoffs,
            input_guardrails: self.input_guardrails,
            output_guardrails: self.output_guardrails,
            output_type: self.output_type,
            hooks: self.hooks,
            tool_use_behavior: self.tool_use_behavior,
            reset_tool_choice: self.reset_tool_choice,
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for AgentBuilder<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("name", &self.name)
            .field("instructions", &self.instructions)
            .field("handoff_description", &self.handoff_description)
            .field("model", &self.model)
            .field("model_settings", &self.model_settings)
            .field("tools", &self.tools)
            .field("handoffs", &self.handoffs)
            .field("input_guardrails", &self.input_guardrails)
            .field("output_guardrails", &self.output_guardrails)
            .field("output_type", &self.output_type)
            .field("hooks", &self.hooks.as_ref().map(|_| "<hooks>"))
            .field("tool_use_behavior", &self.tool_use_behavior)
            .field("reset_tool_choice", &self.reset_tool_choice)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guardrail::GuardrailFunctionOutput;

    // ---- Minimal agent (just a name) ----

    #[test]
    fn builder_minimal_agent() {
        let agent = Agent::<()>::builder("minimal").build();
        assert_eq!(agent.name, "minimal");
        assert!(agent.instructions.is_none());
        assert!(agent.handoff_description.is_none());
        assert!(agent.model.is_none());
        assert!(agent.tools.is_empty());
        assert!(agent.handoffs.is_empty());
        assert!(agent.input_guardrails.is_empty());
        assert!(agent.output_guardrails.is_empty());
        assert!(agent.output_type.is_none());
        assert!(agent.hooks.is_none());
        assert_eq!(agent.tool_use_behavior, ToolUseBehavior::RunLlmAgain);
        assert!(agent.reset_tool_choice);
    }

    // ---- Builder defaults ----

    #[test]
    fn builder_defaults() {
        let agent = Agent::<()>::builder("test").build();
        assert!(agent.tools.is_empty());
        assert!(agent.handoffs.is_empty());
        assert!(agent.input_guardrails.is_empty());
        assert!(agent.output_guardrails.is_empty());
        assert_eq!(agent.tool_use_behavior, ToolUseBehavior::RunLlmAgain);
        assert!(agent.reset_tool_choice);
    }

    // ---- Builder with all fields ----

    #[test]
    fn builder_with_all_fields() {
        use crate::guardrail::{InputGuardrail, OutputGuardrail};
        use crate::handoffs::Handoff;
        use crate::tool::{Tool, WebSearchTool};

        struct TestHooks;
        #[async_trait::async_trait]
        impl AgentHooks<()> for TestHooks {}

        let agent = Agent::<()>::builder("full-agent")
            .instructions("Be helpful.")
            .handoff_description("A fully configured agent.")
            .model("gpt-4o")
            .model_settings(ModelSettings {
                temperature: Some(0.7),
                ..Default::default()
            })
            .tool(Tool::WebSearch(WebSearchTool::default()))
            .handoff(Handoff::to_agent("other").build())
            .input_guardrail(InputGuardrail::new("ig", |_ctx, _agent, _input| {
                Box::pin(async { Ok(GuardrailFunctionOutput::passed(serde_json::json!(null))) })
            }))
            .output_guardrail(OutputGuardrail::new("og", |_ctx, _agent, _output| {
                Box::pin(async { Ok(GuardrailFunctionOutput::passed(serde_json::json!(null))) })
            }))
            .hooks(TestHooks)
            .tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .reset_tool_choice(false)
            .build();

        assert_eq!(agent.name, "full-agent");
        assert!(agent.instructions.is_some());
        assert_eq!(
            agent.handoff_description.as_deref(),
            Some("A fully configured agent.")
        );
        assert!(agent.model.is_some());
        assert_eq!(agent.model_settings.temperature, Some(0.7));
        assert_eq!(agent.tools.len(), 1);
        assert_eq!(agent.handoffs.len(), 1);
        assert_eq!(agent.input_guardrails.len(), 1);
        assert_eq!(agent.output_guardrails.len(), 1);
        assert!(agent.hooks.is_some());
        assert_eq!(agent.tool_use_behavior, ToolUseBehavior::StopOnFirstTool);
        assert!(!agent.reset_tool_choice);
    }

    // ---- get_instructions with static instructions ----

    #[tokio::test]
    async fn get_instructions_static() {
        let agent = Agent::<()>::builder("test")
            .instructions("Hello, world!")
            .build();
        let ctx = RunContextWrapper::new(());
        let result = agent.get_instructions(&ctx).await.unwrap();
        assert_eq!(result, Some("Hello, world!".to_owned()));
    }

    // ---- get_instructions with dynamic instructions ----

    #[tokio::test]
    async fn get_instructions_dynamic() {
        let agent = Agent::<String>::builder("test")
            .dynamic_instructions(|ctx, agent| {
                let lang = ctx.context.clone();
                let name = agent.name.clone();
                Box::pin(async move { Ok(format!("Agent {name}: respond in {lang}.")) })
            })
            .build();
        let ctx = RunContextWrapper::new("French".to_owned());
        let result = agent.get_instructions(&ctx).await.unwrap();
        assert_eq!(result, Some("Agent test: respond in French.".to_owned()));
    }

    // ---- get_instructions with None ----

    #[tokio::test]
    async fn get_instructions_none() {
        let agent = Agent::<()>::builder("test").build();
        let ctx = RunContextWrapper::new(());
        let result = agent.get_instructions(&ctx).await.unwrap();
        assert!(result.is_none());
    }

    // ---- output_type generates correct schema ----

    #[test]
    fn output_type_generates_schema() {
        use schemars::JsonSchema;
        use serde::Deserialize;

        #[derive(Deserialize, JsonSchema)]
        #[allow(dead_code)]
        struct TestOutput {
            answer: String,
            score: f64,
        }

        let agent = Agent::<()>::builder("test")
            .output_type::<TestOutput>()
            .build();

        let schema = agent.output_type.expect("should have output_type");
        assert!(schema.strict);

        // Schema should contain the properties.
        let props = schema.json_schema["properties"]
            .as_object()
            .expect("should have properties");
        assert!(props.contains_key("answer"));
        assert!(props.contains_key("score"));

        // Strict mode properties.
        assert_eq!(schema.json_schema["additionalProperties"], false);
    }

    // ---- ToolUseBehavior variants ----

    #[test]
    fn tool_use_behavior_default_is_run_llm_again() {
        assert_eq!(ToolUseBehavior::default(), ToolUseBehavior::RunLlmAgain);
    }

    #[test]
    fn tool_use_behavior_stop_on_first_tool() {
        let behavior = ToolUseBehavior::StopOnFirstTool;
        assert_eq!(behavior, ToolUseBehavior::StopOnFirstTool);
    }

    #[test]
    fn tool_use_behavior_stop_at_tools() {
        let behavior =
            ToolUseBehavior::StopAtTools(vec!["get_weather".to_owned(), "search".to_owned()]);
        if let ToolUseBehavior::StopAtTools(tools) = &behavior {
            assert_eq!(tools.len(), 2);
            assert_eq!(tools[0], "get_weather");
            assert_eq!(tools[1], "search");
        } else {
            panic!("expected StopAtTools variant");
        }
    }

    // ---- Debug impl ----

    #[test]
    fn agent_debug_impl() {
        let agent = Agent::<()>::builder("debug-test")
            .instructions("Be helpful.")
            .build();
        let debug_str = format!("{agent:?}");
        assert!(debug_str.contains("Agent"));
        assert!(debug_str.contains("debug-test"));
        assert!(debug_str.contains("Static"));
        assert!(debug_str.contains("Be helpful."));
    }

    #[test]
    fn agent_debug_with_dynamic_instructions() {
        let agent = Agent::<()>::builder("dynamic-debug")
            .dynamic_instructions(|_ctx, _agent| Box::pin(async { Ok("dynamic".to_owned()) }))
            .build();
        let debug_str = format!("{agent:?}");
        assert!(debug_str.contains("Dynamic"));
        assert!(debug_str.contains("<closure>"));
    }

    #[test]
    fn agent_debug_with_hooks() {
        struct TestHooks;
        #[async_trait::async_trait]
        impl AgentHooks<()> for TestHooks {}

        let agent = Agent::<()>::builder("hook-debug").hooks(TestHooks).build();
        let debug_str = format!("{agent:?}");
        assert!(debug_str.contains("<hooks>"));
    }

    #[test]
    fn builder_debug_impl() {
        let builder = Agent::<()>::builder("builder-debug").instructions("test");
        let debug_str = format!("{builder:?}");
        assert!(debug_str.contains("AgentBuilder"));
        assert!(debug_str.contains("builder-debug"));
    }

    // ---- Instructions Debug ----

    #[test]
    fn instructions_debug_static() {
        let instr: Instructions<()> = Instructions::Static("hello".to_owned());
        let debug_str = format!("{instr:?}");
        assert!(debug_str.contains("Static"));
        assert!(debug_str.contains("hello"));
    }

    #[test]
    fn instructions_debug_dynamic() {
        let instr: Instructions<()> = Instructions::Dynamic(Arc::new(|_ctx, _agent| {
            Box::pin(async { Ok("dynamic".to_owned()) })
        }));
        let debug_str = format!("{instr:?}");
        assert!(debug_str.contains("Dynamic"));
        assert!(debug_str.contains("<closure>"));
    }

    // ---- Agent is Send + Sync ----

    #[test]
    fn agent_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Agent<()>>();
        assert_sync::<Agent<()>>();
        assert_send::<Agent<String>>();
        assert_sync::<Agent<String>>();
    }

    // ---- AgentBuilder is Send + Sync ----

    #[test]
    fn agent_builder_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<AgentBuilder<()>>();
        assert_sync::<AgentBuilder<()>>();
    }

    // ---- Instructions is Send + Sync ----

    #[test]
    fn instructions_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Instructions<()>>();
        assert_send_sync::<Instructions<String>>();
    }

    // ---- Builder name override ----

    #[test]
    fn builder_name_can_be_overridden() {
        let agent = Agent::<()>::builder("original").name("overridden").build();
        assert_eq!(agent.name, "overridden");
    }

    // ---- Builder tools append vs replace ----

    #[test]
    fn builder_tool_appends() {
        use crate::tool::{FileSearchTool, Tool, WebSearchTool};

        let agent = Agent::<()>::builder("test")
            .tool(Tool::WebSearch(WebSearchTool::default()))
            .tool(Tool::FileSearch(FileSearchTool::default()))
            .build();
        assert_eq!(agent.tools.len(), 2);
    }

    #[test]
    fn builder_tools_replaces() {
        use crate::tool::{FileSearchTool, Tool, WebSearchTool};

        let agent = Agent::<()>::builder("test")
            .tool(Tool::WebSearch(WebSearchTool::default()))
            .tools(vec![Tool::FileSearch(FileSearchTool::default())])
            .build();
        assert_eq!(agent.tools.len(), 1);
        assert_eq!(agent.tools[0].name(), "file_search");
    }

    // ---- Builder handoffs append vs replace ----

    #[test]
    fn builder_handoff_appends() {
        use crate::handoffs::Handoff;

        let agent = Agent::<()>::builder("test")
            .handoff(Handoff::to_agent("a").build())
            .handoff(Handoff::to_agent("b").build())
            .build();
        assert_eq!(agent.handoffs.len(), 2);
    }

    #[test]
    fn builder_handoffs_replaces() {
        use crate::handoffs::Handoff;

        let agent = Agent::<()>::builder("test")
            .handoff(Handoff::to_agent("a").build())
            .handoffs(vec![Handoff::to_agent("b").build()])
            .build();
        assert_eq!(agent.handoffs.len(), 1);
        assert_eq!(agent.handoffs[0].agent_name, "b");
    }

    // ---- OutputSchema direct construction ----

    #[test]
    fn output_schema_direct() {
        let agent = Agent::<()>::builder("test")
            .output_schema(OutputSchema {
                json_schema: serde_json::json!({"type": "object"}),
                strict: false,
            })
            .build();
        let schema = agent.output_type.expect("should have output_type");
        assert!(!schema.strict);
        assert_eq!(schema.json_schema, serde_json::json!({"type": "object"}));
    }

    // ---- ToolUseBehavior Debug ----

    #[test]
    fn tool_use_behavior_debug() {
        let run_again = format!("{:?}", ToolUseBehavior::RunLlmAgain);
        assert!(run_again.contains("RunLlmAgain"));

        let stop_first = format!("{:?}", ToolUseBehavior::StopOnFirstTool);
        assert!(stop_first.contains("StopOnFirstTool"));

        let stop_at = format!(
            "{:?}",
            ToolUseBehavior::StopAtTools(vec!["search".to_owned()])
        );
        assert!(stop_at.contains("StopAtTools"));
        assert!(stop_at.contains("search"));
    }

    // ---- Clone for Instructions ----

    #[test]
    fn instructions_clone_static() {
        let instr: Instructions<()> = Instructions::Static("hello".to_owned());
        #[allow(clippy::redundant_clone)]
        let cloned = instr.clone();
        assert!(matches!(cloned, Instructions::Static(ref s) if s == "hello"));
    }

    #[test]
    fn instructions_clone_dynamic() {
        let instr: Instructions<()> = Instructions::Dynamic(Arc::new(|_ctx, _agent| {
            Box::pin(async { Ok("dynamic".to_owned()) })
        }));
        #[allow(clippy::redundant_clone)]
        let cloned = instr.clone();
        assert!(matches!(cloned, Instructions::Dynamic(_)));
    }
}
