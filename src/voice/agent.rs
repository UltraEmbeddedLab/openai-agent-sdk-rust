//! Realtime agent for voice interactions.
//!
//! A [`RealtimeAgent`] is a specialized agent designed for use within a
//! [`RealtimeSession`](super::session::RealtimeSession) to build voice agents.
//! Unlike the standard [`Agent`](crate::Agent), some configuration options are
//! not supported because all realtime agents within a session share the same
//! model and model settings.
//!
//! This module mirrors the Python SDK's `realtime/agent.py`.

use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use crate::context::RunContextWrapper;
use crate::error::Result;
use crate::guardrail::OutputGuardrail;
use crate::handoffs::Handoff;
use crate::tool::Tool;

/// A boxed, pinned, `Send` future used for async closures.
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Type alias for the dynamic instructions callback on a realtime agent.
type DynamicInstructionsFn<C> = dyn for<'a> Fn(&'a RunContextWrapper<C>, &'a RealtimeAgent<C>) -> BoxFuture<'a, Result<String>>
    + Send
    + Sync;

// ---------------------------------------------------------------------------
// RealtimeInstructions
// ---------------------------------------------------------------------------

/// How instructions are provided to a realtime agent.
#[non_exhaustive]
pub enum RealtimeInstructions<C: Send + Sync + 'static> {
    /// Static string instructions (system prompt).
    Static(String),
    /// Dynamic instructions generated at runtime from the context and agent.
    Dynamic(Arc<DynamicInstructionsFn<C>>),
}

impl<C: Send + Sync + 'static> Clone for RealtimeInstructions<C> {
    fn clone(&self) -> Self {
        match self {
            Self::Static(s) => Self::Static(s.clone()),
            Self::Dynamic(f) => Self::Dynamic(Arc::clone(f)),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for RealtimeInstructions<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(s) => f.debug_tuple("Static").field(s).finish(),
            Self::Dynamic(_) => f.debug_tuple("Dynamic").field(&"<closure>").finish(),
        }
    }
}

// ---------------------------------------------------------------------------
// RealtimeAgent
// ---------------------------------------------------------------------------

/// A specialized agent for voice/realtime interaction.
///
/// Unlike the standard [`Agent`](crate::Agent), a `RealtimeAgent` communicates
/// via audio streams using the `OpenAI` Realtime API.  Because all agents within
/// a single realtime session share the same model and connection, `RealtimeAgent`
/// does not support per-agent model selection or model settings.
///
/// ## Differences from `Agent`
///
/// - **No `model` field** -- the model is chosen at the session level.
/// - **No `model_settings`** -- shared across all agents in a session.
/// - **No `output_type`** -- structured output is not supported in realtime mode.
/// - **No `tool_use_behavior`** -- tool calls are handled uniformly.
/// - **No `input_guardrails`** -- only output guardrails are supported.
///
/// # Example
///
/// ```
/// use openai_agents::voice::RealtimeAgent;
///
/// let agent = RealtimeAgent::<()>::builder("voice-assistant")
///     .instructions("You are a friendly voice assistant.")
///     .build();
///
/// assert_eq!(agent.name, "voice-assistant");
/// ```
pub struct RealtimeAgent<C: Send + Sync + 'static = ()> {
    /// The agent's name, used for identification in logs and handoffs.
    pub name: String,
    /// Instructions (system prompt) for the agent.
    pub instructions: Option<RealtimeInstructions<C>>,
    /// Description used when this agent is a handoff target.
    pub handoff_description: Option<String>,
    /// Handoffs this agent can delegate to.
    pub handoffs: Vec<Handoff<C>>,
    /// Tools available to this agent.
    pub tools: Vec<Tool<C>>,
    /// Output guardrails run after the agent produces output.
    pub output_guardrails: Vec<OutputGuardrail<C>>,
}

impl<C: Send + Sync + 'static> RealtimeAgent<C> {
    /// Create a new [`RealtimeAgentBuilder`] with the given name.
    ///
    /// This is the primary API for constructing realtime agents.
    ///
    /// # Example
    ///
    /// ```
    /// use openai_agents::voice::RealtimeAgent;
    ///
    /// let agent = RealtimeAgent::<()>::builder("my-agent")
    ///     .instructions("Be helpful.")
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder(name: impl Into<String>) -> RealtimeAgentBuilder<C> {
        RealtimeAgentBuilder::new(name.into())
    }

    /// Resolve the instructions for this agent, evaluating dynamic instructions
    /// if needed.
    ///
    /// Returns `Ok(None)` if no instructions are set, `Ok(Some(string))` for
    /// static instructions, or the result of evaluating the dynamic closure.
    ///
    /// # Errors
    ///
    /// Returns an error if the dynamic instructions closure fails.
    pub async fn get_system_prompt(&self, ctx: &RunContextWrapper<C>) -> Result<Option<String>> {
        match &self.instructions {
            None => Ok(None),
            Some(RealtimeInstructions::Static(s)) => Ok(Some(s.clone())),
            Some(RealtimeInstructions::Dynamic(f)) => Ok(Some(f(ctx, self).await?)),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for RealtimeAgent<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeAgent")
            .field("name", &self.name)
            .field("instructions", &self.instructions)
            .field("handoff_description", &self.handoff_description)
            .field("handoffs", &self.handoffs)
            .field("tools", &self.tools)
            .field("output_guardrails", &self.output_guardrails)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RealtimeAgentBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing [`RealtimeAgent<C>`] instances.
///
/// Created via [`RealtimeAgent::builder`].  All fields except `name` have
/// sensible defaults.
pub struct RealtimeAgentBuilder<C: Send + Sync + 'static> {
    name: String,
    instructions: Option<RealtimeInstructions<C>>,
    handoff_description: Option<String>,
    handoffs: Vec<Handoff<C>>,
    tools: Vec<Tool<C>>,
    output_guardrails: Vec<OutputGuardrail<C>>,
    _marker: PhantomData<C>,
}

impl<C: Send + Sync + 'static> RealtimeAgentBuilder<C> {
    const fn new(name: String) -> Self {
        Self {
            name,
            instructions: None,
            handoff_description: None,
            handoffs: Vec::new(),
            tools: Vec::new(),
            output_guardrails: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Set the agent's name, overriding the one provided to [`RealtimeAgent::builder`].
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set static instructions (system prompt) for the agent.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(RealtimeInstructions::Static(instructions.into()));
        self
    }

    /// Set dynamic instructions generated at runtime.
    ///
    /// The closure receives the run context and a reference to the agent, and
    /// must return a boxed future resolving to the instruction string.
    #[must_use]
    pub fn dynamic_instructions<F>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(
                &'a RunContextWrapper<C>,
                &'a RealtimeAgent<C>,
            ) -> BoxFuture<'a, Result<String>>
            + Send
            + Sync
            + 'static,
    {
        self.instructions = Some(RealtimeInstructions::Dynamic(Arc::new(f)));
        self
    }

    /// Set the handoff description used when this agent is a handoff target.
    #[must_use]
    pub fn handoff_description(mut self, desc: impl Into<String>) -> Self {
        self.handoff_description = Some(desc.into());
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

    /// Add an output guardrail to the agent.
    #[must_use]
    pub fn output_guardrail(mut self, guardrail: OutputGuardrail<C>) -> Self {
        self.output_guardrails.push(guardrail);
        self
    }

    /// Build the [`RealtimeAgent`], consuming this builder.
    #[must_use]
    pub fn build(self) -> RealtimeAgent<C> {
        RealtimeAgent {
            name: self.name,
            instructions: self.instructions,
            handoff_description: self.handoff_description,
            handoffs: self.handoffs,
            tools: self.tools,
            output_guardrails: self.output_guardrails,
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for RealtimeAgentBuilder<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealtimeAgentBuilder")
            .field("name", &self.name)
            .field("instructions", &self.instructions)
            .field("handoff_description", &self.handoff_description)
            .field("handoffs", &self.handoffs)
            .field("tools", &self.tools)
            .field("output_guardrails", &self.output_guardrails)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Builder minimal ----

    #[test]
    fn builder_minimal_agent() {
        let agent = RealtimeAgent::<()>::builder("voice-bot").build();
        assert_eq!(agent.name, "voice-bot");
        assert!(agent.instructions.is_none());
        assert!(agent.handoff_description.is_none());
        assert!(agent.handoffs.is_empty());
        assert!(agent.tools.is_empty());
        assert!(agent.output_guardrails.is_empty());
    }

    // ---- Builder with static instructions ----

    #[test]
    fn builder_with_static_instructions() {
        let agent = RealtimeAgent::<()>::builder("test")
            .instructions("Be helpful.")
            .build();
        assert!(matches!(
            agent.instructions,
            Some(RealtimeInstructions::Static(ref s)) if s == "Be helpful."
        ));
    }

    // ---- Builder with dynamic instructions ----

    #[test]
    fn builder_with_dynamic_instructions() {
        let agent = RealtimeAgent::<()>::builder("test")
            .dynamic_instructions(|_ctx, _agent| Box::pin(async { Ok("dynamic".to_owned()) }))
            .build();
        assert!(matches!(
            agent.instructions,
            Some(RealtimeInstructions::Dynamic(_))
        ));
    }

    // ---- get_system_prompt ----

    #[tokio::test]
    async fn get_system_prompt_static() {
        let agent = RealtimeAgent::<()>::builder("test")
            .instructions("Hello!")
            .build();
        let ctx = RunContextWrapper::new(());
        let result = agent.get_system_prompt(&ctx).await.unwrap();
        assert_eq!(result, Some("Hello!".to_owned()));
    }

    #[tokio::test]
    async fn get_system_prompt_dynamic() {
        let agent = RealtimeAgent::<String>::builder("test")
            .dynamic_instructions(|ctx, _agent| {
                let lang = ctx.context.clone();
                Box::pin(async move { Ok(format!("Speak in {lang}.")) })
            })
            .build();
        let ctx = RunContextWrapper::new("French".to_owned());
        let result = agent.get_system_prompt(&ctx).await.unwrap();
        assert_eq!(result, Some("Speak in French.".to_owned()));
    }

    #[tokio::test]
    async fn get_system_prompt_none() {
        let agent = RealtimeAgent::<()>::builder("test").build();
        let ctx = RunContextWrapper::new(());
        let result = agent.get_system_prompt(&ctx).await.unwrap();
        assert!(result.is_none());
    }

    // ---- Builder name override ----

    #[test]
    fn builder_name_can_be_overridden() {
        let agent = RealtimeAgent::<()>::builder("original")
            .name("overridden")
            .build();
        assert_eq!(agent.name, "overridden");
    }

    // ---- Builder handoff description ----

    #[test]
    fn builder_handoff_description() {
        let agent = RealtimeAgent::<()>::builder("test")
            .handoff_description("A voice agent for billing.")
            .build();
        assert_eq!(
            agent.handoff_description.as_deref(),
            Some("A voice agent for billing.")
        );
    }

    // ---- Builder tool append vs replace ----

    #[test]
    fn builder_tool_appends() {
        use crate::tool::{Tool, WebSearchTool};
        let agent = RealtimeAgent::<()>::builder("test")
            .tool(Tool::WebSearch(WebSearchTool::default()))
            .tool(Tool::WebSearch(WebSearchTool::default()))
            .build();
        assert_eq!(agent.tools.len(), 2);
    }

    #[test]
    fn builder_tools_replaces() {
        use crate::tool::{Tool, WebSearchTool};
        let agent = RealtimeAgent::<()>::builder("test")
            .tool(Tool::WebSearch(WebSearchTool::default()))
            .tools(vec![])
            .build();
        assert!(agent.tools.is_empty());
    }

    // ---- Builder handoff append vs replace ----

    #[test]
    fn builder_handoff_appends() {
        let agent = RealtimeAgent::<()>::builder("test")
            .handoff(Handoff::to_agent("a").build())
            .handoff(Handoff::to_agent("b").build())
            .build();
        assert_eq!(agent.handoffs.len(), 2);
    }

    #[test]
    fn builder_handoffs_replaces() {
        let agent = RealtimeAgent::<()>::builder("test")
            .handoff(Handoff::to_agent("a").build())
            .handoffs(vec![Handoff::to_agent("b").build()])
            .build();
        assert_eq!(agent.handoffs.len(), 1);
        assert_eq!(agent.handoffs[0].agent_name, "b");
    }

    // ---- Debug ----

    #[test]
    fn agent_debug_impl() {
        let agent = RealtimeAgent::<()>::builder("dbg-test")
            .instructions("Hello.")
            .build();
        let debug_str = format!("{agent:?}");
        assert!(debug_str.contains("RealtimeAgent"));
        assert!(debug_str.contains("dbg-test"));
        assert!(debug_str.contains("Static"));
    }

    #[test]
    fn agent_debug_dynamic() {
        let agent = RealtimeAgent::<()>::builder("test")
            .dynamic_instructions(|_ctx, _agent| Box::pin(async { Ok("d".to_owned()) }))
            .build();
        let debug_str = format!("{agent:?}");
        assert!(debug_str.contains("Dynamic"));
        assert!(debug_str.contains("<closure>"));
    }

    #[test]
    fn builder_debug_impl() {
        let builder = RealtimeAgent::<()>::builder("builder-dbg").instructions("test");
        let debug_str = format!("{builder:?}");
        assert!(debug_str.contains("RealtimeAgentBuilder"));
        assert!(debug_str.contains("builder-dbg"));
    }

    // ---- Send + Sync ----

    #[test]
    fn realtime_agent_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RealtimeAgent<()>>();
        assert_send_sync::<RealtimeAgent<String>>();
        assert_send_sync::<RealtimeAgentBuilder<()>>();
    }

    // ---- RealtimeInstructions Clone ----

    #[test]
    fn instructions_clone_static() {
        let instr: RealtimeInstructions<()> = RealtimeInstructions::Static("hello".to_owned());
        #[allow(clippy::redundant_clone)]
        let cloned = instr.clone();
        assert!(matches!(cloned, RealtimeInstructions::Static(ref s) if s == "hello"));
    }

    #[test]
    fn instructions_clone_dynamic() {
        let instr: RealtimeInstructions<()> =
            RealtimeInstructions::Dynamic(Arc::new(|_ctx, _agent| {
                Box::pin(async { Ok("d".to_owned()) })
            }));
        #[allow(clippy::redundant_clone)]
        let cloned = instr.clone();
        assert!(matches!(cloned, RealtimeInstructions::Dynamic(_)));
    }
}
