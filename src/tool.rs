//! Tool abstractions for agent execution.
//!
//! This module defines the tool types that agents can use during a run. The main
//! types are:
//!
//! - [`FunctionTool`] — wraps a user-provided async function as an LLM-callable tool.
//! - [`Tool`] — enum covering all tool types (function, web search, file search, code interpreter).
//! - [`ToolContext`] — context passed to tool functions during invocation.
//! - [`FunctionToolResult`] — result from executing a function tool.
//!
//! The [`function_tool`] convenience function is the primary ergonomic API for creating
//! tools from async functions whose input types implement `DeserializeOwned + JsonSchema`.
//!
//! This module mirrors the Python SDK's `tool.py`.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::context::RunContextWrapper;
use crate::error::{AgentError, Result};
use crate::items::ToolOutput;
use crate::schema::{ensure_strict_json_schema, json_schema_for};

/// Type alias for the boxed future returned by tool invoke functions.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Type alias for the invoke function stored inside a [`FunctionTool`].
type InvokeFn<C> =
    Arc<dyn Fn(ToolContext<C>, String) -> BoxFuture<'static, Result<ToolOutput>> + Send + Sync>;

// ---------------------------------------------------------------------------
// ToolContext
// ---------------------------------------------------------------------------

/// Context available to a tool function during execution.
///
/// Provides access to the user-provided run context, the tool name, and the
/// tool call ID from the model response. This is the Rust equivalent of the
/// Python SDK's `ToolContext`.
#[non_exhaustive]
pub struct ToolContext<C: Send + Sync + 'static> {
    /// The user-provided context, wrapped in an `Arc<RwLock>` for shared async access.
    pub context: Arc<tokio::sync::RwLock<RunContextWrapper<C>>>,
    /// Name of the tool being invoked.
    pub tool_name: String,
    /// The tool call ID from the model response.
    pub tool_call_id: String,
}

impl<C: Send + Sync + 'static> ToolContext<C> {
    /// Create a new tool context.
    #[must_use]
    pub fn new(
        context: Arc<tokio::sync::RwLock<RunContextWrapper<C>>>,
        tool_name: impl Into<String>,
        tool_call_id: impl Into<String>,
    ) -> Self {
        Self {
            context,
            tool_name: tool_name.into(),
            tool_call_id: tool_call_id.into(),
        }
    }
}

impl<C: Send + Sync + 'static> Clone for ToolContext<C> {
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            tool_name: self.tool_name.clone(),
            tool_call_id: self.tool_call_id.clone(),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for ToolContext<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolContext")
            .field("tool_name", &self.tool_name)
            .field("tool_call_id", &self.tool_call_id)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// ToolOrigin — metadata describing where a function-tool-backed item came from
// ---------------------------------------------------------------------------

/// Enumerates the runtime source of a function-tool-backed run item.
///
/// Mirrors the Python SDK's `ToolOriginType`. The value is attached to
/// [`ToolCallItem`](crate::items::ToolCallItem) and
/// [`ToolCallOutputItem`](crate::items::ToolCallOutputItem) through the
/// [`ToolOrigin`] struct so downstream consumers can tell whether a tool call
/// originated from a plain function tool, an MCP server tool, or an
/// agent-as-tool wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolOriginType {
    /// The tool is a plain function tool.
    Function,
    /// The tool is backed by an MCP server.
    Mcp,
    /// The tool is an agent-as-tool wrapper.
    AgentAsTool,
}

impl ToolOriginType {
    /// Return the canonical wire string for this origin type.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Mcp => "mcp",
            Self::AgentAsTool => "agent_as_tool",
        }
    }
}

/// Serializable metadata describing where a function-tool-backed item came from.
///
/// Attached to tool call and tool output run items so callers can trace each
/// item back to the specific agent, MCP server, or tool wrapper that produced
/// it. Mirrors the Python SDK's `ToolOrigin` dataclass.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolOrigin {
    /// The origin type of the tool.
    #[serde(rename = "type")]
    pub origin_type: ToolOriginType,
    /// Name of the MCP server, when `origin_type` is `Mcp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_server_name: Option<String>,
    /// Name of the wrapped agent, when `origin_type` is `AgentAsTool`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    /// Exposed tool name for the agent-as-tool, when `origin_type` is `AgentAsTool`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_tool_name: Option<String>,
}

impl ToolOrigin {
    /// Create a new origin marker for a plain function tool.
    #[must_use]
    pub const fn function() -> Self {
        Self {
            origin_type: ToolOriginType::Function,
            mcp_server_name: None,
            agent_name: None,
            agent_tool_name: None,
        }
    }

    /// Create a new origin marker for an MCP-backed tool.
    #[must_use]
    pub fn mcp(server_name: impl Into<String>) -> Self {
        Self {
            origin_type: ToolOriginType::Mcp,
            mcp_server_name: Some(server_name.into()),
            agent_name: None,
            agent_tool_name: None,
        }
    }

    /// Create a new origin marker for an agent-as-tool wrapper.
    #[must_use]
    pub fn agent_as_tool(
        agent_name: impl Into<String>,
        agent_tool_name: impl Into<String>,
    ) -> Self {
        Self {
            origin_type: ToolOriginType::AgentAsTool,
            mcp_server_name: None,
            agent_name: Some(agent_name.into()),
            agent_tool_name: Some(agent_tool_name.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// FunctionTool
// ---------------------------------------------------------------------------

/// A function tool wraps a user-provided async function as an LLM-callable tool.
///
/// The tool exposes a name, description, and JSON schema for its parameters to the
/// model. When the model calls the tool, the `invoke_fn` is called with a
/// [`ToolContext`] and the raw JSON arguments string.
///
/// Use the [`function_tool`] convenience function to create instances from typed
/// async functions.
pub struct FunctionTool<C: Send + Sync + 'static = ()> {
    /// The tool's name as exposed to the LLM.
    pub name: String,
    /// Description of what the tool does.
    pub description: String,
    /// JSON schema for the tool's parameters.
    pub params_json_schema: serde_json::Value,
    /// Whether to use strict JSON schema mode.
    pub strict_json_schema: bool,
    /// Whether the tool is currently enabled.
    pub is_enabled: bool,
    /// Whether the tool requires user approval before execution.
    pub needs_approval: bool,
    /// Optional origin metadata attached to tool call and tool output items.
    ///
    /// When set, the run loop copies this value onto
    /// [`ToolCallItem::tool_origin`](crate::items::ToolCallItem) and
    /// [`ToolCallOutputItem::tool_origin`](crate::items::ToolCallOutputItem)
    /// so callers can see whether the tool originated from an agent-as-tool
    /// wrapper, an MCP server, or a plain function.
    pub tool_origin: Option<ToolOrigin>,
    /// The invoke function, stored as a type-erased async closure.
    invoke_fn: InvokeFn<C>,
}

impl<C: Send + Sync + 'static> Clone for FunctionTool<C> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            params_json_schema: self.params_json_schema.clone(),
            strict_json_schema: self.strict_json_schema,
            is_enabled: self.is_enabled,
            needs_approval: self.needs_approval,
            tool_origin: self.tool_origin.clone(),
            invoke_fn: Arc::clone(&self.invoke_fn),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for FunctionTool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("params_json_schema", &self.params_json_schema)
            .field("strict_json_schema", &self.strict_json_schema)
            .field("is_enabled", &self.is_enabled)
            .field("needs_approval", &self.needs_approval)
            .field("tool_origin", &self.tool_origin)
            .finish_non_exhaustive()
    }
}

impl<C: Send + Sync + 'static> FunctionTool<C> {
    /// Invoke the tool with the given context and raw JSON arguments.
    ///
    /// Delegates to the inner invoke function, which deserializes the JSON string
    /// into the tool's parameter type and calls the user-provided async function.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::ModelBehavior`] if the JSON arguments cannot be deserialized
    /// into the expected parameter type, or propagates any error from the user function.
    pub async fn invoke(&self, ctx: ToolContext<C>, args_json: String) -> Result<ToolOutput> {
        (self.invoke_fn)(ctx, args_json).await
    }

    /// Create a `FunctionTool` that wraps an MCP server tool.
    ///
    /// This is used internally by [`MCPServer::list_tools`](crate::mcp::MCPServer::list_tools)
    /// to convert MCP tool definitions into SDK tools. The resulting tool has a
    /// no-op invoke function because MCP tool calls are dispatched through
    /// [`MCPServer::call_tool`](crate::mcp::MCPServer::call_tool) instead.
    #[must_use]
    pub(crate) fn mcp_tool(
        name: String,
        description: String,
        input_schema: serde_json::Value,
    ) -> Self {
        let tool_name = name.clone();
        let invoke_fn: InvokeFn<C> = Arc::new(move |_ctx, _args_json| {
            let name = tool_name.clone();
            Box::pin(async move {
                // MCP tools are not invoked through this function. The runner
                // dispatches MCP tool calls via MCPServer::call_tool directly.
                Err(AgentError::UserError {
                    message: format!(
                        "MCP tool '{name}' cannot be invoked directly. \
                         Use MCPServer::call_tool instead."
                    ),
                })
            })
        });

        Self {
            name,
            description,
            params_json_schema: input_schema,
            strict_json_schema: false,
            is_enabled: true,
            needs_approval: false,
            tool_origin: Some(ToolOrigin::function()),
            invoke_fn,
        }
    }

    /// Create an MCP-tool variant of [`Self::mcp_tool`] that records the
    /// originating server name on [`Self::tool_origin`].
    #[must_use]
    pub(crate) fn mcp_tool_with_server(
        name: String,
        description: String,
        input_schema: serde_json::Value,
        server_name: impl Into<String>,
    ) -> Self {
        let mut tool = Self::mcp_tool(name, description, input_schema);
        tool.tool_origin = Some(ToolOrigin::mcp(server_name));
        tool
    }
}

// ---------------------------------------------------------------------------
// Hosted tool structs
// ---------------------------------------------------------------------------

/// Web search hosted tool configuration.
///
/// Corresponds to the `OpenAI` Responses API web search tool. When added to an
/// agent, the model can perform web searches during execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WebSearchTool {
    /// User location hint for search result relevance.
    pub user_location: Option<serde_json::Value>,
    /// Controls the amount of search context retrieved. Common values are
    /// `"low"`, `"medium"`, and `"high"`.
    pub search_context_size: Option<String>,
    /// Whether the web search tool may fetch live internet content.
    ///
    /// When `None`, the API default is used. Set to `Some(false)` to request
    /// cached or indexed-only behaviour where supported.
    pub external_web_access: Option<bool>,
}

/// File search hosted tool configuration.
///
/// Corresponds to the `OpenAI` Responses API file search tool. When added to an
/// agent, the model can search over files in the specified vector stores.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FileSearchTool {
    /// IDs of the vector stores to search over.
    pub vector_store_ids: Vec<String>,
    /// Maximum number of search results to return.
    pub max_num_results: Option<u32>,
}

/// Code interpreter hosted tool configuration.
///
/// Corresponds to the `OpenAI` Responses API code interpreter tool. When added
/// to an agent, the model can execute code in a sandboxed environment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CodeInterpreterTool {
    /// Optional container identifier for the code execution environment.
    pub container: Option<String>,
}

/// Tool search hosted tool configuration.
///
/// Corresponds to the `OpenAI` Responses API tool search tool. When added to
/// an agent, the model can search deferred tools by namespace for dynamic
/// tool discovery.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolSearchTool {
    /// Optional description for the tool search.
    pub description: Option<String>,
    /// Execution mode: `"server"` or `"client"`.
    pub execution: Option<String>,
    /// Optional parameters for the tool search.
    pub parameters: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Tool enum
// ---------------------------------------------------------------------------

/// Enum covering all tool types available to an agent.
///
/// This is a closed set of variants representing function tools (user-defined) and
/// hosted tools (provided by the `OpenAI` platform).
#[non_exhaustive]
pub enum Tool<C: Send + Sync + 'static = ()> {
    /// A user-defined function tool.
    Function(FunctionTool<C>),
    /// Web search hosted tool.
    WebSearch(WebSearchTool),
    /// File search hosted tool.
    FileSearch(FileSearchTool),
    /// Code interpreter hosted tool.
    CodeInterpreter(CodeInterpreterTool),
    /// Computer control hosted tool.
    Computer(crate::computer::ComputerTool),
    /// Apply-patch hosted tool for file editing.
    ApplyPatch(crate::editor::ApplyPatchTool),
    /// Tool search hosted tool for dynamic tool discovery.
    ToolSearch(ToolSearchTool),
}

impl<C: Send + Sync + 'static> Clone for Tool<C> {
    fn clone(&self) -> Self {
        match self {
            Self::Function(f) => Self::Function(f.clone()),
            Self::WebSearch(w) => Self::WebSearch(w.clone()),
            Self::FileSearch(f) => Self::FileSearch(f.clone()),
            Self::CodeInterpreter(c) => Self::CodeInterpreter(c.clone()),
            Self::Computer(c) => Self::Computer(c.clone()),
            Self::ApplyPatch(a) => Self::ApplyPatch(a.clone()),
            Self::ToolSearch(t) => Self::ToolSearch(t.clone()),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for Tool<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function(ft) => f.debug_tuple("Function").field(ft).finish(),
            Self::WebSearch(w) => f.debug_tuple("WebSearch").field(w).finish(),
            Self::FileSearch(fs) => f.debug_tuple("FileSearch").field(fs).finish(),
            Self::CodeInterpreter(c) => f.debug_tuple("CodeInterpreter").field(c).finish(),
            Self::Computer(c) => f.debug_tuple("Computer").field(c).finish(),
            Self::ApplyPatch(a) => f.debug_tuple("ApplyPatch").field(a).finish(),
            Self::ToolSearch(t) => f.debug_tuple("ToolSearch").field(t).finish(),
        }
    }
}

impl<C: Send + Sync + 'static> Tool<C> {
    /// Get the tool's name.
    ///
    /// For function tools, returns the user-specified name. For hosted tools,
    /// returns a fixed identifier (`"web_search"`, `"file_search"`, or `"code_interpreter"`).
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Function(f) => &f.name,
            Self::WebSearch(_) => "web_search",
            Self::FileSearch(_) => "file_search",
            Self::CodeInterpreter(_) => "code_interpreter",
            Self::Computer(_) => "computer",
            Self::ApplyPatch(_) => "apply_patch",
            Self::ToolSearch(_) => "tool_search",
        }
    }

    /// Get the tool's description.
    ///
    /// For function tools, returns the user-specified description. For hosted tools,
    /// returns a built-in description.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Function(f) => &f.description,
            Self::WebSearch(_) => "Search the web for information.",
            Self::FileSearch(_) => "Search over files in vector stores.",
            Self::CodeInterpreter(_) => "Execute code in a sandboxed environment.",
            Self::Computer(_) => "Control a computer environment.",
            Self::ApplyPatch(_) => "Apply patches to files.",
            Self::ToolSearch(_) => "Search for deferred tools by namespace.",
        }
    }

    /// Whether this is a hosted tool (not a function tool).
    ///
    /// Hosted tools are provided by the `OpenAI` platform and do not have a
    /// user-defined invoke function.
    #[must_use]
    pub const fn is_hosted(&self) -> bool {
        !matches!(self, Self::Function(_))
    }
}

// ---------------------------------------------------------------------------
// FunctionToolResult
// ---------------------------------------------------------------------------

/// Result from executing a function tool.
///
/// Captures the tool name, call ID, and the output produced by invocation.
/// Used internally by the runner to collect tool execution results.
#[derive(Debug)]
#[non_exhaustive]
pub struct FunctionToolResult {
    /// The name of the tool that was executed.
    pub tool_name: String,
    /// The tool call ID from the model response.
    pub tool_call_id: String,
    /// The output produced by the tool invocation.
    pub output: ToolOutput,
}

impl FunctionToolResult {
    /// Create a new function tool result.
    #[must_use]
    pub fn new(
        tool_name: impl Into<String>,
        tool_call_id: impl Into<String>,
        output: ToolOutput,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            tool_call_id: tool_call_id.into(),
            output,
        }
    }
}

// ---------------------------------------------------------------------------
// function_tool convenience constructor
// ---------------------------------------------------------------------------

/// Create a [`FunctionTool`] from an async function whose input type implements
/// `DeserializeOwned + JsonSchema`.
///
/// The function's input type `T` is automatically used to generate the JSON schema.
/// At invocation time, the raw JSON arguments from the LLM are deserialized into `T`
/// before calling the user function.
///
/// # Errors
///
/// Returns [`AgentError::UserError`] if the generated JSON schema cannot be converted
/// to strict mode (for example, if it contains `additionalProperties: true`).
///
/// # Example
///
/// ```
/// use openai_agents::tool::{function_tool, ToolContext};
/// use openai_agents::items::ToolOutput;
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// #[derive(Deserialize, JsonSchema)]
/// struct WeatherParams {
///     city: String,
/// }
///
/// let tool = function_tool::<(), WeatherParams, _, _>(
///     "get_weather",
///     "Get the weather for a city.",
///     |_ctx: ToolContext<()>, params: WeatherParams| async move {
///         Ok(ToolOutput::Text(format!("Sunny in {}", params.city)))
///     },
/// ).unwrap();
///
/// assert_eq!(tool.name, "get_weather");
/// ```
pub fn function_tool<C, T, F, Fut>(
    name: impl Into<String>,
    description: impl Into<String>,
    func: F,
) -> Result<FunctionTool<C>>
where
    C: Send + Sync + 'static,
    T: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static,
    F: Fn(ToolContext<C>, T) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput>> + Send + 'static,
{
    let raw_schema = json_schema_for::<T>();
    let strict_schema = ensure_strict_json_schema(raw_schema)?;

    let func = Arc::new(func);

    let invoke_fn: InvokeFn<C> = Arc::new(move |ctx, args_json| {
        let func = Arc::clone(&func);
        Box::pin(async move {
            let parsed: T =
                serde_json::from_str(&args_json).map_err(|e| AgentError::ModelBehavior {
                    message: format!("failed to deserialize tool arguments: {e}"),
                })?;
            func(ctx, parsed).await
        })
    });

    Ok(FunctionTool {
        name: name.into(),
        description: description.into(),
        params_json_schema: strict_schema,
        strict_json_schema: true,
        is_enabled: true,
        needs_approval: false,
        tool_origin: Some(ToolOrigin::function()),
        invoke_fn,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct AddParams {
        a: i32,
        b: i32,
    }

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct GreetParams {
        name: String,
    }

    // ---- function_tool creates correct schema ----

    #[test]
    fn function_tool_creates_correct_schema() {
        let tool = function_tool::<(), AddParams, _, _>(
            "add",
            "Add two numbers.",
            |_ctx: ToolContext<()>, _params: AddParams| async move {
                Ok(ToolOutput::Text("result".to_owned()))
            },
        )
        .expect("should create tool");

        assert_eq!(tool.name, "add");
        assert_eq!(tool.description, "Add two numbers.");
        assert!(tool.strict_json_schema);
        assert!(tool.is_enabled);
        assert!(!tool.needs_approval);

        // Schema should have strict mode properties.
        assert_eq!(tool.params_json_schema["strict"], true);
        assert_eq!(tool.params_json_schema["additionalProperties"], false);

        // Schema should contain properties for a and b.
        let props = tool.params_json_schema["properties"]
            .as_object()
            .expect("should have properties");
        assert!(props.contains_key("a"));
        assert!(props.contains_key("b"));
    }

    // ---- FunctionTool invocation ----

    #[tokio::test]
    async fn function_tool_invocation() {
        let tool = function_tool::<(), AddParams, _, _>(
            "add",
            "Add two numbers.",
            |_ctx: ToolContext<()>, params: AddParams| async move {
                Ok(ToolOutput::Text(format!("{}", params.a + params.b)))
            },
        )
        .expect("should create tool");

        let ctx = ToolContext {
            context: Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(()))),
            tool_name: "add".to_owned(),
            tool_call_id: "call_123".to_owned(),
        };

        let result = tool
            .invoke(ctx, r#"{"a": 3, "b": 4}"#.to_owned())
            .await
            .expect("invocation should succeed");

        assert_eq!(result, ToolOutput::Text("7".to_owned()));
    }

    #[tokio::test]
    async fn function_tool_invocation_with_bad_json() {
        let tool = function_tool::<(), AddParams, _, _>(
            "add",
            "Add two numbers.",
            |_ctx: ToolContext<()>, _params: AddParams| async move {
                Ok(ToolOutput::Text("unreachable".to_owned()))
            },
        )
        .expect("should create tool");

        let ctx = ToolContext {
            context: Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(()))),
            tool_name: "add".to_owned(),
            tool_call_id: "call_456".to_owned(),
        };

        let result = tool.invoke(ctx, "not valid json".to_owned()).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentError::ModelBehavior { .. }),
            "expected ModelBehavior error, got: {err}"
        );
    }

    // ---- Tool enum name() and description() ----

    #[test]
    fn tool_function_name_and_description() {
        let ft = function_tool::<(), GreetParams, _, _>(
            "greet",
            "Greet someone.",
            |_ctx: ToolContext<()>, _params: GreetParams| async move {
                Ok(ToolOutput::Text("hi".to_owned()))
            },
        )
        .expect("should create tool");

        let tool = Tool::Function(ft);
        assert_eq!(tool.name(), "greet");
        assert_eq!(tool.description(), "Greet someone.");
        assert!(!tool.is_hosted());
    }

    #[test]
    fn tool_web_search_name_and_description() {
        let tool: Tool<()> = Tool::WebSearch(WebSearchTool::default());
        assert_eq!(tool.name(), "web_search");
        assert!(!tool.description().is_empty());
        assert!(tool.is_hosted());
    }

    #[test]
    fn tool_file_search_name_and_description() {
        let tool: Tool<()> = Tool::FileSearch(FileSearchTool::default());
        assert_eq!(tool.name(), "file_search");
        assert!(!tool.description().is_empty());
        assert!(tool.is_hosted());
    }

    #[test]
    fn tool_code_interpreter_name_and_description() {
        let tool: Tool<()> = Tool::CodeInterpreter(CodeInterpreterTool::default());
        assert_eq!(tool.name(), "code_interpreter");
        assert!(!tool.description().is_empty());
        assert!(tool.is_hosted());
    }

    // ---- FunctionTool Clone and Debug ----

    #[test]
    fn function_tool_clone() {
        let tool = function_tool::<(), GreetParams, _, _>(
            "greet",
            "Greet someone.",
            |_ctx: ToolContext<()>, _params: GreetParams| async move {
                Ok(ToolOutput::Text("hi".to_owned()))
            },
        )
        .expect("should create tool");

        let cloned = tool.clone();
        assert_eq!(cloned.name, tool.name);
        assert_eq!(cloned.description, tool.description);
        assert_eq!(cloned.strict_json_schema, tool.strict_json_schema);
        assert_eq!(cloned.is_enabled, tool.is_enabled);
        assert_eq!(cloned.needs_approval, tool.needs_approval);
    }

    #[test]
    fn function_tool_debug() {
        let tool = function_tool::<(), GreetParams, _, _>(
            "greet",
            "Greet someone.",
            |_ctx: ToolContext<()>, _params: GreetParams| async move {
                Ok(ToolOutput::Text("hi".to_owned()))
            },
        )
        .expect("should create tool");

        let debug_str = format!("{tool:?}");
        assert!(debug_str.contains("greet"));
        assert!(debug_str.contains("Greet someone."));
        // invoke_fn should not appear in debug output.
        assert!(!debug_str.contains("invoke_fn"));
    }

    // ---- Hosted tool Default impls ----

    #[test]
    fn web_search_tool_default() {
        let tool = WebSearchTool::default();
        assert!(tool.user_location.is_none());
        assert!(tool.search_context_size.is_none());
        assert!(tool.external_web_access.is_none());
    }

    #[test]
    fn file_search_tool_default() {
        let tool = FileSearchTool::default();
        assert!(tool.vector_store_ids.is_empty());
        assert!(tool.max_num_results.is_none());
    }

    #[test]
    fn code_interpreter_tool_default() {
        let tool = CodeInterpreterTool::default();
        assert!(tool.container.is_none());
    }

    // ---- Send + Sync assertions ----

    #[test]
    fn tool_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tool<()>>();
        assert_send_sync::<Tool<String>>();
        assert_send_sync::<FunctionTool<()>>();
        assert_send_sync::<ToolContext<()>>();
        assert_send_sync::<FunctionToolResult>();
    }

    // ---- ToolContext Debug ----

    #[test]
    fn tool_context_debug() {
        let ctx = ToolContext::<()> {
            context: Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(()))),
            tool_name: "test_tool".to_owned(),
            tool_call_id: "call_abc".to_owned(),
        };
        let debug_str = format!("{ctx:?}");
        assert!(debug_str.contains("test_tool"));
        assert!(debug_str.contains("call_abc"));
    }

    // ---- Tool enum Clone and Debug ----

    #[test]
    fn tool_enum_clone_and_debug() {
        let tool: Tool<()> = Tool::WebSearch(WebSearchTool {
            user_location: None,
            search_context_size: Some("medium".to_owned()),
            external_web_access: None,
        });
        let cloned = tool.clone();
        assert_eq!(cloned.name(), "web_search");

        let debug_str = format!("{tool:?}");
        assert!(debug_str.contains("WebSearch"));
        assert!(debug_str.contains("medium"));
    }

    // ---- FunctionToolResult construction ----

    #[test]
    fn function_tool_result_fields() {
        let result = FunctionToolResult {
            tool_name: "my_tool".to_owned(),
            tool_call_id: "call_xyz".to_owned(),
            output: ToolOutput::Text("done".to_owned()),
        };
        assert_eq!(result.tool_name, "my_tool");
        assert_eq!(result.tool_call_id, "call_xyz");
        assert_eq!(result.output, ToolOutput::Text("done".to_owned()));
    }

    // ---- Invocation with ToolContext accessing run context ----

    #[tokio::test]
    async fn function_tool_accesses_context() {
        let tool = function_tool::<String, GreetParams, _, _>(
            "greet",
            "Greet someone using context.",
            |ctx: ToolContext<String>, params: GreetParams| async move {
                let greeting = ctx.context.read().await.context.clone();
                Ok(ToolOutput::Text(format!("{greeting}, {}!", params.name)))
            },
        )
        .expect("should create tool");

        let ctx = ToolContext {
            context: Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(
                "Hello".to_owned(),
            ))),
            tool_name: "greet".to_owned(),
            tool_call_id: "call_ctx".to_owned(),
        };

        let result = tool
            .invoke(ctx, r#"{"name": "World"}"#.to_owned())
            .await
            .expect("invocation should succeed");

        assert_eq!(result, ToolOutput::Text("Hello, World!".to_owned()));
    }

    // ---- Computer tool variant ----

    #[test]
    fn tool_computer_name_and_description() {
        let tool: Tool<()> = Tool::Computer(crate::computer::ComputerTool::default());
        assert_eq!(tool.name(), "computer");
        assert_eq!(tool.description(), "Control a computer environment.");
        assert!(tool.is_hosted());
    }

    #[test]
    fn tool_computer_clone_and_debug() {
        let tool: Tool<()> = Tool::Computer(crate::computer::ComputerTool::new(800, 600));
        let cloned = tool.clone();
        assert_eq!(cloned.name(), "computer");

        let debug_str = format!("{tool:?}");
        assert!(debug_str.contains("Computer"));
        assert!(debug_str.contains("800"));
    }

    // ---- ApplyPatch tool variant ----

    #[test]
    fn tool_apply_patch_name_and_description() {
        let tool: Tool<()> = Tool::ApplyPatch(crate::editor::ApplyPatchTool);
        assert_eq!(tool.name(), "apply_patch");
        assert_eq!(tool.description(), "Apply patches to files.");
        assert!(tool.is_hosted());
    }

    #[test]
    fn tool_apply_patch_clone_and_debug() {
        let tool: Tool<()> = Tool::ApplyPatch(crate::editor::ApplyPatchTool);
        let cloned = tool.clone();
        assert_eq!(cloned.name(), "apply_patch");

        let debug_str = format!("{tool:?}");
        assert!(debug_str.contains("ApplyPatch"));
    }

    // ---- ToolSearch tool variant ----

    #[test]
    fn tool_search_default() {
        let tool = ToolSearchTool::default();
        assert!(tool.description.is_none());
        assert!(tool.execution.is_none());
        assert!(tool.parameters.is_none());
    }

    #[test]
    fn tool_search_name_and_description() {
        let tool: Tool<()> = Tool::ToolSearch(ToolSearchTool::default());
        assert_eq!(tool.name(), "tool_search");
        assert_eq!(
            tool.description(),
            "Search for deferred tools by namespace."
        );
        assert!(tool.is_hosted());
    }

    #[test]
    fn tool_search_clone_and_debug() {
        let tool: Tool<()> = Tool::ToolSearch(ToolSearchTool {
            description: Some("search tools".to_owned()),
            execution: Some("server".to_owned()),
            parameters: None,
        });
        let cloned = tool.clone();
        assert_eq!(cloned.name(), "tool_search");

        let debug_str = format!("{tool:?}");
        assert!(debug_str.contains("ToolSearch"));
        assert!(debug_str.contains("search tools"));
    }
}
