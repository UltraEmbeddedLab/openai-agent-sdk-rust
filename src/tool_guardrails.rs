//! Tool-specific input and output guardrails.
//!
//! These guardrails validate individual tool calls, unlike the agent-level
//! [`InputGuardrail`](crate::guardrail::InputGuardrail) and
//! [`OutputGuardrail`](crate::guardrail::OutputGuardrail) which check
//! the overall agent input/output.
//!
//! Tool guardrails use a richer behavior model than agent guardrails:
//!
//! - [`GuardrailBehavior::Allow`] — let the tool call proceed normally.
//! - [`GuardrailBehavior::RejectContent`] — reject the tool call but continue
//!   the run, sending a message back to the model instead of the tool result.
//! - [`GuardrailBehavior::RaiseException`] — halt execution immediately with an error.
//!
//! This module mirrors the Python SDK's `tool_guardrails.py`.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::context::RunContextWrapper;
use crate::error::{AgentError, Result};

/// A boxed future that is `Send`, used internally for type-erased async guardrail functions.
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Type alias for the tool input guardrail closure signature.
///
/// Parameters: `(context, agent_name, tool_name, tool_arguments_json)`.
type ToolInputGuardrailFn<C> = dyn for<'a> Fn(
        &'a RunContextWrapper<C>,
        &'a str,
        &'a str,
        &'a str,
    ) -> BoxFuture<'a, Result<ToolGuardrailFunctionOutput>>
    + Send
    + Sync;

/// Type alias for the tool output guardrail closure signature.
///
/// Parameters: `(context, agent_name, tool_name, tool_output)`.
type ToolOutputGuardrailFn<C> = dyn for<'a> Fn(
        &'a RunContextWrapper<C>,
        &'a str,
        &'a str,
        &'a str,
    ) -> BoxFuture<'a, Result<ToolGuardrailFunctionOutput>>
    + Send
    + Sync;

// ---------------------------------------------------------------------------
// GuardrailBehavior
// ---------------------------------------------------------------------------

/// Defines how the system should respond when a tool guardrail result is processed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum GuardrailBehavior {
    /// Allow normal tool execution to continue without interference.
    #[default]
    Allow,
    /// Reject the tool call/output but continue execution with a message to the model.
    RejectContent {
        /// The message to send to the model instead of the tool result.
        message: String,
    },
    /// Halt execution by raising an exception.
    RaiseException,
}

// ---------------------------------------------------------------------------
// ToolGuardrailFunctionOutput
// ---------------------------------------------------------------------------

/// Output from a tool guardrail function.
///
/// Contains metadata about the guardrail check and a [`GuardrailBehavior`]
/// that determines how the system should react.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolGuardrailFunctionOutput {
    /// Custom metadata or information from the guardrail check.
    ///
    /// For example, the guardrail could include information about the checks
    /// it performed and granular results.
    pub output_info: serde_json::Value,

    /// Defines how the system should respond when this guardrail result is processed.
    pub behavior: GuardrailBehavior,
}

impl ToolGuardrailFunctionOutput {
    /// Create a guardrail output that allows the tool execution to continue normally.
    #[must_use]
    pub const fn allow(output_info: serde_json::Value) -> Self {
        Self {
            output_info,
            behavior: GuardrailBehavior::Allow,
        }
    }

    /// Create a guardrail output that rejects the tool call/output but continues execution.
    ///
    /// The provided `message` is sent to the model instead of the tool result.
    #[must_use]
    pub fn reject_content(message: impl Into<String>, output_info: serde_json::Value) -> Self {
        Self {
            output_info,
            behavior: GuardrailBehavior::RejectContent {
                message: message.into(),
            },
        }
    }

    /// Create a guardrail output that raises an exception to halt execution.
    #[must_use]
    pub const fn raise_exception(output_info: serde_json::Value) -> Self {
        Self {
            output_info,
            behavior: GuardrailBehavior::RaiseException,
        }
    }
}

// ---------------------------------------------------------------------------
// ToolInputGuardrailResult
// ---------------------------------------------------------------------------

/// Result of running a tool input guardrail.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolInputGuardrailResult {
    /// Name of the guardrail that was run.
    pub guardrail_name: String,

    /// Name of the tool whose input was checked.
    pub tool_name: String,

    /// Name of the agent that owns the tool.
    pub agent_name: String,

    /// The guardrail's output.
    pub output: ToolGuardrailFunctionOutput,
}

// ---------------------------------------------------------------------------
// ToolOutputGuardrailResult
// ---------------------------------------------------------------------------

/// Result of running a tool output guardrail.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolOutputGuardrailResult {
    /// Name of the guardrail that was run.
    pub guardrail_name: String,

    /// Name of the tool whose output was checked.
    pub tool_name: String,

    /// Name of the agent that owns the tool.
    pub agent_name: String,

    /// The tool's output that was checked.
    pub tool_output: String,

    /// The guardrail's output.
    pub output: ToolGuardrailFunctionOutput,
}

// ---------------------------------------------------------------------------
// ToolInputGuardrail
// ---------------------------------------------------------------------------

/// A guardrail that validates tool call arguments before execution.
///
/// Tool input guardrails run before a function tool is invoked. They receive the
/// tool's name, the agent name, and the raw JSON arguments string. Based on the
/// [`GuardrailBehavior`] returned:
///
/// - [`Allow`](GuardrailBehavior::Allow) — the tool call proceeds normally.
/// - [`RejectContent`](GuardrailBehavior::RejectContent) — the tool call is
///   skipped and the rejection message is sent to the model instead.
/// - [`RaiseException`](GuardrailBehavior::RaiseException) — the run is aborted
///   with [`AgentError::ToolInputGuardrailTripwire`].
pub struct ToolInputGuardrail<C: Send + Sync + 'static> {
    /// Name of this guardrail (used in error messages and results).
    pub name: String,

    /// The guardrail function, type-erased behind an `Arc` for cloneability.
    guardrail_fn: Arc<ToolInputGuardrailFn<C>>,
}

impl<C: Send + Sync + 'static> ToolInputGuardrail<C> {
    /// Create a new tool input guardrail with the given name and async function.
    ///
    /// The function receives the run context, agent name, tool name, and tool
    /// arguments (as a JSON string), and returns a [`ToolGuardrailFunctionOutput`].
    /// The caller must return a boxed future, e.g. using `Box::pin(async move { ... })`.
    ///
    /// # Example
    ///
    /// ```
    /// use openai_agents::tool_guardrails::{ToolInputGuardrail, ToolGuardrailFunctionOutput};
    /// use serde_json::json;
    ///
    /// let guardrail = ToolInputGuardrail::<()>::new(
    ///     "safe_args",
    ///     |_ctx, _agent, _tool, _args| {
    ///         Box::pin(async { Ok(ToolGuardrailFunctionOutput::allow(json!(null))) })
    ///     },
    /// );
    /// ```
    pub fn new(
        name: impl Into<String>,
        func: impl for<'a> Fn(
            &'a RunContextWrapper<C>,
            &'a str,
            &'a str,
            &'a str,
        ) -> BoxFuture<'a, Result<ToolGuardrailFunctionOutput>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            guardrail_fn: Arc::new(func),
        }
    }

    /// Run the guardrail against the given tool call.
    ///
    /// If the guardrail returns [`GuardrailBehavior::RaiseException`], this method
    /// returns [`Err(AgentError::ToolInputGuardrailTripwire)`](AgentError::ToolInputGuardrailTripwire).
    /// Otherwise, the result is returned for the runner to inspect.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::ToolInputGuardrailTripwire`] if the guardrail raises
    /// an exception. Also propagates any error returned by the guardrail function itself.
    pub async fn run(
        &self,
        ctx: &RunContextWrapper<C>,
        agent_name: &str,
        tool_name: &str,
        tool_arguments: &str,
    ) -> Result<ToolInputGuardrailResult> {
        let output = (self.guardrail_fn)(ctx, agent_name, tool_name, tool_arguments).await?;

        if output.behavior == GuardrailBehavior::RaiseException {
            return Err(AgentError::ToolInputGuardrailTripwire {
                guardrail_name: self.name.clone(),
                tool_name: tool_name.to_owned(),
            });
        }

        Ok(ToolInputGuardrailResult {
            guardrail_name: self.name.clone(),
            tool_name: tool_name.to_owned(),
            agent_name: agent_name.to_owned(),
            output,
        })
    }
}

impl<C: Send + Sync + 'static> Clone for ToolInputGuardrail<C> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            guardrail_fn: Arc::clone(&self.guardrail_fn),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for ToolInputGuardrail<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolInputGuardrail")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// ToolOutputGuardrail
// ---------------------------------------------------------------------------

/// A guardrail that validates tool output after execution.
///
/// Tool output guardrails run after a function tool returns. They receive the
/// tool's name, the agent name, and the tool's output string. Based on the
/// [`GuardrailBehavior`] returned:
///
/// - [`Allow`](GuardrailBehavior::Allow) — the tool output is used normally.
/// - [`RejectContent`](GuardrailBehavior::RejectContent) — the tool output is
///   replaced with the rejection message before being sent to the model.
/// - [`RaiseException`](GuardrailBehavior::RaiseException) — the run is aborted
///   with [`AgentError::ToolOutputGuardrailTripwire`].
pub struct ToolOutputGuardrail<C: Send + Sync + 'static> {
    /// Name of this guardrail (used in error messages and results).
    pub name: String,

    /// The guardrail function, type-erased behind an `Arc` for cloneability.
    guardrail_fn: Arc<ToolOutputGuardrailFn<C>>,
}

impl<C: Send + Sync + 'static> ToolOutputGuardrail<C> {
    /// Create a new tool output guardrail with the given name and async function.
    ///
    /// The function receives the run context, agent name, tool name, and tool
    /// output string, and returns a [`ToolGuardrailFunctionOutput`].
    /// The caller must return a boxed future, e.g. using `Box::pin(async move { ... })`.
    ///
    /// # Example
    ///
    /// ```
    /// use openai_agents::tool_guardrails::{ToolOutputGuardrail, ToolGuardrailFunctionOutput};
    /// use serde_json::json;
    ///
    /// let guardrail = ToolOutputGuardrail::<()>::new(
    ///     "safe_output",
    ///     |_ctx, _agent, _tool, _output| {
    ///         Box::pin(async { Ok(ToolGuardrailFunctionOutput::allow(json!(null))) })
    ///     },
    /// );
    /// ```
    pub fn new(
        name: impl Into<String>,
        func: impl for<'a> Fn(
            &'a RunContextWrapper<C>,
            &'a str,
            &'a str,
            &'a str,
        ) -> BoxFuture<'a, Result<ToolGuardrailFunctionOutput>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            guardrail_fn: Arc::new(func),
        }
    }

    /// Run the guardrail against the given tool output.
    ///
    /// If the guardrail returns [`GuardrailBehavior::RaiseException`], this method
    /// returns [`Err(AgentError::ToolOutputGuardrailTripwire)`](AgentError::ToolOutputGuardrailTripwire).
    /// Otherwise, the result is returned for the runner to inspect.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::ToolOutputGuardrailTripwire`] if the guardrail raises
    /// an exception. Also propagates any error returned by the guardrail function itself.
    pub async fn run(
        &self,
        ctx: &RunContextWrapper<C>,
        agent_name: &str,
        tool_name: &str,
        tool_output: &str,
    ) -> Result<ToolOutputGuardrailResult> {
        let output = (self.guardrail_fn)(ctx, agent_name, tool_name, tool_output).await?;

        if output.behavior == GuardrailBehavior::RaiseException {
            return Err(AgentError::ToolOutputGuardrailTripwire {
                guardrail_name: self.name.clone(),
                tool_name: tool_name.to_owned(),
            });
        }

        Ok(ToolOutputGuardrailResult {
            guardrail_name: self.name.clone(),
            tool_name: tool_name.to_owned(),
            agent_name: agent_name.to_owned(),
            tool_output: tool_output.to_owned(),
            output,
        })
    }
}

impl<C: Send + Sync + 'static> Clone for ToolOutputGuardrail<C> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            guardrail_fn: Arc::clone(&self.guardrail_fn),
        }
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for ToolOutputGuardrail<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolOutputGuardrail")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- ToolGuardrailFunctionOutput constructors ----

    #[test]
    fn output_allow() {
        let output = ToolGuardrailFunctionOutput::allow(json!({"check": "ok"}));
        assert_eq!(output.behavior, GuardrailBehavior::Allow);
        assert_eq!(output.output_info, json!({"check": "ok"}));
    }

    #[test]
    fn output_reject_content() {
        let output =
            ToolGuardrailFunctionOutput::reject_content("not allowed", json!("rejection info"));
        assert_eq!(
            output.behavior,
            GuardrailBehavior::RejectContent {
                message: "not allowed".to_owned()
            }
        );
        assert_eq!(output.output_info, json!("rejection info"));
    }

    #[test]
    fn output_raise_exception() {
        let output = ToolGuardrailFunctionOutput::raise_exception(json!("blocked"));
        assert_eq!(output.behavior, GuardrailBehavior::RaiseException);
        assert_eq!(output.output_info, json!("blocked"));
    }

    // ---- GuardrailBehavior default ----

    #[test]
    fn behavior_default_is_allow() {
        assert_eq!(GuardrailBehavior::default(), GuardrailBehavior::Allow);
    }

    // ---- Tool input guardrail that passes (Allow) ----

    #[tokio::test]
    async fn tool_input_guardrail_allows() {
        let guardrail = ToolInputGuardrail::<()>::new("safe_args", |_ctx, _agent, _tool, _args| {
            Box::pin(async { Ok(ToolGuardrailFunctionOutput::allow(json!("safe"))) })
        });

        let ctx = RunContextWrapper::new(());
        let result = guardrail
            .run(&ctx, "test_agent", "my_tool", r#"{"x": 1}"#)
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.guardrail_name, "safe_args");
        assert_eq!(result.tool_name, "my_tool");
        assert_eq!(result.agent_name, "test_agent");
        assert_eq!(result.output.behavior, GuardrailBehavior::Allow);
        assert_eq!(result.output.output_info, json!("safe"));
    }

    // ---- Tool input guardrail that rejects content ----

    #[tokio::test]
    async fn tool_input_guardrail_rejects_content() {
        let guardrail = ToolInputGuardrail::<()>::new("sql_check", |_ctx, _agent, _tool, _args| {
            Box::pin(async {
                Ok(ToolGuardrailFunctionOutput::reject_content(
                    "SQL injection detected",
                    json!("suspicious pattern"),
                ))
            })
        });

        let ctx = RunContextWrapper::new(());
        let result = guardrail
            .run(&ctx, "test_agent", "db_query", "DROP TABLE users")
            .await;

        // RejectContent should NOT return an error — it returns Ok with the behavior set.
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            result.output.behavior,
            GuardrailBehavior::RejectContent {
                message: "SQL injection detected".to_owned()
            }
        );
    }

    // ---- Tool input guardrail that raises exception ----

    #[tokio::test]
    async fn tool_input_guardrail_raises_exception() {
        let guardrail =
            ToolInputGuardrail::<()>::new("forbidden_check", |_ctx, _agent, _tool, _args| {
                Box::pin(async {
                    Ok(ToolGuardrailFunctionOutput::raise_exception(json!(
                        "forbidden operation"
                    )))
                })
            });

        let ctx = RunContextWrapper::new(());
        let result = guardrail
            .run(&ctx, "test_agent", "dangerous_tool", r"{}")
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                AgentError::ToolInputGuardrailTripwire {
                    ref guardrail_name,
                    ref tool_name,
                } if guardrail_name == "forbidden_check" && tool_name == "dangerous_tool"
            ),
            "expected ToolInputGuardrailTripwire, got: {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "tool input guardrail 'forbidden_check' triggered on tool 'dangerous_tool'"
        );
    }

    // ---- Tool output guardrail that passes (Allow) ----

    #[tokio::test]
    async fn tool_output_guardrail_allows() {
        let guardrail =
            ToolOutputGuardrail::<()>::new("pii_check", |_ctx, _agent, _tool, _output| {
                Box::pin(async { Ok(ToolGuardrailFunctionOutput::allow(json!("no PII found"))) })
            });

        let ctx = RunContextWrapper::new(());
        let result = guardrail
            .run(&ctx, "test_agent", "search_tool", "clean results here")
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.guardrail_name, "pii_check");
        assert_eq!(result.tool_name, "search_tool");
        assert_eq!(result.agent_name, "test_agent");
        assert_eq!(result.tool_output, "clean results here");
        assert_eq!(result.output.behavior, GuardrailBehavior::Allow);
    }

    // ---- Tool output guardrail that rejects content ----

    #[tokio::test]
    async fn tool_output_guardrail_rejects_content() {
        let guardrail =
            ToolOutputGuardrail::<()>::new("pii_filter", |_ctx, _agent, _tool, _output| {
                Box::pin(async {
                    Ok(ToolGuardrailFunctionOutput::reject_content(
                        "PII detected in output, redacted",
                        json!("SSN found"),
                    ))
                })
            });

        let ctx = RunContextWrapper::new(());
        let result = guardrail
            .run(&ctx, "test_agent", "search_tool", "SSN: 123-45-6789")
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            result.output.behavior,
            GuardrailBehavior::RejectContent {
                message: "PII detected in output, redacted".to_owned()
            }
        );
    }

    // ---- Tool output guardrail that raises exception ----

    #[tokio::test]
    async fn tool_output_guardrail_raises_exception() {
        let guardrail =
            ToolOutputGuardrail::<()>::new("malware_check", |_ctx, _agent, _tool, _output| {
                Box::pin(async {
                    Ok(ToolGuardrailFunctionOutput::raise_exception(json!(
                        "malicious content"
                    )))
                })
            });

        let ctx = RunContextWrapper::new(());
        let result = guardrail
            .run(&ctx, "test_agent", "exec_tool", "malicious output")
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                AgentError::ToolOutputGuardrailTripwire {
                    ref guardrail_name,
                    ref tool_name,
                } if guardrail_name == "malware_check" && tool_name == "exec_tool"
            ),
            "expected ToolOutputGuardrailTripwire, got: {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "tool output guardrail 'malware_check' triggered on tool 'exec_tool'"
        );
    }

    // ---- Debug impl ----

    #[test]
    fn tool_input_guardrail_debug() {
        let guardrail = ToolInputGuardrail::<()>::new("my_guard", |_ctx, _agent, _tool, _args| {
            Box::pin(async { Ok(ToolGuardrailFunctionOutput::allow(json!(null))) })
        });
        let debug_str = format!("{guardrail:?}");
        assert!(debug_str.contains("ToolInputGuardrail"));
        assert!(debug_str.contains("my_guard"));
    }

    #[test]
    fn tool_output_guardrail_debug() {
        let guardrail =
            ToolOutputGuardrail::<()>::new("my_output_guard", |_ctx, _agent, _tool, _output| {
                Box::pin(async { Ok(ToolGuardrailFunctionOutput::allow(json!(null))) })
            });
        let debug_str = format!("{guardrail:?}");
        assert!(debug_str.contains("ToolOutputGuardrail"));
        assert!(debug_str.contains("my_output_guard"));
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn tool_input_guardrail_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ToolInputGuardrail<()>>();
        assert_send_sync::<ToolInputGuardrail<String>>();
    }

    #[test]
    fn tool_output_guardrail_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ToolOutputGuardrail<()>>();
        assert_send_sync::<ToolOutputGuardrail<String>>();
    }

    // ---- Clone impl ----

    #[test]
    fn tool_input_guardrail_clone() {
        let guardrail = ToolInputGuardrail::<()>::new("cloneable", |_ctx, _agent, _tool, _args| {
            Box::pin(async { Ok(ToolGuardrailFunctionOutput::allow(json!(null))) })
        });
        let cloned = guardrail.clone();
        assert_eq!(cloned.name, "cloneable");
        // Verify original is still usable after clone.
        assert_eq!(guardrail.name, "cloneable");
    }

    #[test]
    fn tool_output_guardrail_clone() {
        let guardrail =
            ToolOutputGuardrail::<()>::new("cloneable", |_ctx, _agent, _tool, _output| {
                Box::pin(async { Ok(ToolGuardrailFunctionOutput::allow(json!(null))) })
            });
        let cloned = guardrail.clone();
        assert_eq!(cloned.name, "cloneable");
        // Verify original is still usable after clone.
        assert_eq!(guardrail.name, "cloneable");
    }

    // ---- Guardrail function receives correct arguments ----

    #[tokio::test]
    async fn tool_input_guardrail_receives_correct_args() {
        let guardrail =
            ToolInputGuardrail::<String>::new("arg_check", |ctx, agent_name, tool_name, args| {
                let context_val = ctx.context.clone();
                let agent = agent_name.to_owned();
                let tool = tool_name.to_owned();
                let arguments = args.to_owned();
                Box::pin(async move {
                    assert_eq!(context_val, "test_context");
                    assert_eq!(agent, "my_agent");
                    assert_eq!(tool, "my_tool");
                    assert_eq!(arguments, r#"{"key": "value"}"#);
                    Ok(ToolGuardrailFunctionOutput::allow(json!(null)))
                })
            });

        let ctx = RunContextWrapper::new("test_context".to_owned());
        let result = guardrail
            .run(&ctx, "my_agent", "my_tool", r#"{"key": "value"}"#)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn tool_output_guardrail_receives_correct_args() {
        let guardrail = ToolOutputGuardrail::<String>::new(
            "arg_check",
            |ctx, agent_name, tool_name, output| {
                let context_val = ctx.context.clone();
                let agent = agent_name.to_owned();
                let tool = tool_name.to_owned();
                let out = output.to_owned();
                Box::pin(async move {
                    assert_eq!(context_val, "test_context");
                    assert_eq!(agent, "my_agent");
                    assert_eq!(tool, "my_tool");
                    assert_eq!(out, "tool result data");
                    Ok(ToolGuardrailFunctionOutput::allow(json!(null)))
                })
            },
        );

        let ctx = RunContextWrapper::new("test_context".to_owned());
        let result = guardrail
            .run(&ctx, "my_agent", "my_tool", "tool result data")
            .await;
        assert!(result.is_ok());
    }

    // ---- Guardrail function error propagation ----

    #[tokio::test]
    async fn tool_input_guardrail_propagates_error() {
        let guardrail =
            ToolInputGuardrail::<()>::new("error_guard", |_ctx, _agent, _tool, _args| {
                Box::pin(async {
                    Err(AgentError::UserError {
                        message: "something went wrong".to_owned(),
                    })
                })
            });

        let ctx = RunContextWrapper::new(());
        let result = guardrail.run(&ctx, "test_agent", "my_tool", r"{}").await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::UserError { .. }));
    }

    #[tokio::test]
    async fn tool_output_guardrail_propagates_error() {
        let guardrail =
            ToolOutputGuardrail::<()>::new("error_guard", |_ctx, _agent, _tool, _output| {
                Box::pin(async {
                    Err(AgentError::UserError {
                        message: "validation failed".to_owned(),
                    })
                })
            });

        let ctx = RunContextWrapper::new(());
        let result = guardrail.run(&ctx, "test_agent", "my_tool", "output").await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::UserError { .. }));
    }

    // ---- Result types are Clone + Debug ----

    #[test]
    fn tool_input_guardrail_result_clone_and_debug() {
        let result = ToolInputGuardrailResult {
            guardrail_name: "test".to_owned(),
            tool_name: "my_tool".to_owned(),
            agent_name: "agent".to_owned(),
            output: ToolGuardrailFunctionOutput::allow(json!("ok")),
        };
        let cloned = result.clone();
        assert_eq!(cloned.guardrail_name, "test");
        assert_eq!(cloned.tool_name, "my_tool");
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("ToolInputGuardrailResult"));
    }

    #[test]
    fn tool_output_guardrail_result_clone_and_debug() {
        let result = ToolOutputGuardrailResult {
            guardrail_name: "test".to_owned(),
            tool_name: "my_tool".to_owned(),
            agent_name: "agent".to_owned(),
            tool_output: "output data".to_owned(),
            output: ToolGuardrailFunctionOutput::allow(json!("ok")),
        };
        let cloned = result.clone();
        assert_eq!(cloned.agent_name, "agent");
        assert_eq!(cloned.tool_output, "output data");
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("ToolOutputGuardrailResult"));
    }

    // ---- ToolGuardrailFunctionOutput serialization round-trip ----

    #[test]
    fn output_serialization_round_trip() {
        let output = ToolGuardrailFunctionOutput::reject_content(
            "blocked content",
            json!({"reason": "policy violation"}),
        );
        let serialized = serde_json::to_string(&output).expect("should serialize");
        let deserialized: ToolGuardrailFunctionOutput =
            serde_json::from_str(&serialized).expect("should deserialize");
        assert_eq!(deserialized.behavior, output.behavior);
        assert_eq!(deserialized.output_info, output.output_info);
    }

    // ---- GuardrailBehavior serialization ----

    #[test]
    fn behavior_serialization() {
        let allow = GuardrailBehavior::Allow;
        let serialized = serde_json::to_string(&allow).expect("should serialize");
        let deserialized: GuardrailBehavior =
            serde_json::from_str(&serialized).expect("should deserialize");
        assert_eq!(deserialized, GuardrailBehavior::Allow);

        let reject = GuardrailBehavior::RejectContent {
            message: "no".to_owned(),
        };
        let serialized = serde_json::to_string(&reject).expect("should serialize");
        let deserialized: GuardrailBehavior =
            serde_json::from_str(&serialized).expect("should deserialize");
        assert_eq!(deserialized, reject);

        let raise = GuardrailBehavior::RaiseException;
        let serialized = serde_json::to_string(&raise).expect("should serialize");
        let deserialized: GuardrailBehavior =
            serde_json::from_str(&serialized).expect("should deserialize");
        assert_eq!(deserialized, raise);
    }
}
