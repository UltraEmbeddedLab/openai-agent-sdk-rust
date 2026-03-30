//! Guardrails for validating agent inputs and outputs.
//!
//! This module provides [`InputGuardrail`] and [`OutputGuardrail`], which are checks
//! that run before, during, or after agent execution. Input guardrails validate the
//! user input (optionally in parallel with the LLM call), while output guardrails
//! validate the agent's final output.
//!
//! If a guardrail's function returns [`GuardrailFunctionOutput`] with
//! `tripwire_triggered: true`, the runner will abort the run with an
//! [`AgentError::InputGuardrailTripwire`] or [`AgentError::OutputGuardrailTripwire`]
//! error respectively.
//!
//! This module mirrors the Python SDK's `guardrail.py`.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::context::RunContextWrapper;
use crate::error::{AgentError, Result};
use crate::items::InputContent;

/// A boxed future that is `Send`, used internally for type-erased async guardrail functions.
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Type alias for the input guardrail closure signature.
type InputGuardrailFn<C> = dyn for<'a> Fn(
        &'a RunContextWrapper<C>,
        &'a str,
        &'a InputContent,
    ) -> BoxFuture<'a, Result<GuardrailFunctionOutput>>
    + Send
    + Sync;

/// Type alias for the output guardrail closure signature.
type OutputGuardrailFn<C> = dyn for<'a> Fn(
        &'a RunContextWrapper<C>,
        &'a str,
        &'a serde_json::Value,
    ) -> BoxFuture<'a, Result<GuardrailFunctionOutput>>
    + Send
    + Sync;

/// Output from a guardrail function.
///
/// Contains metadata about the guardrail check and whether the tripwire was triggered.
/// When `tripwire_triggered` is `true`, the agent run will be aborted.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GuardrailFunctionOutput {
    /// Custom metadata or information from the guardrail check.
    ///
    /// For example, the guardrail could include information about the checks it
    /// performed and granular results.
    pub output_info: serde_json::Value,

    /// Whether the guardrail's tripwire was triggered (i.e., the check failed).
    ///
    /// If `true`, the agent's execution will be halted immediately.
    pub tripwire_triggered: bool,
}

impl GuardrailFunctionOutput {
    /// Create a new guardrail function output.
    #[must_use]
    pub const fn new(output_info: serde_json::Value, tripwire_triggered: bool) -> Self {
        Self {
            output_info,
            tripwire_triggered,
        }
    }

    /// Create an output indicating the guardrail passed (tripwire not triggered).
    #[must_use]
    pub const fn passed(output_info: serde_json::Value) -> Self {
        Self {
            output_info,
            tripwire_triggered: false,
        }
    }

    /// Create an output indicating the guardrail failed (tripwire triggered).
    #[must_use]
    pub const fn tripwire(output_info: serde_json::Value) -> Self {
        Self {
            output_info,
            tripwire_triggered: true,
        }
    }
}

/// Result of running an input guardrail.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InputGuardrailResult {
    /// Name of the guardrail that was run.
    pub guardrail_name: String,

    /// The guardrail's output.
    pub output: GuardrailFunctionOutput,
}

/// Result of running an output guardrail.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OutputGuardrailResult {
    /// Name of the guardrail that was run.
    pub guardrail_name: String,

    /// Name of the agent whose output was checked.
    pub agent_name: String,

    /// The agent's output that was checked.
    pub agent_output: serde_json::Value,

    /// The guardrail's output.
    pub output: GuardrailFunctionOutput,
}

/// An input guardrail that checks input before or during agent execution.
///
/// Input guardrails are checks that run either in parallel with the agent or before
/// it starts. They can be used to:
/// - Check if input messages are off-topic.
/// - Take over control of the agent's execution if an unexpected input is detected.
///
/// If the guardrail function returns `tripwire_triggered: true`, the runner
/// will raise an [`AgentError::InputGuardrailTripwire`] error.
pub struct InputGuardrail<C: Send + Sync + 'static> {
    /// Name of this guardrail (used in error messages and results).
    pub name: String,

    /// Whether this guardrail can run in parallel with the LLM call.
    ///
    /// When `true` (the default), the guardrail runs concurrently with the agent.
    /// When `false`, the guardrail runs before the agent starts.
    pub run_in_parallel: bool,

    /// The guardrail function, type-erased behind an `Arc` for cloneability.
    guardrail_fn: Arc<InputGuardrailFn<C>>,
}

impl<C: Send + Sync + 'static> InputGuardrail<C> {
    /// Create a new input guardrail with the given name and async function.
    ///
    /// The function receives the run context, the agent name, and the input content,
    /// and returns a [`GuardrailFunctionOutput`]. The caller must return a boxed future,
    /// e.g. using `Box::pin(async move { ... })`.
    ///
    /// By default, `run_in_parallel` is `true`.
    ///
    /// # Example
    ///
    /// ```
    /// use openai_agents::guardrail::{InputGuardrail, GuardrailFunctionOutput};
    /// use serde_json::json;
    ///
    /// let guardrail = InputGuardrail::<()>::new("my_check", |_ctx, _agent, _input| {
    ///     Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
    /// });
    /// ```
    pub fn new(
        name: impl Into<String>,
        func: impl for<'a> Fn(
            &'a RunContextWrapper<C>,
            &'a str,
            &'a InputContent,
        ) -> BoxFuture<'a, Result<GuardrailFunctionOutput>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            run_in_parallel: true,
            guardrail_fn: Arc::new(func),
        }
    }

    /// Create an input guardrail that runs in parallel with the LLM call.
    ///
    /// This is equivalent to calling [`InputGuardrail::new`] since `run_in_parallel`
    /// defaults to `true`, but makes the intent explicit at the call site.
    pub fn parallel(
        name: impl Into<String>,
        func: impl for<'a> Fn(
            &'a RunContextWrapper<C>,
            &'a str,
            &'a InputContent,
        ) -> BoxFuture<'a, Result<GuardrailFunctionOutput>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        let mut guardrail = Self::new(name, func);
        guardrail.run_in_parallel = true;
        guardrail
    }

    /// Create an input guardrail that runs sequentially (before the LLM call).
    ///
    /// Sets `run_in_parallel` to `false`.
    pub fn sequential(
        name: impl Into<String>,
        func: impl for<'a> Fn(
            &'a RunContextWrapper<C>,
            &'a str,
            &'a InputContent,
        ) -> BoxFuture<'a, Result<GuardrailFunctionOutput>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        let mut guardrail = Self::new(name, func);
        guardrail.run_in_parallel = false;
        guardrail
    }

    /// Run the guardrail and return the result.
    ///
    /// If `tripwire_triggered` is `true` in the output, this returns
    /// [`Err(AgentError::InputGuardrailTripwire)`](AgentError::InputGuardrailTripwire).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::InputGuardrailTripwire`] if the guardrail's tripwire is
    /// triggered. Also propagates any error returned by the guardrail function itself.
    pub async fn run(
        &self,
        ctx: &RunContextWrapper<C>,
        agent_name: &str,
        input: &InputContent,
    ) -> Result<InputGuardrailResult> {
        let output = (self.guardrail_fn)(ctx, agent_name, input).await?;

        if output.tripwire_triggered {
            return Err(AgentError::InputGuardrailTripwire {
                guardrail_name: self.name.clone(),
            });
        }

        Ok(InputGuardrailResult {
            guardrail_name: self.name.clone(),
            output,
        })
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for InputGuardrail<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InputGuardrail")
            .field("name", &self.name)
            .field("run_in_parallel", &self.run_in_parallel)
            .finish_non_exhaustive()
    }
}

/// An output guardrail that validates the agent's final output.
///
/// Output guardrails are checks that run on the final output of an agent. They can
/// be used to check if the output passes certain validation criteria.
///
/// If the guardrail function returns `tripwire_triggered: true`, the runner
/// will raise an [`AgentError::OutputGuardrailTripwire`] error.
pub struct OutputGuardrail<C: Send + Sync + 'static> {
    /// Name of this guardrail (used in error messages and results).
    pub name: String,

    /// The guardrail function, type-erased behind an `Arc` for cloneability.
    guardrail_fn: Arc<OutputGuardrailFn<C>>,
}

impl<C: Send + Sync + 'static> OutputGuardrail<C> {
    /// Create a new output guardrail with the given name and async function.
    ///
    /// The function receives the run context, the agent name, and the agent's output
    /// (as a JSON value), and returns a [`GuardrailFunctionOutput`]. The caller must
    /// return a boxed future, e.g. using `Box::pin(async move { ... })`.
    ///
    /// # Example
    ///
    /// ```
    /// use openai_agents::guardrail::{OutputGuardrail, GuardrailFunctionOutput};
    /// use serde_json::json;
    ///
    /// let guardrail = OutputGuardrail::<()>::new("my_check", |_ctx, _agent, _output| {
    ///     Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
    /// });
    /// ```
    pub fn new(
        name: impl Into<String>,
        func: impl for<'a> Fn(
            &'a RunContextWrapper<C>,
            &'a str,
            &'a serde_json::Value,
        ) -> BoxFuture<'a, Result<GuardrailFunctionOutput>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            guardrail_fn: Arc::new(func),
        }
    }

    /// Run the guardrail and return the result.
    ///
    /// If `tripwire_triggered` is `true` in the output, this returns
    /// [`Err(AgentError::OutputGuardrailTripwire)`](AgentError::OutputGuardrailTripwire).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::OutputGuardrailTripwire`] if the guardrail's tripwire is
    /// triggered. Also propagates any error returned by the guardrail function itself.
    pub async fn run(
        &self,
        ctx: &RunContextWrapper<C>,
        agent_name: &str,
        agent_output: &serde_json::Value,
    ) -> Result<OutputGuardrailResult> {
        let output = (self.guardrail_fn)(ctx, agent_name, agent_output).await?;

        if output.tripwire_triggered {
            return Err(AgentError::OutputGuardrailTripwire {
                guardrail_name: self.name.clone(),
            });
        }

        Ok(OutputGuardrailResult {
            guardrail_name: self.name.clone(),
            agent_name: agent_name.to_owned(),
            agent_output: agent_output.clone(),
            output,
        })
    }
}

impl<C: Send + Sync + 'static> fmt::Debug for OutputGuardrail<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputGuardrail")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- GuardrailFunctionOutput constructors ----

    #[test]
    fn guardrail_function_output_new() {
        let output = GuardrailFunctionOutput::new(json!({"check": "ok"}), false);
        assert!(!output.tripwire_triggered);
        assert_eq!(output.output_info, json!({"check": "ok"}));
    }

    #[test]
    fn guardrail_function_output_passed() {
        let output = GuardrailFunctionOutput::passed(json!("all clear"));
        assert!(!output.tripwire_triggered);
        assert_eq!(output.output_info, json!("all clear"));
    }

    #[test]
    fn guardrail_function_output_tripwire() {
        let output = GuardrailFunctionOutput::tripwire(json!("blocked"));
        assert!(output.tripwire_triggered);
        assert_eq!(output.output_info, json!("blocked"));
    }

    // ---- Input guardrail that passes ----

    #[tokio::test]
    async fn input_guardrail_passes() {
        let guardrail = InputGuardrail::<()>::new("safe_check", |_ctx, _agent, _input| {
            Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!("safe"))) })
        });

        let ctx = RunContextWrapper::new(());
        let input = InputContent::Text("hello".to_owned());
        let result = guardrail.run(&ctx, "test_agent", &input).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.guardrail_name, "safe_check");
        assert!(!result.output.tripwire_triggered);
        assert_eq!(result.output.output_info, json!("safe"));
    }

    // ---- Input guardrail that triggers tripwire ----

    #[tokio::test]
    async fn input_guardrail_triggers_tripwire() {
        let guardrail = InputGuardrail::<()>::new("profanity_filter", |_ctx, _agent, _input| {
            Box::pin(async {
                Ok(GuardrailFunctionOutput::tripwire(json!(
                    "profanity detected"
                )))
            })
        });

        let ctx = RunContextWrapper::new(());
        let input = InputContent::Text("bad words".to_owned());
        let result = guardrail.run(&ctx, "test_agent", &input).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentError::InputGuardrailTripwire { ref guardrail_name } if guardrail_name == "profanity_filter"),
            "expected InputGuardrailTripwire, got: {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "input guardrail 'profanity_filter' triggered tripwire"
        );
    }

    // ---- Output guardrail that passes ----

    #[tokio::test]
    async fn output_guardrail_passes() {
        let guardrail = OutputGuardrail::<()>::new("pii_check", |_ctx, _agent, _output| {
            Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!("no PII found"))) })
        });

        let ctx = RunContextWrapper::new(());
        let agent_output = json!("This is a clean response.");
        let result = guardrail.run(&ctx, "test_agent", &agent_output).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.guardrail_name, "pii_check");
        assert_eq!(result.agent_name, "test_agent");
        assert_eq!(result.agent_output, json!("This is a clean response."));
        assert!(!result.output.tripwire_triggered);
    }

    // ---- Output guardrail that triggers tripwire ----

    #[tokio::test]
    async fn output_guardrail_triggers_tripwire() {
        let guardrail = OutputGuardrail::<()>::new("pii_detector", |_ctx, _agent, _output| {
            Box::pin(async { Ok(GuardrailFunctionOutput::tripwire(json!("SSN detected"))) })
        });

        let ctx = RunContextWrapper::new(());
        let agent_output = json!("My SSN is 123-45-6789");
        let result = guardrail.run(&ctx, "test_agent", &agent_output).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentError::OutputGuardrailTripwire { ref guardrail_name } if guardrail_name == "pii_detector"),
            "expected OutputGuardrailTripwire, got: {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "output guardrail 'pii_detector' triggered tripwire"
        );
    }

    // ---- Parallel flag ----

    #[test]
    fn input_guardrail_default_parallel() {
        let guardrail = InputGuardrail::<()>::new("g", |_ctx, _agent, _input| {
            Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
        });
        assert!(guardrail.run_in_parallel);
    }

    #[test]
    fn input_guardrail_parallel_constructor() {
        let guardrail = InputGuardrail::<()>::parallel("g", |_ctx, _agent, _input| {
            Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
        });
        assert!(guardrail.run_in_parallel);
    }

    #[test]
    fn input_guardrail_sequential_constructor() {
        let guardrail = InputGuardrail::<()>::sequential("g", |_ctx, _agent, _input| {
            Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
        });
        assert!(!guardrail.run_in_parallel);
    }

    // ---- Debug impl ----

    #[test]
    fn input_guardrail_debug() {
        let guardrail = InputGuardrail::<()>::new("my_guardrail", |_ctx, _agent, _input| {
            Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
        });
        let debug_str = format!("{guardrail:?}");
        assert!(debug_str.contains("InputGuardrail"));
        assert!(debug_str.contains("my_guardrail"));
        assert!(debug_str.contains("run_in_parallel"));
    }

    #[test]
    fn output_guardrail_debug() {
        let guardrail =
            OutputGuardrail::<()>::new("my_output_guardrail", |_ctx, _agent, _output| {
                Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
            });
        let debug_str = format!("{guardrail:?}");
        assert!(debug_str.contains("OutputGuardrail"));
        assert!(debug_str.contains("my_output_guardrail"));
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn input_guardrail_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InputGuardrail<()>>();
        assert_send_sync::<InputGuardrail<String>>();
    }

    #[test]
    fn output_guardrail_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OutputGuardrail<()>>();
        assert_send_sync::<OutputGuardrail<String>>();
    }

    // ---- Guardrail function receives correct arguments ----

    #[tokio::test]
    async fn input_guardrail_receives_correct_args() {
        let guardrail = InputGuardrail::<String>::new("arg_check", |ctx, agent_name, input| {
            let context_val = ctx.context.clone();
            let agent = agent_name.to_owned();
            let inp = input.clone();
            Box::pin(async move {
                assert_eq!(context_val, "test_context");
                assert_eq!(agent, "my_agent");
                assert!(matches!(inp, InputContent::Text(ref t) if t == "hello"));
                Ok(GuardrailFunctionOutput::passed(json!(null)))
            })
        });

        let ctx = RunContextWrapper::new("test_context".to_owned());
        let input = InputContent::Text("hello".to_owned());
        let result = guardrail.run(&ctx, "my_agent", &input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn output_guardrail_receives_correct_args() {
        let guardrail = OutputGuardrail::<String>::new("arg_check", |ctx, agent_name, output| {
            let context_val = ctx.context.clone();
            let agent = agent_name.to_owned();
            let out = output.clone();
            Box::pin(async move {
                assert_eq!(context_val, "test_context");
                assert_eq!(agent, "my_agent");
                assert_eq!(out, json!({"result": "ok"}));
                Ok(GuardrailFunctionOutput::passed(json!(null)))
            })
        });

        let ctx = RunContextWrapper::new("test_context".to_owned());
        let output = json!({"result": "ok"});
        let result = guardrail.run(&ctx, "my_agent", &output).await;
        assert!(result.is_ok());
    }

    // ---- Guardrail function error propagation ----

    #[tokio::test]
    async fn input_guardrail_propagates_error() {
        let guardrail = InputGuardrail::<()>::new("error_guard", |_ctx, _agent, _input| {
            Box::pin(async {
                Err(AgentError::UserError {
                    message: "something went wrong".to_owned(),
                })
            })
        });

        let ctx = RunContextWrapper::new(());
        let input = InputContent::Text("hello".to_owned());
        let result = guardrail.run(&ctx, "test_agent", &input).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::UserError { .. }));
    }

    #[tokio::test]
    async fn output_guardrail_propagates_error() {
        let guardrail = OutputGuardrail::<()>::new("error_guard", |_ctx, _agent, _output| {
            Box::pin(async {
                Err(AgentError::UserError {
                    message: "validation failed".to_owned(),
                })
            })
        });

        let ctx = RunContextWrapper::new(());
        let output = json!("test");
        let result = guardrail.run(&ctx, "test_agent", &output).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::UserError { .. }));
    }

    // ---- Result types are Clone + Debug ----

    #[test]
    fn input_guardrail_result_clone_and_debug() {
        let result = InputGuardrailResult {
            guardrail_name: "test".to_owned(),
            output: GuardrailFunctionOutput::passed(json!("ok")),
        };
        let cloned = result.clone();
        assert_eq!(cloned.guardrail_name, "test");
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("InputGuardrailResult"));
    }

    #[test]
    fn output_guardrail_result_clone_and_debug() {
        let result = OutputGuardrailResult {
            guardrail_name: "test".to_owned(),
            agent_name: "agent".to_owned(),
            agent_output: json!("output"),
            output: GuardrailFunctionOutput::passed(json!("ok")),
        };
        let cloned = result.clone();
        assert_eq!(cloned.agent_name, "agent");
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("OutputGuardrailResult"));
    }
}
