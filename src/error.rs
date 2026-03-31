//! Error types for the `OpenAI` Agents SDK.
//!
//! This module defines [`AgentError`], the top-level error enum that all SDK operations
//! return, along with a convenient [`Result`] type alias.

/// The top-level error enum for the Agents SDK.
///
/// Every fallible operation in this crate returns [`Result<T>`](crate::error::Result),
/// which uses `AgentError` as the error type. Variants are designed to mirror the
/// exception hierarchy in the Python SDK (`agents.exceptions`).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AgentError {
    /// The agent run exceeded the configured maximum number of turns.
    #[error("max turns ({max_turns}) exceeded")]
    MaxTurnsExceeded {
        /// The configured maximum number of turns that was exceeded.
        max_turns: u32,
    },

    /// The model did something unexpected, such as calling a non-existent tool
    /// or returning malformed JSON.
    #[error("model behavior error: {message}")]
    ModelBehavior {
        /// A human-readable description of the unexpected model behavior.
        message: String,
    },

    /// The caller made an error using the SDK (for example, invalid configuration).
    #[error("user error: {message}")]
    UserError {
        /// A human-readable description of the usage error.
        message: String,
    },

    /// A function tool invocation exceeded its configured timeout.
    #[error("tool '{tool_name}' timed out after {timeout_seconds}s")]
    ToolTimeout {
        /// The name of the tool that timed out.
        tool_name: String,
        /// The timeout duration in seconds that was exceeded.
        timeout_seconds: f64,
    },

    /// An input guardrail triggered its tripwire, aborting the run.
    #[error("input guardrail '{guardrail_name}' triggered tripwire")]
    InputGuardrailTripwire {
        /// The name of the guardrail that triggered.
        guardrail_name: String,
    },

    /// An output guardrail triggered its tripwire, aborting the run.
    #[error("output guardrail '{guardrail_name}' triggered tripwire")]
    OutputGuardrailTripwire {
        /// The name of the guardrail that triggered.
        guardrail_name: String,
    },

    /// A tool input guardrail triggered an exception, aborting the run.
    #[error("tool input guardrail '{guardrail_name}' triggered on tool '{tool_name}'")]
    ToolInputGuardrailTripwire {
        /// The name of the guardrail that triggered.
        guardrail_name: String,
        /// The name of the tool whose input was rejected.
        tool_name: String,
    },

    /// A tool output guardrail triggered an exception, aborting the run.
    #[error("tool output guardrail '{guardrail_name}' triggered on tool '{tool_name}'")]
    ToolOutputGuardrailTripwire {
        /// The name of the guardrail that triggered.
        guardrail_name: String,
        /// The name of the tool whose output was rejected.
        tool_name: String,
    },

    /// An MCP tool call was internally cancelled.
    #[error("MCP tool cancellation: {message}")]
    McpToolCancellation {
        /// A human-readable description of the cancellation reason.
        message: String,
    },

    /// A serialization or deserialization error from `serde_json`.
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),

    /// An HTTP request error from `reqwest`.
    #[error(transparent)]
    Http(#[from] reqwest::Error),

    /// A catch-all for errors that do not fit other variants.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// A convenience alias for `std::result::Result<T, AgentError>`.
pub type Result<T> = std::result::Result<T, AgentError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_max_turns_exceeded() {
        let err = AgentError::MaxTurnsExceeded { max_turns: 10 };
        assert_eq!(err.to_string(), "max turns (10) exceeded");
    }

    #[test]
    fn display_model_behavior() {
        let err = AgentError::ModelBehavior {
            message: "called unknown tool".to_string(),
        };
        assert_eq!(err.to_string(), "model behavior error: called unknown tool");
    }

    #[test]
    fn display_user_error() {
        let err = AgentError::UserError {
            message: "missing api key".to_string(),
        };
        assert_eq!(err.to_string(), "user error: missing api key");
    }

    #[test]
    fn display_tool_timeout() {
        let err = AgentError::ToolTimeout {
            tool_name: "web_search".to_string(),
            timeout_seconds: 30.5,
        };
        assert_eq!(err.to_string(), "tool 'web_search' timed out after 30.5s");
    }

    #[test]
    fn display_input_guardrail_tripwire() {
        let err = AgentError::InputGuardrailTripwire {
            guardrail_name: "profanity_filter".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "input guardrail 'profanity_filter' triggered tripwire"
        );
    }

    #[test]
    fn display_output_guardrail_tripwire() {
        let err = AgentError::OutputGuardrailTripwire {
            guardrail_name: "pii_detector".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "output guardrail 'pii_detector' triggered tripwire"
        );
    }

    #[test]
    fn display_mcp_tool_cancellation() {
        let err = AgentError::McpToolCancellation {
            message: "server disconnected".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "MCP tool cancellation: server disconnected"
        );
    }

    #[test]
    fn from_serde_json_error() {
        let json_err =
            serde_json::from_str::<serde_json::Value>("{{bad}}").expect_err("should fail to parse");
        let agent_err = AgentError::from(json_err);
        assert!(
            matches!(agent_err, AgentError::Serialization(_)),
            "expected Serialization variant"
        );
        // The Display output should come from the inner serde_json error.
        assert!(!agent_err.to_string().is_empty());
    }

    #[test]
    fn from_anyhow_error() {
        let anyhow_err = anyhow::anyhow!("something went wrong");
        let agent_err = AgentError::from(anyhow_err);
        assert!(
            matches!(agent_err, AgentError::Other(_)),
            "expected Other variant"
        );
        assert_eq!(agent_err.to_string(), "something went wrong");
    }

    #[test]
    fn result_type_alias_ok() {
        let ok: Result<i32> = Ok(42);
        assert!(ok.is_ok());
    }

    #[test]
    fn result_type_alias_err() {
        let err: Result<i32> = Err(AgentError::MaxTurnsExceeded { max_turns: 5 });
        assert!(err.is_err());
    }

    /// Ensures that `AgentError` can be sent across threads.
    #[test]
    fn agent_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AgentError>();
    }

    /// Ensures that `AgentError` can be shared across threads.
    #[test]
    fn agent_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<AgentError>();
    }

    /// Verifies that `AgentError` implements `std::error::Error` and can be downcast.
    #[test]
    fn error_downcasting() {
        let err: Box<dyn std::error::Error> =
            Box::new(AgentError::MaxTurnsExceeded { max_turns: 3 });
        let downcast = err.downcast::<AgentError>();
        assert!(downcast.is_ok(), "should downcast to AgentError");
        let inner = downcast.unwrap();
        assert!(matches!(
            *inner,
            AgentError::MaxTurnsExceeded { max_turns: 3 }
        ));
    }

    /// Verifies the error source chain for transparent variants.
    #[test]
    fn serialization_error_source() {
        use std::error::Error;

        let json_err = serde_json::from_str::<serde_json::Value>("not json")
            .expect_err("should fail to parse");
        let agent_err = AgentError::from(json_err);

        // Transparent variants delegate to the inner error's source.
        let source = agent_err.source();
        // For transparent errors, thiserror forwards display and source, so
        // the source should be None (the inner error IS the error, not wrapped).
        // This is correct behavior for `#[error(transparent)]`.
        assert!(
            source.is_none(),
            "transparent variant should delegate source to inner error"
        );
    }

    /// Verifies that the error enum implements the `std::error::Error` trait.
    #[test]
    fn implements_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<AgentError>();
    }
}
