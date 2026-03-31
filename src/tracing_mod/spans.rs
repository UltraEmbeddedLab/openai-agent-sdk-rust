//! Span creation helpers for structured agent tracing.
//!
//! Each function creates a [`tracing::Span`] at the `INFO` level with
//! domain-specific fields. The spans integrate with any subscriber that
//! is configured via `tracing-subscriber`, including OpenTelemetry
//! exporters.
//!
//! All span constructors are cheap and infallible. They never perform I/O
//! or allocate beyond what the `tracing` macros require.

use tracing::{Span, info_span};

/// Create a tracing span for an agent's execution turn.
///
/// The span records the agent name and the current turn number, which is
/// useful for diagnosing multi-turn conversations and handoff chains.
#[must_use]
pub fn agent_span(agent_name: &str, turn: u32) -> Span {
    info_span!(
        "agent",
        agent.name = agent_name,
        agent.turn = turn,
        otel.name = %format!("agent:{agent_name}"),
    )
}

/// Create a tracing span for a model generation (LLM call).
///
/// The span captures which agent initiated the call and which model was
/// used, making it straightforward to correlate latency with model choice.
#[must_use]
pub fn generation_span(agent_name: &str, model: &str) -> Span {
    info_span!(
        "generation",
        agent.name = agent_name,
        model.name = model,
        otel.name = %format!("generation:{model}"),
    )
}

/// Create a tracing span for a function tool invocation.
///
/// The span links the tool call back to the agent that requested it,
/// enabling end-to-end visibility from agent decision to tool result.
#[must_use]
pub fn function_span(agent_name: &str, tool_name: &str) -> Span {
    info_span!(
        "function",
        agent.name = agent_name,
        tool.name = tool_name,
        otel.name = %format!("function:{tool_name}"),
    )
}

/// Create a tracing span for an agent handoff.
///
/// Records both the source and destination agent names so that handoff
/// chains can be reconstructed from trace data.
#[must_use]
pub fn handoff_span(from_agent: &str, to_agent: &str) -> Span {
    info_span!(
        "handoff",
        handoff.from = from_agent,
        handoff.to = to_agent,
        otel.name = %format!("handoff:{from_agent}->{to_agent}"),
    )
}

/// Create a tracing span for a guardrail check.
///
/// The `guardrail_type` parameter should be either `"input"` or `"output"`
/// to distinguish between the two guardrail phases.
#[must_use]
pub fn guardrail_span(guardrail_name: &str, guardrail_type: &str) -> Span {
    info_span!(
        "guardrail",
        guardrail.name = guardrail_name,
        guardrail.type_ = guardrail_type,
        otel.name = %format!("guardrail:{guardrail_name}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_span_does_not_panic() {
        let span = agent_span("test_agent", 1);
        let _guard = span.enter();
    }

    #[test]
    fn generation_span_does_not_panic() {
        let span = generation_span("test_agent", "gpt-4o");
        let _guard = span.enter();
    }

    #[test]
    fn function_span_does_not_panic() {
        let span = function_span("test_agent", "web_search");
        let _guard = span.enter();
    }

    #[test]
    fn handoff_span_does_not_panic() {
        let span = handoff_span("agent_a", "agent_b");
        let _guard = span.enter();
    }

    #[test]
    fn guardrail_span_does_not_panic() {
        let span = guardrail_span("profanity_filter", "input");
        let _guard = span.enter();
    }

    #[test]
    fn spans_can_be_nested() {
        let agent = agent_span("outer_agent", 1);
        let _agent_guard = agent.enter();

        let generation = generation_span("outer_agent", "gpt-4o");
        let _gen_guard = generation.enter();

        let func = function_span("outer_agent", "calculator");
        let _func_guard = func.enter();
    }

    #[test]
    fn agent_span_with_zero_turn() {
        let span = agent_span("agent", 0);
        let _guard = span.enter();
    }

    #[test]
    fn agent_span_with_high_turn() {
        let span = agent_span("agent", u32::MAX);
        let _guard = span.enter();
    }

    #[test]
    fn spans_with_empty_names() {
        let _ = agent_span("", 0);
        let _ = generation_span("", "");
        let _ = function_span("", "");
        let _ = handoff_span("", "");
        let _ = guardrail_span("", "");
    }

    #[test]
    fn spans_with_unicode_names() {
        let _ = agent_span("\u{1F916} bot", 1);
        let _ = function_span("agent", "\u{1F527} wrench_tool");
        let _ = handoff_span("\u{1F1FA}\u{1F1F8} us_agent", "\u{1F1EC}\u{1F1E7} uk_agent");
    }
}
