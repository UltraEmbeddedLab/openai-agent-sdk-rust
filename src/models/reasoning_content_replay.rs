//! Reasoning content replay for chat completions models.
//!
//! When using non-OpenAI providers (e.g. DeepSeek) through the chat completions
//! interface, models may include reasoning/thinking content in their responses.
//! This module provides types and utilities for deciding whether to replay that
//! reasoning content back to the model in subsequent requests, which can improve
//! multi-turn coherence.
//!
//! This mirrors the Python SDK's `models/reasoning_content_replay.py`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Source information for reasoning content extracted from a model response.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ReasoningContentSource {
    /// The raw response item that contained the reasoning content.
    pub item: serde_json::Value,
    /// The model identifier that produced the reasoning content.
    pub origin_model: String,
    /// Optional provider-specific data associated with the reasoning.
    pub provider_data: Option<serde_json::Value>,
}

/// Context passed to the reasoning content replay decision callback.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ReasoningContentReplayContext {
    /// The model being used for the next request.
    pub model: String,
    /// The reasoning content sources from previous responses.
    pub sources: Vec<ReasoningContentSource>,
}

/// Type alias for the callback that decides whether to replay reasoning content.
///
/// Receives a [`ReasoningContentReplayContext`] and returns `true` if reasoning
/// content should be included in the next request.
pub type ShouldReplayReasoningContent =
    Arc<dyn Fn(ReasoningContentReplayContext) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync>;

/// Default implementation for the reasoning content replay decision.
///
/// Returns `true` for DeepSeek models (model names containing `"deepseek"`),
/// which benefit from having their reasoning content replayed. Returns `false`
/// for all other models.
#[must_use]
pub fn default_should_replay_reasoning_content() -> ShouldReplayReasoningContent {
    Arc::new(|ctx: ReasoningContentReplayContext| {
        let should_replay = ctx.model.to_lowercase().contains("deepseek");
        Box::pin(async move { should_replay })
    })
}

/// Extract reasoning content items from a list of response output items.
///
/// Looks for items with `"type": "reasoning"` and returns them along with the
/// model name that produced them.
#[must_use]
pub fn extract_reasoning_content(
    output: &[serde_json::Value],
    model: &str,
) -> Vec<ReasoningContentSource> {
    output
        .iter()
        .filter(|item| {
            item.get("type")
                .and_then(serde_json::Value::as_str)
                == Some("reasoning")
        })
        .map(|item| ReasoningContentSource {
            item: item.clone(),
            origin_model: model.to_owned(),
            provider_data: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reasoning_content_source_debug() {
        let source = ReasoningContentSource {
            item: json!({"type": "reasoning", "text": "thinking..."}),
            origin_model: "deepseek-r1".to_owned(),
            provider_data: None,
        };
        let debug = format!("{source:?}");
        assert!(debug.contains("deepseek-r1"));
    }

    #[test]
    fn extract_reasoning_content_finds_reasoning_items() {
        let output = vec![
            json!({"type": "message", "content": [{"type": "output_text", "text": "hello"}]}),
            json!({"type": "reasoning", "text": "let me think..."}),
            json!({"type": "reasoning", "text": "I should say hello"}),
        ];
        let sources = extract_reasoning_content(&output, "deepseek-r1");
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].origin_model, "deepseek-r1");
        assert_eq!(sources[1].item["text"], "I should say hello");
    }

    #[test]
    fn extract_reasoning_content_empty_when_no_reasoning() {
        let output = vec![
            json!({"type": "message", "content": []}),
            json!({"type": "function_call", "name": "test"}),
        ];
        let sources = extract_reasoning_content(&output, "gpt-4o");
        assert!(sources.is_empty());
    }

    #[tokio::test]
    async fn default_replay_true_for_deepseek() {
        let callback = default_should_replay_reasoning_content();
        let ctx = ReasoningContentReplayContext {
            model: "deepseek-r1".to_owned(),
            sources: vec![],
        };
        assert!(callback(ctx).await);
    }

    #[tokio::test]
    async fn default_replay_false_for_openai() {
        let callback = default_should_replay_reasoning_content();
        let ctx = ReasoningContentReplayContext {
            model: "gpt-4o".to_owned(),
            sources: vec![],
        };
        assert!(!callback(ctx).await);
    }

    #[tokio::test]
    async fn default_replay_case_insensitive() {
        let callback = default_should_replay_reasoning_content();
        let ctx = ReasoningContentReplayContext {
            model: "DeepSeek-V3".to_owned(),
            sources: vec![],
        };
        assert!(callback(ctx).await);
    }

    #[test]
    fn send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ReasoningContentSource>();
        assert_send_sync::<ReasoningContentReplayContext>();
    }
}
