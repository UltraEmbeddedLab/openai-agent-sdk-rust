//! Compacting session that automatically trims old conversation history.
//!
//! Mirrors the Python SDK's `OpenAIResponsesCompactionSession` concept: when the
//! number of stored items exceeds a configurable threshold, older non-user items
//! are replaced with a single compaction summary item, reducing token usage in
//! subsequent model calls.
//!
//! The actual summarisation is performed by a user-supplied async callback
//! (the "compactor"), making it possible to plug in an LLM-based summariser or
//! a simple truncation strategy.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use super::session::Session;
use crate::error::Result;
use crate::items::ResponseInputItem;

/// The default item-count threshold above which compaction is triggered.
pub const DEFAULT_COMPACTION_THRESHOLD: usize = 10;

/// A function that takes the current history and returns a compacted version.
///
/// The compactor receives the full item list and must return a (possibly
/// shorter) replacement list. A typical implementation would:
///
/// 1. Keep all user messages.
/// 2. Summarise assistant / tool messages into one or more compaction items.
/// 3. Return the combined list.
pub type CompactorFn = Arc<dyn Fn(Vec<ResponseInputItem>) -> Vec<ResponseInputItem> + Send + Sync>;

/// A session that compacts old messages when history exceeds a threshold.
///
/// When the number of items exceeds `max_items`, the user-supplied compactor
/// function is called to produce a shorter history, which then replaces the
/// stored items in the underlying session.
///
/// If no compactor is provided, the default strategy keeps the most recent
/// `max_items` items and prepends a JSON marker indicating that earlier
/// history was compacted.
pub struct CompactingSession {
    inner: Arc<dyn Session>,
    max_items: usize,
    compactor: CompactorFn,
    compaction_count: Arc<RwLock<u64>>,
}

impl fmt::Debug for CompactingSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompactingSession")
            .field("session_id", &self.inner.session_id())
            .field("max_items", &self.max_items)
            .finish_non_exhaustive()
    }
}

impl CompactingSession {
    /// Create a new compacting session with the default compaction strategy.
    ///
    /// The default strategy keeps the most recent `max_items` items and
    /// prepends a compaction marker item to indicate that earlier history
    /// was truncated.
    #[must_use]
    pub fn new(inner: Arc<dyn Session>, max_items: usize) -> Self {
        Self {
            inner,
            max_items,
            compactor: Arc::new(default_compactor(max_items)),
            compaction_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Create a compacting session with a custom compaction function.
    ///
    /// The compactor receives the full history when the threshold is exceeded
    /// and must return a replacement history.
    #[must_use]
    pub fn with_compactor(
        inner: Arc<dyn Session>,
        max_items: usize,
        compactor: CompactorFn,
    ) -> Self {
        Self {
            inner,
            max_items,
            compactor,
            compaction_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Get the number of times compaction has been performed.
    pub async fn compaction_count(&self) -> u64 {
        *self.compaction_count.read().await
    }

    /// Run compaction if the current history exceeds the threshold.
    ///
    /// Returns `true` if compaction was actually performed.
    async fn maybe_compact(&self) -> Result<bool> {
        let items = self.inner.get_items(None).await?;
        if items.len() <= self.max_items {
            return Ok(false);
        }

        let compacted = (self.compactor)(items);
        self.inner.set_history(compacted).await?;
        *self.compaction_count.write().await += 1;
        Ok(true)
    }
}

#[async_trait]
impl Session for CompactingSession {
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<ResponseInputItem>> {
        self.inner.get_items(limit).await
    }

    async fn add_items(&self, items: &[ResponseInputItem]) -> Result<()> {
        self.inner.add_items(items).await?;
        self.maybe_compact().await?;
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<ResponseInputItem>> {
        self.inner.pop_item().await
    }

    async fn set_history(&self, items: Vec<ResponseInputItem>) -> Result<()> {
        self.inner.set_history(items).await?;
        self.maybe_compact().await?;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        self.inner.clear().await
    }

    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
}

/// Build the default compactor closure that keeps the last `max_items` items
/// and prepends a compaction marker.
fn default_compactor(
    max_items: usize,
) -> impl Fn(Vec<ResponseInputItem>) -> Vec<ResponseInputItem> + Send + Sync {
    move |items: Vec<ResponseInputItem>| {
        if items.len() <= max_items {
            return items;
        }

        let removed_count = items.len() - max_items;
        let kept = items[removed_count..].to_vec();

        let marker = serde_json::json!({
            "type": "compaction",
            "summary": format!(
                "[Compacted: {removed_count} earlier item(s) were summarised.]"
            ),
        });

        let mut result = Vec::with_capacity(kept.len() + 1);
        result.push(marker);
        result.extend(kept);
        result
    }
}

/// Select items that are candidates for compaction.
///
/// Excludes user messages and existing compaction markers, mirroring the
/// Python SDK's `select_compaction_candidate_items`.
#[must_use]
pub fn select_compaction_candidate_items(items: &[ResponseInputItem]) -> Vec<ResponseInputItem> {
    items
        .iter()
        .filter(|item| {
            let is_user = item
                .get("role")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|r| r == "user");
            let is_compaction = item
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|t| t == "compaction");
            !is_user && !is_compaction
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemorySession;
    use serde_json::json;

    fn make_compacting(max_items: usize) -> CompactingSession {
        let inner = Arc::new(InMemorySession::new("compact-test"));
        CompactingSession::new(inner, max_items)
    }

    #[tokio::test]
    async fn below_threshold_no_compaction() {
        let session = make_compacting(5);
        let items: Vec<_> = (0..3).map(|i| json!({"seq": i})).collect();
        session.add_items(&items).await.unwrap();

        assert_eq!(session.compaction_count().await, 0);
        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved.len(), 3);
    }

    #[tokio::test]
    async fn at_threshold_no_compaction() {
        let session = make_compacting(5);
        let items: Vec<_> = (0..5).map(|i| json!({"seq": i})).collect();
        session.add_items(&items).await.unwrap();

        assert_eq!(session.compaction_count().await, 0);
        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved.len(), 5);
    }

    #[tokio::test]
    async fn above_threshold_triggers_compaction() {
        let session = make_compacting(3);
        let items: Vec<_> = (0..6).map(|i| json!({"seq": i})).collect();
        session.add_items(&items).await.unwrap();

        assert_eq!(session.compaction_count().await, 1);
        let retrieved = session.get_items(None).await.unwrap();
        // Default compactor keeps last 3 items + 1 marker = 4.
        assert_eq!(retrieved.len(), 4);
        assert_eq!(retrieved[0]["type"], "compaction");
        assert_eq!(retrieved[1], json!({"seq": 3}));
        assert_eq!(retrieved[2], json!({"seq": 4}));
        assert_eq!(retrieved[3], json!({"seq": 5}));
    }

    #[tokio::test]
    async fn compaction_count_increments() {
        let inner = Arc::new(InMemorySession::new("count-test"));
        let session = CompactingSession::new(Arc::clone(&inner) as Arc<dyn Session>, 2);

        // First batch: 4 items => compaction.
        session
            .add_items(&[json!(1), json!(2), json!(3), json!(4)])
            .await
            .unwrap();
        assert_eq!(session.compaction_count().await, 1);

        // After first compaction: marker + 2 items = 3 items.
        // Add 1 more => 4 items > 2 => compaction again.
        session.add_items(&[json!(5)]).await.unwrap();
        assert_eq!(session.compaction_count().await, 2);
    }

    #[tokio::test]
    async fn clear_does_not_trigger_compaction() {
        let session = make_compacting(3);
        session
            .add_items(&[json!(1), json!(2), json!(3), json!(4), json!(5)])
            .await
            .unwrap();
        let count_before = session.compaction_count().await;

        session.clear().await.unwrap();
        assert_eq!(session.compaction_count().await, count_before);
        assert!(session.get_items(None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn pop_item_works() {
        let session = make_compacting(10);
        session
            .add_items(&[json!("first"), json!("second")])
            .await
            .unwrap();

        let popped = session.pop_item().await.unwrap();
        assert_eq!(popped, Some(json!("second")));
    }

    #[tokio::test]
    async fn session_id_delegates_to_inner() {
        let session = make_compacting(5);
        assert_eq!(session.session_id(), "compact-test");
    }

    #[tokio::test]
    async fn set_history_can_trigger_compaction() {
        let session = make_compacting(3);
        let items: Vec<_> = (0..6).map(|i| json!({"seq": i})).collect();
        session.set_history(items).await.unwrap();

        assert_eq!(session.compaction_count().await, 1);
    }

    #[tokio::test]
    async fn custom_compactor() {
        let inner = Arc::new(InMemorySession::new("custom"));
        // Custom compactor: just keep last 2 items, no marker.
        let compactor: CompactorFn = Arc::new(|items: Vec<ResponseInputItem>| {
            if items.len() > 2 {
                items[items.len() - 2..].to_vec()
            } else {
                items
            }
        });
        let session =
            CompactingSession::with_compactor(Arc::clone(&inner) as Arc<dyn Session>, 2, compactor);

        session
            .add_items(&[json!("a"), json!("b"), json!("c"), json!("d")])
            .await
            .unwrap();

        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved, vec![json!("c"), json!("d")]);
        assert_eq!(session.compaction_count().await, 1);
    }

    #[test]
    fn select_compaction_candidates_excludes_user_and_compaction() {
        let items = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
            json!({"type": "compaction", "summary": "..."}),
            json!({"type": "function_call", "name": "foo"}),
        ];
        let candidates = select_compaction_candidate_items(&items);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0]["role"], "assistant");
        assert_eq!(candidates[1]["type"], "function_call");
    }

    #[test]
    fn compacting_session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CompactingSession>();
    }

    #[test]
    fn debug_impl_shows_fields() {
        let inner = Arc::new(InMemorySession::new("dbg-test"));
        let session = CompactingSession::new(inner as Arc<dyn Session>, 5);
        let debug_str = format!("{session:?}");
        assert!(debug_str.contains("CompactingSession"));
        assert!(debug_str.contains("max_items: 5"));
    }
}
