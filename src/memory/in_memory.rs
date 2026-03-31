//! In-memory session implementation.
//!
//! [`InMemorySession`] stores conversation history in a `Vec` behind a
//! `tokio::sync::RwLock`, making it safe for concurrent access from multiple
//! async tasks. History is lost when the process exits.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::items::ResponseInputItem;

use super::session::Session;

/// An in-memory session that stores history in a `Vec`.
///
/// This is useful for testing and single-process applications where
/// persistence across restarts is not required. The internal history
/// is protected by a [`tokio::sync::RwLock`] so it can be shared
/// across concurrent tasks.
///
/// # Example
///
/// ```
/// use openai_agents::memory::{InMemorySession, Session};
///
/// # tokio_test::block_on(async {
/// let session = InMemorySession::new("demo");
/// session.add_items(&[serde_json::json!({"role": "user", "content": "hi"})]).await.unwrap();
///
/// let items = session.get_items(None).await.unwrap();
/// assert_eq!(items.len(), 1);
/// # });
/// ```
#[derive(Debug, Clone)]
pub struct InMemorySession {
    id: String,
    history: Arc<RwLock<Vec<ResponseInputItem>>>,
}

impl InMemorySession {
    /// Create a new in-memory session with the given identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            history: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

#[async_trait]
impl Session for InMemorySession {
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<ResponseInputItem>> {
        let history = self.history.read().await;
        limit.map_or_else(
            || Ok(history.clone()),
            |n| {
                let start = history.len().saturating_sub(n);
                Ok(history[start..].to_vec())
            },
        )
    }

    async fn add_items(&self, items: &[ResponseInputItem]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        self.history.write().await.extend_from_slice(items);
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<ResponseInputItem>> {
        Ok(self.history.write().await.pop())
    }

    async fn set_history(&self, items: Vec<ResponseInputItem>) -> Result<()> {
        *self.history.write().await = items;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        self.history.write().await.clear();
        Ok(())
    }

    fn session_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn new_session_has_empty_history() {
        let session = InMemorySession::new("test-1");
        let items = session.get_items(None).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn session_id_returns_correct_id() {
        let session = InMemorySession::new("my-session");
        assert_eq!(session.session_id(), "my-session");
    }

    #[tokio::test]
    async fn add_and_get_items() {
        let session = InMemorySession::new("test-2");
        let item1 = json!({"role": "user", "content": "hello"});
        let item2 = json!({"role": "assistant", "content": "hi there"});

        session
            .add_items(&[item1.clone(), item2.clone()])
            .await
            .unwrap();

        let items = session.get_items(None).await.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], item1);
        assert_eq!(items[1], item2);
    }

    #[tokio::test]
    async fn add_empty_slice_is_noop() {
        let session = InMemorySession::new("test-3");
        session.add_items(&[]).await.unwrap();
        let items = session.get_items(None).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn get_items_with_limit() {
        let session = InMemorySession::new("test-limit");
        let items: Vec<_> = (0..5)
            .map(|i| json!({"role": "user", "content": format!("msg-{i}")}))
            .collect();
        session.add_items(&items).await.unwrap();

        // Limit returns the most recent N items.
        let last_two = session.get_items(Some(2)).await.unwrap();
        assert_eq!(last_two.len(), 2);
        assert_eq!(last_two[0], items[3]);
        assert_eq!(last_two[1], items[4]);
    }

    #[tokio::test]
    async fn get_items_with_limit_exceeding_length() {
        let session = InMemorySession::new("test-limit-overflow");
        let item = json!({"role": "user", "content": "only one"});
        session
            .add_items(std::slice::from_ref(&item))
            .await
            .unwrap();

        let result = session.get_items(Some(100)).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], item);
    }

    #[tokio::test]
    async fn pop_item_returns_most_recent() {
        let session = InMemorySession::new("test-pop");
        let item1 = json!({"role": "user", "content": "first"});
        let item2 = json!({"role": "user", "content": "second"});
        session
            .add_items(&[item1.clone(), item2.clone()])
            .await
            .unwrap();

        let popped = session.pop_item().await.unwrap();
        assert_eq!(popped, Some(item2));

        let remaining = session.get_items(None).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0], item1);
    }

    #[tokio::test]
    async fn pop_item_on_empty_session_returns_none() {
        let session = InMemorySession::new("test-pop-empty");
        let popped = session.pop_item().await.unwrap();
        assert_eq!(popped, None);
    }

    #[tokio::test]
    async fn set_history_replaces_all() {
        let session = InMemorySession::new("test-set");
        let original = json!({"role": "user", "content": "original"});
        session.add_items(&[original]).await.unwrap();

        let replacement = vec![
            json!({"role": "user", "content": "new-1"}),
            json!({"role": "assistant", "content": "new-2"}),
        ];
        session.set_history(replacement.clone()).await.unwrap();

        let items = session.get_items(None).await.unwrap();
        assert_eq!(items, replacement);
    }

    #[tokio::test]
    async fn clear_removes_all_items() {
        let session = InMemorySession::new("test-clear");
        session
            .add_items(&[json!({"role": "user", "content": "data"})])
            .await
            .unwrap();
        assert!(!session.get_items(None).await.unwrap().is_empty());

        session.clear().await.unwrap();
        assert!(session.get_items(None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn multiple_add_calls_append() {
        let session = InMemorySession::new("test-append");
        session
            .add_items(&[json!({"role": "user", "content": "a"})])
            .await
            .unwrap();
        session
            .add_items(&[json!({"role": "user", "content": "b"})])
            .await
            .unwrap();

        let items = session.get_items(None).await.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["content"], "a");
        assert_eq!(items[1]["content"], "b");
    }

    #[tokio::test]
    async fn concurrent_readers() {
        let session = InMemorySession::new("test-concurrent");
        let items: Vec<_> = (0..10)
            .map(|i| json!({"role": "user", "content": format!("msg-{i}")}))
            .collect();
        session.add_items(&items).await.unwrap();

        // Spawn multiple concurrent readers.
        let mut handles = Vec::new();
        for _ in 0..5 {
            let s = session.clone();
            handles.push(tokio::spawn(async move { s.get_items(None).await }));
        }

        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            assert_eq!(result.len(), 10);
        }
    }

    /// Verify that `Session` is object-safe by constructing a trait object.
    #[tokio::test]
    async fn session_trait_is_object_safe() {
        let session: Box<dyn Session> = Box::new(InMemorySession::new("dyn-test"));
        assert_eq!(session.session_id(), "dyn-test");
        let items = session.get_items(None).await.unwrap();
        assert!(items.is_empty());
    }

    /// Verify that `InMemorySession` satisfies `Send + Sync` bounds.
    #[test]
    fn in_memory_session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemorySession>();
    }

    /// Verify that `dyn Session` satisfies `Send + Sync` bounds.
    #[test]
    fn session_trait_object_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn Session>();
    }

    #[tokio::test]
    async fn clone_shares_state() {
        let session = InMemorySession::new("clone-test");
        let cloned = session.clone();

        session
            .add_items(&[json!({"role": "user", "content": "shared"})])
            .await
            .unwrap();

        // The clone should see the same data because they share the Arc.
        let items = cloned.get_items(None).await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["content"], "shared");
    }
}
