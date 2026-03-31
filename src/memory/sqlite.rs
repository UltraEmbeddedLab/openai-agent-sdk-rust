//! `SQLite`-backed session storage.
//!
//! This module is only available with the `sqlite-session` feature flag.
//!
//! Currently provides a stub implementation. A full `SQLite` backend using
//! `rusqlite` or `sqlx` will be added in a future release.

use async_trait::async_trait;

use crate::error::{AgentError, Result};
use crate::items::ResponseInputItem;

use super::session::Session;

/// A session backed by a `SQLite` database.
///
/// Provides persistent conversation history that survives process restarts.
///
/// # Status
///
/// This is a stub implementation. All async methods currently return an error
/// indicating that `SQLite` support is not yet fully implemented. The type and
/// constructor are provided so that downstream code can be written against
/// the `sqlite-session` feature flag today.
#[derive(Debug)]
pub struct SqliteSession {
    id: String,
    #[allow(dead_code)]
    db_path: String,
}

impl SqliteSession {
    /// Create a new `SQLite` session handle.
    ///
    /// The session will target the database at `db_path`. Use `":memory:"` for
    /// an ephemeral in-process database (useful for testing).
    #[must_use]
    pub fn new(id: impl Into<String>, db_path: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            db_path: db_path.into(),
        }
    }
}

#[async_trait]
impl Session for SqliteSession {
    async fn get_items(&self, _limit: Option<usize>) -> Result<Vec<ResponseInputItem>> {
        Err(AgentError::UserError {
            message: "SQLite session not yet implemented".into(),
        })
    }

    async fn add_items(&self, _items: &[ResponseInputItem]) -> Result<()> {
        Err(AgentError::UserError {
            message: "SQLite session not yet implemented".into(),
        })
    }

    async fn pop_item(&self) -> Result<Option<ResponseInputItem>> {
        Err(AgentError::UserError {
            message: "SQLite session not yet implemented".into(),
        })
    }

    async fn set_history(&self, _items: Vec<ResponseInputItem>) -> Result<()> {
        Err(AgentError::UserError {
            message: "SQLite session not yet implemented".into(),
        })
    }

    async fn clear(&self) -> Result<()> {
        Err(AgentError::UserError {
            message: "SQLite session not yet implemented".into(),
        })
    }

    fn session_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqlite_session_new() {
        let session = SqliteSession::new("sess-1", ":memory:");
        assert_eq!(session.session_id(), "sess-1");
    }

    #[tokio::test]
    async fn stub_methods_return_error() {
        let session = SqliteSession::new("sess-2", ":memory:");

        assert!(session.get_items(None).await.is_err());
        assert!(session.add_items(&[]).await.is_err());
        assert!(session.pop_item().await.is_err());
        assert!(session.set_history(vec![]).await.is_err());
        assert!(session.clear().await.is_err());
    }

    #[test]
    fn sqlite_session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SqliteSession>();
    }
}
