//! The [`Session`] trait for conversation history persistence.
//!
//! Implement this trait to provide custom persistence backends (in-memory,
//! `SQLite`, Redis, etc.). The SDK ships with [`super::InMemorySession`] as the
//! default implementation.

use async_trait::async_trait;

use crate::error::Result;
use crate::items::ResponseInputItem;

/// A session stores conversation history for multi-turn interactions.
///
/// Implement this trait to provide custom persistence backends
/// (in-memory, `SQLite`, Redis, etc.).
///
/// All methods are async because real-world backends (databases, network
/// stores) require I/O. The trait requires `Send + Sync` so that sessions
/// can be shared across tasks.
#[async_trait]
pub trait Session: Send + Sync {
    /// Retrieve the stored conversation history for this session.
    ///
    /// When `limit` is `Some(n)`, returns the most recent `n` items in
    /// chronological order. When `limit` is `None`, returns all items.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying storage operation fails.
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<ResponseInputItem>>;

    /// Append new items to the session history.
    ///
    /// Items are appended in the order they appear in the slice.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying storage operation fails.
    async fn add_items(&self, items: &[ResponseInputItem]) -> Result<()>;

    /// Remove and return the most recent item from the session.
    ///
    /// Returns `Ok(None)` if the session history is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying storage operation fails.
    async fn pop_item(&self) -> Result<Option<ResponseInputItem>>;

    /// Replace the entire session history.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying storage operation fails.
    async fn set_history(&self, items: Vec<ResponseInputItem>) -> Result<()>;

    /// Clear all items from the session history.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying storage operation fails.
    async fn clear(&self) -> Result<()>;

    /// Get the session identifier.
    fn session_id(&self) -> &str;
}
