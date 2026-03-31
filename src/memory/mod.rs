//! Session management for persisting conversation history across runs.
//!
//! This module provides the [`Session`] trait for storing and retrieving conversation
//! history, along with concrete implementations:
//!
//! - [`InMemorySession`] -- a simple in-memory store suitable for testing and
//!   single-process applications.
//! - `SqliteSession` (behind the `sqlite-session` feature) -- a stub for future
//!   SQLite-backed persistent storage.
//!
//! # Example
//!
//! ```
//! use openai_agents::memory::{InMemorySession, Session};
//!
//! # tokio_test::block_on(async {
//! let session = InMemorySession::new("sess-001");
//! assert_eq!(session.session_id(), "sess-001");
//!
//! let history = session.get_items(None).await.unwrap();
//! assert!(history.is_empty());
//! # });
//! ```

pub mod in_memory;
pub mod session;

#[cfg(feature = "sqlite-session")]
pub mod sqlite;

pub use in_memory::InMemorySession;
pub use session::Session;

#[cfg(feature = "sqlite-session")]
pub use sqlite::SqliteSession;
