//! `SQLite`-backed session storage.
//!
//! This module is only available with the `sqlite-session` feature flag.
//! It provides persistent conversation history that survives process restarts,
//! backed by a local `SQLite` database via [`rusqlite`].

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use rusqlite::Connection;
use tokio::sync::Mutex;

use super::session::Session;
use crate::error::{AgentError, Result};
use crate::items::ResponseInputItem;

/// A session backed by a `SQLite` database file.
///
/// Provides persistent conversation history that survives process restarts.
/// Each session is identified by a unique session ID and stored in a single
/// `SQLite` database. Multiple sessions can coexist in the same database file.
///
/// # Examples
///
/// ```no_run
/// use openai_agents::memory::{SqliteSession, Session};
///
/// # tokio_test::block_on(async {
/// let session = SqliteSession::open("sess-001", "history.db").unwrap();
/// let items = session.get_items(None).await.unwrap();
/// assert!(items.is_empty());
/// # });
/// ```
#[derive(Debug)]
pub struct SqliteSession {
    id: String,
    conn: Arc<Mutex<Connection>>,
}

impl SqliteSession {
    /// Open or create a `SQLite` session database at the given path.
    ///
    /// Creates the sessions table if it does not already exist. The database
    /// file is created if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or the schema cannot
    /// be initialised.
    pub fn open(session_id: impl Into<String>, db_path: impl AsRef<Path>) -> Result<Self> {
        let id = session_id.into();
        let conn = Connection::open(db_path.as_ref()).map_err(|e| AgentError::UserError {
            message: format!("Failed to open SQLite database: {e}"),
        })?;
        Self::init_schema(&conn)?;
        Ok(Self {
            id,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create an in-memory `SQLite` session (useful for testing).
    ///
    /// The database lives only for the lifetime of this struct and is not
    /// persisted to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the in-memory database cannot be created.
    pub fn in_memory(session_id: impl Into<String>) -> Result<Self> {
        let id = session_id.into();
        let conn = Connection::open_in_memory().map_err(|e| AgentError::UserError {
            message: format!("Failed to open in-memory SQLite: {e}"),
        })?;
        Self::init_schema(&conn)?;
        Ok(Self {
            id,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create the sessions table if it does not already exist.
    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id   TEXT    NOT NULL,
                seq  INTEGER NOT NULL,
                item TEXT    NOT NULL,
                PRIMARY KEY (id, seq)
            );",
        )
        .map_err(|e| AgentError::UserError {
            message: format!("Failed to create sessions table: {e}"),
        })?;
        Ok(())
    }

    /// Return the current maximum sequence number for this session, or `-1` if
    /// no rows exist yet.
    fn max_seq(conn: &Connection, id: &str) -> i64 {
        conn.query_row(
            "SELECT COALESCE(MAX(seq), -1) FROM sessions WHERE id = ?1",
            [id],
            |row| row.get(0),
        )
        .unwrap_or(-1)
    }

    /// Query all JSON strings for this session, optionally limited.
    ///
    /// When `limit` is `Some(n)`, returns the latest `n` rows ordered
    /// descending (caller must reverse). When `None`, returns all rows
    /// ascending.
    fn query_rows(conn: &Connection, id: &str, limit: Option<usize>) -> Result<Vec<String>> {
        if let Some(n) = limit {
            let limit_i64 = i64::try_from(n).unwrap_or(i64::MAX);
            let mut stmt = conn
                .prepare("SELECT item FROM sessions WHERE id = ?1 ORDER BY seq DESC LIMIT ?2")
                .map_err(|e| AgentError::UserError {
                    message: format!("SQLite query error: {e}"),
                })?;
            let rows: std::result::Result<Vec<String>, _> = stmt
                .query_map(rusqlite::params![id, limit_i64], |row| row.get(0))
                .map_err(|e| AgentError::UserError {
                    message: format!("SQLite query error: {e}"),
                })?
                .collect();
            rows.map_err(|e| AgentError::UserError {
                message: format!("SQLite row error: {e}"),
            })
        } else {
            let mut stmt = conn
                .prepare("SELECT item FROM sessions WHERE id = ?1 ORDER BY seq ASC")
                .map_err(|e| AgentError::UserError {
                    message: format!("SQLite query error: {e}"),
                })?;
            let rows: std::result::Result<Vec<String>, _> = stmt
                .query_map([id], |row| row.get(0))
                .map_err(|e| AgentError::UserError {
                    message: format!("SQLite query error: {e}"),
                })?
                .collect();
            rows.map_err(|e| AgentError::UserError {
                message: format!("SQLite row error: {e}"),
            })
        }
    }

    /// Deserialize a list of JSON strings into `ResponseInputItem` values.
    fn deserialize_rows(rows: &[String]) -> Result<Vec<ResponseInputItem>> {
        let mut items = Vec::with_capacity(rows.len());
        for json_str in rows {
            let item: ResponseInputItem = serde_json::from_str(json_str)?;
            items.push(item);
        }
        Ok(items)
    }
}

#[async_trait]
#[allow(clippy::significant_drop_tightening)]
impl Session for SqliteSession {
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<ResponseInputItem>> {
        let rows = {
            let conn = self.conn.lock().await;
            Self::query_rows(&conn, &self.id, limit)?
        };

        if limit.is_some() {
            // Rows came back in descending order; reverse for chronological.
            let reversed: Vec<String> = rows.into_iter().rev().collect();
            Self::deserialize_rows(&reversed)
        } else {
            Self::deserialize_rows(&rows)
        }
    }

    async fn add_items(&self, items: &[ResponseInputItem]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        // Pre-serialize so the lock is held only for writes.
        let serialized: std::result::Result<Vec<String>, _> =
            items.iter().map(serde_json::to_string).collect();
        let serialized = serialized?;

        {
            let conn = self.conn.lock().await;
            let mut seq = Self::max_seq(&conn, &self.id);

            let mut stmt = conn
                .prepare("INSERT INTO sessions (id, seq, item) VALUES (?1, ?2, ?3)")
                .map_err(|e| AgentError::UserError {
                    message: format!("SQLite insert error: {e}"),
                })?;

            for json_str in &serialized {
                seq += 1;
                stmt.execute(rusqlite::params![&self.id, seq, json_str])
                    .map_err(|e| AgentError::UserError {
                        message: format!("SQLite insert error: {e}"),
                    })?;
            }
        }
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<ResponseInputItem>> {
        let json_str = {
            let conn = self.conn.lock().await;
            let result = conn.query_row(
                "SELECT seq, item FROM sessions WHERE id = ?1 ORDER BY seq DESC LIMIT 1",
                [&self.id],
                |row| {
                    let seq: i64 = row.get(0)?;
                    let json_str: String = row.get(1)?;
                    Ok((seq, json_str))
                },
            );

            match result {
                Ok((seq, json_str)) => {
                    conn.execute(
                        "DELETE FROM sessions WHERE id = ?1 AND seq = ?2",
                        rusqlite::params![&self.id, seq],
                    )
                    .map_err(|e| AgentError::UserError {
                        message: format!("SQLite delete error: {e}"),
                    })?;
                    Some(json_str)
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => None,
                Err(e) => {
                    return Err(AgentError::UserError {
                        message: format!("SQLite query error: {e}"),
                    });
                }
            }
        };

        match json_str {
            Some(s) => {
                let item: ResponseInputItem = serde_json::from_str(&s)?;
                Ok(Some(item))
            }
            None => Ok(None),
        }
    }

    async fn set_history(&self, items: Vec<ResponseInputItem>) -> Result<()> {
        self.clear().await?;
        self.add_items(&items).await
    }

    async fn clear(&self) -> Result<()> {
        self.conn
            .lock()
            .await
            .execute("DELETE FROM sessions WHERE id = ?1", [&self.id])
            .map_err(|e| AgentError::UserError {
                message: format!("SQLite delete error: {e}"),
            })?;
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

    /// Helper to create a fresh in-memory session for tests.
    fn make_session(id: &str) -> SqliteSession {
        SqliteSession::in_memory(id).expect("in-memory session should succeed")
    }

    #[test]
    fn in_memory_creation() {
        let session = make_session("sess-1");
        assert_eq!(session.session_id(), "sess-1");
    }

    #[test]
    fn session_id_returned_correctly() {
        let session = make_session("my-unique-id");
        assert_eq!(session.session_id(), "my-unique-id");
    }

    #[tokio::test]
    async fn add_items_and_get_items_round_trip() {
        let session = make_session("rt");
        let items = vec![
            json!({"role": "user", "content": "hello"}),
            json!({"role": "assistant", "content": "hi there"}),
        ];

        session.add_items(&items).await.unwrap();
        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved, items);
    }

    #[tokio::test]
    async fn get_items_empty_session() {
        let session = make_session("empty");
        let items = session.get_items(None).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn get_items_with_limit() {
        let session = make_session("limited");
        let items = vec![
            json!({"seq": 1}),
            json!({"seq": 2}),
            json!({"seq": 3}),
            json!({"seq": 4}),
            json!({"seq": 5}),
        ];
        session.add_items(&items).await.unwrap();

        // Limit 3 should return the last 3 items in chronological order.
        let retrieved = session.get_items(Some(3)).await.unwrap();
        assert_eq!(retrieved.len(), 3);
        assert_eq!(retrieved[0], json!({"seq": 3}));
        assert_eq!(retrieved[1], json!({"seq": 4}));
        assert_eq!(retrieved[2], json!({"seq": 5}));
    }

    #[tokio::test]
    async fn get_items_with_limit_larger_than_history() {
        let session = make_session("big-limit");
        let items = vec![json!({"a": 1}), json!({"b": 2})];
        session.add_items(&items).await.unwrap();

        let retrieved = session.get_items(Some(100)).await.unwrap();
        assert_eq!(retrieved, items);
    }

    #[tokio::test]
    async fn pop_item_returns_most_recent() {
        let session = make_session("pop");
        let items = vec![json!("first"), json!("second"), json!("third")];
        session.add_items(&items).await.unwrap();

        let popped = session.pop_item().await.unwrap();
        assert_eq!(popped, Some(json!("third")));

        // After popping, only two items remain.
        let remaining = session.get_items(None).await.unwrap();
        assert_eq!(remaining, vec![json!("first"), json!("second")]);
    }

    #[tokio::test]
    async fn pop_item_empty_session() {
        let session = make_session("pop-empty");
        let popped = session.pop_item().await.unwrap();
        assert_eq!(popped, None);
    }

    #[tokio::test]
    async fn set_history_replaces_all() {
        let session = make_session("set");
        session
            .add_items(&[json!("old-1"), json!("old-2")])
            .await
            .unwrap();

        let new_history = vec![json!("new-a"), json!("new-b"), json!("new-c")];
        session.set_history(new_history.clone()).await.unwrap();

        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved, new_history);
    }

    #[tokio::test]
    async fn clear_removes_all_items() {
        let session = make_session("clear");
        session.add_items(&[json!("x"), json!("y")]).await.unwrap();

        session.clear().await.unwrap();
        let items = session.get_items(None).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn multiple_sessions_in_same_database() {
        // Two sessions sharing the same underlying connection.
        let conn = Connection::open_in_memory().unwrap();
        SqliteSession::init_schema(&conn).unwrap();
        let shared = Arc::new(Mutex::new(conn));

        let session_a = SqliteSession {
            id: "a".to_owned(),
            conn: Arc::clone(&shared),
        };
        let session_b = SqliteSession {
            id: "b".to_owned(),
            conn: Arc::clone(&shared),
        };

        session_a
            .add_items(&[json!("from-a-1"), json!("from-a-2")])
            .await
            .unwrap();
        session_b.add_items(&[json!("from-b-1")]).await.unwrap();

        let items_a = session_a.get_items(None).await.unwrap();
        let items_b = session_b.get_items(None).await.unwrap();

        assert_eq!(items_a, vec![json!("from-a-1"), json!("from-a-2")]);
        assert_eq!(items_b, vec![json!("from-b-1")]);

        // Clearing one session should not affect the other.
        session_a.clear().await.unwrap();
        assert!(session_a.get_items(None).await.unwrap().is_empty());
        assert_eq!(
            session_b.get_items(None).await.unwrap(),
            vec![json!("from-b-1")]
        );
    }

    #[tokio::test]
    async fn add_items_empty_slice_is_noop() {
        let session = make_session("noop");
        session.add_items(&[]).await.unwrap();
        let items = session.get_items(None).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn incremental_adds_preserve_order() {
        let session = make_session("incr");
        session.add_items(&[json!(1), json!(2)]).await.unwrap();
        session.add_items(&[json!(3)]).await.unwrap();
        session.add_items(&[json!(4), json!(5)]).await.unwrap();

        let items = session.get_items(None).await.unwrap();
        assert_eq!(
            items,
            vec![json!(1), json!(2), json!(3), json!(4), json!(5)]
        );
    }

    #[test]
    fn sqlite_session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SqliteSession>();
    }
}
