# Sessions

A session stores conversation history so that multiple calls to the runner can maintain context
across turns. Without a session, each `Runner::run_*` call is stateless — you must manually
thread the previous output back in as input. With a session, the SDK handles history management
for you.

## The `Session` trait

All session implementations share the `Session` trait from `openai_agents::memory`:

```rust
use async_trait::async_trait;
use openai_agents::error::Result;
use openai_agents::items::ResponseInputItem;

#[async_trait]
pub trait Session: Send + Sync {
    /// Retrieve stored history. `limit` restricts to the most recent N items.
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<ResponseInputItem>>;

    /// Append items to the history.
    async fn add_items(&self, items: &[ResponseInputItem]) -> Result<()>;

    /// Remove and return the most recent item.
    async fn pop_item(&self) -> Result<Option<ResponseInputItem>>;

    /// Replace the entire history.
    async fn set_history(&self, items: Vec<ResponseInputItem>) -> Result<()>;

    /// Clear all items.
    async fn clear(&self) -> Result<()>;

    /// The session's unique identifier.
    fn session_id(&self) -> &str;
}
```

The trait is object-safe, so you can store sessions as `Arc<dyn Session>` and swap
implementations at runtime.

## `InMemorySession`

`InMemorySession` stores history in a `Vec` behind a `tokio::sync::RwLock`. History is lost when
the process exits, making it ideal for tests and single-session applications.

```rust
use openai_agents::memory::{InMemorySession, Session};
use serde_json::json;

# tokio_test::block_on(async {
let session = InMemorySession::new("user-42");

// Add the first user message.
session.add_items(&[json!({"role": "user", "content": "Hello"})]).await.unwrap();

// Retrieve all history.
let items = session.get_items(None).await.unwrap();
assert_eq!(items.len(), 1);

// Retrieve only the most recent 5 items.
let recent = session.get_items(Some(5)).await.unwrap();
# });
```

`InMemorySession` implements `Clone` — both the original and the clone share the same underlying
history via `Arc`.

### Dependency

No additional feature flag is required. `InMemorySession` is always available.

```toml
[dependencies]
openai-agents = "0.1"
```

## `SqliteSession`

`SqliteSession` persists conversation history to an SQLite database file. It is available behind
the `sqlite-session` feature flag.

```toml
[dependencies]
openai-agents = { version = "0.1", features = ["sqlite-session"] }
```

```rust,no_run
use openai_agents::memory::SqliteSession;

# tokio_test::block_on(async {
// Opens or creates an SQLite file at the given path.
let session = SqliteSession::new("conversations.db", "user-42")
    .await
    .expect("failed to open database");

session.add_items(&[serde_json::json!({"role": "user", "content": "Hello"})]).await.unwrap();
# });
```

`SqliteSession` is thread-safe and uses a single connection protected by a `tokio::sync::Mutex`.
The database schema is created automatically on first use.

## `EncryptedSession`

`EncryptedSession` wraps any `Session` implementation with transparent encryption. Items are
serialised to JSON, encrypted with a per-session derived key, and stored as opaque envelopes in
the underlying session.

```rust
use std::sync::Arc;
use openai_agents::memory::{InMemorySession, Session};
use openai_agents::memory::EncryptedSession;
use serde_json::json;

# tokio_test::block_on(async {
let inner = Arc::new(InMemorySession::new("secure-session")) as Arc<dyn Session>;
let session = EncryptedSession::new(inner, b"my-32-byte-master-key-here!!!!!!");

// Items are encrypted before storage.
session.add_items(&[json!({"role": "user", "content": "My SSN is 123-45-6789"})]).await.unwrap();

// Transparent decryption on retrieval.
let items = session.get_items(None).await.unwrap();
assert_eq!(items[0]["content"], "My SSN is 123-45-6789");
# });
```

Key properties:

- A per-session encryption key is derived from the master key and the session ID, so the same
  master key produces different ciphertext across sessions.
- Items encrypted with one key cannot be decrypted with a different key — they are silently
  skipped on retrieval.
- Corrupted items are silently skipped.

> **Security note:** The built-in cipher is an XOR construction intended to demonstrate the API.
> For production use, replace the encryption layer with an AEAD cipher such as AES-GCM (via the
> `aes-gcm` crate).

## `CompactingSession`

`CompactingSession` wraps any `Session` and automatically compacts old history when the number of
stored items exceeds a threshold. This controls token usage in long conversations.

```rust
use std::sync::Arc;
use openai_agents::memory::{InMemorySession, Session};
use openai_agents::memory::CompactingSession;
use serde_json::json;

# tokio_test::block_on(async {
let inner = Arc::new(InMemorySession::new("chat")) as Arc<dyn Session>;

// Compact when items exceed 10.
let session = CompactingSession::new(inner, 10);

// Add messages normally — compaction happens automatically.
for i in 0..15 {
    session.add_items(&[json!({"role": "user", "content": format!("Message {i}")})]).await.unwrap();
}

// History is now compacted.
let count = session.compaction_count().await;
assert!(count >= 1);
# });
```

The default compaction strategy keeps the most recent `max_items` items and prepends a JSON
marker:

```json
{
  "type": "compaction",
  "summary": "[Compacted: 5 earlier item(s) were summarised.]"
}
```

### Custom compactor

Supply a `CompactorFn` to implement your own strategy — for example, calling an LLM to produce
a prose summary:

```rust
use std::sync::Arc;
use openai_agents::memory::{InMemorySession, Session};
use openai_agents::memory::CompactingSession;
use openai_agents::memory::compacting::CompactorFn;
use serde_json::json;

let compactor: CompactorFn = Arc::new(|items| {
    // Keep only user messages; discard assistant turns.
    items
        .into_iter()
        .filter(|item| {
            item.get("role")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|r| r == "user")
        })
        .collect()
});

# tokio_test::block_on(async {
let inner = Arc::new(InMemorySession::new("custom-compact")) as Arc<dyn Session>;
let session = CompactingSession::with_compactor(inner, 10, compactor);
# });
```

`select_compaction_candidate_items` is a utility that returns only the items that are candidates
for removal (excludes user messages and existing compaction markers):

```rust
use openai_agents::memory::compacting::select_compaction_candidate_items;
use serde_json::json;

let items = vec![
    json!({"role": "user", "content": "Hi"}),
    json!({"role": "assistant", "content": "Hello"}),
    json!({"type": "function_call", "name": "search"}),
];
let candidates = select_compaction_candidate_items(&items);
// candidates contains the assistant message and function_call item.
assert_eq!(candidates.len(), 2);
```

## Multi-turn conversations with sessions

The typical pattern for a multi-turn chatbot:

```rust,no_run
use openai_agents::memory::{InMemorySession, Session};
use openai_agents::runner::Runner;
use openai_agents::agent::Agent;
use openai_agents::items::InputContent;
use serde_json::json;

# async fn chat_loop(model: std::sync::Arc<dyn openai_agents::models::Model>) {
let session = InMemorySession::new("conversation-1");
let agent = Agent::<()>::builder("assistant")
    .instructions("You are a helpful assistant.")
    .build();

loop {
    // Read the next user message.
    let user_input = get_user_input(); // your I/O here

    // Prepend session history to this turn's input.
    let mut history = session.get_items(None).await.unwrap();
    history.push(json!({"role": "user", "content": user_input}));
    let input = InputContent::Items(history);

    let result = Runner::run_with_model(&agent, input, (), model.clone(), None, None)
        .await
        .expect("run failed");

    // Persist the new turn to the session.
    session
        .add_items(&[
            json!({"role": "user", "content": user_input}),
            json!({"role": "assistant", "content": result.final_output.to_string()}),
        ])
        .await
        .unwrap();

    println!("Assistant: {}", result.final_output);
}
# }
# fn get_user_input() -> String { String::new() }
```

## Implementing a custom session

Implement `Session` for any storage backend by satisfying the five async methods and
`session_id`:

```rust
use async_trait::async_trait;
use openai_agents::memory::Session;
use openai_agents::error::Result;
use openai_agents::items::ResponseInputItem;

struct RedisSession {
    id: String,
    // redis_client: ...
}

#[async_trait]
impl Session for RedisSession {
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<ResponseInputItem>> {
        // LRANGE key 0 -1 (or last N)
        todo!()
    }

    async fn add_items(&self, items: &[ResponseInputItem]) -> Result<()> {
        // RPUSH key item...
        todo!()
    }

    async fn pop_item(&self) -> Result<Option<ResponseInputItem>> {
        // RPOP key
        todo!()
    }

    async fn set_history(&self, items: Vec<ResponseInputItem>) -> Result<()> {
        // DEL key; RPUSH key items...
        todo!()
    }

    async fn clear(&self) -> Result<()> {
        // DEL key
        todo!()
    }

    fn session_id(&self) -> &str {
        &self.id
    }
}
```

## See also

- [Human in the Loop](./human_in_the_loop.md) — `RunState` for pause/resume workflows.
- [Context](./context.md) — passing per-run application data.
- [Running Agents](./running_agents.md) — the `Runner` API.
