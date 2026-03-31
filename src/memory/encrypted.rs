//! Encrypted session storage wrapper.
//!
//! Wraps any [`Session`] implementation with transparent encryption/decryption
//! of conversation items using a symmetric key derived per-session. Items are
//! serialized to JSON, encrypted with XOR-based key derivation (for
//! demonstration), and stored as hex-encoded strings in the underlying session.
//!
//! **Important:** The XOR cipher used here is for API demonstration only. For
//! production use, replace the `encrypt_bytes` / `decrypt_bytes` methods with a
//! proper AEAD cipher such as AES-GCM (e.g., via the `aes-gcm` crate).

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use super::session::Session;
use crate::error::{AgentError, Result};
use crate::items::ResponseInputItem;

/// Marker key used inside encrypted envelopes so that [`EncryptedSession`] can
/// distinguish its own ciphertext from plain items.
const ENVELOPE_MARKER: &str = "__enc__";

/// Current envelope format version.
const ENVELOPE_VERSION: u64 = 1;

/// A session wrapper that encrypts items before storage and decrypts on retrieval.
///
/// Uses a per-session key derived from the master key and session ID so that
/// the same master key produces different ciphertext across sessions.
///
/// Expired or corrupted items are silently skipped during retrieval, matching
/// the Python SDK's `EncryptedSession` behaviour.
pub struct EncryptedSession {
    inner: Arc<dyn Session>,
    derived_key: Vec<u8>,
    session_id: String,
}

impl fmt::Debug for EncryptedSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncryptedSession")
            .field("session_id", &self.session_id)
            .field("inner", &"<dyn Session>")
            .field("derived_key", &"[REDACTED]")
            .finish()
    }
}

impl EncryptedSession {
    /// Wrap an existing session with encryption using the given master key.
    ///
    /// The master key is combined with the session ID from the inner session
    /// to derive a per-session encryption key.
    #[must_use]
    pub fn new(inner: Arc<dyn Session>, master_key: impl Into<Vec<u8>>) -> Self {
        let session_id = inner.session_id().to_owned();
        let master = master_key.into();
        let derived_key = derive_session_key(&master, &session_id);
        Self {
            inner,
            derived_key,
            session_id,
        }
    }

    /// Encrypt a single [`ResponseInputItem`] into an envelope stored as JSON.
    fn wrap(&self, item: &ResponseInputItem) -> Result<ResponseInputItem> {
        let plaintext = serde_json::to_string(item)?;
        let ciphertext = encrypt_bytes(plaintext.as_bytes(), &self.derived_key);
        let hex_payload = hex_encode(&ciphertext);

        Ok(serde_json::json!({
            ENVELOPE_MARKER: ENVELOPE_VERSION,
            "v": ENVELOPE_VERSION,
            "payload": hex_payload,
        }))
    }

    /// Attempt to decrypt an item. Returns `None` for corrupted or non-envelope items.
    fn unwrap_item(&self, item: &ResponseInputItem) -> Option<ResponseInputItem> {
        // Check if this is an encrypted envelope.
        let marker = item.get(ENVELOPE_MARKER)?;
        if marker.as_u64()? != ENVELOPE_VERSION {
            return None;
        }
        let payload_hex = item.get("payload")?.as_str()?;
        let ciphertext = hex_decode(payload_hex).ok()?;
        let plaintext_bytes = decrypt_bytes(&ciphertext, &self.derived_key);
        let plaintext = String::from_utf8(plaintext_bytes).ok()?;
        serde_json::from_str(&plaintext).ok()
    }
}

#[async_trait]
impl Session for EncryptedSession {
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<ResponseInputItem>> {
        let encrypted_items = self.inner.get_items(limit).await?;
        let mut valid = Vec::with_capacity(encrypted_items.len());
        for enc in &encrypted_items {
            if let Some(item) = self.unwrap_item(enc) {
                valid.push(item);
            }
            // Silently skip corrupted / expired items.
        }
        Ok(valid)
    }

    async fn add_items(&self, items: &[ResponseInputItem]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let wrapped: Result<Vec<ResponseInputItem>> = items.iter().map(|i| self.wrap(i)).collect();
        self.inner.add_items(&wrapped?).await
    }

    async fn pop_item(&self) -> Result<Option<ResponseInputItem>> {
        // Pop items until we find one that decrypts successfully, or the store is empty.
        loop {
            let enc = self.inner.pop_item().await?;
            match enc {
                None => return Ok(None),
                Some(ref e) => {
                    if let Some(item) = self.unwrap_item(e) {
                        return Ok(Some(item));
                    }
                    // Corrupted item; skip and try next.
                }
            }
        }
    }

    async fn set_history(&self, items: Vec<ResponseInputItem>) -> Result<()> {
        let wrapped: Result<Vec<ResponseInputItem>> = items.iter().map(|i| self.wrap(i)).collect();
        self.inner.set_history(wrapped?).await
    }

    async fn clear(&self) -> Result<()> {
        self.inner.clear().await
    }

    fn session_id(&self) -> &str {
        &self.session_id
    }
}

// ---------------------------------------------------------------------------
// Cryptographic helpers (XOR-based — replace for production)
// ---------------------------------------------------------------------------

/// Derive a per-session key by XOR-mixing the master key with the session ID bytes.
///
/// This is a simplified key derivation. In production, use HKDF (e.g. from the
/// `hkdf` crate) with SHA-256.
fn derive_session_key(master: &[u8], session_id: &str) -> Vec<u8> {
    let salt = session_id.as_bytes();
    // Produce a key that is at least as long as the master key.
    let len = master.len().max(32);
    let mut derived = Vec::with_capacity(len);
    for i in 0..len {
        let m = master[i % master.len()];
        let s = salt[i % salt.len()];
        // Mix with index to avoid trivial patterns.
        #[allow(clippy::cast_possible_truncation)]
        let idx_byte = (i & 0xFF) as u8;
        derived.push(m ^ s ^ idx_byte);
    }
    derived
}

/// Encrypt plaintext bytes with XOR against the key (cycled).
fn encrypt_bytes(plaintext: &[u8], key: &[u8]) -> Vec<u8> {
    plaintext
        .iter()
        .zip(key.iter().cycle())
        .map(|(p, k)| p ^ k)
        .collect()
}

/// Decrypt ciphertext bytes with XOR against the key (cycled). Symmetric with `encrypt_bytes`.
fn decrypt_bytes(ciphertext: &[u8], key: &[u8]) -> Vec<u8> {
    encrypt_bytes(ciphertext, key) // XOR is its own inverse.
}

/// Encode bytes as a lowercase hex string.
fn hex_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for &b in data {
        s.push(char::from(b"0123456789abcdef"[(b >> 4) as usize]));
        s.push(char::from(b"0123456789abcdef"[(b & 0x0f) as usize]));
    }
    s
}

/// Decode a hex string into bytes.
fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return Err(AgentError::UserError {
            message: "Invalid hex string: odd length".to_owned(),
        });
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for chunk in hex.as_bytes().chunks(2) {
        let high = hex_digit(chunk[0])?;
        let low = hex_digit(chunk[1])?;
        bytes.push((high << 4) | low);
    }
    Ok(bytes)
}

/// Convert a single ASCII hex digit to its numeric value.
fn hex_digit(c: u8) -> Result<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(AgentError::UserError {
            message: format!("Invalid hex digit: {}", char::from(c)),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemorySession;
    use serde_json::json;

    fn make_encrypted(key: &[u8]) -> (Arc<InMemorySession>, EncryptedSession) {
        let inner = Arc::new(InMemorySession::new("enc-test"));
        let encrypted = EncryptedSession::new(Arc::clone(&inner) as Arc<dyn Session>, key);
        (inner, encrypted)
    }

    #[tokio::test]
    async fn round_trip_single_item() {
        let (_inner, session) = make_encrypted(b"my-secret-key-1234");
        let item = json!({"role": "user", "content": "hello"});
        session
            .add_items(std::slice::from_ref(&item))
            .await
            .unwrap();

        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0], item);
    }

    #[tokio::test]
    async fn round_trip_multiple_items() {
        let (_inner, session) = make_encrypted(b"key-abc");
        let items = vec![
            json!({"role": "user", "content": "first"}),
            json!({"role": "assistant", "content": "second"}),
            json!({"role": "user", "content": "third"}),
        ];
        session.add_items(&items).await.unwrap();

        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved, items);
    }

    #[tokio::test]
    async fn stored_data_differs_from_plaintext() {
        let (inner, session) = make_encrypted(b"secret");
        let item = json!({"role": "user", "content": "sensitive data"});
        session
            .add_items(std::slice::from_ref(&item))
            .await
            .unwrap();

        // Read raw stored items from the inner session.
        let raw = inner.get_items(None).await.unwrap();
        assert_eq!(raw.len(), 1);
        // The stored item should be an encrypted envelope, not the original.
        assert!(raw[0].get(ENVELOPE_MARKER).is_some());
        assert_ne!(raw[0], item);
        // The plaintext content should not appear in the stored value.
        let raw_str = serde_json::to_string(&raw[0]).unwrap();
        assert!(
            !raw_str.contains("sensitive data"),
            "Plaintext should not appear in encrypted storage"
        );
    }

    #[tokio::test]
    async fn wrong_key_produces_no_items() {
        let (inner, session_a) = make_encrypted(b"key-A");
        let item = json!({"role": "user", "content": "secret"});
        session_a.add_items(&[item]).await.unwrap();

        // Create a new encrypted session with a different key over the same inner store.
        let session_b = EncryptedSession::new(Arc::clone(&inner) as Arc<dyn Session>, b"key-B");
        let retrieved = session_b.get_items(None).await.unwrap();
        // Items encrypted with key-A should not decrypt with key-B (they produce
        // invalid JSON and are silently skipped).
        assert!(
            retrieved.is_empty(),
            "Items should not decrypt with wrong key"
        );
    }

    #[tokio::test]
    async fn pop_item_round_trip() {
        let (_inner, session) = make_encrypted(b"pop-key");
        let items = vec![json!({"seq": 1}), json!({"seq": 2})];
        session.add_items(&items).await.unwrap();

        let popped = session.pop_item().await.unwrap();
        assert_eq!(popped, Some(json!({"seq": 2})));

        let remaining = session.get_items(None).await.unwrap();
        assert_eq!(remaining, vec![json!({"seq": 1})]);
    }

    #[tokio::test]
    async fn pop_item_empty_returns_none() {
        let (_inner, session) = make_encrypted(b"empty-key");
        let popped = session.pop_item().await.unwrap();
        assert_eq!(popped, None);
    }

    #[tokio::test]
    async fn set_history_replaces_all() {
        let (_inner, session) = make_encrypted(b"set-key");
        session.add_items(&[json!({"old": true})]).await.unwrap();

        let new_history = vec![json!({"new": 1}), json!({"new": 2})];
        session.set_history(new_history.clone()).await.unwrap();

        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved, new_history);
    }

    #[tokio::test]
    async fn clear_removes_all() {
        let (_inner, session) = make_encrypted(b"clear-key");
        session.add_items(&[json!({"data": true})]).await.unwrap();
        session.clear().await.unwrap();

        let items = session.get_items(None).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn session_id_matches_inner() {
        let (_inner, session) = make_encrypted(b"id-key");
        assert_eq!(session.session_id(), "enc-test");
    }

    #[tokio::test]
    async fn add_empty_slice_is_noop() {
        let (_inner, session) = make_encrypted(b"noop-key");
        session.add_items(&[]).await.unwrap();
        let items = session.get_items(None).await.unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn encrypted_session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EncryptedSession>();
    }

    #[test]
    fn hex_round_trip() {
        let data = b"hello world";
        let encoded = hex_encode(data);
        let decoded = hex_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn hex_decode_invalid_char() {
        let result = hex_decode("zz");
        assert!(result.is_err());
    }

    #[test]
    fn hex_decode_odd_length() {
        let result = hex_decode("abc");
        assert!(result.is_err());
    }

    #[test]
    fn derive_session_key_different_sessions_produce_different_keys() {
        let master = b"master-secret";
        let key_a = derive_session_key(master, "session-a");
        let key_b = derive_session_key(master, "session-b");
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn encrypt_decrypt_round_trip() {
        let key = b"test-key-12345";
        let plaintext = b"The quick brown fox jumps over the lazy dog";
        let ciphertext = encrypt_bytes(plaintext, key);
        assert_ne!(ciphertext, plaintext.to_vec());
        let decrypted = decrypt_bytes(&ciphertext, key);
        assert_eq!(decrypted, plaintext.to_vec());
    }

    #[test]
    fn debug_impl_redacts_key() {
        let inner = Arc::new(InMemorySession::new("dbg"));
        let session = EncryptedSession::new(inner as Arc<dyn Session>, b"secret");
        let debug_str = format!("{session:?}");
        assert!(debug_str.contains("[REDACTED]"));
        assert!(!debug_str.contains("secret"));
    }
}
