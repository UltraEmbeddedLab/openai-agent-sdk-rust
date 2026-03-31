//! Data types for realtime conversation items.
//!
//! These types represent the messages, tool calls, and responses that flow
//! through a realtime session.  They mirror the Python SDK's `realtime/items.py`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Content types
// ---------------------------------------------------------------------------

/// Text input content in a realtime message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct InputTextContent {
    /// The text content.
    pub text: Option<String>,
}

/// Audio input content in a realtime message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct InputAudioContent {
    /// Base64-encoded audio data.
    pub audio: Option<String>,
    /// Transcript of the audio, if available.
    pub transcript: Option<String>,
}

/// Text content from the assistant in realtime responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AssistantTextContent {
    /// The text content from the assistant.
    pub text: Option<String>,
}

/// Audio content from the assistant in realtime responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AssistantAudioContent {
    /// Base64-encoded audio data from the assistant.
    pub audio: Option<String>,
    /// Transcript of the audio response.
    pub transcript: Option<String>,
}

// ---------------------------------------------------------------------------
// User input content enum
// ---------------------------------------------------------------------------

/// Content that can appear in a user message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum UserContent {
    /// Text input.
    #[serde(rename = "input_text")]
    Text(InputTextContent),
    /// Audio input.
    #[serde(rename = "input_audio")]
    Audio(InputAudioContent),
}

/// Content that can appear in an assistant message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AssistantContent {
    /// Text content.
    #[serde(rename = "text")]
    Text(AssistantTextContent),
    /// Audio content.
    #[serde(rename = "audio")]
    Audio(AssistantAudioContent),
}

// ---------------------------------------------------------------------------
// Message items
// ---------------------------------------------------------------------------

/// A system message item in a realtime conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SystemMessageItem {
    /// Unique identifier for this message item.
    pub item_id: String,
    /// ID of the previous item in the conversation.
    pub previous_item_id: Option<String>,
    /// The text content of the system message.
    pub content: Vec<InputTextContent>,
}

/// A user message item in a realtime conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct UserMessageItem {
    /// Unique identifier for this message item.
    pub item_id: String,
    /// ID of the previous item in the conversation.
    pub previous_item_id: Option<String>,
    /// The content items (text or audio) in the user message.
    pub content: Vec<UserContent>,
}

/// Status of an assistant's response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResponseStatus {
    /// The response is still being generated.
    #[serde(rename = "in_progress")]
    InProgress,
    /// The response has been fully generated.
    #[serde(rename = "completed")]
    Completed,
    /// The response was cut short.
    #[serde(rename = "incomplete")]
    Incomplete,
}

/// An assistant message item in a realtime conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AssistantMessageItem {
    /// Unique identifier for this message item.
    pub item_id: String,
    /// ID of the previous item in the conversation.
    pub previous_item_id: Option<String>,
    /// The status of the assistant's response.
    pub status: Option<ResponseStatus>,
    /// The content items (text or audio) from the assistant.
    pub content: Vec<AssistantContent>,
}

/// A message item in a realtime conversation, from any role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role")]
#[non_exhaustive]
pub enum RealtimeMessageItem {
    /// A system message.
    #[serde(rename = "system")]
    System(SystemMessageItem),
    /// A user message.
    #[serde(rename = "user")]
    User(UserMessageItem),
    /// An assistant message.
    #[serde(rename = "assistant")]
    Assistant(AssistantMessageItem),
}

// ---------------------------------------------------------------------------
// Tool call items
// ---------------------------------------------------------------------------

/// Status of a tool call execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolCallStatus {
    /// The tool call is currently being executed.
    #[serde(rename = "in_progress")]
    InProgress,
    /// The tool call has completed.
    #[serde(rename = "completed")]
    Completed,
}

/// A tool call item in a realtime conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RealtimeToolCallItem {
    /// Unique identifier for this tool call item.
    pub item_id: String,
    /// ID of the previous item in the conversation.
    pub previous_item_id: Option<String>,
    /// The call ID for this tool invocation.
    pub call_id: Option<String>,
    /// The status of the tool call execution.
    pub status: ToolCallStatus,
    /// The JSON string arguments passed to the tool.
    pub arguments: String,
    /// The name of the tool being called.
    pub name: String,
    /// The output result from the tool execution, if available.
    pub output: Option<String>,
}

// ---------------------------------------------------------------------------
// RealtimeItem
// ---------------------------------------------------------------------------

/// A realtime item that can be a message or a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum RealtimeItem {
    /// A message item (system, user, or assistant).
    #[serde(rename = "message")]
    Message(RealtimeMessageItem),
    /// A function call item.
    #[serde(rename = "function_call")]
    ToolCall(RealtimeToolCallItem),
}

// ---------------------------------------------------------------------------
// RealtimeResponse
// ---------------------------------------------------------------------------

/// A response from the realtime model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RealtimeResponse {
    /// Unique identifier for this response.
    pub id: String,
    /// The message items in the response.
    pub output: Vec<RealtimeMessageItem>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- InputTextContent ----

    #[test]
    fn input_text_content() {
        let content = InputTextContent {
            text: Some("Hello".to_owned()),
        };
        let json = serde_json::to_string(&content).expect("serialize");
        let deserialized: InputTextContent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, content);
    }

    // ---- InputAudioContent ----

    #[test]
    fn input_audio_content() {
        let content = InputAudioContent {
            audio: Some("base64data".to_owned()),
            transcript: Some("hello".to_owned()),
        };
        assert_eq!(content.transcript.as_deref(), Some("hello"));
    }

    // ---- ToolCallStatus ----

    #[test]
    fn tool_call_status_serialization() {
        let status = ToolCallStatus::InProgress;
        let json = serde_json::to_string(&status).expect("serialize");
        assert_eq!(json, r#""in_progress""#);
    }

    #[test]
    fn tool_call_status_equality() {
        assert_eq!(ToolCallStatus::Completed, ToolCallStatus::Completed);
        assert_ne!(ToolCallStatus::InProgress, ToolCallStatus::Completed);
    }

    // ---- ResponseStatus ----

    #[test]
    fn response_status_serialization() {
        let status = ResponseStatus::Completed;
        let json = serde_json::to_string(&status).expect("serialize");
        assert_eq!(json, r#""completed""#);
    }

    // ---- RealtimeToolCallItem ----

    #[test]
    fn tool_call_item() {
        let item = RealtimeToolCallItem {
            item_id: "item-1".to_owned(),
            previous_item_id: None,
            call_id: Some("call-1".to_owned()),
            status: ToolCallStatus::Completed,
            arguments: r#"{"city": "NYC"}"#.to_owned(),
            name: "get_weather".to_owned(),
            output: Some("Sunny".to_owned()),
        };
        assert_eq!(item.name, "get_weather");
        assert_eq!(item.status, ToolCallStatus::Completed);
    }

    // ---- RealtimeResponse ----

    #[test]
    fn realtime_response() {
        let response = RealtimeResponse {
            id: "resp-1".to_owned(),
            output: vec![],
        };
        assert_eq!(response.id, "resp-1");
        assert!(response.output.is_empty());
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn item_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InputTextContent>();
        assert_send_sync::<InputAudioContent>();
        assert_send_sync::<AssistantTextContent>();
        assert_send_sync::<AssistantAudioContent>();
        assert_send_sync::<SystemMessageItem>();
        assert_send_sync::<UserMessageItem>();
        assert_send_sync::<AssistantMessageItem>();
        assert_send_sync::<RealtimeMessageItem>();
        assert_send_sync::<RealtimeToolCallItem>();
        assert_send_sync::<RealtimeItem>();
        assert_send_sync::<RealtimeResponse>();
        assert_send_sync::<ToolCallStatus>();
        assert_send_sync::<ResponseStatus>();
    }
}
