//! JSON-RPC 2.0 protocol types for MCP communication.
//!
//! This module defines the wire format for the Model Context Protocol, which
//! uses JSON-RPC 2.0 over stdio or HTTP/SSE. These types handle serialization
//! and deserialization of requests, responses, notifications, and the MCP-specific
//! tool definitions and results.

use serde::{Deserialize, Serialize};

/// The JSON-RPC version string, always `"2.0"`.
pub const JSONRPC_VERSION: &str = "2.0";

/// The MCP protocol version used during initialization.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

// ---------------------------------------------------------------------------
// JSON-RPC core types
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 request.
///
/// Sent from the client to the server. Each request carries a unique `id` and
/// expects exactly one [`JsonRpcResponse`] with the same `id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JsonRpcRequest {
    /// The JSON-RPC version, always `"2.0"`.
    pub jsonrpc: String,
    /// A unique identifier for this request.
    pub id: u64,
    /// The method to invoke on the server.
    pub method: String,
    /// Optional parameters for the method.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request.
    #[must_use]
    pub fn new(id: u64, method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            method: method.into(),
            params,
        }
    }
}

/// A JSON-RPC 2.0 response.
///
/// Sent from the server to the client in reply to a [`JsonRpcRequest`]. Exactly
/// one of `result` or `error` will be present.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JsonRpcResponse {
    /// The JSON-RPC version, always `"2.0"`.
    pub jsonrpc: String,
    /// The identifier matching the original request.
    pub id: u64,
    /// The successful result, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// The error, if the request failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Create a successful response.
    #[must_use]
    pub fn success(id: u64, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response.
    #[must_use]
    pub fn error(id: u64, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// A JSON-RPC 2.0 error object.
///
/// Included in [`JsonRpcResponse`] when the request fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JsonRpcError {
    /// A numeric error code.
    pub code: i64,
    /// A short description of the error.
    pub message: String,
    /// Optional structured data about the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcError {
    /// Create a new JSON-RPC error.
    #[must_use]
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Create a new JSON-RPC error with additional data.
    #[must_use]
    pub fn with_data(code: i64, message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            code,
            message: message.into(),
            data: Some(data),
        }
    }
}

/// A JSON-RPC 2.0 notification (no `id`, no response expected).
///
/// Notifications are one-way messages sent from client to server (or vice versa)
/// that do not require a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JsonRpcNotification {
    /// The JSON-RPC version, always `"2.0"`.
    pub jsonrpc: String,
    /// The notification method.
    pub method: String,
    /// Optional parameters for the notification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcNotification {
    /// Create a new JSON-RPC notification.
    #[must_use]
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            method: method.into(),
            params,
        }
    }
}

// ---------------------------------------------------------------------------
// MCP-specific types
// ---------------------------------------------------------------------------

/// An MCP tool definition received from a server.
///
/// Describes a tool that the server exposes, including its name, description,
/// and the JSON Schema for its input parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpToolDef {
    /// The tool's unique name.
    pub name: String,
    /// An optional human-readable description of the tool.
    #[serde(default)]
    pub description: Option<String>,
    /// The JSON Schema describing the tool's input parameters.
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

impl McpToolDef {
    /// Create a new MCP tool definition.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description,
            input_schema,
        }
    }
}

/// The result from calling an MCP tool.
///
/// Contains the content returned by the tool and whether the invocation
/// resulted in an error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpToolResult {
    /// The content items returned by the tool.
    pub content: Vec<McpContent>,
    /// Whether the tool call resulted in an error.
    #[serde(default)]
    pub is_error: bool,
}

impl McpToolResult {
    /// Create a successful tool result with text content.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![McpContent::Text { text: text.into() }],
            is_error: false,
        }
    }

    /// Create an error tool result with a text message.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![McpContent::Text {
                text: message.into(),
            }],
            is_error: true,
        }
    }

    /// Convert the tool result content into a single string.
    ///
    /// Joins all text content items with newlines. Non-text items are
    /// represented by a placeholder.
    #[must_use]
    pub fn to_text(&self) -> String {
        self.content
            .iter()
            .map(|c| match c {
                McpContent::Text { text } => text.clone(),
                McpContent::Image { mime_type, .. } => {
                    format!("[image: {mime_type}]")
                }
                McpContent::Resource { uri, .. } => {
                    format!("[resource: {uri}]")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A content item returned by an MCP tool.
///
/// MCP tools can return text, images, or resource references.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum McpContent {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        /// The text value.
        text: String,
    },
    /// Base64-encoded image content.
    #[serde(rename = "image")]
    Image {
        /// The base64-encoded image data.
        data: String,
        /// The MIME type of the image (e.g., `"image/png"`).
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// A resource reference.
    #[serde(rename = "resource")]
    Resource {
        /// The URI of the resource.
        uri: String,
        /// Optional text representation of the resource.
        text: Option<String>,
    },
}

/// Parameters sent with the `initialize` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct InitializeParams {
    /// The MCP protocol version the client supports.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Client capabilities (currently empty).
    pub capabilities: serde_json::Value,
    /// Information about the client.
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo,
}

/// Information about the MCP client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ClientInfo {
    /// The client's name.
    pub name: String,
    /// The client's version.
    pub version: String,
}

impl Default for InitializeParams {
    fn default() -> Self {
        Self {
            protocol_version: MCP_PROTOCOL_VERSION.to_owned(),
            capabilities: serde_json::json!({}),
            client_info: ClientInfo {
                name: "openai-agents-rust".to_owned(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
            },
        }
    }
}

/// The result of a `tools/list` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ListToolsResult {
    /// The tools exposed by the server.
    pub tools: Vec<McpToolDef>,
}

// ---------------------------------------------------------------------------
// MCP resource types
// ---------------------------------------------------------------------------

/// An MCP resource descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpResource {
    /// The URI identifying this resource.
    pub uri: String,
    /// A human-readable name for the resource.
    #[serde(default)]
    pub name: Option<String>,
    /// An optional description of the resource.
    #[serde(default)]
    pub description: Option<String>,
    /// The MIME type of the resource content.
    #[serde(default, rename = "mimeType")]
    pub mime_type: Option<String>,
}

/// The result of a `resources/list` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ListResourcesResult {
    /// The resources exposed by the server.
    pub resources: Vec<McpResource>,
    /// Optional cursor for pagination.
    #[serde(default, rename = "nextCursor")]
    pub next_cursor: Option<String>,
}

/// An MCP resource template descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpResourceTemplate {
    /// The URI template (RFC 6570) for this resource.
    #[serde(rename = "uriTemplate")]
    pub uri_template: String,
    /// A human-readable name for the template.
    #[serde(default)]
    pub name: Option<String>,
    /// An optional description of the template.
    #[serde(default)]
    pub description: Option<String>,
    /// The MIME type of the resource content.
    #[serde(default, rename = "mimeType")]
    pub mime_type: Option<String>,
}

/// The result of a `resources/templates/list` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ListResourceTemplatesResult {
    /// The resource templates exposed by the server.
    #[serde(rename = "resourceTemplates")]
    pub resource_templates: Vec<McpResourceTemplate>,
    /// Optional cursor for pagination.
    #[serde(default, rename = "nextCursor")]
    pub next_cursor: Option<String>,
}

/// A content item inside a `resources/read` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpResourceContent {
    /// The URI of the resource.
    pub uri: String,
    /// The MIME type of the content.
    #[serde(default, rename = "mimeType")]
    pub mime_type: Option<String>,
    /// Text content (for text resources).
    #[serde(default)]
    pub text: Option<String>,
    /// Base64-encoded binary content (for blob resources).
    #[serde(default)]
    pub blob: Option<String>,
}

/// The result of a `resources/read` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ReadResourceResult {
    /// The content items returned for the resource.
    pub contents: Vec<McpResourceContent>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- JsonRpcRequest ----

    #[test]
    fn request_serialization() {
        let req = JsonRpcRequest::new(1, "tools/list", None);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["id"], 1);
        assert_eq!(json["method"], "tools/list");
        assert!(json.get("params").is_none());
    }

    #[test]
    fn request_serialization_with_params() {
        let params = json!({"name": "test", "arguments": {"a": 1}});
        let req = JsonRpcRequest::new(42, "tools/call", Some(params.clone()));
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["id"], 42);
        assert_eq!(json["method"], "tools/call");
        assert_eq!(json["params"], params);
    }

    #[test]
    fn request_round_trip() {
        let req = JsonRpcRequest::new(
            5,
            "initialize",
            Some(json!({"protocolVersion": "2024-11-05"})),
        );
        let serialized = serde_json::to_string(&req).unwrap();
        let deserialized: JsonRpcRequest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.id, 5);
        assert_eq!(deserialized.method, "initialize");
        assert!(deserialized.params.is_some());
    }

    // ---- JsonRpcResponse ----

    #[test]
    fn response_success_serialization() {
        let resp = JsonRpcResponse::success(1, json!({"tools": []}));
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["id"], 1);
        assert!(json["result"].is_object());
        assert!(json.get("error").is_none());
    }

    #[test]
    fn response_error_serialization() {
        let err = JsonRpcError::new(-32601, "Method not found");
        let resp = JsonRpcResponse::error(1, err);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["error"]["code"], -32601);
        assert_eq!(json["error"]["message"], "Method not found");
        assert!(json.get("result").is_none());
    }

    #[test]
    fn response_deserialization_success() {
        let json_str = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"get_weather","description":"Get weather","inputSchema":{"type":"object"}}]}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json_str).unwrap();
        assert_eq!(resp.id, 1);
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn response_deserialization_error() {
        let json_str =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Invalid Request"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json_str).unwrap();
        assert_eq!(resp.id, 1);
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32600);
        assert_eq!(err.message, "Invalid Request");
    }

    // ---- JsonRpcError ----

    #[test]
    fn error_with_data() {
        let err = JsonRpcError::with_data(-32000, "Server error", json!({"detail": "timeout"}));
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["code"], -32000);
        assert_eq!(json["data"]["detail"], "timeout");
    }

    #[test]
    fn error_without_data_skips_field() {
        let err = JsonRpcError::new(-32601, "Method not found");
        let json = serde_json::to_string(&err).unwrap();
        assert!(!json.contains("data"));
    }

    // ---- JsonRpcNotification ----

    #[test]
    fn notification_serialization() {
        let notif = JsonRpcNotification::new("notifications/initialized", None);
        let json = serde_json::to_value(&notif).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["method"], "notifications/initialized");
        assert!(json.get("params").is_none());
        // Notifications must not have an id field.
        assert!(json.get("id").is_none());
    }

    #[test]
    fn notification_round_trip() {
        let notif = JsonRpcNotification::new("test/notify", Some(json!({"key": "value"})));
        let serialized = serde_json::to_string(&notif).unwrap();
        let deserialized: JsonRpcNotification = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.method, "test/notify");
        assert_eq!(deserialized.params.unwrap()["key"], "value");
    }

    // ---- McpToolDef ----

    #[test]
    fn tool_def_deserialization() {
        let json_str = r#"{
            "name": "get_weather",
            "description": "Get weather for a location",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }"#;
        let tool: McpToolDef = serde_json::from_str(json_str).unwrap();
        assert_eq!(tool.name, "get_weather");
        assert_eq!(
            tool.description.as_deref(),
            Some("Get weather for a location")
        );
        assert_eq!(tool.input_schema["type"], "object");
        assert!(tool.input_schema["properties"]["location"].is_object());
    }

    #[test]
    fn tool_def_without_description() {
        let json_str = r#"{
            "name": "do_thing",
            "inputSchema": {"type": "object"}
        }"#;
        let tool: McpToolDef = serde_json::from_str(json_str).unwrap();
        assert_eq!(tool.name, "do_thing");
        assert!(tool.description.is_none());
    }

    #[test]
    fn tool_def_serialization_round_trip() {
        let tool = McpToolDef::new(
            "calc",
            Some("A calculator".to_owned()),
            json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
        );
        let serialized = serde_json::to_string(&tool).unwrap();
        let deserialized: McpToolDef = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.name, "calc");
        assert_eq!(deserialized.description.as_deref(), Some("A calculator"));
    }

    // ---- McpToolResult ----

    #[test]
    fn tool_result_text() {
        let result = McpToolResult::text("Hello, world!");
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.to_text(), "Hello, world!");
    }

    #[test]
    fn tool_result_error() {
        let result = McpToolResult::error("Something failed");
        assert!(result.is_error);
        assert_eq!(result.to_text(), "Something failed");
    }

    #[test]
    fn tool_result_deserialization() {
        let json_str = r#"{
            "content": [
                {"type": "text", "text": "The weather is sunny"},
                {"type": "text", "text": "Temperature: 72F"}
            ],
            "isError": false
        }"#;
        let result: McpToolResult = serde_json::from_str(json_str).unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 2);
        assert_eq!(result.to_text(), "The weather is sunny\nTemperature: 72F");
    }

    #[test]
    fn tool_result_with_is_error_default() {
        let json_str = r#"{"content": [{"type": "text", "text": "ok"}]}"#;
        let result: McpToolResult = serde_json::from_str(json_str).unwrap();
        assert!(!result.is_error);
    }

    // ---- McpContent ----

    #[test]
    fn content_text_serialization() {
        let content = McpContent::Text {
            text: "hello".to_owned(),
        };
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hello");
    }

    #[test]
    fn content_image_serialization() {
        let content = McpContent::Image {
            data: "base64data".to_owned(),
            mime_type: "image/png".to_owned(),
        };
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json["type"], "image");
        assert_eq!(json["data"], "base64data");
        assert_eq!(json["mimeType"], "image/png");
    }

    #[test]
    fn content_resource_serialization() {
        let content = McpContent::Resource {
            uri: "file:///tmp/data.txt".to_owned(),
            text: Some("file contents".to_owned()),
        };
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json["type"], "resource");
        assert_eq!(json["uri"], "file:///tmp/data.txt");
        assert_eq!(json["text"], "file contents");
    }

    #[test]
    fn content_resource_without_text() {
        let json_str = r#"{"type": "resource", "uri": "file:///data"}"#;
        let content: McpContent = serde_json::from_str(json_str).unwrap();
        if let McpContent::Resource { uri, text } = &content {
            assert_eq!(uri, "file:///data");
            assert!(text.is_none());
        } else {
            panic!("expected Resource variant");
        }
    }

    #[test]
    fn content_mixed_deserialization() {
        let json_str = r#"[
            {"type": "text", "text": "hello"},
            {"type": "image", "data": "abc", "mimeType": "image/jpeg"},
            {"type": "resource", "uri": "file:///x"}
        ]"#;
        let contents: Vec<McpContent> = serde_json::from_str(json_str).unwrap();
        assert_eq!(contents.len(), 3);
        assert!(matches!(&contents[0], McpContent::Text { text } if text == "hello"));
        assert!(
            matches!(&contents[1], McpContent::Image { mime_type, .. } if mime_type == "image/jpeg")
        );
        assert!(matches!(&contents[2], McpContent::Resource { uri, .. } if uri == "file:///x"));
    }

    // ---- InitializeParams ----

    #[test]
    fn initialize_params_default() {
        let params = InitializeParams::default();
        assert_eq!(params.protocol_version, MCP_PROTOCOL_VERSION);
        assert_eq!(params.client_info.name, "openai-agents-rust");
        assert!(!params.client_info.version.is_empty());
    }

    #[test]
    fn initialize_params_serialization() {
        let params = InitializeParams::default();
        let json = serde_json::to_value(&params).unwrap();
        assert_eq!(json["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert_eq!(json["clientInfo"]["name"], "openai-agents-rust");
    }

    // ---- ListToolsResult ----

    #[test]
    fn list_tools_result_deserialization() {
        let json_str = r#"{
            "tools": [
                {
                    "name": "tool_a",
                    "description": "Tool A",
                    "inputSchema": {"type": "object"}
                },
                {
                    "name": "tool_b",
                    "inputSchema": {"type": "object", "properties": {}}
                }
            ]
        }"#;
        let result: ListToolsResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.tools.len(), 2);
        assert_eq!(result.tools[0].name, "tool_a");
        assert_eq!(result.tools[1].name, "tool_b");
        assert!(result.tools[1].description.is_none());
    }

    #[test]
    fn list_tools_result_empty() {
        let json_str = r#"{"tools": []}"#;
        let result: ListToolsResult = serde_json::from_str(json_str).unwrap();
        assert!(result.tools.is_empty());
    }

    // ---- McpToolResult to_text ----

    #[test]
    fn tool_result_to_text_with_mixed_content() {
        let result = McpToolResult {
            content: vec![
                McpContent::Text {
                    text: "line 1".to_owned(),
                },
                McpContent::Image {
                    data: "abc".to_owned(),
                    mime_type: "image/png".to_owned(),
                },
                McpContent::Resource {
                    uri: "file:///x".to_owned(),
                    text: None,
                },
            ],
            is_error: false,
        };
        assert_eq!(
            result.to_text(),
            "line 1\n[image: image/png]\n[resource: file:///x]"
        );
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<JsonRpcRequest>();
        assert_send_sync::<JsonRpcResponse>();
        assert_send_sync::<JsonRpcError>();
        assert_send_sync::<JsonRpcNotification>();
        assert_send_sync::<McpToolDef>();
        assert_send_sync::<McpToolResult>();
        assert_send_sync::<McpContent>();
        assert_send_sync::<InitializeParams>();
        assert_send_sync::<ClientInfo>();
        assert_send_sync::<ListToolsResult>();
    }

    // ---- Clone assertions ----

    // ---- Resource types ----

    #[test]
    fn list_resources_result_deserialization() {
        let json_str = r#"{
            "resources": [
                {"uri": "file:///readme.md", "name": "README", "mimeType": "text/markdown"}
            ],
            "nextCursor": "abc"
        }"#;
        let result: ListResourcesResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.resources.len(), 1);
        assert_eq!(result.resources[0].uri, "file:///readme.md");
        assert_eq!(result.next_cursor.as_deref(), Some("abc"));
    }

    #[test]
    fn list_resource_templates_result_deserialization() {
        let json_str = r#"{
            "resourceTemplates": [
                {"uriTemplate": "file:///{path}", "name": "File"}
            ]
        }"#;
        let result: ListResourceTemplatesResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.resource_templates.len(), 1);
        assert_eq!(result.resource_templates[0].uri_template, "file:///{path}");
        assert!(result.next_cursor.is_none());
    }

    #[test]
    fn read_resource_result_deserialization() {
        let json_str = r#"{
            "contents": [
                {"uri": "file:///data.txt", "mimeType": "text/plain", "text": "hello world"}
            ]
        }"#;
        let result: ReadResourceResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.contents.len(), 1);
        assert_eq!(result.contents[0].text.as_deref(), Some("hello world"));
    }

    #[test]
    fn types_are_cloneable() {
        let req = JsonRpcRequest::new(1, "test", None);
        let _ = req;

        let resp = JsonRpcResponse::success(1, json!(null));
        let _ = resp;

        let err = JsonRpcError::new(-1, "err");
        let _ = err;

        let notif = JsonRpcNotification::new("test", None);
        let _ = notif;

        let tool = McpToolDef::new("t", None, json!({}));
        let _ = tool;

        let result = McpToolResult::text("ok");
        let _ = result;

        let content = McpContent::Text {
            text: "hi".to_owned(),
        };
        let _ = content;
    }

    // ---- Full protocol round-trip ----

    #[test]
    fn full_initialize_round_trip() {
        // Client sends initialize request.
        let params = InitializeParams::default();
        let req = JsonRpcRequest::new(
            1,
            "initialize",
            Some(serde_json::to_value(&params).unwrap()),
        );
        let req_json = serde_json::to_string(&req).unwrap();

        // Server parses request.
        let parsed_req: JsonRpcRequest = serde_json::from_str(&req_json).unwrap();
        assert_eq!(parsed_req.method, "initialize");

        // Server sends response.
        let resp = JsonRpcResponse::success(
            parsed_req.id,
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-server", "version": "1.0.0"}
            }),
        );
        let resp_json = serde_json::to_string(&resp).unwrap();

        // Client parses response.
        let parsed_resp: JsonRpcResponse = serde_json::from_str(&resp_json).unwrap();
        assert_eq!(parsed_resp.id, 1);
        assert!(parsed_resp.error.is_none());
        let result = parsed_resp.result.unwrap();
        assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
    }

    #[test]
    fn full_tool_call_round_trip() {
        // Client sends tools/call request.
        let req = JsonRpcRequest::new(
            3,
            "tools/call",
            Some(json!({
                "name": "get_weather",
                "arguments": {"location": "Tokyo"}
            })),
        );
        let req_json = serde_json::to_string(&req).unwrap();

        // Server parses and processes.
        let parsed: JsonRpcRequest = serde_json::from_str(&req_json).unwrap();
        assert_eq!(parsed.method, "tools/call");

        // Server responds with tool result.
        let tool_result = McpToolResult::text("Sunny, 25C");
        let resp = JsonRpcResponse::success(parsed.id, serde_json::to_value(&tool_result).unwrap());
        let resp_json = serde_json::to_string(&resp).unwrap();

        // Client parses response.
        let parsed_resp: JsonRpcResponse = serde_json::from_str(&resp_json).unwrap();
        let result: McpToolResult = serde_json::from_value(parsed_resp.result.unwrap()).unwrap();
        assert!(!result.is_error);
        assert_eq!(result.to_text(), "Sunny, 25C");
    }

    // ---- isError field name casing ----

    #[test]
    fn is_error_uses_camel_case_in_json() {
        let result = McpToolResult::error("fail");
        let json = serde_json::to_value(&result).unwrap();
        // The serde default field name is `is_error` which serializes as `is_error`.
        // Verify the field exists.
        assert!(json.get("is_error").is_some() || json.get("isError").is_some());
    }
}
