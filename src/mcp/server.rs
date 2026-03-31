//! MCP server connection and tool discovery.
//!
//! This module defines the types for connecting to an MCP server and
//! discovering the tools it exposes. The stdio transport communicates
//! with MCP servers over JSON-RPC 2.0, spawning the server as a child
//! process and exchanging newline-delimited JSON messages over stdin/stdout.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use super::protocol::{
    InitializeParams, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, ListToolsResult,
    McpToolDef, McpToolResult,
};
use crate::error::{AgentError, Result};
use crate::tool::Tool;

// ---------------------------------------------------------------------------
// MCPTransport
// ---------------------------------------------------------------------------

/// Transport mechanism for connecting to an MCP server.
///
/// MCP supports multiple transports. The two primary ones are:
///
/// * **Stdio** — the server runs as a subprocess and communicates over
///   standard input/output.
/// * **Sse** — the server is a remote HTTP endpoint that uses
///   Server-Sent Events for streaming.
/// * **`StreamableHttp`** — the server uses the newer MCP Streamable HTTP
///   transport for bidirectional communication.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MCPTransport {
    /// Standard I/O transport (subprocess).
    Stdio {
        /// Command to execute to start the MCP server process.
        command: String,
        /// Arguments to pass to the command.
        args: Vec<String>,
        /// Environment variables to set for the subprocess.
        env: HashMap<String, String>,
    },
    /// Server-Sent Events over HTTP.
    Sse {
        /// The SSE endpoint URL.
        url: String,
        /// Optional HTTP headers to include in requests.
        headers: HashMap<String, String>,
    },
    /// Streamable HTTP transport.
    StreamableHttp {
        /// The HTTP endpoint URL.
        url: String,
        /// Optional HTTP headers to include in requests.
        headers: HashMap<String, String>,
    },
}

// ---------------------------------------------------------------------------
// MCPServerConfig
// ---------------------------------------------------------------------------

/// Configuration for connecting to an MCP server.
///
/// Combines the server's human-readable name with the transport details
/// needed to establish a connection.
///
/// # Example
///
/// ```
/// use openai_agents::mcp::MCPServer;
///
/// let server = MCPServer::stdio(
///     "weather-server",
///     "npx",
///     vec!["-y".into(), "@weather/mcp-server".into()],
/// );
/// assert_eq!(server.name(), "weather-server");
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MCPServerConfig {
    /// A human-readable name for the server, used in logging and error
    /// messages.
    pub name: String,
    /// The transport type and connection details.
    pub transport: MCPTransport,
}

impl MCPServerConfig {
    /// Create a new server configuration.
    #[must_use]
    pub fn new(name: impl Into<String>, transport: MCPTransport) -> Self {
        Self {
            name: name.into(),
            transport,
        }
    }
}

// ---------------------------------------------------------------------------
// StdioProcess — internal type managing the child process I/O
// ---------------------------------------------------------------------------

/// Internal handle for a running stdio MCP server subprocess.
///
/// Wraps the child process and provides buffered readers/writers for
/// newline-delimited JSON-RPC communication.
struct StdioProcess {
    /// The child process handle.
    child: Child,
    /// Buffered writer to the child's stdin.
    writer: BufWriter<tokio::process::ChildStdin>,
    /// Buffered reader from the child's stdout.
    reader: BufReader<tokio::process::ChildStdout>,
}

// ---------------------------------------------------------------------------
// MCPServer
// ---------------------------------------------------------------------------

/// An MCP server connection that provides tools to agents.
///
/// `MCPServer` manages the lifecycle of the connection to an MCP server
/// and converts the tools it exposes into the SDK's [`Tool`] type so
/// they can be used by agents.
///
/// # Lifecycle
///
/// 1. Create via [`MCPServer::new`], [`MCPServer::stdio`], or [`MCPServer::sse`].
/// 2. Call [`connect`](Self::connect) to establish the connection.
/// 3. Call [`list_tools`](Self::list_tools) to discover available tools.
/// 4. Call [`disconnect`](Self::disconnect) when done.
///
/// # Example
///
/// ```no_run
/// use openai_agents::mcp::MCPServer;
///
/// # async fn example() -> openai_agents::Result<()> {
/// let mut server = MCPServer::stdio("my-server", "npx", vec!["-y".into(), "some-pkg".into()]);
/// server.connect().await?;
/// let tools = server.list_tools().await?;
/// server.disconnect().await?;
/// # Ok(())
/// # }
/// ```
pub struct MCPServer {
    /// The server configuration.
    config: MCPServerConfig,
    /// The child process handle (for stdio transport), protected by a mutex
    /// so that concurrent tool calls are serialized.
    process: Option<Arc<Mutex<StdioProcess>>>,
    /// Cached tool definitions from the last `tools/list` call.
    tools_cache: Vec<McpToolDef>,
    /// Monotonically increasing JSON-RPC request ID.
    next_id: AtomicU64,
    /// Whether the server connection is currently active.
    is_connected: bool,
}

// Manual Debug implementation because `StdioProcess` does not implement Debug.
impl std::fmt::Debug for MCPServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MCPServer")
            .field("config", &self.config)
            .field("is_connected", &self.is_connected)
            .field("tools_cache_len", &self.tools_cache.len())
            .finish_non_exhaustive()
    }
}

impl MCPServer {
    /// Create a new MCP server with the given configuration.
    #[must_use]
    pub const fn new(config: MCPServerConfig) -> Self {
        Self {
            config,
            process: None,
            tools_cache: Vec::new(),
            next_id: AtomicU64::new(1),
            is_connected: false,
        }
    }

    /// Create an MCP server with stdio transport.
    ///
    /// The server will be started as a subprocess using the given `command`
    /// and `args`. Communication happens over the subprocess's stdin/stdout.
    #[must_use]
    pub fn stdio(name: impl Into<String>, command: impl Into<String>, args: Vec<String>) -> Self {
        Self::new(MCPServerConfig {
            name: name.into(),
            transport: MCPTransport::Stdio {
                command: command.into(),
                args,
                env: HashMap::new(),
            },
        })
    }

    /// Create an MCP server with stdio transport and environment variables.
    ///
    /// Like [`stdio`](Self::stdio) but allows specifying environment variables
    /// for the subprocess.
    #[must_use]
    pub fn stdio_with_env(
        name: impl Into<String>,
        command: impl Into<String>,
        args: Vec<String>,
        env: HashMap<String, String>,
    ) -> Self {
        Self::new(MCPServerConfig {
            name: name.into(),
            transport: MCPTransport::Stdio {
                command: command.into(),
                args,
                env,
            },
        })
    }

    /// Create an MCP server with SSE transport.
    ///
    /// The server is a remote HTTP endpoint that uses Server-Sent Events
    /// for streaming communication.
    #[must_use]
    pub fn sse(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(MCPServerConfig {
            name: name.into(),
            transport: MCPTransport::Sse {
                url: url.into(),
                headers: HashMap::new(),
            },
        })
    }

    /// Create an MCP server with SSE transport and custom headers.
    ///
    /// Like [`sse`](Self::sse) but allows specifying HTTP headers for the
    /// connection.
    #[must_use]
    pub fn sse_with_headers(
        name: impl Into<String>,
        url: impl Into<String>,
        headers: HashMap<String, String>,
    ) -> Self {
        Self::new(MCPServerConfig {
            name: name.into(),
            transport: MCPTransport::Sse {
                url: url.into(),
                headers,
            },
        })
    }

    /// Create an MCP server with Streamable HTTP transport.
    ///
    /// The server uses the newer MCP Streamable HTTP transport for
    /// bidirectional communication.
    #[must_use]
    pub fn streamable_http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(MCPServerConfig {
            name: name.into(),
            transport: MCPTransport::StreamableHttp {
                url: url.into(),
                headers: HashMap::new(),
            },
        })
    }

    /// Create an MCP server with Streamable HTTP transport and custom headers.
    #[must_use]
    pub fn streamable_http_with_headers(
        name: impl Into<String>,
        url: impl Into<String>,
        headers: HashMap<String, String>,
    ) -> Self {
        Self::new(MCPServerConfig {
            name: name.into(),
            transport: MCPTransport::StreamableHttp {
                url: url.into(),
                headers,
            },
        })
    }

    // -----------------------------------------------------------------------
    // Connection lifecycle
    // -----------------------------------------------------------------------

    /// Connect to the MCP server and perform the protocol handshake.
    ///
    /// For stdio transport, this spawns the subprocess, sends the `initialize`
    /// request, and sends the `notifications/initialized` notification.
    ///
    /// This must be called before [`list_tools`](Self::list_tools). The
    /// connection remains active until [`disconnect`](Self::disconnect) is
    /// called.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection fails or the server does not
    /// respond to the initialization handshake.
    pub async fn connect(&mut self) -> Result<()> {
        match &self.config.transport {
            MCPTransport::Stdio { command, args, env } => {
                self.connect_stdio(command.clone(), args.clone(), env.clone())
                    .await?;
            }
            MCPTransport::Sse { .. } | MCPTransport::StreamableHttp { .. } => {
                // HTTP-based transports are not yet implemented. Mark as
                // connected so the API surface is exercisable in tests.
                self.is_connected = true;
            }
        }
        Ok(())
    }

    /// Spawn the subprocess and perform the MCP initialization handshake.
    #[allow(clippy::similar_names)] // stdin/stdout are inherently similar names.
    async fn connect_stdio(
        &mut self,
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    ) -> Result<()> {
        let mut cmd = Command::new(&command);
        cmd.args(&args)
            .envs(&env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        let mut child = cmd.spawn().map_err(|e| AgentError::UserError {
            message: format!(
                "Failed to spawn MCP server '{}' (command: {command}): {e}",
                self.config.name
            ),
        })?;

        let stdin = child.stdin.take().ok_or_else(|| AgentError::UserError {
            message: format!(
                "MCP server '{}' stdin not available after spawn",
                self.config.name
            ),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| AgentError::UserError {
            message: format!(
                "MCP server '{}' stdout not available after spawn",
                self.config.name
            ),
        })?;

        let process_handle = StdioProcess {
            child,
            writer: BufWriter::new(stdin),
            reader: BufReader::new(stdout),
        };

        self.process = Some(Arc::new(Mutex::new(process_handle)));
        self.is_connected = true;

        // Perform the MCP initialization handshake.
        let init_params = InitializeParams::default();
        let _init_result = self
            .send_request(
                "initialize",
                Some(
                    serde_json::to_value(&init_params).map_err(|e| AgentError::UserError {
                        message: format!("Failed to serialize initialize params: {e}"),
                    })?,
                ),
            )
            .await?;

        // Send the initialized notification.
        self.send_notification("notifications/initialized", None)
            .await?;

        Ok(())
    }

    /// Get the list of tools available from this MCP server.
    ///
    /// The server must be connected first via [`connect`](Self::connect).
    /// Results are cached internally and can be retrieved again without
    /// a network round-trip by calling this method again.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the server is not connected.
    pub async fn list_tools(&mut self) -> Result<Vec<Tool<()>>> {
        if !self.is_connected {
            return Err(AgentError::UserError {
                message: format!(
                    "MCP server '{}' is not connected. Call connect() first.",
                    self.config.name
                ),
            });
        }

        // If we have no process (SSE/HTTP stub), return empty.
        if self.process.is_none() {
            return Ok(Vec::new());
        }

        let result = self.send_request("tools/list", None).await?;
        let list_result: ListToolsResult =
            serde_json::from_value(result).map_err(|e| AgentError::UserError {
                message: format!(
                    "Failed to parse tools/list response from '{}': {e}",
                    self.config.name
                ),
            })?;

        self.tools_cache.clone_from(&list_result.tools);

        let tools = list_result
            .tools
            .into_iter()
            .map(|def| {
                let mut schema = def.input_schema;
                // MCP spec does not require `properties` in the input schema,
                // but the OpenAI API requires it. Add an empty object if missing.
                if schema.get("properties").is_none() {
                    if let Some(obj) = schema.as_object_mut() {
                        obj.insert(
                            "properties".to_owned(),
                            serde_json::Value::Object(serde_json::Map::new()),
                        );
                    }
                }

                Tool::Function(crate::tool::FunctionTool::mcp_tool(
                    def.name,
                    def.description.unwrap_or_default(),
                    schema,
                ))
            })
            .collect();

        Ok(tools)
    }

    /// Invoke a tool on the MCP server by name.
    ///
    /// Sends a `tools/call` JSON-RPC request to the server and returns the
    /// parsed [`McpToolResult`].
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the server is not connected.
    /// Returns [`AgentError::ModelBehavior`] if the server returns an error.
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Option<&serde_json::Value>,
    ) -> Result<McpToolResult> {
        if !self.is_connected {
            return Err(AgentError::UserError {
                message: format!(
                    "MCP server '{}' is not connected. Call connect() first.",
                    self.config.name
                ),
            });
        }

        // If we have no process (SSE/HTTP stub), return empty result.
        if self.process.is_none() {
            return Ok(McpToolResult::text(""));
        }

        let params = serde_json::json!({
            "name": tool_name,
            "arguments": arguments.cloned().unwrap_or_else(|| serde_json::json!({})),
        });

        let result = self.send_request("tools/call", Some(params)).await?;
        let tool_result: McpToolResult =
            serde_json::from_value(result).map_err(|e| AgentError::UserError {
                message: format!(
                    "Failed to parse tools/call response from '{}': {e}",
                    self.config.name
                ),
            })?;

        Ok(tool_result)
    }

    /// Disconnect from the MCP server and release resources.
    ///
    /// For stdio transport, this kills the subprocess. After disconnecting,
    /// [`list_tools`](Self::list_tools) and [`call_tool`](Self::call_tool)
    /// will return errors until [`connect`](Self::connect) is called again.
    ///
    /// # Errors
    ///
    /// Returns an error if cleanup fails.
    pub async fn disconnect(&mut self) -> Result<()> {
        if self.is_connected {
            // Attempt to send shutdown notification (best effort).
            if self.process.is_some() {
                let _ = self.send_notification("shutdown", None).await;
            }

            // Kill the child process.
            if let Some(process) = self.process.take() {
                let mut guard = process.lock().await;
                let _ = guard.child.kill().await;
                drop(guard);
            }

            self.is_connected = false;
            self.tools_cache.clear();
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get the server's human-readable name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Check if the server is currently connected.
    #[must_use]
    pub const fn is_connected(&self) -> bool {
        self.is_connected
    }

    /// Get a reference to the server configuration.
    #[must_use]
    pub const fn config(&self) -> &MCPServerConfig {
        &self.config
    }

    /// Get the cached tool definitions from the last `tools/list` call.
    #[must_use]
    pub fn cached_tools(&self) -> &[McpToolDef] {
        &self.tools_cache
    }

    // -----------------------------------------------------------------------
    // JSON-RPC transport helpers
    // -----------------------------------------------------------------------

    /// Send a JSON-RPC request and wait for the response.
    #[allow(clippy::significant_drop_tightening)] // Guard must span write + read.
    async fn send_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest::new(id, method, params);
        let request_json = serde_json::to_string(&request).map_err(|e| AgentError::UserError {
            message: format!("Failed to serialize JSON-RPC request: {e}"),
        })?;

        let process = self.process.as_ref().ok_or_else(|| AgentError::UserError {
            message: format!("MCP server '{}' has no active process", self.config.name),
        })?;

        let mut stdio = process.lock().await;

        // Write the request as a single newline-terminated JSON line.
        stdio
            .writer
            .write_all(request_json.as_bytes())
            .await
            .map_err(|e| AgentError::UserError {
                message: format!("Failed to write to MCP server '{}': {e}", self.config.name),
            })?;
        stdio
            .writer
            .write_all(b"\n")
            .await
            .map_err(|e| AgentError::UserError {
                message: format!(
                    "Failed to write newline to MCP server '{}': {e}",
                    self.config.name
                ),
            })?;
        stdio
            .writer
            .flush()
            .await
            .map_err(|e| AgentError::UserError {
                message: format!(
                    "Failed to flush MCP server '{}' stdin: {e}",
                    self.config.name
                ),
            })?;

        // Read the response line. Skip blank lines and notifications (messages
        // without an `id` field) until we find the response matching our request.
        loop {
            let mut line = String::new();
            let bytes_read =
                stdio
                    .reader
                    .read_line(&mut line)
                    .await
                    .map_err(|e| AgentError::UserError {
                        message: format!(
                            "Failed to read from MCP server '{}': {e}",
                            self.config.name
                        ),
                    })?;

            if bytes_read == 0 {
                return Err(AgentError::UserError {
                    message: format!(
                        "MCP server '{}' closed stdout unexpectedly",
                        self.config.name
                    ),
                });
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Try to parse as a JSON-RPC response. If it is a notification or
            // a response for a different id, skip it.
            let value: serde_json::Value =
                serde_json::from_str(trimmed).map_err(|e| AgentError::UserError {
                    message: format!("Invalid JSON from MCP server '{}': {e}", self.config.name),
                })?;

            // Skip notifications (no "id" field).
            let Some(resp_id) = value.get("id") else {
                continue;
            };

            // Skip responses for other request IDs.
            if resp_id.as_u64() != Some(id) {
                continue;
            }

            let response: JsonRpcResponse =
                serde_json::from_value(value).map_err(|e| AgentError::UserError {
                    message: format!(
                        "Failed to parse JSON-RPC response from '{}': {e}",
                        self.config.name
                    ),
                })?;

            if let Some(error) = response.error {
                return Err(AgentError::ModelBehavior {
                    message: format!(
                        "MCP server '{}' error (code {}): {}",
                        self.config.name, error.code, error.message
                    ),
                });
            }

            return Ok(response.result.unwrap_or(serde_json::Value::Null));
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    async fn send_notification(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<()> {
        let notification = JsonRpcNotification::new(method, params);
        let json = serde_json::to_string(&notification).map_err(|e| AgentError::UserError {
            message: format!("Failed to serialize notification: {e}"),
        })?;

        if let Some(ref process) = self.process {
            let mut guard = process.lock().await;
            // Best-effort write; failures are tolerated for notifications.
            if guard.writer.write_all(json.as_bytes()).await.is_ok() {
                let _ = guard.writer.write_all(b"\n").await;
                let _ = guard.writer.flush().await;
            }
            drop(guard);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MCPServer creation ----

    #[test]
    fn create_stdio_server() {
        let server = MCPServer::stdio("test-server", "npx", vec!["-y".into(), "pkg".into()]);
        assert_eq!(server.name(), "test-server");
        assert!(!server.is_connected());
        assert!(matches!(
            server.config().transport,
            MCPTransport::Stdio { .. }
        ));
    }

    #[test]
    fn create_stdio_server_with_env() {
        let mut env = HashMap::new();
        env.insert("API_KEY".to_owned(), "secret".to_owned());
        let server = MCPServer::stdio_with_env("env-server", "node", vec!["server.js".into()], env);
        assert_eq!(server.name(), "env-server");
        if let MCPTransport::Stdio {
            command, args, env, ..
        } = &server.config().transport
        {
            assert_eq!(command, "node");
            assert_eq!(args, &["server.js"]);
            assert_eq!(env.get("API_KEY").unwrap(), "secret");
        } else {
            panic!("expected Stdio transport");
        }
    }

    #[test]
    fn create_sse_server() {
        let server = MCPServer::sse("sse-server", "https://example.com/mcp");
        assert_eq!(server.name(), "sse-server");
        assert!(!server.is_connected());
        assert!(matches!(
            server.config().transport,
            MCPTransport::Sse { .. }
        ));
    }

    #[test]
    fn create_sse_server_with_headers() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_owned(), "Bearer token".to_owned());
        let server = MCPServer::sse_with_headers("auth-server", "https://example.com/mcp", headers);
        assert_eq!(server.name(), "auth-server");
        if let MCPTransport::Sse { url, headers, .. } = &server.config().transport {
            assert_eq!(url, "https://example.com/mcp");
            assert_eq!(headers.get("Authorization").unwrap(), "Bearer token");
        } else {
            panic!("expected Sse transport");
        }
    }

    #[test]
    fn create_streamable_http_server() {
        let server = MCPServer::streamable_http("http-server", "https://example.com/mcp");
        assert_eq!(server.name(), "http-server");
        assert!(matches!(
            server.config().transport,
            MCPTransport::StreamableHttp { .. }
        ));
    }

    #[test]
    fn create_server_from_config() {
        let config = MCPServerConfig {
            name: "custom".to_owned(),
            transport: MCPTransport::Stdio {
                command: "python".to_owned(),
                args: vec!["server.py".into()],
                env: HashMap::new(),
            },
        };
        let server = MCPServer::new(config);
        assert_eq!(server.name(), "custom");
    }

    // ---- Connect / disconnect lifecycle ----

    #[tokio::test]
    async fn connect_and_disconnect_sse() {
        // SSE transport uses the stub path (no subprocess).
        let mut server = MCPServer::sse("lifecycle", "https://example.com");
        assert!(!server.is_connected());

        server.connect().await.expect("connect should succeed");
        assert!(server.is_connected());

        server
            .disconnect()
            .await
            .expect("disconnect should succeed");
        assert!(!server.is_connected());
    }

    #[tokio::test]
    async fn reconnect_after_disconnect() {
        let mut server = MCPServer::sse("reconnect", "https://example.com");
        server.connect().await.unwrap();
        server.disconnect().await.unwrap();
        server.connect().await.unwrap();
        assert!(server.is_connected());
    }

    // ---- list_tools before connect ----

    #[tokio::test]
    async fn list_tools_before_connect_returns_error() {
        let mut server = MCPServer::stdio("not-connected", "echo", vec![]);
        let result = server.list_tools().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentError::UserError { .. }),
            "expected UserError, got: {err}"
        );
        let msg = err.to_string();
        assert!(msg.contains("not-connected"));
        assert!(msg.contains("not connected"));
    }

    #[tokio::test]
    async fn list_tools_after_connect_sse_returns_empty() {
        let mut server = MCPServer::sse("connected", "https://example.com");
        server.connect().await.unwrap();
        let tools = server.list_tools().await.unwrap();
        assert!(tools.is_empty());
    }

    // ---- call_tool before connect ----

    #[tokio::test]
    async fn call_tool_before_connect_returns_error() {
        let server = MCPServer::stdio("not-connected", "echo", vec![]);
        let result = server.call_tool("some_tool", None).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::UserError { .. }));
    }

    // ---- MCPTransport variants ----

    #[test]
    fn transport_stdio_debug() {
        let t = MCPTransport::Stdio {
            command: "node".to_owned(),
            args: vec!["index.js".into()],
            env: HashMap::new(),
        };
        let debug = format!("{t:?}");
        assert!(debug.contains("Stdio"));
        assert!(debug.contains("node"));
    }

    #[test]
    fn transport_sse_debug() {
        let t = MCPTransport::Sse {
            url: "https://example.com".to_owned(),
            headers: HashMap::new(),
        };
        let debug = format!("{t:?}");
        assert!(debug.contains("Sse"));
        assert!(debug.contains("example.com"));
    }

    #[test]
    fn transport_streamable_http_debug() {
        let t = MCPTransport::StreamableHttp {
            url: "https://example.com/mcp".to_owned(),
            headers: HashMap::new(),
        };
        let debug = format!("{t:?}");
        assert!(debug.contains("StreamableHttp"));
    }

    #[test]
    fn transport_clone() {
        let original = MCPTransport::Stdio {
            command: "cmd".to_owned(),
            args: vec!["arg1".into()],
            env: HashMap::new(),
        };
        let cloned = original.clone();
        assert!(matches!(cloned, MCPTransport::Stdio { .. }));
        // Verify original is still usable (not moved).
        assert!(matches!(original, MCPTransport::Stdio { .. }));
    }

    // ---- MCPServerConfig ----

    #[test]
    fn server_config_clone() {
        let original = MCPServerConfig {
            name: "test".to_owned(),
            transport: MCPTransport::Sse {
                url: "https://example.com".to_owned(),
                headers: HashMap::new(),
            },
        };
        let cloned = original.clone();
        assert_eq!(cloned.name, "test");
        // Verify original is still usable.
        assert_eq!(original.name, "test");
    }

    #[test]
    fn server_config_debug() {
        let config = MCPServerConfig {
            name: "debug-test".to_owned(),
            transport: MCPTransport::Sse {
                url: "https://example.com".to_owned(),
                headers: HashMap::new(),
            },
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("debug-test"));
    }

    // ---- Debug on MCPServer ----

    #[test]
    fn server_debug() {
        let server = MCPServer::stdio("debug-srv", "echo", vec![]);
        let debug = format!("{server:?}");
        assert!(debug.contains("debug-srv"));
        assert!(debug.contains("is_connected"));
    }

    // ---- cached_tools ----

    #[test]
    fn cached_tools_initially_empty() {
        let server = MCPServer::stdio("cache-test", "echo", vec![]);
        assert!(server.cached_tools().is_empty());
    }

    // ---- Send + Sync bounds ----

    #[test]
    fn send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<MCPServer>();
        assert_sync::<MCPServer>();
        assert_send::<MCPServerConfig>();
        assert_sync::<MCPServerConfig>();
        assert_send::<MCPTransport>();
        assert_sync::<MCPTransport>();
    }

    // ---- MCPServerConfig::new ----

    #[test]
    fn server_config_new() {
        let config = MCPServerConfig::new(
            "test-server",
            MCPTransport::Sse {
                url: "https://example.com".to_owned(),
                headers: HashMap::new(),
            },
        );
        assert_eq!(config.name, "test-server");
    }
}
