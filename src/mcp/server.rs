//! MCP server connection and tool discovery.
//!
//! This module defines the types for connecting to an MCP server and
//! discovering the tools it exposes. Three transports are supported:
//!
//! * **Stdio** — JSON-RPC over stdin/stdout of a spawned subprocess.
//! * **SSE** — Server-Sent Events over HTTP (GET for receiving, POST for sending).
//! * **`StreamableHttp`** — Bidirectional HTTP with streaming responses.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures::StreamExt;
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
// HttpTransportState — shared state for SSE and StreamableHTTP transports
// ---------------------------------------------------------------------------

/// Internal state for HTTP-based MCP transports (SSE and `StreamableHttp`).
///
/// Both SSE and `StreamableHttp` transports send JSON-RPC requests via HTTP
/// POST and receive responses in the response body. The SSE transport
/// additionally negotiates a POST endpoint via an initial SSE stream.
struct HttpTransportState {
    /// The URL to POST JSON-RPC requests to.
    post_url: String,
    /// HTTP headers to include in every request.
    headers: HashMap<String, String>,
    /// The shared HTTP client.
    client: reqwest::Client,
    /// The MCP session ID captured from response headers.
    session_id: Mutex<Option<String>>,
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
    /// HTTP transport state (for SSE and `StreamableHttp` transports).
    http_state: Option<Arc<HttpTransportState>>,
    /// Cached tool definitions from the last `tools/list` call.
    tools_cache: Vec<McpToolDef>,
    /// Monotonically increasing JSON-RPC request ID.
    next_id: AtomicU64,
    /// Whether the server connection is currently active.
    is_connected: bool,
    /// The MCP session ID assigned by the server (StreamableHttp only).
    ///
    /// Populated from the `Mcp-Session-Id` response header during the
    /// initialization handshake. Can be used to resume sessions across
    /// connection restarts.
    session_id: Option<String>,
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
            http_state: None,
            tools_cache: Vec::new(),
            next_id: AtomicU64::new(1),
            is_connected: false,
            session_id: None,
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
            MCPTransport::Sse { url, headers } => {
                self.connect_sse(url.clone(), headers.clone()).await?;
            }
            MCPTransport::StreamableHttp { url, headers } => {
                self.connect_streamable_http(url.clone(), headers.clone())
                    .await?;
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

    /// Open an SSE connection and perform the MCP initialization handshake.
    ///
    /// The SSE transport works as follows:
    /// 1. Open a GET request to the SSE endpoint with `Accept: text/event-stream`.
    /// 2. Read the SSE stream until an `endpoint` event is received, which
    ///    provides the POST URL for sending JSON-RPC requests.
    /// 3. Send the `initialize` request and `notifications/initialized`
    ///    notification via the POST endpoint.
    async fn connect_sse(&mut self, url: String, headers: HashMap<String, String>) -> Result<()> {
        let client = reqwest::Client::new();
        let mut request = client.get(&url).header("Accept", "text/event-stream");

        for (key, value) in &headers {
            request = request.header(key.as_str(), value.as_str());
        }

        let response = request.send().await.map_err(|e| AgentError::UserError {
            message: format!("SSE connection to '{}' failed: {e}", self.config.name),
        })?;

        if !response.status().is_success() {
            return Err(AgentError::UserError {
                message: format!(
                    "SSE connection to '{}' failed with status {}",
                    self.config.name,
                    response.status()
                ),
            });
        }

        // Read the SSE stream to find the endpoint event.
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut post_endpoint: Option<String> = None;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| AgentError::UserError {
                message: format!("Failed to read SSE stream from '{}': {e}", self.config.name),
            })?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Parse SSE events looking for the endpoint event.
            while let Some((event_type, data)) = parse_sse_event(&mut buffer) {
                if event_type == "endpoint" {
                    post_endpoint = Some(resolve_url(&url, &data));
                    break;
                }
            }

            if post_endpoint.is_some() {
                break;
            }
        }

        let post_url = post_endpoint.ok_or_else(|| AgentError::UserError {
            message: format!(
                "SSE server '{}' did not provide an endpoint event",
                self.config.name
            ),
        })?;

        self.http_state = Some(Arc::new(HttpTransportState {
            post_url,
            headers,
            client,
            session_id: Mutex::new(None),
        }));
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

        self.send_notification("notifications/initialized", None)
            .await?;

        Ok(())
    }

    /// Connect using the `StreamableHttp` transport and perform the MCP initialization handshake.
    ///
    /// The `StreamableHttp` transport is simpler than SSE: JSON-RPC requests are
    /// sent as HTTP POST and responses come back in the response body.
    async fn connect_streamable_http(
        &mut self,
        url: String,
        headers: HashMap<String, String>,
    ) -> Result<()> {
        let client = reqwest::Client::new();

        self.http_state = Some(Arc::new(HttpTransportState {
            post_url: url,
            headers,
            client,
            session_id: Mutex::new(None),
        }));
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

        // Capture the session ID from the HTTP state after initialization.
        if let Some(ref http) = self.http_state {
            if let Ok(guard) = http.session_id.lock() {
                self.session_id = guard.clone();
            }
        }

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

    /// List the resources available from this MCP server.
    ///
    /// Sends a `resources/list` JSON-RPC request. An optional cursor can be
    /// provided for pagination.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the server is not connected.
    pub async fn list_resources(
        &self,
        cursor: Option<&str>,
    ) -> Result<super::protocol::ListResourcesResult> {
        if !self.is_connected {
            return Err(AgentError::UserError {
                message: format!(
                    "MCP server '{}' is not connected. Call connect() first.",
                    self.config.name
                ),
            });
        }

        let params = cursor.map(|c| serde_json::json!({ "cursor": c }));
        let result = self.send_request("resources/list", params).await?;
        serde_json::from_value(result).map_err(|e| AgentError::UserError {
            message: format!(
                "Failed to parse resources/list response from '{}': {e}",
                self.config.name
            ),
        })
    }

    /// List the resource templates available from this MCP server.
    ///
    /// Sends a `resources/templates/list` JSON-RPC request. An optional cursor
    /// can be provided for pagination.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the server is not connected.
    pub async fn list_resource_templates(
        &self,
        cursor: Option<&str>,
    ) -> Result<super::protocol::ListResourceTemplatesResult> {
        if !self.is_connected {
            return Err(AgentError::UserError {
                message: format!(
                    "MCP server '{}' is not connected. Call connect() first.",
                    self.config.name
                ),
            });
        }

        let params = cursor.map(|c| serde_json::json!({ "cursor": c }));
        let result = self
            .send_request("resources/templates/list", params)
            .await?;
        serde_json::from_value(result).map_err(|e| AgentError::UserError {
            message: format!(
                "Failed to parse resources/templates/list response from '{}': {e}",
                self.config.name
            ),
        })
    }

    /// Read the content of a resource by URI.
    ///
    /// Sends a `resources/read` JSON-RPC request and returns the resource
    /// content.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the server is not connected.
    pub async fn read_resource(
        &self,
        uri: &str,
    ) -> Result<super::protocol::ReadResourceResult> {
        if !self.is_connected {
            return Err(AgentError::UserError {
                message: format!(
                    "MCP server '{}' is not connected. Call connect() first.",
                    self.config.name
                ),
            });
        }

        let params = serde_json::json!({ "uri": uri });
        let result = self.send_request("resources/read", Some(params)).await?;
        serde_json::from_value(result).map_err(|e| AgentError::UserError {
            message: format!(
                "Failed to parse resources/read response from '{}': {e}",
                self.config.name
            ),
        })
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
            let _ = self.send_notification("shutdown", None).await;

            // Kill the child process (stdio transport).
            if let Some(process) = self.process.take() {
                let mut guard = process.lock().await;
                let _ = guard.child.kill().await;
                drop(guard);
            }

            // Drop HTTP state (SSE / StreamableHttp transports).
            self.http_state = None;

            self.is_connected = false;
            self.tools_cache.clear();
            self.session_id = None;
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

    /// Get the MCP session ID assigned by the server.
    ///
    /// Only populated for `StreamableHttp` transports when the server returns
    /// an `Mcp-Session-Id` header during initialization. Returns `None` for
    /// stdio and SSE transports.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    // -----------------------------------------------------------------------
    // JSON-RPC transport helpers
    // -----------------------------------------------------------------------

    /// Send a JSON-RPC request and wait for the response.
    ///
    /// Routes to the appropriate transport: stdio (child process) or HTTP
    /// (SSE / `StreamableHttp`).
    async fn send_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest::new(id, method, params);

        if let Some(ref http) = self.http_state {
            return Self::send_request_http(http, &self.config.name, &request).await;
        }

        if let Some(ref process) = self.process {
            return Self::send_request_stdio(process, &self.config.name, id, &request).await;
        }

        Err(AgentError::UserError {
            message: format!("MCP server '{}' has no active transport", self.config.name),
        })
    }

    /// Send a JSON-RPC request over the stdio transport.
    #[allow(clippy::significant_drop_tightening)] // Guard must span write + read.
    async fn send_request_stdio(
        process: &Arc<Mutex<StdioProcess>>,
        server_name: &str,
        id: u64,
        request: &JsonRpcRequest,
    ) -> Result<serde_json::Value> {
        let request_json = serde_json::to_string(request).map_err(|e| AgentError::UserError {
            message: format!("Failed to serialize JSON-RPC request: {e}"),
        })?;

        let mut stdio = process.lock().await;

        // Write the request as a single newline-terminated JSON line.
        stdio
            .writer
            .write_all(request_json.as_bytes())
            .await
            .map_err(|e| AgentError::UserError {
                message: format!("Failed to write to MCP server '{server_name}': {e}"),
            })?;
        stdio
            .writer
            .write_all(b"\n")
            .await
            .map_err(|e| AgentError::UserError {
                message: format!("Failed to write newline to MCP server '{server_name}': {e}",),
            })?;
        stdio
            .writer
            .flush()
            .await
            .map_err(|e| AgentError::UserError {
                message: format!("Failed to flush MCP server '{server_name}' stdin: {e}",),
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
                        message: format!("Failed to read from MCP server '{server_name}': {e}",),
                    })?;

            if bytes_read == 0 {
                return Err(AgentError::UserError {
                    message: format!("MCP server '{server_name}' closed stdout unexpectedly",),
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
                    message: format!("Invalid JSON from MCP server '{server_name}': {e}"),
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
                        "Failed to parse JSON-RPC response from '{server_name}': {e}",
                    ),
                })?;

            if let Some(error) = response.error {
                return Err(AgentError::ModelBehavior {
                    message: format!(
                        "MCP server '{server_name}' error (code {}): {}",
                        error.code, error.message
                    ),
                });
            }

            return Ok(response.result.unwrap_or(serde_json::Value::Null));
        }
    }

    /// Send a JSON-RPC request over an HTTP transport (SSE or `StreamableHttp`).
    async fn send_request_http(
        state: &HttpTransportState,
        server_name: &str,
        request: &JsonRpcRequest,
    ) -> Result<serde_json::Value> {
        let mut req = state
            .client
            .post(&state.post_url)
            .header("Content-Type", "application/json");

        for (key, value) in &state.headers {
            req = req.header(key.as_str(), value.as_str());
        }

        let response = req
            .json(request)
            .send()
            .await
            .map_err(|e| AgentError::UserError {
                message: format!("MCP HTTP request to '{server_name}' failed: {e}",),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::UserError {
                message: format!("MCP HTTP request to '{server_name}' failed ({status}): {body}",),
            });
        }

        // Capture the Mcp-Session-Id header if present.
        if let Some(session_id_value) = response.headers().get("mcp-session-id") {
            if let Ok(sid) = session_id_value.to_str() {
                if let Ok(mut guard) = state.session_id.lock() {
                    *guard = Some(sid.to_owned());
                }
            }
        }

        // The response body may be JSON-RPC directly, or SSE-wrapped.
        // Try to parse as JSON-RPC first.
        let body = response.text().await.map_err(|e| AgentError::UserError {
            message: format!("Failed to read HTTP response from '{server_name}': {e}",),
        })?;

        // Some SSE-style responses wrap the JSON in `data: ` lines.
        let json_text = extract_json_from_sse_or_raw(&body);

        let json_response: JsonRpcResponse =
            serde_json::from_str(json_text).map_err(|e| AgentError::UserError {
                message: format!(
                    "Failed to parse JSON-RPC response from '{server_name}': {e}\nBody: {body}",
                ),
            })?;

        if let Some(error) = json_response.error {
            return Err(AgentError::ModelBehavior {
                message: format!(
                    "MCP server '{server_name}' error (code {}): {}",
                    error.code, error.message
                ),
            });
        }

        Ok(json_response.result.unwrap_or(serde_json::Value::Null))
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
        } else if let Some(ref http) = self.http_state {
            // Best-effort POST for HTTP transports.
            let mut req = http
                .client
                .post(&http.post_url)
                .header("Content-Type", "application/json");

            for (key, value) in &http.headers {
                req = req.header(key.as_str(), value.as_str());
            }

            let _ = req.body(json).send().await;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SSE and URL helper functions
// ---------------------------------------------------------------------------

/// Parse a single SSE event from the buffer, returning the event type and data.
///
/// SSE events are terminated by a double newline (`\n\n`). Each event can
/// contain `event:` and `data:` fields. Returns `None` if no complete event
/// is available in the buffer yet.
fn parse_sse_event(buffer: &mut String) -> Option<(String, String)> {
    let pos = buffer.find("\n\n")?;
    let event_block = buffer[..pos].to_string();
    *buffer = buffer[pos + 2..].to_string();

    let mut event_type = String::new();
    let mut data = String::new();

    for line in event_block.lines() {
        if let Some(t) = line.strip_prefix("event: ") {
            event_type = t.trim().to_string();
        } else if let Some(t) = line.strip_prefix("event:") {
            event_type = t.trim().to_string();
        } else if let Some(d) = line.strip_prefix("data: ") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(d.trim());
        } else if let Some(d) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(d.trim());
        }
    }

    if data.is_empty() && event_type.is_empty() {
        None
    } else {
        Some((event_type, data))
    }
}

/// Resolve a potentially relative URL against a base URL.
///
/// If `relative` is already an absolute URL (starts with `http://` or `https://`),
/// it is returned as-is. Otherwise, the path portion of `base` is replaced.
fn resolve_url(base: &str, relative: &str) -> String {
    if relative.starts_with("http://") || relative.starts_with("https://") {
        return relative.to_string();
    }

    // Extract the origin (scheme + host + port) from the base URL.
    // For a URL like "https://example.com:8080/sse", we want "https://example.com:8080".
    let Some(scheme_end) = base.find("://") else {
        // Fallback: just concatenate.
        return format!("{base}/{relative}");
    };

    let after_scheme = &base[scheme_end + 3..];
    let origin_end = after_scheme.find('/').map_or(after_scheme.len(), |p| p);
    let origin = &base[..scheme_end + 3 + origin_end];

    if relative.starts_with('/') {
        format!("{origin}{relative}")
    } else {
        // Relative to the current path directory.
        base.rfind('/').map_or_else(
            || format!("{origin}/{relative}"),
            |last_slash| {
                if last_slash > scheme_end + 2 {
                    format!("{}/{relative}", &base[..last_slash])
                } else {
                    format!("{origin}/{relative}")
                }
            },
        )
    }
}

/// Extract JSON from a response body that may be raw JSON or SSE-formatted.
///
/// Some MCP servers return the JSON-RPC response wrapped in SSE `data:` lines.
/// This function strips the SSE framing if present, returning the raw JSON.
fn extract_json_from_sse_or_raw(body: &str) -> &str {
    let trimmed = body.trim();

    // If it looks like JSON already, return it.
    if trimmed.starts_with('{') {
        return trimmed;
    }

    // Try to extract from SSE `data:` lines.
    for line in trimmed.lines() {
        let data = line
            .strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:"));
        if let Some(d) = data {
            let d = d.trim();
            if d.starts_with('{') {
                return d;
            }
        }
    }

    trimmed
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

    // ---- Connect / disconnect lifecycle (wiremock) ----

    /// Helper to create a wiremock SSE endpoint that returns an endpoint event
    /// and responds to the initialize and tools/list requests.
    async fn setup_sse_mock() -> (wiremock::MockServer, String) {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let mock_server = wiremock::MockServer::start().await;
        let post_path = "/messages";

        // GET /sse -> SSE stream with endpoint event.
        let sse_body = format!("event: endpoint\ndata: {post_path}\n\n");
        Mock::given(method("GET"))
            .and(path("/sse"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .insert_header("Content-Type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        // POST /messages for initialize -> success response.
        Mock::given(method("POST"))
            .and(path(post_path))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock-server", "version": "1.0.0"}
                }
            })))
            .mount(&mock_server)
            .await;

        let url = format!("{}/sse", mock_server.uri());
        (mock_server, url)
    }

    /// Helper to create a wiremock `StreamableHTTP` endpoint.
    async fn setup_http_mock() -> (wiremock::MockServer, String) {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let mock_server = wiremock::MockServer::start().await;

        // POST /mcp -> JSON-RPC response.
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock-http-server", "version": "1.0.0"}
                }
            })))
            .mount(&mock_server)
            .await;

        let url = format!("{}/mcp", mock_server.uri());
        (mock_server, url)
    }

    #[tokio::test]
    async fn connect_and_disconnect_sse() {
        let (_mock, url) = setup_sse_mock().await;
        let mut server = MCPServer::sse("lifecycle", &url);
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
    async fn connect_and_disconnect_streamable_http() {
        let (_mock, url) = setup_http_mock().await;
        let mut server = MCPServer::streamable_http("lifecycle-http", &url);
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
        let (_mock, url) = setup_http_mock().await;
        let mut server = MCPServer::streamable_http("reconnect", &url);
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

    // ---- parse_sse_event ----

    #[test]
    fn parse_sse_event_basic() {
        let mut buf = "event: endpoint\ndata: /messages\n\n".to_string();
        let result = parse_sse_event(&mut buf);
        assert_eq!(
            result,
            Some(("endpoint".to_string(), "/messages".to_string()))
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn parse_sse_event_no_space_after_colon() {
        let mut buf = "event:endpoint\ndata:/messages\n\n".to_string();
        let result = parse_sse_event(&mut buf);
        assert_eq!(
            result,
            Some(("endpoint".to_string(), "/messages".to_string()))
        );
    }

    #[test]
    fn parse_sse_event_data_only() {
        let mut buf = "data: {\"key\":\"value\"}\n\n".to_string();
        let result = parse_sse_event(&mut buf);
        assert_eq!(
            result,
            Some((String::new(), "{\"key\":\"value\"}".to_string()))
        );
    }

    #[test]
    fn parse_sse_event_incomplete() {
        let mut buf = "event: endpoint\ndata: /messages\n".to_string();
        let result = parse_sse_event(&mut buf);
        assert!(result.is_none());
        // Buffer should be unchanged.
        assert!(buf.contains("endpoint"));
    }

    #[test]
    fn parse_sse_event_multiple_events() {
        let mut buf = "event: ping\ndata: first\n\nevent: endpoint\ndata: /msg\n\n".to_string();
        let first = parse_sse_event(&mut buf);
        assert_eq!(first, Some(("ping".to_string(), "first".to_string())));
        let second = parse_sse_event(&mut buf);
        assert_eq!(second, Some(("endpoint".to_string(), "/msg".to_string())));
        assert!(parse_sse_event(&mut buf).is_none());
    }

    #[test]
    fn parse_sse_event_empty_block_returns_none() {
        let mut buf = "\n\n".to_string();
        let result = parse_sse_event(&mut buf);
        assert!(result.is_none());
    }

    #[test]
    fn parse_sse_event_multiline_data() {
        let mut buf = "data: line1\ndata: line2\n\n".to_string();
        let result = parse_sse_event(&mut buf);
        assert_eq!(result, Some((String::new(), "line1\nline2".to_string())));
    }

    // ---- resolve_url ----

    #[test]
    fn resolve_url_absolute() {
        let result = resolve_url("https://example.com/sse", "https://other.com/messages");
        assert_eq!(result, "https://other.com/messages");
    }

    #[test]
    fn resolve_url_absolute_path() {
        let result = resolve_url("https://example.com/sse", "/messages");
        assert_eq!(result, "https://example.com/messages");
    }

    #[test]
    fn resolve_url_relative_path() {
        let result = resolve_url("https://example.com/api/sse", "messages");
        assert_eq!(result, "https://example.com/api/messages");
    }

    #[test]
    fn resolve_url_with_port() {
        let result = resolve_url("http://localhost:8080/sse", "/messages");
        assert_eq!(result, "http://localhost:8080/messages");
    }

    #[test]
    fn resolve_url_base_no_path() {
        let result = resolve_url("https://example.com", "/messages");
        assert_eq!(result, "https://example.com/messages");
    }

    // ---- extract_json_from_sse_or_raw ----

    #[test]
    fn extract_json_raw() {
        let body = r#"{"jsonrpc":"2.0","id":1,"result":null}"#;
        assert_eq!(extract_json_from_sse_or_raw(body), body);
    }

    #[test]
    fn extract_json_from_sse_data_line() {
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":null}\n\n";
        assert_eq!(
            extract_json_from_sse_or_raw(body),
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":null}"
        );
    }

    #[test]
    fn extract_json_from_sse_no_space() {
        let body = "data:{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":null}\n";
        assert_eq!(
            extract_json_from_sse_or_raw(body),
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":null}"
        );
    }

    #[test]
    fn extract_json_with_whitespace() {
        let body = "  {\"id\":1}  ";
        assert_eq!(extract_json_from_sse_or_raw(body), "{\"id\":1}");
    }

    // ---- HTTP transport integration tests (wiremock) ----

    #[tokio::test]
    async fn streamable_http_list_tools() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let mock_server = wiremock::MockServer::start().await;

        // Respond to initialize.
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock", "version": "1.0.0"}
                }
            })))
            .up_to_n_times(2) // initialize + notification
            .mount(&mock_server)
            .await;

        // Respond to tools/list.
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get weather for a location",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"}
                                }
                            }
                        }
                    ]
                }
            })))
            .mount(&mock_server)
            .await;

        let url = format!("{}/mcp", mock_server.uri());
        let mut server = MCPServer::streamable_http("test-http", &url);
        server.connect().await.unwrap();

        let tools = server.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);

        server.disconnect().await.unwrap();
    }

    #[tokio::test]
    async fn streamable_http_call_tool() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let mock_server = wiremock::MockServer::start().await;

        // Respond to all POSTs (initialize + tool call).
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "mock", "version": "1.0.0"}
                }
            })))
            .up_to_n_times(2)
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "content": [{"type": "text", "text": "Sunny, 25C"}],
                    "isError": false
                }
            })))
            .mount(&mock_server)
            .await;

        let url = format!("{}/mcp", mock_server.uri());
        let mut server = MCPServer::streamable_http("tool-call", &url);
        server.connect().await.unwrap();

        let result = server
            .call_tool(
                "get_weather",
                Some(&serde_json::json!({"location": "Tokyo"})),
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(result.to_text(), "Sunny, 25C");

        server.disconnect().await.unwrap();
    }

    #[tokio::test]
    async fn http_transport_error_response() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let mock_server = wiremock::MockServer::start().await;

        // Return 500 error.
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&mock_server)
            .await;

        let url = format!("{}/mcp", mock_server.uri());
        let mut server = MCPServer::streamable_http("error-test", &url);

        let result = server.connect().await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("500"),
            "error should mention status: {err_msg}"
        );
    }

    #[tokio::test]
    async fn http_transport_jsonrpc_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let mock_server = wiremock::MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32601, "message": "Method not found"}
            })))
            .mount(&mock_server)
            .await;

        let url = format!("{}/mcp", mock_server.uri());
        let mut server = MCPServer::streamable_http("rpc-error", &url);

        let result = server.connect().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentError::ModelBehavior { .. }),
            "expected ModelBehavior, got: {err}"
        );
        assert!(err.to_string().contains("Method not found"));
    }

    #[tokio::test]
    async fn sse_connect_no_endpoint_event() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let mock_server = wiremock::MockServer::start().await;

        // SSE stream that never sends an endpoint event.
        Mock::given(method("GET"))
            .and(path("/sse"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("event: ping\ndata: hello\n\n")
                    .insert_header("Content-Type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        let url = format!("{}/sse", mock_server.uri());
        let mut server = MCPServer::sse("no-endpoint", &url);

        let result = server.connect().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("endpoint"));
    }

    #[tokio::test]
    async fn sse_connect_with_custom_headers() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, ResponseTemplate};

        let mock_server = wiremock::MockServer::start().await;

        // Verify the custom header is present.
        Mock::given(method("GET"))
            .and(path("/sse"))
            .and(header("X-Custom", "test-value"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("event: endpoint\ndata: /msg\n\n")
                    .insert_header("Content-Type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/msg"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "mock", "version": "1.0.0"}
                }
            })))
            .mount(&mock_server)
            .await;

        let mut headers = HashMap::new();
        headers.insert("X-Custom".to_owned(), "test-value".to_owned());

        let url = format!("{}/sse", mock_server.uri());
        let mut server = MCPServer::sse_with_headers("header-test", &url, headers);
        server
            .connect()
            .await
            .expect("connect with custom headers should succeed");
        assert!(server.is_connected());
        server.disconnect().await.unwrap();
    }

    #[tokio::test]
    async fn sse_connect_resolves_absolute_endpoint() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let mock_server = wiremock::MockServer::start().await;
        let abs_url = format!("{}/absolute-messages", mock_server.uri());

        // SSE sends an absolute URL as endpoint.
        let sse_body = format!("event: endpoint\ndata: {abs_url}\n\n");
        Mock::given(method("GET"))
            .and(path("/sse"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .insert_header("Content-Type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/absolute-messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "mock", "version": "1.0.0"}
                }
            })))
            .mount(&mock_server)
            .await;

        let url = format!("{}/sse", mock_server.uri());
        let mut server = MCPServer::sse("abs-endpoint", &url);
        server
            .connect()
            .await
            .expect("connect should succeed with absolute endpoint");
        assert!(server.is_connected());
        server.disconnect().await.unwrap();
    }
}
