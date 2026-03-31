//! MCP server connection and tool discovery.
//!
//! This module defines the types for connecting to an MCP server and
//! discovering the tools it exposes. The actual MCP protocol wire format
//! is not yet implemented; the current version provides the full public
//! API surface with stub implementations that will be filled in once the
//! MCP client library is integrated.

use std::collections::HashMap;

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
#[derive(Debug)]
pub struct MCPServer {
    /// The server configuration.
    config: MCPServerConfig,
    /// Whether the server connection is currently active.
    is_connected: bool,
}

impl MCPServer {
    /// Create a new MCP server with the given configuration.
    #[must_use]
    pub const fn new(config: MCPServerConfig) -> Self {
        Self {
            config,
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

    /// Connect to the MCP server and discover available tools.
    ///
    /// This must be called before [`list_tools`](Self::list_tools). The
    /// connection remains active until [`disconnect`](Self::disconnect) is
    /// called.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection fails or the server does not
    /// respond.
    #[allow(clippy::unused_async)] // Will be async when MCP protocol is implemented.
    pub async fn connect(&mut self) -> Result<()> {
        // TODO: Implement actual MCP protocol connection.
        // For now, mark as connected so the rest of the API surface works.
        self.is_connected = true;
        Ok(())
    }

    /// Get the list of tools available from this MCP server.
    ///
    /// The server must be connected first via [`connect`](Self::connect).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the server is not connected.
    #[allow(clippy::unused_async)] // Will be async when MCP protocol is implemented.
    pub async fn list_tools(&self) -> Result<Vec<Tool<()>>> {
        if !self.is_connected {
            return Err(AgentError::UserError {
                message: format!(
                    "MCP server '{}' is not connected. Call connect() first.",
                    self.config.name
                ),
            });
        }
        // TODO: Implement actual tool listing via MCP protocol.
        Ok(Vec::new())
    }

    /// Invoke a tool on the MCP server by name.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::UserError`] if the server is not connected.
    /// Returns [`AgentError::ModelBehavior`] if the arguments are invalid
    /// JSON.
    #[allow(clippy::unused_async)] // Will be async when MCP protocol is implemented.
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value> {
        if !self.is_connected {
            return Err(AgentError::UserError {
                message: format!(
                    "MCP server '{}' is not connected. Call connect() first.",
                    self.config.name
                ),
            });
        }
        // TODO: Implement actual MCP tool invocation.
        let _ = tool_name;
        let _ = arguments;
        Ok(serde_json::Value::Null)
    }

    /// Disconnect from the MCP server and release resources.
    ///
    /// After disconnecting, [`list_tools`](Self::list_tools) and
    /// [`call_tool`](Self::call_tool) will return errors until
    /// [`connect`](Self::connect) is called again.
    ///
    /// # Errors
    ///
    /// Returns an error if cleanup fails.
    #[allow(clippy::unused_async)] // Will be async when MCP protocol is implemented.
    pub async fn disconnect(&mut self) -> Result<()> {
        // TODO: Implement actual MCP protocol disconnection.
        self.is_connected = false;
        Ok(())
    }

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
    async fn connect_and_disconnect() {
        let mut server = MCPServer::stdio("lifecycle", "echo", vec![]);
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
        let server = MCPServer::stdio("not-connected", "echo", vec![]);
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
    async fn list_tools_after_connect_returns_empty() {
        let mut server = MCPServer::stdio("connected", "echo", vec![]);
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

    // ---- Send + Sync bounds ----

    #[test]
    fn send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MCPServer>();
        assert_send_sync::<MCPServerConfig>();
        assert_send_sync::<MCPTransport>();
    }
}
