//! Model Context Protocol (MCP) integration.
//!
//! This module provides support for connecting to MCP servers and using their
//! tools within agent workflows. MCP servers expose tools over a standardised
//! JSON-RPC 2.0 protocol, and this module converts them into the SDK's
//! [`Tool`](crate::tool::Tool) type so they can be used seamlessly by agents.
//!
//! Enable the **`mcp`** feature flag to use this module.
//!
//! # Overview
//!
//! * [`MCPServer`] — represents a connection to a single MCP server.
//! * [`MCPServerConfig`] — transport and identification details for a server.
//! * [`MCPTransport`] — the transport mechanism (stdio subprocess or SSE/HTTP).
//! * [`MCPConfig`] — per-agent MCP configuration (error handling, timeouts).
//! * [`ToolFilter`] — controls which MCP tools are exposed to the agent.
//! * [`protocol`] — JSON-RPC 2.0 wire types and MCP-specific message types.
//!
//! # Example
//!
//! ```no_run
//! use openai_agents::mcp::{MCPServer, MCPServerConfig, MCPTransport};
//!
//! # async fn example() -> openai_agents::Result<()> {
//! let mut server = MCPServer::stdio("my-server", "npx", vec!["-y".into(), "some-mcp-server".into()]);
//! server.connect().await?;
//! let tools = server.list_tools().await?;
//! println!("discovered {} tools", tools.len());
//! server.disconnect().await?;
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod protocol;
pub mod server;
pub mod util;

pub use config::{ApprovalCallable, ApprovalPolicy, ApprovalPolicySetting, MCPConfig};
pub use protocol::{
    ClientInfo, InitializeParams, JsonRpcError, JsonRpcNotification, JsonRpcRequest,
    JsonRpcResponse, ListResourceTemplatesResult, ListResourcesResult, ListToolsResult, McpContent,
    McpResource, McpResourceContent, McpResourceTemplate, McpToolDef, McpToolResult,
    ReadResourceResult,
};
pub use server::{MCPServer, MCPServerConfig, MCPTransport};
pub use util::{ToolFilter, ToolFilterContext, ToolFilterStatic};
