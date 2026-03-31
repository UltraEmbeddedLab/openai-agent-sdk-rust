# Model Context Protocol (MCP)

The Model Context Protocol (MCP) is an open standard that lets agents connect to external
tool servers using a common JSON-RPC 2.0 wire format. An MCP server exposes a catalogue of
tools — functions the agent can call — without you writing any glue code. The SDK discovers
those tools automatically during the connection handshake and surfaces them to agents exactly
like hand-written function tools.

> **Feature flag.** MCP support is compiled in by default. No feature flag is required.

## What is MCP?

MCP defines a client/server architecture:

- **Server** — a process or HTTP endpoint that owns one or more tools and executes them.
- **Client** — the SDK, which connects to the server, lists available tools, and calls them on
  behalf of the agent.
- **Transport** — the communication channel between client and server.

The SDK implements three transports:

| Transport | When to use |
|---|---|
| `Stdio` | Spawn a local subprocess (Node, Python, Go binary, etc.) |
| `Sse` | Connect to a remote HTTP server using Server-Sent Events |
| `StreamableHttp` | Connect to a remote HTTP server using the newer MCP Streamable HTTP transport |

## Creating an MCPServer

### Stdio (subprocess)

The stdio transport spawns a child process and communicates over its stdin/stdout. This is the
most common pattern for local MCP servers distributed as npm packages or other executables.

```rust
use openai_agents::mcp::MCPServer;

// Simplest form: command + args.
let server = MCPServer::stdio(
    "filesystem",                         // human-readable name
    "npx",                                // executable
    vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into()],
);

// With environment variables.
use std::collections::HashMap;
let mut env = HashMap::new();
env.insert("FS_ROOT".into(), "/tmp/workspace".into());

let server = MCPServer::stdio_with_env(
    "filesystem",
    "npx",
    vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into()],
    env,
);
```

### SSE (HTTP Server-Sent Events)

```rust
use openai_agents::mcp::MCPServer;

// Plain SSE endpoint.
let server = MCPServer::sse("my-server", "https://mcp.example.com/sse");

// With custom headers (e.g., bearer token authentication).
use std::collections::HashMap;
let mut headers = HashMap::new();
headers.insert("Authorization".into(), "Bearer my-token".into());

let server = MCPServer::sse_with_headers(
    "auth-server",
    "https://mcp.example.com/sse",
    headers,
);
```

### Streamable HTTP

```rust
use openai_agents::mcp::MCPServer;

let server = MCPServer::streamable_http(
    "my-server",
    "https://mcp.example.com/mcp",
);

// With custom headers.
use std::collections::HashMap;
let mut headers = HashMap::new();
headers.insert("X-API-Key".into(), "secret".into());

let server = MCPServer::streamable_http_with_headers(
    "my-server",
    "https://mcp.example.com/mcp",
    headers,
);
```

## Connecting and Discovering Tools

The connection lifecycle is explicit: you call `connect`, use the server, then `disconnect`.
The handshake sends an `initialize` request and a `notifications/initialized` notification,
which causes the server to advertise its capabilities.

```rust,no_run
use openai_agents::mcp::MCPServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut server = MCPServer::stdio(
        "search",
        "npx",
        vec!["-y".into(), "@mcp/search-server".into()],
    );

    // 1. Connect (sends initialize handshake).
    server.connect().await?;

    // 2. Discover tools.
    let tools = server.list_tools().await?;
    for tool in &tools {
        println!(
            "Tool: {} — {}",
            tool.name,
            tool.description.as_deref().unwrap_or("(no description)")
        );
    }

    // 3. Disconnect when done.
    server.disconnect().await?;

    Ok(())
}
```

`list_tools` returns a `Vec<McpToolDef>` where each entry has:

- `name: String` — the unique tool name.
- `description: Option<String>` — a human-readable description.
- `input_schema: serde_json::Value` — JSON Schema describing the parameters.

## Calling Tools

Once connected, call any tool by name with a JSON arguments object:

```rust,no_run
use openai_agents::mcp::MCPServer;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut server = MCPServer::stdio(
        "calculator",
        "python",
        vec!["-m".into(), "calculator_mcp".into()],
    );
    server.connect().await?;

    let result = server.call_tool("add", json!({"a": 3, "b": 4})).await?;
    println!("Result: {}", result.to_text());
    // => "Result: 7"

    server.disconnect().await?;
    Ok(())
}
```

`call_tool` returns a `McpToolResult` with:

- `content: Vec<McpContent>` — a list of content items (text, image, resource).
- `is_error: bool` — whether the tool invocation produced an error.
- `to_text() -> String` — convenience method that joins all text content with newlines.

### McpContent variants

```rust
use openai_agents::mcp::McpContent;

// Text content — the most common case.
let text = McpContent::Text { text: "Hello from the tool".to_owned() };

// Base64-encoded image.
let image = McpContent::Image {
    data: "<base64 data>".to_owned(),
    mime_type: "image/png".to_owned(),
};

// A resource reference.
let resource = McpContent::Resource {
    uri: "file:///tmp/output.json".to_owned(),
    text: Some("{ \"result\": 42 }".to_owned()),
};
```

## MCPConfig Options

`MCPConfig` controls per-agent error handling and timeout behaviour for MCP tool calls.

```rust
use openai_agents::mcp::MCPConfig;

let config = MCPConfig::new()
    // When true, tool errors are returned to the model as a text message
    // so it can recover. When false (default), errors propagate as AgentError.
    .with_convert_errors(true)
    // Timeout for each tool call, in seconds.
    .with_timeout(30.0);
```

Attach `MCPConfig` when building an agent:

```rust,no_run
use openai_agents::{Agent, mcp::{MCPConfig, MCPServer}};

let server = MCPServer::sse("my-server", "https://mcp.example.com/sse");
let config = MCPConfig::new().with_convert_errors(true).with_timeout(15.0);

let agent = Agent::<()>::builder("assistant")
    .instructions("You are a helpful assistant.")
    .mcp_server(server)
    .mcp_config(config)
    .build();
```

## Tool Filtering

By default all tools from an MCP server are exposed to the agent. You can restrict this with
a `ToolFilter`.

### Allowlist

Only the listed tools are available:

```rust
use openai_agents::mcp::ToolFilterStatic;

let filter = ToolFilterStatic::allow(vec![
    "search".to_owned(),
    "fetch_url".to_owned(),
]);

assert!(filter.is_allowed("search"));
assert!(!filter.is_allowed("delete_file")); // blocked by allowlist
```

### Blocklist

All tools except the listed ones are available:

```rust
use openai_agents::mcp::ToolFilterStatic;

let filter = ToolFilterStatic::block(vec![
    "delete_file".to_owned(),
    "execute_shell".to_owned(),
]);

assert!(filter.is_allowed("search"));
assert!(!filter.is_allowed("delete_file")); // explicitly blocked
```

### Combined filter

When both lists are specified, the allowlist is applied first, then the blocklist removes
any remaining disallowed tools:

```rust
use openai_agents::mcp::{ToolFilter, ToolFilterStatic};
use std::collections::HashSet;

let filter = ToolFilterStatic {
    allowed_tool_names: Some(["read_file", "write_file", "delete_file"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()),
    blocked_tool_names: Some(std::iter::once("delete_file".to_owned()).collect()),
};

assert!(filter.is_allowed("read_file"));
assert!(filter.is_allowed("write_file"));
assert!(!filter.is_allowed("delete_file")); // blocked overrides allowed
```

Use `ToolFilter::from_lists` for a convenient constructor:

```rust
use openai_agents::mcp::ToolFilter;

let filter = ToolFilter::from_lists(
    Some(vec!["read_file".to_owned(), "list_dir".to_owned()]),
    None,
);
```

## JSON-RPC Protocol Details

MCP uses JSON-RPC 2.0. The SDK protocol types are in `openai_agents::mcp::protocol`.

### Request

```rust
use openai_agents::mcp::protocol::JsonRpcRequest;
use serde_json::json;

let request = JsonRpcRequest::new(
    1,                    // unique request ID
    "tools/call",         // method name
    Some(json!({
        "name": "get_weather",
        "arguments": { "city": "Tokyo" }
    })),
);
```

### Response

```rust
use openai_agents::mcp::protocol::{JsonRpcResponse, JsonRpcError};
use serde_json::json;

// Success.
let ok = JsonRpcResponse::success(1, json!({ "result": "sunny" }));

// Error.
let err = JsonRpcResponse::error(
    1,
    JsonRpcError::new(-32601, "Method not found"),
);
```

### MCP method reference

| Method | Direction | Purpose |
|---|---|---|
| `initialize` | Client → Server | Start the session and negotiate capabilities |
| `notifications/initialized` | Client → Server | Confirm initialization complete |
| `tools/list` | Client → Server | Request the tool catalogue |
| `tools/call` | Client → Server | Invoke a specific tool |

## Complete Example

```rust,no_run
use openai_agents::{Agent, Runner};
use openai_agents::mcp::{MCPConfig, MCPServer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create and connect to the MCP server.
    let mut server = MCPServer::stdio(
        "filesystem",
        "npx",
        vec![
            "-y".into(),
            "@modelcontextprotocol/server-filesystem".into(),
            "/tmp".into(),
        ],
    );
    server.connect().await?;

    // Print discovered tools.
    let tools = server.list_tools().await?;
    println!("Found {} MCP tools", tools.len());
    for t in &tools {
        println!("  - {}: {}", t.name, t.description.as_deref().unwrap_or(""));
    }

    // Attach the server to an agent.
    let config = MCPConfig::new()
        .with_convert_errors(true)
        .with_timeout(30.0);

    let agent = Agent::<()>::builder("file-assistant")
        .instructions("You help users read and explore files. Use the MCP tools.")
        .mcp_server(server)
        .mcp_config(config)
        .build();

    // Run the agent.
    let result = Runner::run(&agent, "List the files in /tmp", ()).await?;
    println!("{}", result.final_output);

    Ok(())
}
```

## See Also

- [Tools](./tools.md) — hand-written function tools.
- [Configuration](./config.md) — `RunConfig` and `ModelSettings`.
- [Examples](./examples.md) — runnable examples.
