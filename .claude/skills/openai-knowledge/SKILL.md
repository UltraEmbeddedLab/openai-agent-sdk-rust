---
name: openai-knowledge
description: Pull authoritative OpenAI API documentation when working on API integrations (Responses API, Chat Completions, tools, streaming, models, MCP). Uses the OpenAI Developer Docs MCP server.
disable-model-invocation: false
user-invocable: true
---

# OpenAI Knowledge

When working on OpenAI API integrations, use the OpenAI Developer Docs MCP server to get authoritative documentation.

## When to Use

- Implementing or modifying model provider code (`src/models/`).
- Working with tool definitions, function calling, or structured outputs.
- Implementing streaming (SSE, WebSocket).
- Working with the Responses API or Chat Completions API.
- Implementing MCP integration.
- Any work involving OpenAI API request/response formats.

## How to Use

1. Use the `mcp__openaiDeveloperDocs__search_openai_docs` tool to search for relevant documentation.
2. Use the `mcp__openaiDeveloperDocs__fetch_openai_doc` tool to fetch specific documentation pages.
3. Use the `mcp__openaiDeveloperDocs__list_api_endpoints` tool to browse available API endpoints.
4. Use the `mcp__openaiDeveloperDocs__get_openapi_spec` tool for the detailed API specification.

## Reference Implementation

Always cross-reference with the Python SDK implementation at `F:\forks\openai-agents-python\src\agents\models\` to ensure the Rust implementation matches the expected behavior.

## Key API Areas

- **Responses API**: `POST /responses` — the newer API that supports tools, structured output, and streaming natively.
- **Chat Completions API**: `POST /chat/completions` — the classic API with function calling.
- **Models**: Model listing, capabilities, and configuration.
- **Tools**: Function tools, hosted tools (web search, file search, code interpreter).
- **Streaming**: Server-Sent Events for real-time responses.
