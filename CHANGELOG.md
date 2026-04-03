# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-03

### Added

- **WebSearchTool `external_web_access`**: Added `Option<bool>` field to `WebSearchTool` to control whether the tool may fetch live internet content, matching the Python SDK's `WebSearchTool.external_web_access`.
- **MCP resource methods**: Added `list_resources()`, `list_resource_templates()`, and `read_resource()` methods to `MCPServer` for discovering and reading MCP server resources. Added corresponding protocol types: `McpResource`, `McpResourceTemplate`, `McpResourceContent`, `ListResourcesResult`, `ListResourceTemplatesResult`, `ReadResourceResult`.
- **MCP session ID**: Added `session_id()` accessor to `MCPServer` that returns the `Mcp-Session-Id` assigned by `StreamableHttp` servers during initialization, enabling session resumption across restarts.
- **MCP callable approval policies**: Added `ApprovalPolicy`, `ApprovalPolicySetting`, and `ApprovalCallable` types to `mcp::config`. Approval policies can be `Always`, `Never`, per-tool mappings, or async callable functions. Added `MCPConfig::with_approval_policy()` builder method.
- **Reasoning content replay**: Added `models::reasoning_content_replay` module with `ReasoningContentSource`, `ReasoningContentReplayContext`, `ShouldReplayReasoningContent` callback type, `default_should_replay_reasoning_content()` (returns true for DeepSeek models), and `extract_reasoning_content()` utility.
- **`remove_all_tools` handoff filter**: Added `extensions::handoff_filters::remove_all_tools()` which creates a `HandoffInputFilter` that strips tool calls, tool outputs, reasoning items, handoff items, and MCP metadata from conversation history before agent-to-agent transfers.

### Alignment with Python SDK

These changes bring the Rust SDK up to feature parity with the Python SDK v0.13.x releases (0.13.0 through 0.13.4).

## [0.1.0] - 2026-03-15

### Added

- Initial release of the OpenAI Agents SDK for Rust.
- Core agent framework with `Agent<C>`, `AgentBuilder<C>`, and generic context support.
- `Runner` execution loop with tool dispatch, handoff resolution, and guardrail enforcement.
- `Model` and `ModelProvider` traits for pluggable LLM backends.
- Built-in `OpenAIResponsesModel` and `OpenAIChatCompletionsModel` implementations.
- `AnyProvider` and `LiteLLMModel` adapters for non-OpenAI providers.
- Function tools via `function_tool()` with automatic JSON Schema generation.
- Hosted tools: `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ComputerTool`, `ApplyPatchTool`.
- Input and output guardrails with `InputGuardrail<C>` and `OutputGuardrail<C>`.
- Tool-level guardrails with `ToolInputGuardrail<C>` and `ToolOutputGuardrail<C>`.
- Agent-to-agent handoffs with `Handoff<C>` and configurable input filters.
- MCP server integration with stdio, SSE, and StreamableHttp transports.
- Session-based memory with `InMemorySession`, `CompactingSession`, and `EncryptedSession`.
- Distributed tracing with OpenTelemetry OTLP export support.
- Streaming support via `StreamEvent` and `RunResultStreaming`.
- Run state serialization with `RunState` for checkpointing.
- Retry policies for transient failures.
- Feature flags: `mcp`, `voice`, `tracing-otlp`, `sqlite-session`.
