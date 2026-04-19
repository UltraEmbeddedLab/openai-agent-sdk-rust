# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-19

### Added

- **`ToolOrigin` / `ToolOriginType`**: Origin metadata attached to `FunctionTool`, `ToolCallItem`, and `ToolCallOutputItem` so callers can tell whether a tool call originated from a plain function tool, an MCP server, or an agent-as-tool wrapper. Matches Python SDK v0.14.2's `ToolOrigin` feature (issue #2228).
- **`Computer` modifier-key variants**: Added `click_with_modifiers`, `double_click_with_modifiers`, `scroll_with_modifiers`, `move_cursor_with_modifiers`, and `drag_with_modifiers` default methods to the `Computer` trait. Drivers can override to support held modifier keys (Ctrl+click, etc.); the defaults forward to the base methods so existing implementations keep working. Mirrors Python SDK fix #2873.
- **`flush_traces()` public API**: Top-level `flush_traces()` function in `tracing_mod` force-flushes buffered OTLP traces without requiring the caller to hold an `OtlpGuard`. Mirrors Python SDK v0.13.5's `flush_traces()` (issue #2135).
- **`RunResultStreaming::run_loop_exception` / `take_run_loop_exception`**: New accessors that surface errors raised by the background streaming run loop (for example during early model setup) after `stream_events()` has completed. Mirrors Python SDK v0.14.2 fix for issue #2929.
- **Streaming input guardrails**: The streaming run loop now evaluates input guardrails before any tool execution and halts the run with `AgentError::InputGuardrailTripwire` when a tripwire triggers. Mirrors Python SDK v0.14.2 fix for issue #2688.

### Fixed

- **`remove_all_tools` handoff filter** now also strips hosted tool types (`code_interpreter_call`, `image_generation_call`, `local_shell_call`, `shell_call`, `apply_patch_call`, and their `*_output` counterparts) from input history before handoff. Matches Python SDK fix #2885.
- **ChatCompletion empty-choices handling** now surfaces any provider error payload in the `ModelBehavior` error message so callers can diagnose upstream failures. Matches Python SDK fix for issue #604.
- **`ItemHelpers::extract_text`** tolerates `null` / missing `text` fields in `output_text` content items (can appear during partial streaming or via LiteLLM gateways). Matches Python SDK fix #2883.
- **Tool-name sanitization** now emits a warning via `tracing::warn!` when non-alphanumeric characters are replaced during registration, matching Python SDK fix #2951.

### Alignment with Python SDK

These changes bring the Rust SDK up to feature parity with the Python SDK v0.14.x releases (0.14.0 through 0.14.2), except for the `sandbox` module (opt-in major feature) and MongoDB session backend (extension).

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
