# Tracing

The SDK emits structured tracing spans for every significant event in an agent run: each agent
turn, model generation, tool call, handoff, and guardrail check. Spans integrate with the Rust
`tracing` ecosystem, which means they work with any `tracing-subscriber` layer — including
OpenTelemetry exporters.

The tracing API lives in `openai_agents::tracing_support` (re-exported from the crate root as
`openai_agents::tracing_support`).

## Built-in spans

Five span creation helpers are provided:

| Function | Span name | Key fields |
|---|---|---|
| `agent_span(agent_name, turn)` | `"agent"` | `agent.name`, `agent.turn` |
| `generation_span(agent_name, model)` | `"generation"` | `agent.name`, `model.name` |
| `function_span(agent_name, tool_name)` | `"function"` | `agent.name`, `tool.name` |
| `handoff_span(from_agent, to_agent)` | `"handoff"` | `handoff.from`, `handoff.to` |
| `guardrail_span(guardrail_name, guardrail_type)` | `"guardrail"` | `guardrail.name`, `guardrail.type_` |

All spans also carry an `otel.name` field in `"category:name"` format for OpenTelemetry backends.

### Example: manual instrumentation

```rust
use openai_agents::tracing_support::{agent_span, generation_span, function_span};

let span = agent_span("my_agent", 1);
let _guard = span.enter();

// Nest a generation span inside the agent span.
let gen_span = generation_span("my_agent", "gpt-4o");
let _gen_guard = gen_span.enter();

// A tool call spans can be nested further.
let tool_span = function_span("my_agent", "web_search");
let _tool_guard = tool_span.enter();
```

The spans follow standard `tracing` scoping rules: they are active until the guard is dropped.
The runner creates and manages these spans automatically during a run — manual creation is only
needed for custom instrumentation.

## `TracingConfig`

`TracingConfig` controls what data is recorded in spans and whether tracing is active for a
specific run:

```rust
use openai_agents::tracing_support::TracingConfig;
use serde_json::json;

let config = TracingConfig::new()
    .with_workflow_name("order-processing")
    .with_group_id("session-abc123")
    .with_metadata(json!({ "environment": "production" }))
    .with_sensitive_data(false)   // omit prompts and outputs from spans
    .with_api_key("sk-my-trace-key")
    .with_disabled(false);
```

| Field | Default | Description |
|---|---|---|
| `include_sensitive_data` | `true` | Include prompt/output text in span fields. |
| `workflow_name` | `None` | Custom workflow label attached to the trace. |
| `group_id` | `None` | Group identifier linking related traces. |
| `metadata` | `None` | Arbitrary JSON metadata attached to the trace. |
| `api_key` | `None` | API key for trace export. |
| `disabled` | `false` | If `true`, this trace is suppressed. |

Attach `TracingConfig` to a run via `RunConfig`:

```rust
use openai_agents::config::RunConfig;
use openai_agents::tracing_support::TracingConfig;

let run_config = RunConfig {
    tracing: Some(TracingConfig::new().with_workflow_name("customer-chat")),
    ..Default::default()
};
```

## Global disable switch

`set_tracing_disabled` / `is_tracing_disabled` provide a coarse-grained way to suppress all
tracing output process-wide:

```rust
use openai_agents::tracing_support::{set_tracing_disabled, is_tracing_disabled};

// Suppress all tracing (e.g. in tests or CI).
set_tracing_disabled(true);
assert!(is_tracing_disabled());

// Re-enable.
set_tracing_disabled(false);
assert!(!is_tracing_disabled());
```

When disabled, span creation helpers still return valid `tracing::Span` values (they become
no-ops internally), so callers do not need conditional logic.

## Development logging with `enable_verbose_stdout_logging`

For local development, call `enable_verbose_stdout_logging()` at startup to get human-readable
output filtered by the `RUST_LOG` environment variable, defaulting to `debug` for the SDK and
`info` for dependencies:

```rust,no_run
use openai_agents::logger::enable_verbose_stdout_logging;

fn main() {
    enable_verbose_stdout_logging();
    // ... run your agents ...
}
```

Use `RUST_LOG` to fine-tune what is shown:

```bash
RUST_LOG=openai_agents=trace cargo run   # everything
RUST_LOG=openai_agents::runner=debug     # runner module only
RUST_LOG=warn                            # warnings and errors only
```

To set a specific filter programmatically:

```rust,no_run
use openai_agents::logger::setup_logging;

setup_logging("openai_agents=debug,info");
```

Both functions are idempotent: if a global subscriber has already been registered they silently
do nothing.

## OTLP export with the `tracing-otlp` feature

For production observability, enable the `tracing-otlp` feature to export spans to any
OpenTelemetry-compatible backend (Jaeger, Grafana Tempo, Zipkin, etc.):

```toml
[dependencies]
openai-agents = { version = "0.1", features = ["tracing-otlp"] }
```

Then initialise the exporter at application startup:

```rust,no_run
use openai_agents::tracing_support::OtlpExporterConfig;
use openai_agents::tracing_support::exporter::init_otlp_tracing;

fn main() -> openai_agents::Result<()> {
    let config = OtlpExporterConfig::new()
        .with_endpoint("http://localhost:4317")
        .with_service_name("my-agent-service");

    // Keep the guard alive for the lifetime of the process.
    // Dropping it flushes buffered spans and shuts down the trace provider.
    let _guard = init_otlp_tracing(config)?;

    // ... run your agents ...
    Ok(())
}
```

`init_otlp_tracing` sets up:

1. A `tracing-subscriber` registry with an OpenTelemetry layer.
2. A gRPC OTLP exporter sending spans to the configured endpoint.
3. A console logging layer filtered by `RUST_LOG`.

### `OtlpExporterConfig`

| Field | Default | Description |
|---|---|---|
| `endpoint` | `"http://localhost:4317"` | OTLP collector endpoint (gRPC). |
| `service_name` | `"openai-agents"` | Service name reported in traces. |

```rust
use openai_agents::tracing_support::OtlpExporterConfig;

let config = OtlpExporterConfig::new()
    .with_endpoint("http://otel-collector.internal:4317")
    .with_service_name("order-service");
```

Without the `tracing-otlp` feature, `init_otlp_tracing` returns `AgentError::UserError` directing
you to enable the flag.

## Integrating with an existing `tracing-subscriber`

If your application already sets up its own `tracing-subscriber`, simply ensure it is registered
before starting any agent runs. The SDK's span helpers call standard `tracing` macros and will
automatically integrate with whatever subscriber is active.

For OpenTelemetry without the built-in OTLP helper, configure `tracing-opentelemetry` manually:

```toml
[dependencies]
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
tracing-opentelemetry = "0.27"
```

```rust,no_run
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

fn init_tracing() {
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(my_tracer()))
        .with(tracing_subscriber::fmt::layer())
        .init();
}
```

## Span field reference

### `agent_span`

```
agent
  agent.name  = "triage_agent"
  agent.turn  = 1
  otel.name   = "agent:triage_agent"
```

### `generation_span`

```
generation
  agent.name  = "triage_agent"
  model.name  = "gpt-4o"
  otel.name   = "generation:gpt-4o"
```

### `function_span`

```
function
  agent.name  = "triage_agent"
  tool.name   = "web_search"
  otel.name   = "function:web_search"
```

### `handoff_span`

```
handoff
  handoff.from  = "triage_agent"
  handoff.to    = "billing_agent"
  otel.name     = "handoff:triage_agent->billing_agent"
```

### `guardrail_span`

```
guardrail
  guardrail.name   = "profanity_filter"
  guardrail.type_  = "input"
  otel.name        = "guardrail:profanity_filter"
```

`guardrail_type` is either `"input"` or `"output"`.

## See also

- [Configuration](./config.md) — attaching `TracingConfig` to `RunConfig`.
- [Running Agents](./running_agents.md) — the `Runner` API.
- [Guardrails](./guardrails.md) — guardrail spans.
- [Handoffs](./handoffs.md) — handoff spans.
