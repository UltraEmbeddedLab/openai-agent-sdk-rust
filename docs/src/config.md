# Configuration

The SDK provides two configuration types: `ModelSettings` for individual LLM
calls and `RunConfig` for an entire agent run. Both use fluent builder APIs.

## ModelSettings

`ModelSettings` holds optional parameters for a single model call. Every field
defaults to `None`, meaning the model's own defaults apply.

```rust
pub struct ModelSettings {
    pub temperature:          Option<f64>,
    pub top_p:                Option<f64>,
    pub frequency_penalty:    Option<f64>,
    pub presence_penalty:     Option<f64>,
    pub tool_choice:          Option<ToolChoice>,
    pub parallel_tool_calls:  Option<bool>,
    pub truncation:           Option<Truncation>,
    pub max_tokens:           Option<u32>,
    pub metadata:             Option<HashMap<String, String>>,
    pub store:                Option<bool>,
    pub extra_body:           Option<serde_json::Value>,
    pub extra_headers:        Option<HashMap<String, String>>,
    pub extra_args:           Option<HashMap<String, serde_json::Value>>,
}
```

### Fluent builder methods

`ModelSettings` exposes `with_*` methods that consume and return `self`:

```rust
use openai_agents::ModelSettings;
use openai_agents::config::ToolChoice;

let settings = ModelSettings::new()
    .with_temperature(0.7)
    .with_top_p(0.95)
    .with_max_tokens(1024)
    .with_tool_choice(ToolChoice::Auto)
    .with_store(true);
```

### Struct literal construction

Since every field is `Option`, you can also use a struct literal with
`..Default::default()` to set only the fields you need:

```rust
use openai_agents::ModelSettings;

let settings = ModelSettings {
    temperature: Some(0.2),
    presence_penalty: Some(0.1),
    ..Default::default()
};
```

### resolve() — layered settings

`ModelSettings::resolve` merges two settings objects, giving precedence to the
override. `None` fields in the override fall back to the base values. The
`extra_args` map is merged (union) rather than replaced:

```rust
let base = ModelSettings::new().with_temperature(0.7);
let per_agent = ModelSettings::new().with_temperature(0.3).with_max_tokens(512);

// per_agent wins on temperature; base contributes nothing else.
let effective = base.resolve(Some(&per_agent));
assert_eq!(effective.temperature, Some(0.3));
assert_eq!(effective.max_tokens, Some(512));
```

The runner applies this merge automatically: the agent's `model_settings` is
resolved against the run-level `RunConfig::model_settings`.

### ModelSettings fields reference

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | `f64` | Sampling temperature (0–2). Higher = more random. |
| `top_p` | `f64` | Nucleus sampling. Model uses top-p probability mass. |
| `frequency_penalty` | `f64` | Penalises repeated tokens (–2 to 2). |
| `presence_penalty` | `f64` | Penalises tokens that already appeared (–2 to 2). |
| `tool_choice` | `ToolChoice` | See below. |
| `parallel_tool_calls` | `bool` | Allow the model to call multiple tools in one turn. |
| `truncation` | `Truncation` | `Auto` or `Disabled`. |
| `max_tokens` | `u32` | Maximum completion tokens. |
| `metadata` | `HashMap<String, String>` | Arbitrary metadata stored with the response. |
| `store` | `bool` | Whether to store the response server-side. |
| `extra_body` | `serde_json::Value` | Extra fields merged into the request body. |
| `extra_headers` | `HashMap<String, String>` | Extra HTTP headers for this call. |
| `extra_args` | `HashMap<String, serde_json::Value>` | Arbitrary extra arguments. |

## ToolChoice

`ToolChoice` controls how the model selects tools on each call:

```rust
pub enum ToolChoice {
    /// Model decides (default API behaviour).
    Auto,
    /// Model must use at least one tool.
    Required,
    /// Model must not use any tools.
    None,
    /// Model must use the tool with this exact name.
    Named(String),
}
```

```rust
use openai_agents::config::ToolChoice;
use openai_agents::ModelSettings;

// Force the model to always call a tool.
let settings = ModelSettings::new().with_tool_choice(ToolChoice::Required);

// Force a specific tool.
let settings = ModelSettings::new()
    .with_tool_choice(ToolChoice::Named("get_weather".to_string()));
```

## RunConfig

`RunConfig` controls an entire agent run. Build it with `RunConfig::builder()`:

```rust
use openai_agents::config::{RunConfig, ModelSettings};

let config = RunConfig::builder()
    .max_turns(5)
    .workflow_name("invoice-processing")
    .model("gpt-4o")
    .model_settings(ModelSettings::new().with_temperature(0.0))
    .tracing_disabled(false)
    .trace_id("trace-abc-123")
    .group_id("conversation-456")
    .build();
```

### RunConfig fields reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `Option<ModelRef>` | `None` | Overrides per-agent model for the whole run. |
| `model_settings` | `Option<ModelSettings>` | `None` | Run-level settings merged with agent settings. |
| `max_turns` | `u32` | `10` (`DEFAULT_MAX_TURNS`) | Maximum turns before `MaxTurnsExceeded`. |
| `tracing_disabled` | `bool` | `false` | Disable span emission for this run. |
| `workflow_name` | `String` | `"agent_workflow"` | Label shown in traces. |
| `trace_id` | `Option<String>` | `None` | Custom trace ID; auto-generated if absent. |
| `group_id` | `Option<String>` | `None` | Links related traces (e.g. a conversation ID). |

### DEFAULT_MAX_TURNS

```rust
use openai_agents::config::DEFAULT_MAX_TURNS;

assert_eq!(DEFAULT_MAX_TURNS, 10);
```

When `RunConfig` is not supplied, or when `max_turns` is not set on the builder,
the runner uses this default.

## Global config

These functions configure SDK-wide defaults before constructing any model:

```rust
use openai_agents::{
    set_default_openai_key, set_default_model,
    set_default_base_url, set_default_openai_api,
    OpenAiApi,
};

// API key (overrides OPENAI_API_KEY env var).
set_default_openai_key("sk-...");

// Default model name.
set_default_model("gpt-4o-mini");

// Base URL (useful for proxies or compatible providers).
set_default_base_url("https://my-proxy.example.com/v1");

// Switch to Chat Completions instead of Responses API.
set_default_openai_api(OpenAiApi::ChatCompletions);
```

Corresponding getters:

```rust
use openai_agents::{
    get_default_openai_key, get_default_model,
    get_default_base_url, get_default_openai_api,
};

if let Some(key) = get_default_openai_key() {
    println!("API key is set ({} chars)", key.len());
}
```

Global config is stored in process-wide `RwLock` statics and is safe to call
from multiple threads.

## Applying config to a run

Pass a `RunConfig` as the last argument to any runner method:

```rust
use std::sync::Arc;
use openai_agents::{Agent, Runner};
use openai_agents::config::{RunConfig, ModelSettings};
use openai_agents::models::openai_responses::OpenAIResponsesModel;

let config = RunConfig::builder()
    .max_turns(3)
    .model_settings(ModelSettings::new().with_temperature(0.5))
    .build();

let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);
let agent = Agent::<()>::builder("configured").instructions("Be brief.").build();

let result = Runner::run_with_model(
    &agent, "Summarise quantum computing.", (), model, None, Some(config),
).await?;
```
