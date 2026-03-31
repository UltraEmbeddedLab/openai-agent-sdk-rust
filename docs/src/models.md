# Models and Providers

The SDK separates the concern of *calling a model* from the concern of *selecting which model
to use*. Two traits govern this:

- `Model` — knows how to make a single API call (blocking or streaming).
- `ModelProvider` — maps a model name string to a `Model` instance.

This design lets you swap in any LLM backend — OpenAI, local Ollama, LiteLLM, or your own —
without changing agent code.

## The `Model` Trait

```rust
use openai_agents::models::{Model, ModelTracing, ToolSpec, HandoffToolSpec, OutputSchemaSpec};
use openai_agents::config::ModelSettings;
use openai_agents::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use openai_agents::error::Result;
use std::pin::Pin;
use tokio_stream::Stream;
use async_trait::async_trait;

#[async_trait]
pub trait Model: Send + Sync {
    /// Non-streaming call. Returns the complete response.
    async fn get_response(
        &self,
        system_instructions: Option<&str>,
        input: &[ResponseInputItem],
        model_settings: &ModelSettings,
        tools: &[ToolSpec],
        output_schema: Option<&OutputSchemaSpec>,
        handoffs: &[HandoffToolSpec],
        tracing: ModelTracing,
        previous_response_id: Option<&str>,
    ) -> Result<ModelResponse>;

    /// Streaming call. Returns a stream of incremental events.
    fn stream_response<'a>(
        &'a self,
        system_instructions: Option<&'a str>,
        input: &'a [ResponseInputItem],
        model_settings: &'a ModelSettings,
        tools: &'a [ToolSpec],
        output_schema: Option<&'a OutputSchemaSpec>,
        handoffs: &'a [HandoffToolSpec],
        tracing: ModelTracing,
        previous_response_id: Option<&'a str>,
    ) -> Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent>> + Send + 'a>>;

    /// Release resources. Default is a no-op.
    async fn close(&self) {}
}
```

All inputs and outputs use the OpenAI Responses API JSON format so that the runner loop works
uniformly regardless of the backend.

## The `ModelProvider` Trait

```rust
use openai_agents::models::{Model, ModelProvider};
use openai_agents::error::Result;
use std::sync::Arc;
use async_trait::async_trait;

#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Resolve a model name to a `Model` instance.
    /// Pass `None` to get the provider's default model.
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>>;

    /// Release resources. Default is a no-op.
    async fn close(&self) {}
}
```

## ModelTracing

`ModelTracing` controls whether API request/response data is included in traces.

```rust
use openai_agents::models::ModelTracing;

// Trace everything (default).
let tracing = ModelTracing::Enabled;

// Disable tracing entirely.
let tracing = ModelTracing::Disabled;

// Emit spans but omit the request and response payloads (useful when inputs contain PII).
let tracing = ModelTracing::EnabledWithoutData;

// Check flags.
assert!(!ModelTracing::Enabled.is_disabled());
assert!(ModelTracing::Enabled.include_data());
assert!(!ModelTracing::EnabledWithoutData.include_data());
```

---

## OpenAIResponsesModel

`OpenAIResponsesModel` calls the OpenAI `/v1/responses` endpoint. This is the preferred
backend for OpenAI models because it natively supports tool use, structured output, and
multi-turn history via `previous_response_id`.

```rust,no_run
use openai_agents::models::openai_responses::OpenAIResponsesModel;

// Read API key from OPENAI_API_KEY environment variable.
let model = OpenAIResponsesModel::new("gpt-4o")?;

// Explicit key and custom base URL (e.g. Azure OpenAI or a proxy).
let model = OpenAIResponsesModel::with_config(
    "gpt-4o",
    "sk-...",
    "https://my-proxy.example.com/v1",
);
```

`OpenAIProvider` is the corresponding `ModelProvider` for this backend:

```rust,no_run
use openai_agents::models::OpenAIProvider;
use openai_agents::models::ModelProvider;

let provider = OpenAIProvider::from_env()?;
let model = provider.get_model(Some("gpt-4o-mini"))?;
```

---

## OpenAIChatCompletionsModel

`OpenAIChatCompletionsModel` uses the `/v1/chat/completions` endpoint. This is supported by
Azure OpenAI, many third-party gateways, and local servers that expose an OpenAI-compatible
API.

```rust,no_run
use openai_agents::models::openai_chatcompletions::OpenAIChatCompletionsModel;

// Default OpenAI endpoint.
let model = OpenAIChatCompletionsModel::new("gpt-4o", "sk-...")?;

// Custom base URL (Azure, vLLM, etc.).
let model = OpenAIChatCompletionsModel::with_base_url(
    "gpt-35-turbo",
    "sk-...",
    "https://my-resource.openai.azure.com/openai/deployments/my-deployment",
);
```

> **Note.** The Chat Completions backend performs an internal conversion between the Responses
> API format used by the runner and the Chat Completions `messages` format. The conversion is
> transparent to agent code.

---

## MultiProvider

`MultiProvider` routes model name strings to the correct backend automatically. This is the
most convenient provider for production use when you need to mix Responses API and Chat
Completions models in the same application.

### Routing rules

| Model name | Backend |
|---|---|
| `"gpt-4o"`, `"o3-mini"`, or any bare name | OpenAI Responses API |
| `"openai/gpt-4o"` | OpenAI Responses API (prefix stripped) |
| `"chatcompletions/gpt-4o"` | OpenAI Chat Completions API |
| `"myprovider/..."` | Custom registered provider |

```rust,no_run
use openai_agents::models::MultiProvider;
use openai_agents::models::ModelProvider;

// Read key from OPENAI_API_KEY.
let provider = MultiProvider::from_env()?;

// Explicit key.
let provider = MultiProvider::new("sk-...");

// Custom base URL.
let provider = MultiProvider::with_base_url("sk-...", "https://my-proxy.example.com/v1");

// Route to the Responses API.
let model = provider.get_model(Some("gpt-4o"))?;

// Force Chat Completions.
let model = provider.get_model(Some("chatcompletions/gpt-4o-mini"))?;
```

### Registering custom providers

```rust,no_run
use openai_agents::models::{MultiProvider, ModelProvider};
use std::sync::Arc;

let mut provider = MultiProvider::from_env()?;

// Register a custom provider for the "my-backend" prefix.
// Any call to get_model("my-backend/...") will be routed here.
provider.register_provider(
    "my-backend",
    Arc::new(my_custom_provider),
);

let model = provider.get_model(Some("my-backend/large-v2"))?;
```

---

## LiteLLMModel / LiteLLMProvider

[LiteLLM](https://github.com/BerriAI/litellm) is a proxy that exposes an OpenAI-compatible
endpoint and routes to Anthropic, Google, AWS Bedrock, Azure, and many others behind the
scenes. The SDK provides `LiteLLMModel` and `LiteLLMProvider` as first-class wrappers.

```rust,no_run
use openai_agents::models::litellm::{LiteLLMModel, LiteLLMProvider};
use openai_agents::models::ModelProvider;

// Direct model (delegates to OpenAIChatCompletionsModel internally).
let model = LiteLLMModel::new(
    "anthropic/claude-sonnet-4-20250514",
    "my-litellm-key",
    "http://localhost:4000/v1",
);

// Provider for use with agents.
let provider = LiteLLMProvider::new("my-litellm-key", "http://localhost:4000/v1");
let model = provider.get_model(Some("anthropic/claude-sonnet-4-20250514"))?;
```

The default base URL is `http://localhost:4000/v1`, which is where `litellm --port 4000`
listens by default.

---

## AnyProvider

`AnyProvider` is the simplest way to connect to any OpenAI-compatible API endpoint — local
or remote — without any prefix routing logic.

```rust,no_run
use openai_agents::models::any_provider::AnyProvider;
use openai_agents::models::ModelProvider;

// Local Ollama server (no authentication needed).
let provider = AnyProvider::unauthenticated("http://localhost:11434/v1");
let model = provider.get_model(Some("llama3"))?;

// vLLM or TGI with an API key.
let provider = AnyProvider::new("my-key", "http://vllm-server:8000/v1");
let model = provider.get_model(Some("meta-llama/Llama-3-8B-Instruct"))?;
```

`unauthenticated` sends `"no-key"` as the bearer token, which most local servers ignore.

---

## Implementing a Custom Model

To add support for a new LLM backend, implement the `Model` trait:

```rust,no_run
use openai_agents::models::{
    Model, ModelTracing, ToolSpec, HandoffToolSpec, OutputSchemaSpec,
};
use openai_agents::config::ModelSettings;
use openai_agents::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use openai_agents::error::Result;
use std::pin::Pin;
use tokio_stream::Stream;
use async_trait::async_trait;

pub struct MyModel {
    endpoint: String,
    api_key: String,
}

#[async_trait]
impl Model for MyModel {
    async fn get_response(
        &self,
        system_instructions: Option<&str>,
        input: &[ResponseInputItem],
        model_settings: &ModelSettings,
        _tools: &[ToolSpec],
        _output_schema: Option<&OutputSchemaSpec>,
        _handoffs: &[HandoffToolSpec],
        _tracing: ModelTracing,
        _previous_response_id: Option<&str>,
    ) -> Result<ModelResponse> {
        // Call your API here and convert the response into ModelResponse.
        todo!()
    }

    fn stream_response<'a>(
        &'a self,
        _system_instructions: Option<&'a str>,
        _input: &'a [ResponseInputItem],
        _model_settings: &'a ModelSettings,
        _tools: &'a [ToolSpec],
        _output_schema: Option<&'a OutputSchemaSpec>,
        _handoffs: &'a [HandoffToolSpec],
        _tracing: ModelTracing,
        _previous_response_id: Option<&'a str>,
    ) -> Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent>> + Send + 'a>> {
        // Return a stream of incremental events.
        Box::pin(tokio_stream::empty())
    }
}
```

Then wrap it in a `ModelProvider`:

```rust,no_run
use openai_agents::models::{Model, ModelProvider};
use openai_agents::error::Result;
use std::sync::Arc;
use async_trait::async_trait;

pub struct MyProvider {
    api_key: String,
}

#[async_trait]
impl ModelProvider for MyProvider {
    fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>> {
        let name = model_name.unwrap_or("default");
        Ok(Arc::new(MyModel {
            endpoint: format!("https://my-api.example.com/v1/{name}"),
            api_key: self.api_key.clone(),
        }))
    }
}
```

### Using your custom provider with an agent

```rust,no_run
use openai_agents::{Agent, Runner};
use std::sync::Arc;

let provider = Arc::new(MyProvider { api_key: "secret".into() });
let model = provider.get_model(Some("large-v2"))?;

let agent = Agent::<()>::builder("assistant")
    .instructions("You are helpful.")
    .build();

let result = Runner::run_with_model(&agent, "Hello!", (), model, None, None).await?;
println!("{}", result.final_output);
```

---

## See Also

- [Configuration](./config.md) — `ModelSettings`, temperature, tool choice, etc.
- [Tracing](./tracing.md) — observability and `ModelTracing`.
- [Running Agents](./running_agents.md) — how the runner uses models.
