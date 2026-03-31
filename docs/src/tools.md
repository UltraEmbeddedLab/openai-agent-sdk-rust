# Tools

Tools give agents the ability to take actions — calling APIs, querying
databases, running computations, or anything else you can express as an async
Rust function. The `Tool<C>` enum covers every tool type the runner supports.

## Tool enum variants

```rust
pub enum Tool<C: Send + Sync + 'static> {
    Function(FunctionTool<C>),
    WebSearch(WebSearchTool),
    FileSearch(FileSearchTool),
    CodeInterpreter(CodeInterpreterTool),
    Computer(ComputerTool),
    ApplyPatch(ApplyPatchTool),
}
```

`Function` is for user-defined tools. All other variants are hosted tools
managed by the OpenAI platform and require no invoke function.

## function_tool — the primary API

`function_tool` creates a `FunctionTool<C>` from any async closure or function.
The input type `T` must implement `serde::Deserialize` and `schemars::JsonSchema`
so the SDK can generate the parameter schema automatically:

```rust
pub fn function_tool<C, T, F, Fut>(
    name: impl Into<String>,
    description: impl Into<String>,
    func: F,
) -> Result<FunctionTool<C>>
where
    C: Send + Sync + 'static,
    T: DeserializeOwned + JsonSchema + Send + 'static,
    F: Fn(ToolContext<C>, T) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput>> + Send + 'static,
```

### Full example

```rust
use schemars::JsonSchema;
use serde::Deserialize;
use openai_agents::{Agent, Runner, Tool};
use openai_agents::items::ToolOutput;
use openai_agents::tool::{ToolContext, function_tool};

/// Input schema for the weather tool.
#[derive(Deserialize, JsonSchema)]
struct WeatherInput {
    /// The city to look up.
    city: String,
}

let weather_tool = function_tool::<(), WeatherInput, _, _>(
    "get_weather",
    "Get the current weather for a city.",
    |_ctx: ToolContext<()>, input: WeatherInput| async move {
        // Call a real API here in production.
        Ok(ToolOutput::Text(format!("Sunny, 22 °C in {}", input.city)))
    },
)?;

let agent = Agent::<()>::builder("weather-bot")
    .instructions("Help users check the weather.")
    .tool(Tool::Function(weather_tool))
    .build();
```

The `?` at the end propagates `AgentError::UserError` if the generated schema
cannot be converted to strict mode (a rare edge case for types with
`additionalProperties: true`).

## Input types with JsonSchema derive

Tool input types must derive both `Deserialize` and `JsonSchema`. Document each
field with a `///` doc comment — the comment becomes the field description in
the schema shown to the model:

```rust
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct SearchInput {
    /// The search query string.
    query: String,
    /// Maximum number of results to return. Defaults to 10.
    limit: Option<u32>,
}
```

Nested structs, enums, and `Vec<T>` are all supported as long as every nested
type also derives `JsonSchema`.

## ToolOutput variants

Tool functions return `Result<ToolOutput>`. Three output variants are available:

```rust
pub enum ToolOutput {
    /// Plain text output.
    Text(String),
    /// Image identified by URL or file ID.
    Image {
        image_url: Option<String>,
        file_id:   Option<String>,
    },
    /// File identified by data, URL, or file ID.
    File {
        file_data: Option<String>,
        file_url:  Option<String>,
        file_id:   Option<String>,
        filename:  Option<String>,
    },
}
```

Most tools return `ToolOutput::Text`. Return `Image` or `File` when the tool
produces media that the model or the application should reference:

```rust
Ok(ToolOutput::Image {
    image_url: Some("https://example.com/chart.png".to_string()),
    file_id: None,
})
```

## ToolContext — accessing the context in a tool

Every tool function receives a `ToolContext<C>`, which carries the shared run
context, the tool name, and the tool call ID:

```rust
pub struct ToolContext<C: Send + Sync + 'static> {
    pub context:      Arc<RwLock<RunContextWrapper<C>>>,
    pub tool_name:    String,
    pub tool_call_id: String,
}
```

Read the context with `ctx.context.read().await`:

```rust
use schemars::JsonSchema;
use serde::Deserialize;
use openai_agents::items::ToolOutput;
use openai_agents::tool::{ToolContext, function_tool};

struct MyCtx { api_base: String }

#[derive(Deserialize, JsonSchema)]
struct FetchInput { path: String }

let fetch_tool = function_tool::<MyCtx, FetchInput, _, _>(
    "fetch",
    "Fetch a resource relative to the API base URL.",
    |ctx: ToolContext<MyCtx>, input: FetchInput| async move {
        let base = ctx.context.read().await.context.api_base.clone();
        let url = format!("{}/{}", base, input.path);
        // Make the HTTP request...
        Ok(ToolOutput::Text(format!("fetched: {url}")))
    },
)?;
```

See [Context](./context.md) for more on `RunContextWrapper`.

## Hosted tools

Hosted tools are provided by the OpenAI platform and do not require a Rust
invoke function. Construct them and wrap in `Tool::*`:

### WebSearchTool

```rust
use openai_agents::{Tool, tool::WebSearchTool};

let tool = Tool::WebSearch(WebSearchTool {
    user_location: None,
    search_context_size: Some("medium".to_string()),
});
```

### FileSearchTool

```rust
use openai_agents::{Tool, tool::FileSearchTool};

let tool = Tool::FileSearch(FileSearchTool {
    vector_store_ids: vec!["vs_abc123".to_string()],
    max_num_results: Some(5),
});
```

### CodeInterpreterTool

```rust
use openai_agents::{Tool, tool::CodeInterpreterTool};

let tool = Tool::CodeInterpreter(CodeInterpreterTool {
    container: None,
});
```

### ComputerTool

```rust
use openai_agents::{Tool, ComputerTool};

let tool = Tool::Computer(ComputerTool::new(1024, 768));
```

### ApplyPatchTool

```rust
use openai_agents::{Tool, ApplyPatchTool};

let tool = Tool::ApplyPatch(ApplyPatchTool);
```

## Multiple tools on one agent

Call `.tool()` multiple times or pass a `Vec` to `.tools()`:

```rust
let agent = Agent::<()>::builder("researcher")
    .instructions("Answer questions using all available tools.")
    .tool(Tool::WebSearch(WebSearchTool::default()))
    .tool(Tool::Function(my_function_tool))
    .tool(Tool::FileSearch(FileSearchTool {
        vector_store_ids: vec!["vs_xyz".to_string()],
        max_num_results: Some(3),
    }))
    .build();
```

## Tool guardrails

Wrap a function tool with a `ToolInputGuardrail` or `ToolOutputGuardrail` to
validate arguments before invocation or sanitise output before it reaches the
model. See [Guardrails](./guardrails.md) for details.
