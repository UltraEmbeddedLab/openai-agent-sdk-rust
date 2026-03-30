# Tools

Tools give agents the ability to take actions — calling APIs, querying databases, running computations, or anything else you can express as a function.

## Function Tools

The simplest way to create a tool is with the `#[function_tool]` attribute macro:

```rust
use openai_agents::function_tool;

#[function_tool]
/// Get the current weather for a location.
async fn get_weather(location: String) -> String {
    format!("The weather in {location} is sunny, 22°C")
}
```

The macro automatically:
- Generates a JSON Schema from the function signature.
- Wraps the function as a `FunctionTool`.
- Uses the doc comment as the tool description.

## Tool Trait

For more control, implement the `Tool` trait directly:

```rust
use openai_agents::{Tool, ToolContext};
use async_trait::async_trait;

struct MyTool;

#[async_trait]
impl<C: Send + Sync + 'static> Tool<C> for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "Does something useful." }
    fn parameters_schema(&self) -> serde_json::Value { todo!() }

    async fn execute(
        &self,
        ctx: &ToolContext<C>,
        args: serde_json::Value,
    ) -> Result<String, AgentError> {
        todo!()
    }
}
```

## Tool Types

| Type | Description |
|------|-------------|
| **Function tools** | Rust functions wrapped with `#[function_tool]` or `FunctionTool` |
| **Hosted tools** | OpenAI-hosted tools (web search, file search, code interpreter) |
| **MCP tools** | Tools from Model Context Protocol servers |

## Adding Tools to Agents

```rust
let agent = Agent::builder()
    .name("ToolBot")
    .instructions("Use your tools to help the user.")
    .tool(get_weather())
    .tool(search_database())
    .build();
```
