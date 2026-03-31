# Running Agents

The `Runner` struct is the main entry point for executing agents. It provides
three static methods covering the three main execution patterns: single-model
runs, multi-agent runs with handoff registries, and streaming runs.

## Runner::run_with_model

The simplest way to run an agent to completion:

```rust
pub async fn run_with_model<C: Send + Sync + 'static>(
    agent: &Agent<C>,
    input: impl Into<InputContent>,
    context: C,
    model: Arc<dyn Model>,
    hooks: Option<Arc<dyn RunHooks<C>>>,
    config: Option<RunConfig>,
) -> Result<RunResult>
```

The runner drives the full agent loop — resolving instructions, calling the
model, executing tools, running guardrails — until a final output is produced
or the turn limit is reached:

```rust
use std::sync::Arc;
use openai_agents::{Agent, Runner};
use openai_agents::models::openai_responses::OpenAIResponsesModel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::<()>::builder("assistant")
        .instructions("You are a helpful assistant.")
        .build();

    let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

    let result = Runner::run_with_model(
        &agent,
        "What is the capital of France?",
        (),    // context
        model,
        None,  // hooks
        None,  // config (defaults: 10 max turns)
    )
    .await?;

    println!("{}", result.final_output);
    Ok(())
}
```

### Passing a context

Any `Send + Sync + 'static` type can serve as the context. Tools and hooks
receive it through a `RunContextWrapper<C>`:

```rust
struct AppContext {
    user_id: String,
}

let agent = Agent::<AppContext>::builder("ctx-agent")
    .instructions("Be helpful.")
    .build();

let ctx = AppContext { user_id: "usr_42".to_string() };

let result = Runner::run_with_model(
    &agent, "Hello", ctx, model, None, None,
).await?;
```

## Runner::run_with_agents

Use this variant when handoffs are involved. Supply a registry that maps agent
names to `&Agent<C>` references. The starting agent does not need to be in the
registry:

```rust
pub async fn run_with_agents<C: Send + Sync + 'static>(
    starting_agent: &Agent<C>,
    agents: &HashMap<String, &Agent<C>>,
    input: impl Into<InputContent>,
    context: C,
    model: Arc<dyn Model>,
    hooks: Option<Arc<dyn RunHooks<C>>>,
    config: Option<RunConfig>,
) -> Result<RunResult>
```

```rust
use std::collections::HashMap;
use std::sync::Arc;
use openai_agents::{Agent, Handoff, Runner};
use openai_agents::models::openai_responses::OpenAIResponsesModel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let billing = Agent::<()>::builder("billing")
        .instructions("Handle all billing questions.")
        .build();

    let support = Agent::<()>::builder("support")
        .instructions("Handle technical support.")
        .build();

    let triage = Agent::<()>::builder("triage")
        .instructions("Route users to billing or support.")
        .handoff(Handoff::to_agent("billing").build())
        .handoff(Handoff::to_agent("support").build())
        .build();

    let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

    let agents: HashMap<String, &Agent<()>> = HashMap::from([
        ("billing".to_string(), &billing),
        ("support".to_string(), &support),
    ]);

    let result = Runner::run_with_agents(
        &triage,
        &agents,
        "I was charged twice last month.",
        (),
        model,
        None,
        None,
    )
    .await?;

    println!("Handled by: {}", result.last_agent_name);
    println!("{}", result.final_output);
    Ok(())
}
```

See [Handoffs](./handoffs.md) for more on multi-agent routing.

## Runner::run_streamed

Returns a `RunResultStreaming` immediately and drives the agent loop in the
background. The agent reference must be `'static`:

```rust
pub fn run_streamed<C: Send + Sync + 'static>(
    agent: &'static Agent<C>,
    input: impl Into<InputContent> + Send + 'static,
    context: C,
    model: Arc<dyn Model>,
    hooks: Option<Arc<dyn RunHooks<C>>>,
    config: Option<RunConfig>,
) -> RunResultStreaming
```

```rust
use std::sync::Arc;
use tokio_stream::StreamExt;
use openai_agents::{Agent, Runner};
use openai_agents::stream_events::StreamEvent;
use openai_agents::models::openai_responses::OpenAIResponsesModel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Box::leak produces a 'static reference required by run_streamed.
    let agent: &'static Agent<()> = Box::leak(Box::new(
        Agent::<()>::builder("streaming-agent")
            .instructions("Answer concisely.")
            .build(),
    ));

    let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

    let mut streaming = Runner::run_streamed(
        agent, "What is Rust?", (), model, None, None,
    );

    let mut stream = streaming.stream_events();
    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::RunItemCreated { name, .. } => println!("item: {name}"),
            StreamEvent::AgentUpdated { new_agent_name } => println!("agent: {new_agent_name}"),
            _ => {}
        }
    }
    drop(stream);

    println!("{}", streaming.final_output);
    Ok(())
}
```

See [Streaming](./streaming.md) for the full event reference.

## RunConfig options

Pass a `RunConfig` to any runner method to control turn limits, tracing, and
model overrides at the run level:

```rust
use openai_agents::config::{RunConfig, ModelSettings};

let config = RunConfig::builder()
    .max_turns(5)
    .workflow_name("my-workflow")
    .model_settings(ModelSettings::new().with_temperature(0.3))
    .tracing_disabled(true)
    .build();

let result = Runner::run_with_model(
    &agent, input, ctx, model, None, Some(config),
).await?;
```

### DEFAULT_MAX_TURNS

When no `RunConfig` is supplied the runner defaults to
`DEFAULT_MAX_TURNS = 10`. If the agent loop reaches that limit without producing
a final output, the runner returns `AgentError::MaxTurnsExceeded`.

## Multi-turn conversations

Use `result.to_input_list()` to carry conversation history into the next run.
The returned `Vec<ResponseInputItem>` holds all model outputs from the previous
run in a format the API accepts as input:

```rust
use std::sync::Arc;
use openai_agents::{Agent, Runner};
use openai_agents::items::InputContent;
use openai_agents::models::openai_responses::OpenAIResponsesModel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::<()>::builder("chat")
        .instructions("You are a friendly chat assistant.")
        .build();

    let model: Arc<dyn openai_agents::models::Model> =
        Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

    // Turn 1.
    let result1 = Runner::run_with_model(
        &agent, "My name is Alice.", (), Arc::clone(&model), None, None,
    ).await?;
    println!("Turn 1: {}", result1.final_output);

    // Turn 2 — append history from turn 1.
    let mut history = result1.to_input_list();
    history.push(serde_json::json!({
        "role": "user",
        "content": "What is my name?"
    }));

    let result2 = Runner::run_with_model(
        &agent,
        InputContent::Items(history),
        (),
        Arc::clone(&model),
        None,
        None,
    ).await?;
    println!("Turn 2: {}", result2.final_output);

    Ok(())
}
```

## Error handling

All runner methods return `Result<RunResult>` (a type alias for
`Result<RunResult, AgentError>`). Match on the error variants to handle specific
failure modes:

```rust
use openai_agents::error::AgentError;

match Runner::run_with_model(&agent, input, ctx, model, None, None).await {
    Ok(result) => println!("{}", result.final_output),
    Err(AgentError::MaxTurnsExceeded { max_turns }) => {
        eprintln!("Agent did not finish within {max_turns} turns");
    }
    Err(AgentError::GuardrailTripwire { guardrail_name, .. }) => {
        eprintln!("Blocked by guardrail: {guardrail_name}");
    }
    Err(e) => eprintln!("Run failed: {e}"),
}
```

Common `AgentError` variants:

| Variant | When it occurs |
|---------|----------------|
| `MaxTurnsExceeded` | Turn limit reached without a final output. |
| `GuardrailTripwire` | An input or output guardrail tripped. |
| `ModelBehavior` | The model returned malformed output. |
| `UserError` | Configuration error, e.g., missing handoff target. |
| `ApiError` | HTTP error from the model provider. |
| `Serialization` | JSON deserialization failure. |
