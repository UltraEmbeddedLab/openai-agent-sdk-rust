# Context

The context is a user-defined value that flows through every tool invocation,
guardrail check, and lifecycle hook during an agent run. It is never sent to
the LLM; it is purely a way to pass your application's dependencies (database
connections, configuration, request metadata) into your callback code.

## RunContextWrapper\<C\>

The SDK wraps your context value in `RunContextWrapper<C>` before passing it to
tools and hooks:

```rust
pub struct RunContextWrapper<C: Send + Sync + 'static> {
    /// Your application context.
    pub context: C,
    /// Accumulated token usage across all model calls in this run.
    pub usage: Usage,
}
```

`C` must be `Send + Sync + 'static`. This is required because the runner may
schedule work across Tokio tasks.

## Using () as context

If your tools and hooks do not need shared state, use the unit type:

```rust
use std::sync::Arc;
use openai_agents::{Agent, Runner};
use openai_agents::models::openai_responses::OpenAIResponsesModel;

let agent = Agent::<()>::builder("simple")
    .instructions("Be helpful.")
    .build();

let model = Arc::new(OpenAIResponsesModel::new("gpt-4o-mini")?);

let result = Runner::run_with_model(
    &agent,
    "Hello!",
    (),  // <-- context
    model,
    None,
    None,
).await?;
```

## Custom context types

Define any struct as your context. Fields should be cheap to clone or wrapped
in `Arc` for shared ownership:

```rust
use std::sync::Arc;

struct AppContext {
    /// A database connection pool.
    db: Arc<DbPool>,
    /// The authenticated user's ID.
    user_id: String,
    /// A feature-flag client.
    flags: Arc<FeatureFlags>,
}
```

Pass an instance when calling the runner:

```rust
let ctx = AppContext {
    db: Arc::clone(&db_pool),
    user_id: "usr_42".to_string(),
    flags: Arc::clone(&flags),
};

let result = Runner::run_with_model(
    &agent, input, ctx, model, None, None,
).await?;
```

## Accessing context in tools

Tools receive a `ToolContext<C>`. The run context is behind a shared
`Arc<RwLock<RunContextWrapper<C>>>` so multiple tools can be awaited
concurrently without data races:

```rust
use std::sync::Arc;
use schemars::JsonSchema;
use serde::Deserialize;
use openai_agents::items::ToolOutput;
use openai_agents::tool::{ToolContext, function_tool};

#[derive(Deserialize, JsonSchema)]
struct LookupInput { id: u64 }

let lookup_tool = function_tool::<AppContext, LookupInput, _, _>(
    "lookup_user",
    "Look up a user record by ID.",
    |ctx: ToolContext<AppContext>, input: LookupInput| async move {
        // Acquire a read lock to borrow the context.
        let wrapper = ctx.context.read().await;
        let db = Arc::clone(&wrapper.context.db);
        drop(wrapper); // release the lock before the async DB call

        let record = db.find_user(input.id).await?;
        Ok(ToolOutput::Text(format!("{record:?}")))
    },
)?;
```

> **Note:** Release the `RwLock` guard before any `.await` point, or you will
> hold the lock across an await and potentially deadlock.

## Accessing context in dynamic instructions

Dynamic instruction closures also receive a `&RunContextWrapper<C>`:

```rust
use openai_agents::Agent;

struct UserContext { language: String }

let agent = Agent::<UserContext>::builder("multilingual")
    .dynamic_instructions(|ctx, _agent| {
        let lang = ctx.context.language.clone();
        Box::pin(async move {
            Ok(format!("You are a helpful assistant. Always respond in {lang}."))
        })
    })
    .build();
```

## Usage tracking via context

The `usage` field on `RunContextWrapper` accumulates token counts as the run
progresses. In hooks and tools you can read it to implement soft token budgets:

```rust
use openai_agents::lifecycle::RunHooks;
use openai_agents::context::RunContextWrapper;
use async_trait::async_trait;

struct BudgetHooks { max_tokens: u32 }

#[async_trait]
impl RunHooks<AppContext> for BudgetHooks {
    async fn on_tool_end(
        &self,
        ctx: &RunContextWrapper<AppContext>,
        _tool_name: &str,
        _output: &str,
    ) {
        if ctx.usage.total_tokens > self.max_tokens {
            println!("Warning: token budget exceeded ({} tokens)", ctx.usage.total_tokens);
        }
    }
}
```

The `Usage` struct exposes:

| Field | Description |
|-------|-------------|
| `requests` | Number of model calls made. |
| `input_tokens` | Total prompt tokens consumed. |
| `output_tokens` | Total completion tokens generated. |
| `total_tokens` | Sum of input and output tokens. |
| `input_tokens_details.cached_tokens` | Prompt tokens served from cache. |
| `output_tokens_details.reasoning_tokens` | Reasoning tokens (o-series models). |

After the run completes, the same accumulated usage is available on
`result.usage`.

## Context and thread safety

The context type `C` must satisfy `Send + Sync + 'static`. If you need interior
mutability (e.g., a counter), use `Arc<Mutex<T>>` or `Arc<AtomicUsize>`:

```rust
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

struct CountingContext {
    tool_call_count: Arc<AtomicUsize>,
}

// In a tool:
// ctx.context.read().await.context.tool_call_count.fetch_add(1, Ordering::Relaxed);
```
