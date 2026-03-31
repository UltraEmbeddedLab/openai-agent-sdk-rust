# Handoffs

A handoff lets one agent transfer control to another agent mid-run. From the LLM's perspective a
handoff is just a tool call — when the model invokes the handoff tool, the runner switches
execution to the target agent, carrying the conversation history with it.

Handoffs are the building block for [multi-agent workflows](./multi_agent.md). They allow you to
build systems where a general-purpose triage agent routes requests to specialists without the
caller needing to know which agent will ultimately handle the request.

## Creating a handoff

Use `Handoff::to_agent` to obtain a builder, then call `build()`:

```rust
use openai_agents::handoffs::Handoff;

let handoff: Handoff<()> = Handoff::to_agent("billing_agent").build();

assert_eq!(handoff.tool_name, "transfer_to_billing_agent");
assert_eq!(
    handoff.tool_description,
    "Handoff to the billing_agent agent to handle the request."
);
```

The default tool name is `transfer_to_{agent_name}` (lowercased, spaces replaced with
underscores). The default description is `"Handoff to the {agent_name} agent to handle the
request."`.

## Custom tool names and descriptions

Override either field to give the LLM more precise guidance:

```rust
use openai_agents::handoffs::Handoff;

let handoff: Handoff<()> = Handoff::to_agent("billing_agent")
    .tool_name("escalate_to_billing")
    .tool_description(
        "Use this when the user has a billing question, \
         payment dispute, or invoice request.",
    )
    .build();
```

## Structured handoff arguments with `input_type`

By default a handoff tool has no parameters — the LLM calls it without any arguments. You can
add a structured payload by calling `.input_type::<T>()`, where `T` implements
`schemars::JsonSchema`. The generated JSON schema is sent to the model as the tool's `parameters`.

```rust
use openai_agents::handoffs::Handoff;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct EscalationArgs {
    /// The reason for escalating to the billing agent.
    reason: String,
    /// Priority level: 1 (low) to 3 (high).
    priority: u8,
}

let handoff: Handoff<()> = Handoff::to_agent("billing_agent")
    .input_type::<EscalationArgs>()
    .build();
```

When the model calls this tool, the runner passes the raw JSON argument string to the `on_invoke`
callback (see below).

You can also supply a raw JSON schema directly:

```rust
use openai_agents::handoffs::Handoff;
use serde_json::json;

let handoff: Handoff<()> = Handoff::to_agent("billing_agent")
    .input_json_schema(json!({
        "type": "object",
        "properties": {
            "reason": { "type": "string" }
        },
        "required": ["reason"]
    }))
    .build();
```

### Strict schema enforcement

`strict_json_schema` defaults to `true`, which adds `additionalProperties: false` and marks all
fields as required. Disable it only when your schema is incompatible with strict mode:

```rust
use openai_agents::handoffs::Handoff;

let handoff: Handoff<()> = Handoff::to_agent("billing_agent")
    .strict_json_schema(false)
    .build();
```

## Custom `on_invoke` callback

The default `on_invoke` implementation simply returns the agent name unchanged. Override it to
implement dynamic routing or to inspect the handoff arguments before routing:

```rust
use openai_agents::handoffs::Handoff;
use serde::Deserialize;

#[derive(Deserialize)]
struct HandoffArgs {
    reason: String,
}

let handoff: Handoff<()> = Handoff::to_agent("billing_agent")
    .on_invoke(|_ctx, args_json| {
        Box::pin(async move {
            if let Ok(args) = serde_json::from_str::<HandoffArgs>(&args_json) {
                println!("Handoff reason: {}", args.reason);
            }
            // Return the name of the agent to activate.
            Ok("billing_agent".to_string())
        })
    })
    .build();
```

The callback signature is `Fn(&RunContextWrapper<C>, String) -> BoxFuture<Result<String>>`. The
`String` argument is the raw JSON args; the return value is the target agent name.

## Input filters

An input filter transforms `HandoffInputData` before the receiving agent sees it. This is useful
for truncating long histories, summarising previous turns, or stripping sensitive content.

```rust
use std::sync::Arc;
use openai_agents::handoffs::{Handoff, HandoffInputData};
use openai_agents::items::InputContent;

let filter = Arc::new(|data: HandoffInputData| {
    Box::pin(async move {
        // Give the new agent a clean slate — no prior history.
        HandoffInputData {
            input_history: InputContent::Items(vec![]),
            pre_handoff_items: vec![],
            new_items: data.new_items,
        }
    })
}) as openai_agents::handoffs::HandoffInputFilter;

let handoff: Handoff<()> = Handoff::to_agent("fresh_start_agent")
    .input_filter(filter)
    .build();
```

`HandoffInputData` contains three fields:

| Field | Description |
|---|---|
| `input_history` | The full conversation history up to this point. |
| `pre_handoff_items` | Items generated in previous turns, before the current turn. |
| `new_items` | Items generated in the current turn, including the handoff call itself. |

## Handoff history management

When multiple handoffs occur in a chain, each receiving agent would otherwise see the full raw
transcript from every previous agent. The `handoffs::history` module provides utilities to
summarise this transcript into a single compact assistant message, reducing token usage.

### `nest_handoff_history`

`nest_handoff_history` normalises the history, flattens any previously nested summaries, and
applies a mapper to produce a compact representation. Use it directly inside an input filter:

```rust
use std::sync::Arc;
use openai_agents::handoffs::{Handoff, HandoffInputData, nest_handoff_history};

let filter = Arc::new(|data: HandoffInputData| {
    Box::pin(async move { nest_handoff_history(&data, None) })
}) as openai_agents::handoffs::HandoffInputFilter;

let handoff: Handoff<()> = Handoff::to_agent("specialist_agent")
    .input_filter(filter)
    .build();
```

The default mapper produces an assistant message like:

```
For context, here is the conversation so far between the user and the previous agent:
<CONVERSATION HISTORY>
1. user: What is my balance?
2. assistant (triage): Let me transfer you to billing.
</CONVERSATION HISTORY>
```

### Custom history mappers

Supply a `HandoffHistoryMapper` to replace the default summarisation logic:

```rust
use openai_agents::handoffs::{
    HandoffInputData, HandoffHistoryMapper, nest_handoff_history,
};

let mapper: HandoffHistoryMapper = Box::new(|transcript| {
    vec![serde_json::json!({
        "role": "developer",
        "content": format!("Context: {} prior turns.", transcript.len()),
    })]
});

let filtered = nest_handoff_history(&handoff_input_data, Some(&mapper));
```

### Customising the history markers

The `<CONVERSATION HISTORY>` / `</CONVERSATION HISTORY>` delimiters are configurable at the
process level:

```rust
use openai_agents::handoffs::{
    set_conversation_history_wrappers,
    reset_conversation_history_wrappers,
    get_conversation_history_wrappers,
};

// Set custom markers.
set_conversation_history_wrappers(Some("[HISTORY START]"), Some("[HISTORY END]"));

let (start, end) = get_conversation_history_wrappers();
assert_eq!(start, "[HISTORY START]");
assert_eq!(end, "[HISTORY END]");

// Restore the defaults when done.
reset_conversation_history_wrappers();
```

Pass `None` for either argument to leave that marker unchanged.

## Disabling a handoff at runtime

Set `is_enabled(false)` to hide the handoff tool from the model. This is useful for feature
flags or conditional routing:

```rust
use openai_agents::handoffs::Handoff;

let handoff: Handoff<()> = Handoff::to_agent("premium_agent")
    .is_enabled(false)
    .build();

assert!(!handoff.is_enabled);
```

## Attaching handoffs to an agent

Register handoffs on an `Agent` via the builder:

```rust
use openai_agents::agent::Agent;
use openai_agents::handoffs::Handoff;

let billing_handoff = Handoff::<()>::to_agent("billing_agent").build();
let support_handoff = Handoff::<()>::to_agent("support_agent").build();

let triage = Agent::<()>::builder("triage")
    .instructions("Route the user to the right specialist.")
    .handoff(billing_handoff)
    .handoff(support_handoff)
    .build();
```

## See also

- [Multi-Agent Workflows](./multi_agent.md) — orchestrating agents with `Runner::run_with_agents`.
- [Running Agents](./running_agents.md) — the `Runner` API.
- [Agents](./agents.md) — lifecycle hooks fired on handoff events.
