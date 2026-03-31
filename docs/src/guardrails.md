# Guardrails

Guardrails are validation checks that run alongside or around agent execution. They let you
enforce policies — blocking off-topic inputs, detecting PII in outputs, or preventing dangerous
tool calls — without coupling the policy logic to the agent's instructions.

There are two categories:

- **Agent guardrails** (`InputGuardrail`, `OutputGuardrail`) — check the overall agent
  input or final output.
- **Tool guardrails** (`ToolInputGuardrail`, `ToolOutputGuardrail`) — check individual tool
  call arguments or results.

## Agent input guardrails

An `InputGuardrail` runs before or concurrently with the LLM call. If its function returns
`tripwire_triggered: true`, the runner aborts with `AgentError::InputGuardrailTripwire`.

```rust
use openai_agents::guardrail::{InputGuardrail, GuardrailFunctionOutput};
use serde_json::json;

let guardrail = InputGuardrail::<()>::new("topic_check", |_ctx, _agent_name, input| {
    let input = input.clone();
    Box::pin(async move {
        let text = match &input {
            openai_agents::items::InputContent::Text(t) => t.as_str(),
            _ => "",
        };

        if text.to_lowercase().contains("competitor") {
            // Trigger the tripwire to abort the run.
            Ok(GuardrailFunctionOutput::tripwire(
                json!({ "reason": "competitor mention" }),
            ))
        } else {
            Ok(GuardrailFunctionOutput::passed(json!(null)))
        }
    })
});
```

The guardrail function receives:

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `&RunContextWrapper<C>` | Access to the run's application context. |
| `agent_name` | `&str` | The name of the agent being guarded. |
| `input` | `&InputContent` | The user input for this run. |

### `GuardrailFunctionOutput` constructors

| Constructor | `tripwire_triggered` | Use case |
|---|---|---|
| `passed(info)` | `false` | The check passed; continue normally. |
| `tripwire(info)` | `true` | The check failed; abort the run. |
| `new(info, bool)` | user-supplied | Explicit control. |

The `output_info` field (`serde_json::Value`) is available on the error and can carry structured
metadata about what the guardrail detected.

### Parallel vs sequential execution

By default, input guardrails run **in parallel** with the LLM call to minimise latency. Use
`InputGuardrail::sequential` to make a guardrail run **before** the LLM starts — for example
when the guardrail must complete before the model is allowed to see the input:

```rust
use openai_agents::guardrail::{InputGuardrail, GuardrailFunctionOutput};
use serde_json::json;

// Runs before the LLM call.
let pre_check = InputGuardrail::<()>::sequential("auth_check", |_ctx, _agent, _input| {
    Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
});

// Runs in parallel with the LLM call (default).
let parallel_check = InputGuardrail::<()>::parallel("topic_check", |_ctx, _agent, _input| {
    Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
});
```

The `run_in_parallel` field is `true` for both `InputGuardrail::new` and
`InputGuardrail::parallel`, and `false` for `InputGuardrail::sequential`.

### Attaching input guardrails to an agent

```rust
use openai_agents::agent::Agent;
use openai_agents::guardrail::{InputGuardrail, GuardrailFunctionOutput};
use serde_json::json;

let guardrail = InputGuardrail::<()>::new("language_check", |_ctx, _agent, _input| {
    Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
});

let agent = Agent::<()>::builder("assistant")
    .instructions("You are a helpful assistant.")
    .input_guardrail(guardrail)
    .build();
```

### Handling `InputGuardrailTripwire`

```rust,no_run
use openai_agents::error::AgentError;
use openai_agents::runner::Runner;

match Runner::run_with_model(&agent, input, (), model, None, None).await {
    Ok(result) => println!("{}", result.final_output),
    Err(AgentError::InputGuardrailTripwire { guardrail_name }) => {
        println!("Guardrail '{guardrail_name}' blocked the request.");
    }
    Err(e) => eprintln!("Run failed: {e}"),
}
```

## Agent output guardrails

An `OutputGuardrail` runs after the agent produces its final output. It receives the output as a
`serde_json::Value`. If it returns `tripwire_triggered: true`, the runner aborts with
`AgentError::OutputGuardrailTripwire`.

```rust
use openai_agents::guardrail::{OutputGuardrail, GuardrailFunctionOutput};
use serde_json::json;

let pii_check = OutputGuardrail::<()>::new("pii_detector", |_ctx, _agent, output| {
    let output = output.clone();
    Box::pin(async move {
        let text = output.as_str().unwrap_or("");

        // Naive SSN pattern check.
        let has_ssn = text.contains("SSN") || text.contains("social security");
        if has_ssn {
            Ok(GuardrailFunctionOutput::tripwire(
                json!({ "reason": "potential PII in output" }),
            ))
        } else {
            Ok(GuardrailFunctionOutput::passed(json!(null)))
        }
    })
});
```

Attach to an agent:

```rust
use openai_agents::agent::Agent;
use openai_agents::guardrail::{OutputGuardrail, GuardrailFunctionOutput};
use serde_json::json;

let guardrail = OutputGuardrail::<()>::new("pii_check", |_ctx, _agent, _output| {
    Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
});

let agent = Agent::<()>::builder("assistant")
    .output_guardrail(guardrail)
    .build();
```

### `InputGuardrailResult` and `OutputGuardrailResult`

When a guardrail passes, `guardrail.run(...)` returns an `InputGuardrailResult` or
`OutputGuardrailResult` containing the guardrail name, the agent name (output only), the checked
data, and the `GuardrailFunctionOutput`. These results are also stored on `RunState` for
inspection after the run.

## Tool guardrails

Tool guardrails operate at the individual tool-call level, providing a richer three-way behavior
model compared to the binary pass/fail of agent guardrails.

### `GuardrailBehavior`

```rust
use openai_agents::tool_guardrails::GuardrailBehavior;

// Allow the tool call to proceed.
let allow = GuardrailBehavior::Allow;

// Reject the call but continue the run — the model receives the message instead.
let reject = GuardrailBehavior::RejectContent {
    message: "That query is not allowed.".to_string(),
};

// Halt execution immediately with an error.
let halt = GuardrailBehavior::RaiseException;
```

`GuardrailBehavior::Allow` is the default.

### `ToolInputGuardrail`

A `ToolInputGuardrail` runs before a tool is invoked. It receives the tool name, agent name, and
the raw JSON arguments string.

```rust
use openai_agents::tool_guardrails::{ToolInputGuardrail, ToolGuardrailFunctionOutput};
use serde_json::json;

let sql_guard = ToolInputGuardrail::<()>::new("sql_injection", |_ctx, _agent, tool_name, args| {
    let tool = tool_name.to_owned();
    let args = args.to_owned();
    Box::pin(async move {
        if tool == "run_query" && args.contains("DROP") {
            Ok(ToolGuardrailFunctionOutput::reject_content(
                "Destructive SQL statements are not permitted.",
                json!({ "detected": "DROP statement" }),
            ))
        } else {
            Ok(ToolGuardrailFunctionOutput::allow(json!(null)))
        }
    })
});
```

`ToolGuardrailFunctionOutput` constructors:

| Constructor | Behavior | Description |
|---|---|---|
| `allow(info)` | `Allow` | Tool proceeds normally. |
| `reject_content(msg, info)` | `RejectContent` | Call is skipped; model receives `msg`. |
| `raise_exception(info)` | `RaiseException` | Run is aborted with `ToolInputGuardrailTripwire`. |

### `ToolOutputGuardrail`

A `ToolOutputGuardrail` runs after a tool returns. It receives the tool's output as a `&str`.

```rust
use openai_agents::tool_guardrails::{ToolOutputGuardrail, ToolGuardrailFunctionOutput};
use serde_json::json;

let output_filter = ToolOutputGuardrail::<()>::new("pii_filter", |_ctx, _agent, _tool, output| {
    let output = output.to_owned();
    Box::pin(async move {
        // Redact if output contains an SSN pattern (simplified).
        if output.contains("SSN") {
            Ok(ToolGuardrailFunctionOutput::reject_content(
                "[PII redacted]",
                json!({ "reason": "SSN detected in tool output" }),
            ))
        } else {
            Ok(ToolGuardrailFunctionOutput::allow(json!(null)))
        }
    })
});
```

### Attaching tool guardrails to a tool

Tool guardrails are attached to individual `FunctionTool` instances, not to the agent directly:

```rust,no_run
use openai_agents::tool::FunctionTool;
use openai_agents::tool_guardrails::{ToolInputGuardrail, ToolGuardrailFunctionOutput};
use serde_json::json;

let guard = ToolInputGuardrail::<()>::new("rate_limit", |_ctx, _agent, _tool, _args| {
    Box::pin(async { Ok(ToolGuardrailFunctionOutput::allow(json!(null))) })
});

// Attach the guardrail when building the tool.
// (See the Tools documentation for the full FunctionTool API.)
```

### Handling tool guardrail errors

```rust,no_run
use openai_agents::error::AgentError;

match result {
    Err(AgentError::ToolInputGuardrailTripwire { guardrail_name, tool_name }) => {
        println!("Guardrail '{guardrail_name}' blocked tool '{tool_name}'.");
    }
    Err(AgentError::ToolOutputGuardrailTripwire { guardrail_name, tool_name }) => {
        println!("Output guardrail '{guardrail_name}' triggered for tool '{tool_name}'.");
    }
    _ => {}
}
```

## Running multiple guardrails

Multiple guardrails can be registered on the same agent. Parallel input guardrails all start
concurrently. If any single guardrail triggers its tripwire, the run aborts immediately — the
remaining guardrails may still be in flight but their results are discarded.

```rust
use openai_agents::agent::Agent;
use openai_agents::guardrail::{InputGuardrail, OutputGuardrail, GuardrailFunctionOutput};
use serde_json::json;

fn make_pass_guard(name: &str) -> InputGuardrail<()> {
    let name = name.to_string();
    InputGuardrail::new(name, |_ctx, _agent, _input| {
        Box::pin(async { Ok(GuardrailFunctionOutput::passed(json!(null))) })
    })
}

let agent = Agent::<()>::builder("assistant")
    .input_guardrail(make_pass_guard("length_check"))
    .input_guardrail(make_pass_guard("topic_check"))
    .input_guardrail(make_pass_guard("language_check"))
    .build();
```

## See also

- [Human in the Loop](./human_in_the_loop.md) — pausing runs for manual tool approval.
- [Tools](./tools.md) — attaching tool guardrails to `FunctionTool`.
- [Running Agents](./running_agents.md) — how the runner invokes guardrails.
