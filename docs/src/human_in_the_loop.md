# Human in the Loop

Human-in-the-loop (HITL) workflows allow you to pause a run, persist its state, review pending
tool calls, and then resume execution — potentially from a different process or after an
arbitrary delay. This is useful for approving high-stakes actions, auditing LLM decisions, or
building asynchronous task queues.

The primary building block is `RunState`: a fully serializable snapshot of an agent run.

## `RunState`

`RunState` captures everything needed to resume a run:

| Field | Type | Description |
|---|---|---|
| `agent_name` | `String` | The currently active agent. |
| `input` | `InputContent` | The original input to the run. |
| `new_items` | `Vec<RunItem>` | Items generated so far. |
| `raw_responses` | `Vec<ModelResponse>` | Raw model responses. |
| `usage` | `Usage` | Accumulated token usage. |
| `turn` | `u32` | Current turn number. |
| `next_step` | `NextStep` | What the runner should do next. |
| `input_guardrail_results` | `Vec<InputGuardrailResult>` | Guardrail results collected. |
| `output_guardrail_results` | `Vec<OutputGuardrailResult>` | Output guardrail results. |

Create a fresh state for a new run:

```rust
use openai_agents::run_state::RunState;
use openai_agents::items::InputContent;

let state = RunState::new("my_agent", InputContent::Text("Hello".to_owned()));
```

## `NextStep`

The `NextStep` enum encodes what happens when a run resumes:

```rust
use openai_agents::run_state::{NextStep, PendingToolCall};

// The run should continue with another LLM turn.
let step = NextStep::ContinueLoop;

// The run is waiting for tool calls to be approved.
let step = NextStep::ExecuteTools(vec![
    PendingToolCall::new("call_abc", "delete_record", r#"{"id": 42}"#),
]);

// The run produced a final answer.
let step = NextStep::FinalOutput(serde_json::json!("Done."));
```

## `PendingToolCall`

Each tool call awaiting approval is represented by `PendingToolCall`:

```rust
use openai_agents::run_state::PendingToolCall;

let call = PendingToolCall::new(
    "call_abc",              // call_id from the model response
    "delete_record",         // tool_name
    r#"{"id": 42}"#,         // raw JSON arguments
);

assert_eq!(call.call_id, "call_abc");
assert_eq!(call.tool_name, "delete_record");
assert!(!call.approved);
assert!(call.rejection_message.is_none());
```

## Serialization: `to_json` / `from_json`

`RunState` implements full serde serialisation. Persist the state with `to_json` and restore it
later with `from_json`. The schema version is validated on deserialisation.

```rust
use openai_agents::run_state::RunState;
use openai_agents::items::InputContent;

let state = RunState::new("my_agent", InputContent::Text("Process order #42".to_owned()));

// Serialize to a pretty-printed JSON string.
let json = state.to_json().expect("serialization should succeed");

// Store `json` in a database, file, or message queue...

// Later, in another process or after a restart:
let restored = RunState::from_json(&json).expect("deserialization should succeed");
assert_eq!(restored.agent_name, "my_agent");
```

`from_json` returns `AgentError::UserError` if the `schema_version` field in the JSON does not
match the library's `CURRENT_SCHEMA_VERSION` constant, preventing stale state from being used
after an upgrade.

## Tool approval workflow

The typical HITL approval flow follows these steps:

1. The runner pauses with `NextStep::ExecuteTools(pending_calls)`.
2. Your application serialises `RunState` and stores it.
3. A human (or automated policy) reviews each `PendingToolCall`.
4. For each call, the application calls `state.approve_tool(call_id)` or
   `state.reject_tool(call_id, reason)`.
5. The application resumes the run by passing the updated state back to the runner.

### `approve_tool` and `reject_tool`

Both methods mutate the `RunState` in place and return `Result<()>`:

```rust
use openai_agents::run_state::{RunState, NextStep, PendingToolCall};
use openai_agents::items::InputContent;

let mut state = RunState::new("agent", InputContent::Text("delete all logs".to_owned()));
state.next_step = NextStep::ExecuteTools(vec![
    PendingToolCall::new("call_1", "delete_logs", r#"{"scope": "all"}"#),
    PendingToolCall::new("call_2", "notify_admin", r#"{"msg": "done"}"#),
]);

// Approve the notification but reject the deletion.
state.reject_tool("call_1", "Deletion requires director approval").expect("ok");
state.approve_tool("call_2").expect("ok");

if let NextStep::ExecuteTools(ref calls) = state.next_step {
    assert!(!calls[0].approved);
    assert_eq!(calls[0].rejection_message.as_deref(), Some("Deletion requires director approval"));
    assert!(calls[1].approved);
}
```

Both methods return `AgentError::UserError` if:
- `next_step` is not `NextStep::ExecuteTools`, or
- the given `call_id` is not found in the pending calls list.

## Full example: tool approval flow

The following example demonstrates the complete pause-review-resume cycle using `RunState` as an
in-memory checkpoint. In production, replace the in-process channel with a database or queue.

```rust,no_run
use openai_agents::run_state::{RunState, NextStep, PendingToolCall};
use openai_agents::items::InputContent;

/// Simulate a human approval gate.
fn human_approval(calls: &[PendingToolCall]) -> Vec<(&str, bool)> {
    calls
        .iter()
        .map(|call| {
            // In production, display the call details to a human reviewer.
            println!(
                "Approve tool '{}' with args {}? (auto-approving notify, rejecting delete)",
                call.tool_name, call.arguments
            );
            let approved = call.tool_name == "notify_admin";
            (call.call_id.as_str(), approved)
        })
        .collect()
}

fn process_state(state_json: &str) -> String {
    let mut state = RunState::from_json(state_json).expect("valid state");

    if let NextStep::ExecuteTools(ref calls) = state.next_step.clone() {
        for (call_id, approved) in human_approval(calls) {
            if approved {
                state.approve_tool(call_id).expect("approve ok");
            } else {
                state.reject_tool(call_id, "Rejected by reviewer").expect("reject ok");
            }
        }
    }

    state.to_json().expect("serialize ok")
}
```

## `to_input_list`

`RunState::to_input_list` converts the state into the flat input list the runner needs for the
next model call. You rarely need this directly — the runner calls it internally — but it is useful
for debugging:

```rust
use openai_agents::run_state::RunState;
use openai_agents::items::InputContent;

let state = RunState::new("agent", InputContent::Text("hello".to_owned()));
let items = state.to_input_list();

// Items contains the original input converted to a ResponseInputItem.
assert_eq!(items[0]["role"], "user");
assert_eq!(items[0]["content"], "hello");
```

## See also

- [Guardrails](./guardrails.md) — blocking requests at the input/output level.
- [Running Agents](./running_agents.md) — `Runner::run_with_model` and `RunConfig`.
- [Results](./results.md) — `RunResult` and its fields.
