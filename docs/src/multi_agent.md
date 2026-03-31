# Multi-Agent Workflows

The Rust SDK supports composing multiple agents into a single run. Agents communicate through
[handoffs](./handoffs.md): one agent transfers control to another by calling a handoff tool. The
runner automatically looks up the target agent and continues execution.

## `Runner::run_with_agents`

For single-agent runs, `Runner::run_with_model` is sufficient. When you have multiple agents that
may hand off to each other, use `Runner::run_with_agents` and pass a registry:

```rust
use std::collections::HashMap;
use std::sync::Arc;
use openai_agents::agent::Agent;
use openai_agents::runner::Runner;
use openai_agents::handoffs::Handoff;

// Build three agents.
let billing = Agent::<()>::builder("billing_agent")
    .instructions("Answer billing questions.")
    .build();

let support = Agent::<()>::builder("support_agent")
    .instructions("Handle technical support.")
    .build();

let triage = Agent::<()>::builder("triage_agent")
    .instructions("Route users to billing or support.")
    .handoff(Handoff::<()>::to_agent("billing_agent").build())
    .handoff(Handoff::<()>::to_agent("support_agent").build())
    .build();

// Build the agent registry.
let mut agents: HashMap<String, &Agent<()>> = HashMap::new();
agents.insert("billing_agent".to_string(), &billing);
agents.insert("support_agent".to_string(), &support);
// The starting agent does not need to be in the registry.

// Run starting from triage.
# async fn run(model: Arc<dyn openai_agents::models::Model>) {
# let billing = Agent::<()>::builder("billing_agent").build();
# let support = Agent::<()>::builder("support_agent").build();
# let triage = Agent::<()>::builder("triage_agent").build();
# let mut agents: HashMap<String, &Agent<()>> = HashMap::new();
# agents.insert("billing_agent".to_string(), &billing);
# agents.insert("support_agent".to_string(), &support);
let result = Runner::run_with_agents(
    &triage,
    &agents,
    "My invoice is wrong.",
    (),
    model,
    None,  // no lifecycle hooks
    None,  // default RunConfig
)
.await
.expect("run should succeed");

println!("{}", result.final_output);
# }
```

The `starting_agent` does not have to appear in the `agents` map. The registry is only consulted
when a handoff targets a named agent.

### Error behaviour

If a handoff names an agent that is not in the registry, `run_with_agents` returns
`AgentError::UserError` with a message identifying the missing agent name.

## The triage pattern

The triage pattern uses a lightweight front-line agent to classify the request and route it to the
right specialist. It keeps each specialist focused on a narrow domain.

```rust
use openai_agents::agent::Agent;
use openai_agents::handoffs::Handoff;

let refunds = Agent::<()>::builder("refunds_agent")
    .instructions("Process refund requests. Ask for the order number.")
    .build();

let shipping = Agent::<()>::builder("shipping_agent")
    .instructions("Track shipments and resolve delivery issues.")
    .build();

let accounts = Agent::<()>::builder("accounts_agent")
    .instructions("Help with account settings, passwords, and billing details.")
    .build();

let triage = Agent::<()>::builder("triage")
    .instructions(
        "Classify the user's request and transfer to the correct specialist. \
         Use transfer_to_refunds_agent for refund requests, \
         transfer_to_shipping_agent for delivery questions, and \
         transfer_to_accounts_agent for account issues.",
    )
    .handoff(Handoff::<()>::to_agent("refunds_agent").build())
    .handoff(Handoff::<()>::to_agent("shipping_agent").build())
    .handoff(Handoff::<()>::to_agent("accounts_agent").build())
    .build();
```

## The chain pattern

In the chain pattern, agents form a sequential pipeline. Each agent performs a transformation and
then hands off to the next one. This is useful for workflows like extract → validate → format.

```rust
use openai_agents::agent::Agent;
use openai_agents::handoffs::Handoff;

// Step 3: format the validated data.
let formatter = Agent::<()>::builder("formatter_agent")
    .instructions("Format the validated data as a JSON report.")
    .build();

// Step 2: validate the extracted data, then pass to formatter.
let validator = Agent::<()>::builder("validator_agent")
    .instructions(
        "Validate the extracted fields. Reject invalid entries. \
         When validation is complete, hand off to formatter_agent.",
    )
    .handoff(Handoff::<()>::to_agent("formatter_agent").build())
    .build();

// Step 1: extract data from raw input, then pass to validator.
let extractor = Agent::<()>::builder("extractor_agent")
    .instructions(
        "Extract name, email, and date from the user's message. \
         When done, hand off to validator_agent.",
    )
    .handoff(Handoff::<()>::to_agent("validator_agent").build())
    .build();
```

The chain starts at `extractor_agent`. The registry contains `validator_agent` and
`formatter_agent`, so the runner can resolve both handoffs.

## Lifecycle hooks across agents

The `RunHooks` trait fires events across all agents in a run. Use `on_handoff` to observe
transfers and `on_agent_start` / `on_agent_end` to measure per-agent latency:

```rust
use async_trait::async_trait;
use openai_agents::context::RunContextWrapper;
use openai_agents::lifecycle::RunHooks;
use openai_agents::items::{ModelResponse, ResponseInputItem};

struct AuditHooks;

#[async_trait]
impl RunHooks<()> for AuditHooks {
    async fn on_agent_start(&self, _ctx: &RunContextWrapper<()>, agent_name: &str) {
        println!("[START] {agent_name}");
    }

    async fn on_agent_end(
        &self,
        _ctx: &RunContextWrapper<()>,
        agent_name: &str,
        _output: &serde_json::Value,
    ) {
        println!("[END]   {agent_name}");
    }

    async fn on_handoff(
        &self,
        _ctx: &RunContextWrapper<()>,
        from_agent: &str,
        to_agent: &str,
    ) {
        println!("[HANDOFF] {from_agent} → {to_agent}");
    }
}
```

Pass the hooks to `run_with_agents`:

```rust,no_run
use std::sync::Arc;
use openai_agents::runner::Runner;

# async fn example(
#     triage: &openai_agents::agent::Agent<()>,
#     agents: &std::collections::HashMap<String, &openai_agents::agent::Agent<()>>,
#     model: Arc<dyn openai_agents::models::Model>,
# ) {
# struct AuditHooks;
# #[async_trait::async_trait]
# impl openai_agents::lifecycle::RunHooks<()> for AuditHooks {}
let result = Runner::run_with_agents(
    triage,
    agents,
    "I need a refund for order #12345.",
    (),
    model,
    Some(Arc::new(AuditHooks) as Arc<dyn openai_agents::lifecycle::RunHooks<()>>),
    None,
)
.await
.expect("run should succeed");
# }
```

## Practical example: three-agent customer service

The following example shows a complete three-agent setup. Triage routes to billing or support;
each specialist handles a narrow domain.

```rust,no_run
use std::collections::HashMap;
use std::sync::Arc;
use openai_agents::agent::Agent;
use openai_agents::handoffs::Handoff;
use openai_agents::runner::Runner;

#[tokio::main]
async fn main() {
    let model = /* your model */
#       todo!();

    // Specialist agents.
    let billing = Agent::<()>::builder("billing_agent")
        .instructions(
            "You handle billing questions. \
             Ask for the account number if needed.",
        )
        .build();

    let support = Agent::<()>::builder("support_agent")
        .instructions(
            "You handle technical support. \
             Collect error messages and device information.",
        )
        .build();

    // Triage agent with handoffs to both specialists.
    let triage = Agent::<()>::builder("triage_agent")
        .instructions(
            "Decide whether the user needs billing or technical support, \
             then transfer them. Do not answer questions yourself.",
        )
        .handoff(
            Handoff::<()>::to_agent("billing_agent")
                .tool_description(
                    "Transfer to billing for invoice, payment, or account charge questions.",
                )
                .build(),
        )
        .handoff(
            Handoff::<()>::to_agent("support_agent")
                .tool_description(
                    "Transfer to support for technical problems, errors, or how-to questions.",
                )
                .build(),
        )
        .build();

    // Registry for handoff resolution.
    let mut agents: HashMap<String, &Agent<()>> = HashMap::new();
    agents.insert("billing_agent".to_string(), &billing);
    agents.insert("support_agent".to_string(), &support);

    let result = Runner::run_with_agents(
        &triage,
        &agents,
        "My credit card was charged twice.",
        (),
        model,
        None,
        None,
    )
    .await
    .expect("run should succeed");

    println!("Final answer: {}", result.final_output);
}
```

## Turn limits and configuration

By default the runner allows up to 10 turns before returning
`AgentError::MaxTurnsExceeded`. Each agent transition counts as part of the same run
and shares the turn budget. Increase the limit via `RunConfig`:

```rust
use openai_agents::config::RunConfig;

let config = RunConfig {
    max_turns: 25,
    ..Default::default()
};
```

## See also

- [Handoffs](./handoffs.md) — building individual handoff objects.
- [Running Agents](./running_agents.md) — the full `Runner` API.
- [Agents](./agents.md) — `RunHooks` and `AgentHooks` traits.
- [Configuration](./config.md) — `RunConfig` options.
