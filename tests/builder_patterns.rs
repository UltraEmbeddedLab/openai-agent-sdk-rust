//! Integration tests for the builder APIs: Agent, `RunConfig`, and Handoff.
//!
//! Validates that all builder methods produce correct values and that defaults
//! match expectations. Tests do not require a model — they only exercise the
//! construction side of the public API.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use openai_agents::agent::{Agent, Instructions, OutputSchema, ToolUseBehavior};
use openai_agents::config::{
    DEFAULT_MAX_TURNS, ModelRef, ModelSettings, RunConfig, ToolChoice, Truncation,
};
use openai_agents::context::RunContextWrapper;
use openai_agents::guardrail::{GuardrailFunctionOutput, InputGuardrail, OutputGuardrail};
use openai_agents::handoffs::Handoff;
use openai_agents::items::ToolOutput;
use openai_agents::lifecycle::AgentHooks;
use openai_agents::tool::{
    CodeInterpreterTool, FileSearchTool, Tool, ToolContext, WebSearchTool, function_tool,
};

// ---------------------------------------------------------------------------
// Section 1: Agent builder
// ---------------------------------------------------------------------------

#[test]
fn test_agent_builder_minimal_defaults() {
    let agent = Agent::<()>::builder("minimal").build();

    assert_eq!(agent.name, "minimal");
    assert!(
        agent.instructions.is_none(),
        "instructions should default to None"
    );
    assert!(agent.handoff_description.is_none());
    assert!(agent.model.is_none(), "model should default to None");
    assert!(agent.tools.is_empty(), "tools should default to empty");
    assert!(
        agent.handoffs.is_empty(),
        "handoffs should default to empty"
    );
    assert!(agent.input_guardrails.is_empty());
    assert!(agent.output_guardrails.is_empty());
    assert!(agent.output_type.is_none());
    assert!(agent.hooks.is_none());
    assert_eq!(
        agent.tool_use_behavior,
        ToolUseBehavior::RunLlmAgain,
        "default tool use behavior should be RunLlmAgain"
    );
    assert!(
        agent.reset_tool_choice,
        "reset_tool_choice should default to true"
    );
}

#[test]
fn test_agent_builder_with_static_instructions() {
    let agent = Agent::<()>::builder("bot")
        .instructions("You are a helpful assistant.")
        .build();

    assert!(
        matches!(&agent.instructions, Some(Instructions::Static(s)) if s == "You are a helpful assistant."),
        "instructions should be Static"
    );
}

#[tokio::test]
async fn test_agent_builder_with_dynamic_instructions() {
    let agent = Agent::<String>::builder("dynamic-bot")
        .dynamic_instructions(|ctx, _agent| {
            let lang = ctx.context.clone();
            Box::pin(async move { Ok(format!("Respond in {lang}.")) })
        })
        .build();

    let ctx = RunContextWrapper::new("English".to_owned());
    let result = agent.get_instructions(&ctx).await.unwrap();
    assert_eq!(result, Some("Respond in English.".to_owned()));
}

#[test]
fn test_agent_builder_model_override() {
    let agent = Agent::<()>::builder("bot").model("gpt-4o").build();

    assert!(
        matches!(&agent.model, Some(ModelRef::Name(n)) if n == "gpt-4o"),
        "model should be set to gpt-4o"
    );
}

#[test]
fn test_agent_builder_model_settings() {
    let settings = ModelSettings::new()
        .with_temperature(0.3)
        .with_top_p(0.85)
        .with_max_tokens(512);

    let agent = Agent::<()>::builder("bot").model_settings(settings).build();

    assert_eq!(agent.model_settings.temperature, Some(0.3));
    assert_eq!(agent.model_settings.top_p, Some(0.85));
    assert_eq!(agent.model_settings.max_tokens, Some(512));
}

#[test]
fn test_agent_builder_single_tool() {
    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct P {
        x: i32,
    }

    let tool =
        function_tool::<(), P, _, _>("my_fn", "desc", |_ctx: ToolContext<()>, _p: P| async move {
            Ok(ToolOutput::Text("ok".to_owned()))
        })
        .expect("tool creation");

    let agent = Agent::<()>::builder("bot")
        .tool(Tool::Function(tool))
        .build();

    assert_eq!(agent.tools.len(), 1);
    assert_eq!(agent.tools[0].name(), "my_fn");
}

#[test]
fn test_agent_builder_multiple_tools_append() {
    let t1: Tool<()> = Tool::WebSearch(WebSearchTool::default());
    let t2: Tool<()> = Tool::FileSearch(FileSearchTool::default());
    let t3: Tool<()> = Tool::CodeInterpreter(CodeInterpreterTool::default());

    let agent = Agent::<()>::builder("multi-tool")
        .tool(t1)
        .tool(t2)
        .tool(t3)
        .build();

    assert_eq!(agent.tools.len(), 3);
}

#[test]
fn test_agent_builder_tools_replaces_previous() {
    let t1: Tool<()> = Tool::WebSearch(WebSearchTool::default());
    let t2: Tool<()> = Tool::FileSearch(FileSearchTool::default());

    let agent = Agent::<()>::builder("bot")
        .tool(t1)
        // Replace all tools with just t2.
        .tools(vec![t2])
        .build();

    assert_eq!(agent.tools.len(), 1);
    assert_eq!(agent.tools[0].name(), "file_search");
}

#[test]
fn test_agent_builder_single_handoff() {
    let handoff: Handoff<()> = Handoff::to_agent("billing").build();

    let agent = Agent::<()>::builder("bot").handoff(handoff).build();

    assert_eq!(agent.handoffs.len(), 1);
    assert_eq!(agent.handoffs[0].agent_name, "billing");
}

#[test]
fn test_agent_builder_multiple_handoffs() {
    let h1: Handoff<()> = Handoff::to_agent("billing").build();
    let h2: Handoff<()> = Handoff::to_agent("support").build();
    let h3: Handoff<()> = Handoff::to_agent("triage").build();

    let agent = Agent::<()>::builder("router")
        .handoff(h1)
        .handoff(h2)
        .handoff(h3)
        .build();

    assert_eq!(agent.handoffs.len(), 3);
    let names: Vec<&str> = agent
        .handoffs
        .iter()
        .map(|h| h.agent_name.as_str())
        .collect();
    assert!(names.contains(&"billing"));
    assert!(names.contains(&"support"));
    assert!(names.contains(&"triage"));
}

#[test]
fn test_agent_builder_handoffs_replaces_previous() {
    let h1: Handoff<()> = Handoff::to_agent("a").build();
    let h2: Handoff<()> = Handoff::to_agent("b").build();

    let agent = Agent::<()>::builder("bot")
        .handoff(h1)
        .handoffs(vec![h2])
        .build();

    assert_eq!(agent.handoffs.len(), 1);
    assert_eq!(agent.handoffs[0].agent_name, "b");
}

#[test]
fn test_agent_builder_input_guardrail() {
    let g = InputGuardrail::<()>::new("check", |_ctx, _a, _i| {
        Box::pin(async { Ok(GuardrailFunctionOutput::passed(serde_json::json!(null))) })
    });

    let agent = Agent::<()>::builder("bot").input_guardrail(g).build();

    assert_eq!(agent.input_guardrails.len(), 1);
    assert_eq!(agent.input_guardrails[0].name, "check");
}

#[test]
fn test_agent_builder_output_guardrail() {
    let g = OutputGuardrail::<()>::new("pii_check", |_ctx, _a, _o| {
        Box::pin(async { Ok(GuardrailFunctionOutput::passed(serde_json::json!(null))) })
    });

    let agent = Agent::<()>::builder("bot").output_guardrail(g).build();

    assert_eq!(agent.output_guardrails.len(), 1);
    assert_eq!(agent.output_guardrails[0].name, "pii_check");
}

#[test]
fn test_agent_builder_output_type_generates_strict_schema() {
    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct Answer {
        text: String,
        confidence: f64,
    }

    let agent = Agent::<()>::builder("structured")
        .output_type::<Answer>()
        .build();

    let schema = agent.output_type.expect("should have output type");
    assert!(schema.strict, "strict should be true");
    assert_eq!(
        schema.json_schema["additionalProperties"], false,
        "schema should forbid additional properties"
    );

    let props = schema.json_schema["properties"]
        .as_object()
        .expect("schema should have properties");
    assert!(props.contains_key("text"));
    assert!(props.contains_key("confidence"));
}

#[test]
fn test_agent_builder_output_schema_direct() {
    let schema = OutputSchema::new(serde_json::json!({"type": "string"}), false);

    let agent = Agent::<()>::builder("raw-schema")
        .output_schema(schema)
        .build();

    let stored = agent.output_type.expect("should have output type");
    assert!(!stored.strict);
    assert_eq!(stored.json_schema["type"], "string");
}

#[test]
fn test_agent_builder_hooks() {
    struct NoOpHooks;
    #[async_trait]
    impl AgentHooks<()> for NoOpHooks {}

    let agent = Agent::<()>::builder("with-hooks").hooks(NoOpHooks).build();

    assert!(agent.hooks.is_some(), "agent should have hooks");
}

#[test]
fn test_agent_builder_tool_use_behavior_stop_on_first_tool() {
    let agent = Agent::<()>::builder("bot")
        .tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
        .build();

    assert_eq!(agent.tool_use_behavior, ToolUseBehavior::StopOnFirstTool);
}

#[test]
fn test_agent_builder_tool_use_behavior_stop_at_tools() {
    let agent = Agent::<()>::builder("bot")
        .tool_use_behavior(ToolUseBehavior::StopAtTools(vec![
            "search".to_owned(),
            "calculator".to_owned(),
        ]))
        .build();

    match &agent.tool_use_behavior {
        ToolUseBehavior::StopAtTools(tools) => {
            assert_eq!(tools.len(), 2);
            assert!(tools.contains(&"search".to_owned()));
            assert!(tools.contains(&"calculator".to_owned()));
        }
        other => panic!("expected StopAtTools, got: {other:?}"),
    }
}

#[test]
fn test_agent_builder_reset_tool_choice_false() {
    let agent = Agent::<()>::builder("bot").reset_tool_choice(false).build();

    assert!(!agent.reset_tool_choice);
}

#[test]
fn test_agent_builder_handoff_description() {
    let agent = Agent::<()>::builder("specialist")
        .handoff_description("Use this agent for billing questions.")
        .build();

    assert_eq!(
        agent.handoff_description.as_deref(),
        Some("Use this agent for billing questions.")
    );
}

#[test]
fn test_agent_builder_name_override() {
    let agent = Agent::<()>::builder("original-name")
        .name("overridden-name")
        .build();

    assert_eq!(agent.name, "overridden-name");
}

#[test]
fn test_agent_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Agent<()>>();
    assert_send_sync::<Agent<String>>();
}

// ---------------------------------------------------------------------------
// Section 2: RunConfig builder
// ---------------------------------------------------------------------------

#[test]
fn test_run_config_builder_defaults() {
    let config = RunConfig::builder().build();

    assert!(config.model.is_none(), "model should default to None");
    assert!(
        config.model_settings.is_none(),
        "model_settings should default to None"
    );
    assert_eq!(
        config.max_turns, DEFAULT_MAX_TURNS,
        "max_turns should use default"
    );
    assert!(
        !config.tracing_disabled,
        "tracing should be enabled by default"
    );
    assert_eq!(
        config.workflow_name, "agent_workflow",
        "workflow name should use default"
    );
    assert!(config.trace_id.is_none());
    assert!(config.group_id.is_none());
}

#[test]
fn test_run_config_default_impl_matches_builder_default() {
    let from_default = RunConfig::default();
    let from_builder = RunConfig::builder().build();

    assert_eq!(from_default.max_turns, from_builder.max_turns);
    assert_eq!(from_default.tracing_disabled, from_builder.tracing_disabled);
    assert_eq!(from_default.workflow_name, from_builder.workflow_name);
}

#[test]
fn test_run_config_builder_model() {
    let config = RunConfig::builder().model("gpt-4o-mini").build();

    assert!(
        matches!(&config.model, Some(ModelRef::Name(n)) if n == "gpt-4o-mini"),
        "model should be set"
    );
}

#[test]
fn test_run_config_builder_model_settings() {
    let settings = ModelSettings::new()
        .with_temperature(0.9)
        .with_max_tokens(1024);

    let config = RunConfig::builder().model_settings(settings).build();

    let stored = config.model_settings.expect("should have model settings");
    assert_eq!(stored.temperature, Some(0.9));
    assert_eq!(stored.max_tokens, Some(1024));
}

#[test]
fn test_run_config_builder_max_turns() {
    let config = RunConfig::builder().max_turns(3).build();
    assert_eq!(config.max_turns, 3);
}

#[test]
fn test_run_config_builder_max_turns_one() {
    let config = RunConfig::builder().max_turns(1).build();
    assert_eq!(config.max_turns, 1);
}

#[test]
fn test_run_config_builder_tracing_disabled_true() {
    let config = RunConfig::builder().tracing_disabled(true).build();
    assert!(config.tracing_disabled);
}

#[test]
fn test_run_config_builder_tracing_disabled_false() {
    let config = RunConfig::builder().tracing_disabled(false).build();
    assert!(!config.tracing_disabled);
}

#[test]
fn test_run_config_builder_workflow_name() {
    let config = RunConfig::builder()
        .workflow_name("my_custom_workflow")
        .build();
    assert_eq!(config.workflow_name, "my_custom_workflow");
}

#[test]
fn test_run_config_builder_trace_id() {
    let config = RunConfig::builder().trace_id("trace-abc-123").build();
    assert_eq!(config.trace_id.as_deref(), Some("trace-abc-123"));
}

#[test]
fn test_run_config_builder_group_id() {
    let config = RunConfig::builder().group_id("session-group-456").build();
    assert_eq!(config.group_id.as_deref(), Some("session-group-456"));
}

#[test]
fn test_run_config_builder_all_fields() {
    let config = RunConfig::builder()
        .model("gpt-4o")
        .model_settings(
            ModelSettings::new()
                .with_temperature(0.5)
                .with_tool_choice(ToolChoice::Auto),
        )
        .max_turns(7)
        .tracing_disabled(true)
        .workflow_name("full_test")
        .trace_id("t-123")
        .group_id("g-456")
        .build();

    assert!(matches!(&config.model, Some(ModelRef::Name(n)) if n == "gpt-4o"));
    let settings = config.model_settings.unwrap();
    assert_eq!(settings.temperature, Some(0.5));
    assert_eq!(settings.tool_choice, Some(ToolChoice::Auto));
    assert_eq!(config.max_turns, 7);
    assert!(config.tracing_disabled);
    assert_eq!(config.workflow_name, "full_test");
    assert_eq!(config.trace_id.as_deref(), Some("t-123"));
    assert_eq!(config.group_id.as_deref(), Some("g-456"));
}

// ---------------------------------------------------------------------------
// Section 3: ModelSettings builder / field access
// ---------------------------------------------------------------------------

#[test]
fn test_model_settings_all_fields_none_by_default() {
    let s = ModelSettings::new();
    assert!(s.temperature.is_none());
    assert!(s.top_p.is_none());
    assert!(s.frequency_penalty.is_none());
    assert!(s.presence_penalty.is_none());
    assert!(s.tool_choice.is_none());
    assert!(s.parallel_tool_calls.is_none());
    assert!(s.truncation.is_none());
    assert!(s.max_tokens.is_none());
    assert!(s.metadata.is_none());
    assert!(s.store.is_none());
    assert!(s.extra_body.is_none());
    assert!(s.extra_headers.is_none());
    assert!(s.extra_args.is_none());
}

#[test]
fn test_model_settings_resolve_override_wins() {
    let base = ModelSettings::new().with_temperature(0.7).with_top_p(0.9);
    let overrides = ModelSettings::new()
        .with_temperature(0.1)
        .with_max_tokens(256);

    let resolved = base.resolve(Some(&overrides));

    // Override wins for temperature.
    assert_eq!(resolved.temperature, Some(0.1));
    // Base value is preserved when override is None.
    assert_eq!(resolved.top_p, Some(0.9));
    // Override adds new value.
    assert_eq!(resolved.max_tokens, Some(256));
}

#[test]
fn test_model_settings_resolve_with_no_override_is_clone() {
    let base = ModelSettings::new()
        .with_temperature(0.7)
        .with_store(true)
        .with_truncation(Truncation::Auto);

    let resolved = base.resolve(None);
    assert_eq!(resolved.temperature, Some(0.7));
    assert_eq!(resolved.store, Some(true));
    assert_eq!(resolved.truncation, Some(Truncation::Auto));
}

#[test]
fn test_model_settings_tool_choice_variants() {
    let auto = ModelSettings::new().with_tool_choice(ToolChoice::Auto);
    let required = ModelSettings::new().with_tool_choice(ToolChoice::Required);
    let none = ModelSettings::new().with_tool_choice(ToolChoice::None);
    let named =
        ModelSettings::new().with_tool_choice(ToolChoice::Named("specific_tool".to_owned()));

    assert_eq!(auto.tool_choice, Some(ToolChoice::Auto));
    assert_eq!(required.tool_choice, Some(ToolChoice::Required));
    assert_eq!(none.tool_choice, Some(ToolChoice::None));
    assert_eq!(
        named.tool_choice,
        Some(ToolChoice::Named("specific_tool".to_owned()))
    );
}

// ---------------------------------------------------------------------------
// Section 4: Handoff builder
// ---------------------------------------------------------------------------

#[test]
fn test_handoff_builder_defaults() {
    let handoff: Handoff<()> = Handoff::to_agent("billing").build();

    assert_eq!(handoff.agent_name, "billing");
    assert_eq!(handoff.tool_name, "transfer_to_billing");
    assert_eq!(
        handoff.tool_description,
        "Handoff to the billing agent to handle the request."
    );
    assert!(handoff.strict_json_schema);
    assert!(handoff.input_filter.is_none());
    assert!(handoff.is_enabled);
}

#[test]
fn test_handoff_builder_custom_tool_name() {
    let handoff: Handoff<()> = Handoff::to_agent("billing")
        .tool_name("route_to_billing")
        .build();

    assert_eq!(handoff.tool_name, "route_to_billing");
}

#[test]
fn test_handoff_builder_custom_description() {
    let handoff: Handoff<()> = Handoff::to_agent("support")
        .tool_description("Use for customer support escalations.")
        .build();

    assert_eq!(
        handoff.tool_description,
        "Use for customer support escalations."
    );
}

#[test]
fn test_handoff_builder_disabled() {
    let handoff: Handoff<()> = Handoff::to_agent("billing").is_enabled(false).build();

    assert!(!handoff.is_enabled);
}

#[test]
fn test_handoff_builder_non_strict_schema() {
    let handoff: Handoff<()> = Handoff::to_agent("router")
        .strict_json_schema(false)
        .build();

    assert!(!handoff.strict_json_schema);
    // Non-strict schema returns the raw (empty) schema as-is.
    assert_eq!(handoff.input_json_schema, serde_json::json!({}));
}

#[test]
fn test_handoff_builder_with_typed_input_schema() {
    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct HandoffArgs {
        reason: String,
        priority: u32,
    }

    let handoff: Handoff<()> = Handoff::to_agent("specialized")
        .input_type::<HandoffArgs>()
        .build();

    let props = handoff.input_json_schema["properties"]
        .as_object()
        .expect("schema should have properties");
    assert!(props.contains_key("reason"));
    assert!(props.contains_key("priority"));

    // Strict mode should be applied.
    assert_eq!(handoff.input_json_schema["additionalProperties"], false);
}

#[test]
fn test_handoff_builder_with_raw_json_schema() {
    let raw_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "context": {"type": "string"}
        }
    });

    let handoff: Handoff<()> = Handoff::to_agent("agent")
        .input_json_schema(raw_schema)
        .strict_json_schema(false)
        .build();

    // Non-strict mode, so schema is used as-is.
    assert_eq!(
        handoff.input_json_schema["properties"]["context"]["type"],
        "string"
    );
}

#[tokio::test]
async fn test_handoff_default_invoke_returns_agent_name() {
    let handoff: Handoff<()> = Handoff::to_agent("target").build();
    let ctx = openai_agents::context::RunContextWrapper::new(());

    let result = handoff
        .invoke(&ctx, String::new())
        .await
        .expect("invoke should succeed");

    assert_eq!(result, "target");
}

#[tokio::test]
async fn test_handoff_custom_invoke_callback() {
    let handoff: Handoff<String> = Handoff::to_agent("default_target")
        .on_invoke(|ctx: &RunContextWrapper<String>, args| {
            let prefix = ctx.context.clone();
            Box::pin(async move { Ok(format!("{prefix}:{args}")) })
        })
        .build();

    let ctx = openai_agents::context::RunContextWrapper::new("PREFIX".to_owned());
    let result = handoff
        .invoke(&ctx, "TARGET".to_owned())
        .await
        .expect("invoke should succeed");

    assert_eq!(result, "PREFIX:TARGET");
}

#[test]
fn test_handoff_get_transfer_message() {
    let handoff: Handoff<()> = Handoff::to_agent("billing").build();
    let msg = handoff.get_transfer_message("triage");
    assert_eq!(msg, "Transferred from 'triage' to 'billing'.");
}

#[test]
fn test_handoff_default_tool_name_normalisation() {
    // Spaces → underscores, uppercase → lowercase.
    assert_eq!(
        Handoff::<()>::default_tool_name("Billing Support"),
        "transfer_to_billing_support"
    );
    assert_eq!(
        Handoff::<()>::default_tool_name("URGENT_TRIAGE"),
        "transfer_to_urgent_triage"
    );
    assert_eq!(
        Handoff::<()>::default_tool_name("simple"),
        "transfer_to_simple"
    );
}

#[test]
fn test_handoff_default_description() {
    assert_eq!(
        Handoff::<()>::default_tool_description("billing"),
        "Handoff to the billing agent to handle the request."
    );
}

#[test]
fn test_handoff_debug_impl() {
    let handoff: Handoff<()> = Handoff::to_agent("billing").build();
    let debug_str = format!("{handoff:?}");
    assert!(debug_str.contains("Handoff"));
    assert!(debug_str.contains("billing"));
    assert!(debug_str.contains("transfer_to_billing"));
}

#[test]
fn test_handoff_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Handoff<()>>();
    assert_send_sync::<Handoff<String>>();
}

// ---------------------------------------------------------------------------
// Section 5: Agent builder combined with RunConfig builder
// ---------------------------------------------------------------------------

#[test]
fn test_agent_and_config_compose_correctly() {
    let agent = Agent::<()>::builder("combined")
        .instructions("Be thorough.")
        .model("gpt-4o")
        .model_settings(ModelSettings::new().with_temperature(0.2))
        .tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
        .build();

    let config = RunConfig::builder()
        .max_turns(5)
        .workflow_name("combined_test")
        .build();

    // Verify agent and config fields are independent.
    assert_eq!(agent.name, "combined");
    assert!(matches!(&agent.model, Some(ModelRef::Name(n)) if n == "gpt-4o"));
    assert_eq!(agent.model_settings.temperature, Some(0.2));
    assert_eq!(agent.tool_use_behavior, ToolUseBehavior::StopOnFirstTool);

    assert_eq!(config.max_turns, 5);
    assert_eq!(config.workflow_name, "combined_test");
}

// ---------------------------------------------------------------------------
// Section 6: Agent builder with context type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct AppContext {
    user_id: String,
    tier: u8,
}

#[test]
fn test_agent_builder_with_custom_context_type() {
    let agent = Agent::<AppContext>::builder("premium-agent")
        .instructions("Serve premium users.")
        .build();

    assert_eq!(agent.name, "premium-agent");
}

#[tokio::test]
async fn test_agent_dynamic_instructions_use_custom_context() {
    let agent = Agent::<AppContext>::builder("personalised")
        .dynamic_instructions(|ctx, _agent| {
            let tier = ctx.context.tier;
            let user_id = ctx.context.user_id.clone();
            Box::pin(async move { Ok(format!("Serve user {user_id} at tier {tier}.")) })
        })
        .build();

    let ctx = openai_agents::context::RunContextWrapper::new(AppContext {
        user_id: "user_42".to_owned(),
        tier: 3,
    });

    let instructions = agent
        .get_instructions(&ctx)
        .await
        .expect("should produce instructions");

    assert_eq!(
        instructions,
        Some("Serve user user_42 at tier 3.".to_owned())
    );
}

// ---------------------------------------------------------------------------
// Section 7: RunConfig DEFAULT_MAX_TURNS constant
// ---------------------------------------------------------------------------

#[test]
fn test_default_max_turns_is_reasonable() {
    // The default should match what the RunConfig returns.
    let config = RunConfig::default();
    assert_eq!(config.max_turns, DEFAULT_MAX_TURNS);
    // Ensure it is a sane value (not 0, not unreasonably large).
    const {
        assert!(
            DEFAULT_MAX_TURNS > 0,
            "DEFAULT_MAX_TURNS should be positive"
        );
    };
    const {
        assert!(
            DEFAULT_MAX_TURNS <= 100,
            "DEFAULT_MAX_TURNS should be at most 100"
        );
    };
}

// ---------------------------------------------------------------------------
// Section 8: ModelRef conversions
// ---------------------------------------------------------------------------

#[test]
fn test_model_ref_from_str_slice() {
    let r: ModelRef = "gpt-4o".into();
    assert!(matches!(r, ModelRef::Name(ref n) if n == "gpt-4o"));
}

#[test]
fn test_model_ref_from_string() {
    let r: ModelRef = "gpt-4o-mini".to_owned().into();
    assert!(matches!(r, ModelRef::Name(ref n) if n == "gpt-4o-mini"));
}

#[test]
fn test_model_ref_debug() {
    let r = ModelRef::Name("test-model".to_owned());
    let debug_str = format!("{r:?}");
    assert!(debug_str.contains("Name"));
    assert!(debug_str.contains("test-model"));
}

// ---------------------------------------------------------------------------
// Section 9: ToolUseBehavior
// ---------------------------------------------------------------------------

#[test]
fn test_tool_use_behavior_run_llm_again_is_default() {
    assert_eq!(ToolUseBehavior::default(), ToolUseBehavior::RunLlmAgain);
}

#[test]
fn test_tool_use_behavior_equality() {
    assert_eq!(
        ToolUseBehavior::StopOnFirstTool,
        ToolUseBehavior::StopOnFirstTool
    );
    assert_ne!(
        ToolUseBehavior::RunLlmAgain,
        ToolUseBehavior::StopOnFirstTool
    );
}

#[test]
fn test_tool_use_behavior_stop_at_tools_equality() {
    let a = ToolUseBehavior::StopAtTools(vec!["foo".to_owned()]);
    let b = ToolUseBehavior::StopAtTools(vec!["foo".to_owned()]);
    let c = ToolUseBehavior::StopAtTools(vec!["bar".to_owned()]);

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_tool_use_behavior_clone() {
    let behavior = ToolUseBehavior::StopAtTools(vec!["search".to_owned()]);
    #[allow(clippy::redundant_clone)]
    let cloned = behavior.clone();
    assert_eq!(behavior, cloned);
}

// ---------------------------------------------------------------------------
// Section 10: Agent with multiple guardrails
// ---------------------------------------------------------------------------

#[test]
fn test_agent_with_both_guardrail_types() {
    let input_g = InputGuardrail::<()>::new("input_check", |_ctx, _a, _i| {
        Box::pin(async { Ok(GuardrailFunctionOutput::passed(serde_json::json!(null))) })
    });

    let output_g = OutputGuardrail::<()>::new("output_check", |_ctx, _a, _o| {
        Box::pin(async { Ok(GuardrailFunctionOutput::passed(serde_json::json!(null))) })
    });

    let agent = Agent::<()>::builder("guarded")
        .input_guardrail(input_g)
        .output_guardrail(output_g)
        .build();

    assert_eq!(agent.input_guardrails.len(), 1);
    assert_eq!(agent.output_guardrails.len(), 1);
    assert_eq!(agent.input_guardrails[0].name, "input_check");
    assert_eq!(agent.output_guardrails[0].name, "output_check");
}
