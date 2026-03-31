//! Integration tests for tool creation and invocation patterns.
//!
//! Tests cover `function_tool` helper creation, schema generation, invocation,
//! multiple tools on an agent, and all `ToolOutput` variants.

#![allow(dead_code)]

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;

use openai_agents::context::RunContextWrapper;
use openai_agents::error::AgentError;
use openai_agents::items::ToolOutput;
use openai_agents::tool::{
    CodeInterpreterTool, FileSearchTool, FunctionToolResult, Tool, ToolContext, WebSearchTool,
    function_tool,
};

// ---------------------------------------------------------------------------
// Test 1: function_tool helper creates correct schema
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct WeatherParams {
    city: String,
    units: Option<String>,
}

#[test]
fn test_function_tool_schema_has_correct_properties() {
    let tool = function_tool::<(), WeatherParams, _, _>(
        "get_weather",
        "Get weather for a city.",
        |_ctx: ToolContext<()>, _params: WeatherParams| async move {
            Ok(ToolOutput::Text("sunny".to_owned()))
        },
    )
    .expect("tool creation should succeed");

    assert_eq!(tool.name, "get_weather");
    assert_eq!(tool.description, "Get weather for a city.");
    assert!(tool.strict_json_schema);
    assert!(tool.is_enabled);
    assert!(!tool.needs_approval);

    // Schema should have strict mode properties.
    assert_eq!(
        tool.params_json_schema["strict"], true,
        "schema should have strict: true"
    );
    assert_eq!(
        tool.params_json_schema["additionalProperties"], false,
        "schema should have additionalProperties: false"
    );

    // Schema should include both properties.
    let props = tool.params_json_schema["properties"]
        .as_object()
        .expect("schema should have properties");
    assert!(props.contains_key("city"), "should have 'city' property");
    assert!(props.contains_key("units"), "should have 'units' property");
}

#[test]
fn test_function_tool_schema_is_object_type() {
    #[derive(Deserialize, JsonSchema)]
    struct SimpleParams {
        value: i32,
    }

    let tool =
        function_tool::<(), SimpleParams, _, _>(
            "simple",
            "A simple tool.",
            |_ctx: ToolContext<()>, _p: SimpleParams| async move {
                Ok(ToolOutput::Text("ok".to_owned()))
            },
        )
        .expect("tool creation should succeed");

    assert_eq!(
        tool.params_json_schema["type"], "object",
        "root schema should be an object type"
    );
}

#[test]
fn test_function_tool_all_properties_required_in_strict_mode() {
    #[derive(Deserialize, JsonSchema)]
    struct MultiParam {
        x: f64,
        y: f64,
        label: String,
    }

    let tool = function_tool::<(), MultiParam, _, _>(
        "plot",
        "Plot a point.",
        |_ctx: ToolContext<()>, _p: MultiParam| async move {
            Ok(ToolOutput::Text("plotted".to_owned()))
        },
    )
    .expect("tool creation should succeed");

    let required = tool.params_json_schema["required"]
        .as_array()
        .expect("schema should have required array");
    let required_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();

    assert!(required_strs.contains(&"x"), "x should be required");
    assert!(required_strs.contains(&"y"), "y should be required");
    assert!(required_strs.contains(&"label"), "label should be required");
}

// ---------------------------------------------------------------------------
// Test 2: function_tool invocation — happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_function_tool_invocation_returns_correct_output() {
    #[derive(Deserialize, JsonSchema)]
    struct MulParams {
        x: i64,
        y: i64,
    }

    let tool = function_tool::<(), MulParams, _, _>(
        "multiply",
        "Multiply two numbers.",
        |_ctx: ToolContext<()>, params: MulParams| async move {
            Ok(ToolOutput::Text(format!("{}", params.x * params.y)))
        },
    )
    .expect("tool creation should succeed");

    let ctx = ToolContext::new(
        Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(()))),
        "multiply",
        "call_mul",
    );

    let result = tool
        .invoke(ctx, r#"{"x": 6, "y": 7}"#.to_owned())
        .await
        .expect("invocation should succeed");

    assert_eq!(result, ToolOutput::Text("42".to_owned()));
}

#[tokio::test]
async fn test_function_tool_invocation_bad_json_returns_model_behavior_error() {
    #[derive(Deserialize, JsonSchema)]
    struct AnyParam {
        value: String,
    }

    let tool = function_tool::<(), AnyParam, _, _>(
        "tool",
        "desc",
        |_ctx: ToolContext<()>, _p: AnyParam| async move { Ok(ToolOutput::Text("ok".to_owned())) },
    )
    .expect("tool creation should succeed");

    let ctx = ToolContext::new(
        Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(()))),
        "tool",
        "call_bad",
    );

    let err = tool
        .invoke(ctx, "not valid json".to_owned())
        .await
        .expect_err("should fail with bad JSON");

    assert!(
        matches!(err, AgentError::ModelBehavior { .. }),
        "expected ModelBehavior error, got: {err:?}"
    );
    assert!(
        err.to_string().contains("failed to deserialize"),
        "error message should mention deserialization failure"
    );
}

#[tokio::test]
async fn test_function_tool_invocation_missing_field_returns_error() {
    #[derive(Deserialize, JsonSchema)]
    struct RequiredParam {
        required_field: String,
    }

    let tool =
        function_tool::<(), RequiredParam, _, _>(
            "tool",
            "desc",
            |_ctx: ToolContext<()>, _p: RequiredParam| async move {
                Ok(ToolOutput::Text("ok".to_owned()))
            },
        )
        .expect("tool creation should succeed");

    let ctx = ToolContext::new(
        Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(()))),
        "tool",
        "c1",
    );

    // JSON is valid but missing the required field.
    let err = tool
        .invoke(ctx, r"{}".to_owned())
        .await
        .expect_err("should fail: missing required field");

    assert!(
        matches!(err, AgentError::ModelBehavior { .. }),
        "expected ModelBehavior error, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: function_tool accesses run context
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_function_tool_reads_context_value() {
    #[derive(Deserialize, JsonSchema)]
    struct EchoParams {
        suffix: String,
    }

    let tool = function_tool::<String, EchoParams, _, _>(
        "echo_with_context",
        "Echo with context prefix.",
        |ctx: ToolContext<String>, params: EchoParams| async move {
            let prefix = ctx.context.read().await.context.clone();
            Ok(ToolOutput::Text(format!("{prefix} {}", params.suffix)))
        },
    )
    .expect("tool creation should succeed");

    let ctx = ToolContext::new(
        Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(
            "PREFIX".to_owned(),
        ))),
        "echo_with_context",
        "c1",
    );

    let result = tool
        .invoke(ctx, r#"{"suffix": "SUFFIX"}"#.to_owned())
        .await
        .expect("should succeed");

    assert_eq!(result, ToolOutput::Text("PREFIX SUFFIX".to_owned()));
}

// ---------------------------------------------------------------------------
// Test 4: Multiple tools on an agent — verify tool specs
// ---------------------------------------------------------------------------

use async_trait::async_trait;
use openai_agents::config::ModelSettings;
use openai_agents::items::{ModelResponse, ResponseInputItem, ResponseStreamEvent};
use openai_agents::models::{HandoffToolSpec, Model, ModelTracing, OutputSchemaSpec, ToolSpec};
use openai_agents::new_model_response;
use openai_agents::usage::Usage;
use std::pin::Pin;
use tokio_stream::Stream;

#[derive(Deserialize, JsonSchema)]
struct NoParams {}

#[test]
fn test_agent_with_multiple_tools_all_appear_in_spec() {
    let t1 = function_tool::<(), NoParams, _, _>(
        "tool_alpha",
        "Alpha tool.",
        |_ctx: ToolContext<()>, _p: NoParams| async move { Ok(ToolOutput::Text("a".to_owned())) },
    )
    .expect("tool creation");

    let t2 = function_tool::<(), NoParams, _, _>(
        "tool_beta",
        "Beta tool.",
        |_ctx: ToolContext<()>, _p: NoParams| async move { Ok(ToolOutput::Text("b".to_owned())) },
    )
    .expect("tool creation");

    let agent = openai_agents::agent::Agent::<()>::builder("multi-tool")
        .tool(Tool::Function(t1))
        .tool(Tool::Function(t2))
        .tool(Tool::WebSearch(WebSearchTool::default()))
        .build();

    assert_eq!(agent.tools.len(), 3);

    let names: Vec<&str> = agent.tools.iter().map(openai_agents::Tool::name).collect();
    assert!(names.contains(&"tool_alpha"), "should have tool_alpha");
    assert!(names.contains(&"tool_beta"), "should have tool_beta");
    assert!(names.contains(&"web_search"), "should have web_search");
}

#[test]
fn test_function_tool_is_not_hosted() {
    let t = function_tool::<(), NoParams, _, _>(
        "my_fn",
        "desc",
        |_ctx: ToolContext<()>, _p: NoParams| async move { Ok(ToolOutput::Text("ok".to_owned())) },
    )
    .expect("tool creation");

    let tool = Tool::Function(t);
    assert!(!tool.is_hosted());
}

#[test]
fn test_hosted_tools_are_marked_as_hosted() {
    let web: Tool<()> = Tool::WebSearch(WebSearchTool::default());
    let file: Tool<()> = Tool::FileSearch(FileSearchTool::default());
    let code: Tool<()> = Tool::CodeInterpreter(CodeInterpreterTool::default());

    assert!(web.is_hosted());
    assert!(file.is_hosted());
    assert!(code.is_hosted());
}

// ---------------------------------------------------------------------------
// Test 5: ToolOutput variants
// ---------------------------------------------------------------------------

#[test]
fn test_tool_output_text_variant() {
    let output = ToolOutput::Text("hello world".to_owned());
    match &output {
        ToolOutput::Text(s) => assert_eq!(s, "hello world"),
        _ => panic!("expected Text variant"),
    }
}

#[test]
fn test_tool_output_image_with_url() {
    let output = ToolOutput::Image {
        image_url: Some("https://example.com/photo.png".to_owned()),
        file_id: None,
    };
    match &output {
        ToolOutput::Image { image_url, file_id } => {
            assert_eq!(image_url.as_deref(), Some("https://example.com/photo.png"));
            assert!(file_id.is_none());
        }
        _ => panic!("expected Image variant"),
    }
}

#[test]
fn test_tool_output_image_with_file_id() {
    let output = ToolOutput::Image {
        image_url: None,
        file_id: Some("file_abc123".to_owned()),
    };
    match &output {
        ToolOutput::Image { image_url, file_id } => {
            assert!(image_url.is_none());
            assert_eq!(file_id.as_deref(), Some("file_abc123"));
        }
        _ => panic!("expected Image variant"),
    }
}

#[test]
fn test_tool_output_file_with_all_fields() {
    let output = ToolOutput::File {
        file_data: Some("base64encodeddata".to_owned()),
        file_url: Some("https://example.com/file.pdf".to_owned()),
        file_id: Some("file_xyz".to_owned()),
        filename: Some("report.pdf".to_owned()),
    };
    match &output {
        ToolOutput::File {
            file_data,
            file_url,
            file_id,
            filename,
        } => {
            assert_eq!(file_data.as_deref(), Some("base64encodeddata"));
            assert_eq!(file_url.as_deref(), Some("https://example.com/file.pdf"));
            assert_eq!(file_id.as_deref(), Some("file_xyz"));
            assert_eq!(filename.as_deref(), Some("report.pdf"));
        }
        _ => panic!("expected File variant"),
    }
}

#[test]
fn test_tool_output_file_with_partial_fields() {
    let output = ToolOutput::File {
        file_data: None,
        file_url: None,
        file_id: Some("file_only".to_owned()),
        filename: None,
    };
    match &output {
        ToolOutput::File {
            file_data,
            file_url,
            file_id,
            filename,
        } => {
            assert!(file_data.is_none());
            assert!(file_url.is_none());
            assert_eq!(file_id.as_deref(), Some("file_only"));
            assert!(filename.is_none());
        }
        _ => panic!("expected File variant"),
    }
}

// ---------------------------------------------------------------------------
// Test 6: FunctionToolResult construction
// ---------------------------------------------------------------------------

#[test]
fn test_function_tool_result_stores_all_fields() {
    let result = FunctionToolResult::new(
        "my_tool",
        "call_xyz",
        ToolOutput::Text("result_value".to_owned()),
    );

    assert_eq!(result.tool_name, "my_tool");
    assert_eq!(result.tool_call_id, "call_xyz");
    assert_eq!(result.output, ToolOutput::Text("result_value".to_owned()));
}

// ---------------------------------------------------------------------------
// Test 7: Tool clone behavior
// ---------------------------------------------------------------------------

#[test]
fn test_function_tool_clone_preserves_fields() {
    let tool =
        function_tool::<(), NoParams, _, _>(
            "cloneable",
            "A cloneable tool.",
            |_ctx: ToolContext<()>, _p: NoParams| async move {
                Ok(ToolOutput::Text("cloned".to_owned()))
            },
        )
        .expect("tool creation");

    let cloned = tool.clone();
    assert_eq!(cloned.name, tool.name);
    assert_eq!(cloned.description, tool.description);
    assert_eq!(cloned.strict_json_schema, tool.strict_json_schema);
    assert_eq!(cloned.is_enabled, tool.is_enabled);
    assert_eq!(cloned.needs_approval, tool.needs_approval);
}

#[tokio::test]
async fn test_cloned_tool_still_invokable() {
    let tool =
        function_tool::<(), NoParams, _, _>(
            "cloneable",
            "A cloneable tool.",
            |_ctx: ToolContext<()>, _p: NoParams| async move {
                Ok(ToolOutput::Text("invoked".to_owned()))
            },
        )
        .expect("tool creation");

    let cloned = tool.clone();

    let ctx = ToolContext::new(
        Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(()))),
        "cloneable",
        "c1",
    );

    let result = cloned
        .invoke(ctx, "{}".to_owned())
        .await
        .expect("should succeed");

    assert_eq!(result, ToolOutput::Text("invoked".to_owned()));
}

// ---------------------------------------------------------------------------
// Test 8: Tool integration with runner — unknown tool produces error output
// ---------------------------------------------------------------------------

/// A mock model that returns a function call for a tool not on the agent.
struct UnknownToolModel;

#[async_trait]
impl Model for UnknownToolModel {
    async fn get_response(
        &self,
        _system_instructions: Option<&str>,
        _input: &[ResponseInputItem],
        _model_settings: &ModelSettings,
        _tools: &[ToolSpec],
        _output_schema: Option<&OutputSchemaSpec>,
        _handoffs: &[HandoffToolSpec],
        _tracing: ModelTracing,
        _previous_response_id: Option<&str>,
    ) -> openai_agents::error::Result<ModelResponse> {
        static CALL_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        if count == 0 {
            // First call: request unknown tool.
            Ok(new_model_response(
                vec![serde_json::json!({
                    "type": "function_call",
                    "name": "does_not_exist",
                    "call_id": "c1",
                    "arguments": "{}",
                })],
                Usage::default(),
                None,
                None,
            ))
        } else {
            // Second call: produce final output.
            Ok(new_model_response(
                vec![serde_json::json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Tool not found, sorry."}]
                })],
                Usage::default(),
                None,
                None,
            ))
        }
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
    ) -> Pin<Box<dyn Stream<Item = openai_agents::error::Result<ResponseStreamEvent>> + Send + 'a>>
    {
        Box::pin(tokio_stream::empty())
    }
}

#[tokio::test]
async fn test_unknown_tool_produces_error_in_output_item() {
    let agent = openai_agents::agent::Agent::<()>::builder("no-tools").build();
    let model = Arc::new(UnknownToolModel);

    let result =
        openai_agents::runner::Runner::run_with_model(&agent, "test", (), model, None, None)
            .await
            .expect("run should succeed even with unknown tool");

    // The runner produces a ToolCallOutput with an error message for unknown tools.
    let tool_outputs: Vec<_> = result
        .new_items
        .iter()
        .filter(|item| matches!(item, openai_agents::items::RunItem::ToolCallOutput(_)))
        .collect();

    assert!(!tool_outputs.is_empty(), "should have a tool output item");

    if let openai_agents::items::RunItem::ToolCallOutput(tco) = &tool_outputs[0] {
        let raw_str = tco.raw_item.to_string();
        // The error output should mention that the tool was not found.
        assert!(
            raw_str.contains("not found") || raw_str.to_lowercase().contains("error"),
            "tool output should indicate error: {raw_str}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 9: Tool spec descriptions match tool descriptions
// ---------------------------------------------------------------------------

#[test]
fn test_web_search_tool_description() {
    let tool: Tool<()> = Tool::WebSearch(WebSearchTool::default());
    assert_eq!(tool.description(), "Search the web for information.");
    assert_eq!(tool.name(), "web_search");
}

#[test]
fn test_file_search_tool_description() {
    let mut fs = FileSearchTool::default();
    fs.vector_store_ids = vec!["vs_123".to_owned()];
    fs.max_num_results = Some(10);
    let tool: Tool<()> = Tool::FileSearch(fs);
    assert_eq!(tool.description(), "Search over files in vector stores.");
    assert_eq!(tool.name(), "file_search");
}

#[test]
fn test_code_interpreter_tool_description() {
    let mut ci = CodeInterpreterTool::default();
    ci.container = Some("container_abc".to_owned());
    let tool: Tool<()> = Tool::CodeInterpreter(ci);
    assert_eq!(
        tool.description(),
        "Execute code in a sandboxed environment."
    );
    assert_eq!(tool.name(), "code_interpreter");
}

// ---------------------------------------------------------------------------
// Test 10: ToolContext is Debug and Clone
// ---------------------------------------------------------------------------

#[test]
fn test_tool_context_debug_contains_fields() {
    let ctx = ToolContext::<()>::new(
        Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(()))),
        "debug_tool",
        "call_dbg",
    );

    let debug_str = format!("{ctx:?}");
    assert!(
        debug_str.contains("debug_tool"),
        "debug output should contain tool name"
    );
    assert!(
        debug_str.contains("call_dbg"),
        "debug output should contain call_id"
    );
}

#[test]
fn test_tool_context_clone_shares_arc() {
    let inner = Arc::new(tokio::sync::RwLock::new(RunContextWrapper::new(42_u32)));
    let ctx = ToolContext::new(Arc::clone(&inner), "t", "c");

    let cloned = ctx.clone();

    // Both contexts should point to the same Arc.
    assert!(Arc::ptr_eq(&ctx.context, &cloned.context));
    assert_eq!(cloned.tool_name, "t");
    assert_eq!(cloned.tool_call_id, "c");
}

// ---------------------------------------------------------------------------
// Test 11: FunctionTool debug output does not leak invoke_fn
// ---------------------------------------------------------------------------

#[test]
fn test_function_tool_debug_does_not_leak_impl() {
    let tool = function_tool::<(), NoParams, _, _>(
        "debug_test",
        "A tool for debug testing.",
        |_ctx: ToolContext<()>, _p: NoParams| async move { Ok(ToolOutput::Text("x".to_owned())) },
    )
    .expect("tool creation");

    let debug_str = format!("{tool:?}");
    assert!(debug_str.contains("debug_test"), "should contain tool name");
    assert!(
        debug_str.contains("A tool for debug testing."),
        "should contain description"
    );
    // The internal invoke_fn closure should not appear in debug output.
    assert!(
        !debug_str.contains("invoke_fn"),
        "invoke_fn should not appear in debug: {debug_str}"
    );
}
