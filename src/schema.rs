//! JSON Schema generation and strict-mode enforcement for function tools.
//!
//! This module provides utilities to generate JSON schemas from Rust types (via
//! [`schemars`]) and to transform those schemas into the "strict" format that the
//! `OpenAI` API expects. The strict format ensures deterministic structured output
//! from the model by requiring `"additionalProperties": false` on every object,
//! listing all properties in `"required"`, and setting `"strict": true` at the
//! top level.
//!
//! # Example
//!
//! ```
//! use openai_agents::schema::{json_schema_for, ensure_strict_json_schema};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct MyParams {
//!     name: String,
//!     age: u32,
//! }
//!
//! let schema = json_schema_for::<MyParams>();
//! let strict = ensure_strict_json_schema(schema).unwrap();
//!
//! assert_eq!(strict["additionalProperties"], false);
//! assert_eq!(strict["strict"], true);
//! ```

use serde_json::{Map, Value};

use crate::error::AgentError;

/// Schema information for a function tool.
///
/// Captures the name, description, JSON schema, and strict-mode flag that are
/// sent to the `OpenAI` API when registering a tool.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FuncSchema {
    /// The name of the function tool.
    pub name: String,
    /// An optional human-readable description of what the tool does.
    pub description: Option<String>,
    /// The JSON schema describing the tool's parameters.
    pub params_json_schema: Value,
    /// Whether the schema has been transformed into `OpenAI` strict mode.
    pub strict_json_schema: bool,
}

/// Generate a JSON schema for a Rust type using `schemars`.
///
/// Uses the `OpenAPI` 3.0 schema settings, which produce schemas compatible with
/// the `OpenAI` API. The returned [`Value`] is a JSON object representing the
/// root schema for `T`.
#[must_use]
pub fn json_schema_for<T: schemars::JsonSchema>() -> Value {
    let generator = schemars::generate::SchemaSettings::openapi3().into_generator();
    let schema = generator.into_root_schema_for::<T>();
    Value::from(schema)
}

/// Ensure the schema conforms to the `OpenAI` strict structured-output format.
///
/// This function recursively walks the given JSON schema and applies the
/// following transformations:
///
/// - Adds `"additionalProperties": false` to every object type.
/// - Sets `"required"` to the full list of property names for every object.
/// - Converts `"oneOf"` to `"anyOf"` (`OpenAI` does not support `"oneOf"` in
///   nested contexts).
/// - Flattens single-element `"allOf"` arrays.
/// - Removes `null` defaults (the model will default to `null` for nullable
///   fields anyway).
/// - Inlines `"$ref"` when other properties are present alongside it.
/// - Recurses into `"$defs"`, `"definitions"`, `"properties"`, `"items"`,
///   `"anyOf"`, `"oneOf"`, and `"allOf"`.
/// - Sets `"strict": true` at the top level.
///
/// # Errors
///
/// Returns [`AgentError::UserError`] if the schema contains
/// `"additionalProperties": true`, which is incompatible with strict mode.
pub fn ensure_strict_json_schema(mut schema: Value) -> Result<Value, AgentError> {
    if schema == Value::Object(Map::new()) {
        return Ok(serde_json::json!({
            "additionalProperties": false,
            "type": "object",
            "properties": {},
            "required": [],
        }));
    }

    let root = schema.clone();
    ensure_strict_recursive(&mut schema, &root)?;
    schema["strict"] = Value::Bool(true);
    Ok(schema)
}

/// Recursively transform a schema node into strict mode.
fn ensure_strict_recursive(schema: &mut Value, root: &Value) -> Result<(), AgentError> {
    let Some(obj) = schema.as_object_mut() else {
        return Ok(());
    };

    process_defs(obj, root)?;
    enforce_additional_properties(obj)?;
    process_properties(obj, root)?;
    process_items(obj, root)?;
    process_any_of(obj, root)?;
    process_one_of(obj, root)?;
    process_all_of(obj, root)?;
    strip_null_default(obj);
    inline_ref(obj, root)?;

    Ok(())
}

/// Process `$defs`, `definitions`, and `components/schemas` sections.
fn process_defs(obj: &mut Map<String, Value>, root: &Value) -> Result<(), AgentError> {
    for key in &["$defs", "definitions"] {
        if let Some(defs) = obj.get_mut(*key) {
            if let Some(defs_map) = defs.as_object_mut() {
                let keys: Vec<String> = defs_map.keys().cloned().collect();
                for k in keys {
                    if let Some(def_schema) = defs_map.get_mut(&k) {
                        ensure_strict_recursive(def_schema, root)?;
                    }
                }
            }
        }
    }

    // Handle OpenAPI 3.0 style components/schemas.
    if let Some(components) = obj.get_mut("components") {
        if let Some(schemas) = components
            .as_object_mut()
            .and_then(|c| c.get_mut("schemas"))
        {
            if let Some(schemas_map) = schemas.as_object_mut() {
                let keys: Vec<String> = schemas_map.keys().cloned().collect();
                for k in keys {
                    if let Some(def_schema) = schemas_map.get_mut(&k) {
                        ensure_strict_recursive(def_schema, root)?;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Enforce `additionalProperties: false` on object types.
fn enforce_additional_properties(obj: &mut Map<String, Value>) -> Result<(), AgentError> {
    let is_object_type = obj.get("type").and_then(Value::as_str) == Some("object");
    if !is_object_type {
        return Ok(());
    }

    match obj.get("additionalProperties") {
        None => {
            obj.insert("additionalProperties".to_string(), Value::Bool(false));
        }
        Some(Value::Bool(true) | Value::Object(_)) => {
            return Err(AgentError::UserError {
                message: "additionalProperties should not be set for object types. \
                          This could be because you configured additional properties \
                          to be allowed. If you really need this, update the function \
                          or output tool to not use a strict schema."
                    .to_string(),
            });
        }
        _ => {}
    }
    Ok(())
}

/// Set required to all property names and recurse into each property schema.
fn process_properties(obj: &mut Map<String, Value>, root: &Value) -> Result<(), AgentError> {
    if let Some(properties) = obj.get("properties") {
        if let Some(props_map) = properties.as_object() {
            let keys: Vec<String> = props_map.keys().cloned().collect();
            let required: Vec<Value> = keys.iter().map(|k| Value::String(k.clone())).collect();
            obj.insert("required".to_string(), Value::Array(required));
        }
    }

    if let Some(properties) = obj.get_mut("properties") {
        if let Some(props_map) = properties.as_object_mut() {
            let keys: Vec<String> = props_map.keys().cloned().collect();
            for key in keys {
                if let Some(prop_schema) = props_map.get_mut(&key) {
                    ensure_strict_recursive(prop_schema, root)?;
                }
            }
        }
    }
    Ok(())
}

/// Recurse into array `items`.
fn process_items(obj: &mut Map<String, Value>, root: &Value) -> Result<(), AgentError> {
    if let Some(items) = obj.get_mut("items") {
        if items.is_object() {
            ensure_strict_recursive(items, root)?;
        }
    }
    Ok(())
}

/// Recurse into `anyOf` variants.
fn process_any_of(obj: &mut Map<String, Value>, root: &Value) -> Result<(), AgentError> {
    if let Some(any_of) = obj.get_mut("anyOf") {
        if let Some(arr) = any_of.as_array_mut() {
            for variant in arr.iter_mut() {
                ensure_strict_recursive(variant, root)?;
            }
        }
    }
    Ok(())
}

/// Convert `oneOf` to `anyOf` since the `OpenAI` API does not support `oneOf` in nested contexts.
fn process_one_of(obj: &mut Map<String, Value>, root: &Value) -> Result<(), AgentError> {
    let Some(one_of) = obj.remove("oneOf") else {
        return Ok(());
    };
    let Some(one_of_arr) = one_of.as_array() else {
        return Ok(());
    };

    let mut converted: Vec<Value> = one_of_arr.clone();
    for variant in &mut converted {
        ensure_strict_recursive(variant, root)?;
    }

    let existing = obj
        .get_mut("anyOf")
        .and_then(Value::as_array_mut)
        .map(std::mem::take);

    let mut merged = existing.unwrap_or_default();
    merged.extend(converted);
    obj.insert("anyOf".to_string(), Value::Array(merged));
    Ok(())
}

/// Process `allOf`: flatten single-element arrays, recurse into multi-element arrays.
fn process_all_of(obj: &mut Map<String, Value>, root: &Value) -> Result<(), AgentError> {
    let Some(all_of) = obj.remove("allOf") else {
        return Ok(());
    };
    let Some(all_of_arr) = all_of.as_array() else {
        return Ok(());
    };

    if all_of_arr.len() == 1 {
        let mut single = all_of_arr[0].clone();
        ensure_strict_recursive(&mut single, root)?;
        if let Some(single_map) = single.as_object() {
            for (k, v) in single_map {
                obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    } else {
        let mut new_all_of: Vec<Value> = Vec::with_capacity(all_of_arr.len());
        for entry in all_of_arr {
            let mut entry = entry.clone();
            ensure_strict_recursive(&mut entry, root)?;
            new_all_of.push(entry);
        }
        obj.insert("allOf".to_string(), Value::Array(new_all_of));
    }
    Ok(())
}

/// Strip `null` defaults since the model defaults to `null` for nullable fields.
fn strip_null_default(obj: &mut Map<String, Value>) {
    if obj.get("default") == Some(&Value::Null) {
        obj.remove("default");
    }
}

/// Inline `$ref` when other properties are present alongside it.
fn inline_ref(obj: &mut Map<String, Value>, root: &Value) -> Result<(), AgentError> {
    let Some(Value::String(ref_str)) = obj.get("$ref").cloned() else {
        return Ok(());
    };

    if obj.len() <= 1 {
        return Ok(());
    }

    let resolved = resolve_ref(root, &ref_str)?;
    if let Some(resolved_map) = resolved.as_object() {
        for (k, v) in resolved_map {
            obj.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    obj.remove("$ref");

    // Re-run strict enforcement on the now-inlined schema.
    let root_clone = root.clone();
    let mut temp = Value::Object(std::mem::take(obj));
    ensure_strict_recursive(&mut temp, &root_clone)?;
    if let Value::Object(m) = temp {
        *obj = m;
    }
    Ok(())
}

/// Resolve a JSON `$ref` pointer against the root schema.
fn resolve_ref(root: &Value, reference: &str) -> Result<Value, AgentError> {
    let path = reference
        .strip_prefix("#/")
        .ok_or_else(|| AgentError::UserError {
            message: format!("unexpected $ref format {reference:?}; does not start with #/"),
        })?;

    let mut resolved = root;
    for key in path.split('/') {
        resolved = resolved.get(key).ok_or_else(|| AgentError::UserError {
            message: format!("could not resolve $ref {reference:?}: missing key {key:?}"),
        })?;
    }

    Ok(resolved.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct SimpleStruct {
        name: String,
        age: u32,
    }

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct Address {
        street: String,
        city: String,
    }

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct NestedStruct {
        person_name: String,
        address: Address,
    }

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct WithOptionals {
        required_field: String,
        optional_field: Option<i32>,
    }

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct WithVec {
        tags: Vec<String>,
    }

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    enum Color {
        Red,
        Green,
        Blue,
    }

    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct ComplexStruct {
        name: String,
        age: u32,
        email: Option<String>,
        tags: Vec<String>,
        address: Address,
        color: Color,
    }

    #[test]
    fn json_schema_for_simple_struct() {
        let schema = json_schema_for::<SimpleStruct>();
        let obj = schema.as_object().expect("schema should be an object");
        assert_eq!(obj.get("type").and_then(Value::as_str), Some("object"));
        let props = obj.get("properties").and_then(Value::as_object).unwrap();
        assert!(props.contains_key("name"));
        assert!(props.contains_key("age"));
    }

    #[test]
    fn json_schema_for_nested_struct() {
        let schema = json_schema_for::<NestedStruct>();
        let obj = schema.as_object().expect("schema should be an object");
        let props = obj.get("properties").and_then(Value::as_object).unwrap();
        assert!(props.contains_key("person_name"));
        assert!(props.contains_key("address"));
    }

    #[test]
    fn json_schema_for_optional_fields() {
        let schema = json_schema_for::<WithOptionals>();
        let obj = schema.as_object().expect("schema should be an object");
        let props = obj.get("properties").and_then(Value::as_object).unwrap();
        assert!(props.contains_key("required_field"));
        assert!(props.contains_key("optional_field"));
    }

    #[test]
    fn strict_adds_additional_properties_false() {
        let schema = json_schema_for::<SimpleStruct>();
        let strict = ensure_strict_json_schema(schema).unwrap();
        assert_eq!(strict["additionalProperties"], Value::Bool(false));
    }

    #[test]
    fn strict_sets_all_properties_required() {
        let schema = json_schema_for::<WithOptionals>();
        let strict = ensure_strict_json_schema(schema).unwrap();
        let required = strict["required"].as_array().unwrap();
        let required_strs: Vec<&str> = required.iter().filter_map(Value::as_str).collect();
        assert!(required_strs.contains(&"required_field"));
        assert!(required_strs.contains(&"optional_field"));
    }

    #[test]
    fn strict_with_nested_objects() {
        let schema = json_schema_for::<NestedStruct>();
        let strict = ensure_strict_json_schema(schema).unwrap();

        // Top level should have additionalProperties: false.
        assert_eq!(strict["additionalProperties"], Value::Bool(false));

        // Every object with properties should have additionalProperties: false.
        check_all_objects_strict(&strict);
    }

    #[test]
    fn strict_sets_strict_true_at_top_level() {
        let schema = json_schema_for::<SimpleStruct>();
        let strict = ensure_strict_json_schema(schema).unwrap();
        assert_eq!(strict["strict"], Value::Bool(true));
    }

    #[test]
    fn strict_with_array_items() {
        let schema = json_schema_for::<WithVec>();
        let strict = ensure_strict_json_schema(schema).unwrap();
        assert_eq!(strict["additionalProperties"], Value::Bool(false));
        assert_eq!(strict["strict"], Value::Bool(true));
    }

    #[test]
    fn strict_empty_schema() {
        let schema = Value::Object(Map::new());
        let strict = ensure_strict_json_schema(schema).unwrap();
        assert_eq!(strict["additionalProperties"], Value::Bool(false));
        assert_eq!(strict["type"], "object");
        assert_eq!(strict["properties"], Value::Object(Map::new()));
        assert_eq!(strict["required"], Value::Array(vec![]));
    }

    #[test]
    fn strict_rejects_additional_properties_true() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "additionalProperties": true
        });
        let result = ensure_strict_json_schema(schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("additionalProperties"),
            "error message should mention additionalProperties: {err}"
        );
    }

    #[test]
    fn strict_converts_one_of_to_any_of() {
        let schema = serde_json::json!({
            "oneOf": [
                { "type": "string" },
                { "type": "integer" }
            ]
        });
        let strict = ensure_strict_json_schema(schema).unwrap();
        assert!(strict.get("oneOf").is_none(), "oneOf should be removed");
        let any_of = strict["anyOf"].as_array().unwrap();
        assert_eq!(any_of.len(), 2);
    }

    #[test]
    fn strict_flattens_single_all_of() {
        let schema = serde_json::json!({
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                }
            ]
        });
        let strict = ensure_strict_json_schema(schema).unwrap();
        assert!(
            strict.get("allOf").is_none(),
            "single allOf should be flattened"
        );
        assert_eq!(strict["type"], "object");
        assert!(strict.get("properties").is_some());
    }

    #[test]
    fn strict_removes_null_defaults() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "default": null
                }
            }
        });
        let strict = ensure_strict_json_schema(schema).unwrap();
        let name_schema = &strict["properties"]["name"];
        assert!(
            name_schema.get("default").is_none(),
            "null default should be removed"
        );
    }

    #[test]
    fn snapshot_complex_schema() {
        let schema = json_schema_for::<ComplexStruct>();
        let strict = ensure_strict_json_schema(schema).unwrap();
        insta::assert_json_snapshot!(strict);
    }

    #[test]
    fn resolve_ref_basic() {
        let root = serde_json::json!({
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": { "type": "string" }
                    }
                }
            }
        });
        let resolved = resolve_ref(&root, "#/$defs/Address").unwrap();
        assert_eq!(resolved["type"], "object");
    }

    #[test]
    fn resolve_ref_invalid_format() {
        let root = serde_json::json!({});
        let result = resolve_ref(&root, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn func_schema_struct_fields() {
        let fs = FuncSchema {
            name: "test_tool".to_string(),
            description: Some("A test tool.".to_string()),
            params_json_schema: serde_json::json!({"type": "object"}),
            strict_json_schema: true,
        };
        assert_eq!(fs.name, "test_tool");
        assert_eq!(fs.description.as_deref(), Some("A test tool."));
        assert!(fs.strict_json_schema);
    }

    /// Recursively verify that every object with properties has
    /// `"additionalProperties": false`.
    fn check_all_objects_strict(value: &Value) {
        if let Some(obj) = value.as_object() {
            if obj.get("type").and_then(Value::as_str) == Some("object")
                && obj.contains_key("properties")
            {
                assert_eq!(
                    obj.get("additionalProperties"),
                    Some(&Value::Bool(false)),
                    "object missing additionalProperties: false"
                );
            }
            for v in obj.values() {
                check_all_objects_strict(v);
            }
        } else if let Some(arr) = value.as_array() {
            for v in arr {
                check_all_objects_strict(v);
            }
        }
    }
}
