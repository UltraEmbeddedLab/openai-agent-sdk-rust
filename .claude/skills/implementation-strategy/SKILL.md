---
name: implementation-strategy
description: Plan the implementation approach before changing runtime code, public APIs, or user-facing behavior. Decides compatibility boundaries and implementation shape. Use before any significant code change.
disable-model-invocation: false
user-invocable: true
---

# Implementation Strategy

Before changing runtime code, exported APIs, or user-facing behavior, use this skill to plan the approach.

## Steps

1. **Identify the scope**: What files and modules are affected? List them.

2. **Check the Python reference**: Read the corresponding Python SDK code at `F:\forks\openai-agents-python\src\agents\` to understand:
   - The exact behavior being implemented.
   - The public API surface.
   - Edge cases and error handling.
   - How it integrates with other modules.

3. **Design the Rust equivalent**: Map the Python patterns to idiomatic Rust:
   - ABC/abstract classes → traits with `#[async_trait]`.
   - Dataclasses with many fields → builder pattern.
   - Union types → enums.
   - Exceptions → `thiserror` error enum variants.
   - `Optional[T]` → `Option<T>`.
   - `list[T]` → `Vec<T>`.
   - Context managers → `Drop` trait or RAII patterns.
   - Decorators → procedural macros or trait implementations.

4. **Assess compatibility**:
   - Is this a new feature or a change to existing API?
   - If changing existing API, what's the migration path?
   - Should `#[non_exhaustive]` be used?
   - Are there breaking changes to consider?

5. **Write the plan**: Document the approach before implementing:
   - List all files to create/modify.
   - Define the public API surface (structs, traits, functions).
   - Note any design decisions and their rationale.
   - Identify test cases needed.

6. **Get approval**: Present the plan for review before implementing.

## Key Principles

- Prefer the simplest correct implementation.
- Match the Python SDK's behavior exactly unless there's a strong Rust-specific reason to diverge.
- Document any intentional divergences with comments explaining why.
- Use `#[non_exhaustive]` on all public enums and structs with named fields.
- Prefer generics over trait objects unless dynamic dispatch is genuinely needed.
