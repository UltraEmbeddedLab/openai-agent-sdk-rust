---
name: rust-python-mapping
description: Map Python SDK patterns to idiomatic Rust. Use when porting a specific module from the Python SDK to help translate Python abstractions, type patterns, and async patterns to their Rust equivalents.
disable-model-invocation: false
user-invocable: true
argument-hint: "[python-module-path]"
---

# Rust-Python Mapping

Translate a Python module from the OpenAI Agents SDK to idiomatic Rust.

## Usage

Pass the Python module path as an argument:
```
/rust-python-mapping src/agents/agent.py
```

## Steps

1. **Read the Python source** at `F:\forks\openai-agents-python\$ARGUMENTS`.

2. **Analyze the module**:
   - List all classes, functions, and type aliases.
   - Identify the public API surface.
   - Note inheritance hierarchies and protocols.
   - Find all imports and dependencies on other SDK modules.

3. **Generate the Rust mapping**:

   For each Python construct, provide the Rust equivalent:

   | Python | Rust |
   |--------|------|
   | `class Foo(ABC)` | `trait Foo` with `#[async_trait]` |
   | `@dataclass class Bar` | `#[derive(Debug, Clone)] struct Bar` with builder |
   | `def method(self)` | `fn method(&self)` |
   | `async def method(self)` | `async fn method(&self)` in trait via `#[async_trait]` |
   | `Foo | Bar | Baz` | `enum FooOrBarOrBaz { Foo(Foo), Bar(Bar), Baz(Baz) }` |
   | `Optional[T] = None` | `Option<T>` with `#[builder(default)]` |
   | `list[T]` | `Vec<T>` |
   | `dict[K, V]` | `HashMap<K, V>` |
   | `TypeVar("T")` | Generic `<T>` |
   | `Callable[[A], R]` | `Box<dyn Fn(A) -> R + Send + Sync>` or `impl Fn(A) -> R` |
   | `AsyncIterator[T]` | `Pin<Box<dyn Stream<Item = T> + Send>>` |
   | `@property` | `pub fn field(&self) -> &T` |
   | `raise SomeError` | `return Err(AgentError::SomeError)` |
   | `try/except` | `match result { Ok(v) => ..., Err(e) => ... }` |
   | `with context_manager` | RAII / `Drop` / explicit scope |
   | `logging.getLogger` | `tracing::instrument`, `tracing::debug!()` |

4. **Output the skeleton Rust code** with:
   - All struct/enum/trait definitions.
   - Method signatures (no bodies — just `todo!()`).
   - Doc comments explaining what each item does.
   - `use` imports for dependencies.

5. **Note any design decisions** where Rust diverges from Python and why.
