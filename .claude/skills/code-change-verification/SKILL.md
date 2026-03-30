---
name: code-change-verification
description: Run the full verification stack (fmt, clippy, test) for the Rust SDK. Use after any changes to src/, tests/, examples/, Cargo.toml, or CI workflows.
disable-model-invocation: false
user-invocable: true
---

# Code Change Verification

Run the full verification stack from the repository root. Rerun the full stack after applying fixes.

## Steps

1. **Format check**:
   ```bash
   cargo fmt --check
   ```
   If formatting issues are found, run `cargo fmt` to fix them, then re-check.

2. **Clippy lint**:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```
   Fix any clippy warnings before proceeding.

3. **Run tests**:
   ```bash
   cargo test --all-features
   ```
   All tests must pass.

4. **Check documentation**:
   ```bash
   cargo doc --no-deps --all-features
   ```
   Documentation must build without warnings (RUSTDOCFLAGS=-Dwarnings).

## Failure Handling

- If any step fails, fix the issue and rerun the **entire** stack from step 1.
- Do not skip steps or mark work as complete until all steps pass.
- Report any persistent failures to the user with the full error output.
