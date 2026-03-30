---
name: pr-draft-summary
description: Generate a PR summary block with branch suggestion, title, and draft description after completing code changes. Use as the final step after runtime code, tests, examples, or build config changes.
disable-model-invocation: false
user-invocable: true
---

# PR Draft Summary

Generate a pull request summary as the final handoff step after code changes.

## When to Use

After completing work that involves:
- Runtime code changes (`src/`).
- Test changes (`tests/`).
- Example changes (`examples/`).
- Build/test configuration changes (`Cargo.toml`, `Makefile`, CI workflows).
- Documentation with behavior impact.

## Skip When

- Trivial or conversation-only tasks.
- Repo-meta/doc-only changes without behavior impact.
- User explicitly says not to include PR draft.

## Output Format

Generate a summary block in this format:

```
## PR Draft

**Branch**: `feat/<short-description>` or `fix/<short-description>`
**Title**: <concise title under 70 characters>

### Summary
- <bullet point 1: what changed and why>
- <bullet point 2: what changed and why>
- <bullet point 3: if needed>

### Test Plan
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy -- -W clippy::all` passes
- [ ] `cargo test` passes
- [ ] <specific test cases relevant to this change>

### Notes
<any additional context for reviewers>
```

## Steps

1. Review all changes made in the current session (use `git diff`).
2. Summarize the changes focusing on *what* and *why*, not *how*.
3. Suggest a branch name following the convention: `feat/`, `fix/`, `refactor/`, `docs/`.
4. Write a concise PR title (under 70 characters).
5. List specific test cases that validate the changes.
6. Note any breaking changes, migration steps, or reviewer focus areas.
