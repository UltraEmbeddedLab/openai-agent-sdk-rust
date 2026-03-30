# ExecPlan Template

Use this template when work is multi-step, spans several files, involves new features or refactors, or is likely to take more than about an hour.

## Plan: [Title]

### Goal
<!-- What are we trying to achieve? -->

### Context
<!-- What Python SDK module(s) are we porting? Links to relevant files. -->

### Scope
<!-- What's in scope and what's explicitly out of scope? -->

### Implementation Steps

1. [ ] Step 1 — Description
2. [ ] Step 2 — Description
3. [ ] Step 3 — Description

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/module.rs` | Create | New module |
| `src/lib.rs` | Modify | Add module export |

### Public API Surface

```rust
// Key types and functions this change introduces
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Decision 1 | Why |

### Test Cases

- [ ] Test case 1
- [ ] Test case 2

### Compatibility Notes
<!-- Any breaking changes or migration concerns -->

---

## Living Sections (update as you execute)

### Progress
- [ ] Started
- [ ] Implementation complete
- [ ] Tests passing
- [ ] Verification stack green

### Surprises & Discoveries
<!-- Things that came up during implementation -->

### Decision Log
<!-- Decisions made during execution with rationale -->

### Outcomes & Retrospective
<!-- What worked, what didn't, what to do differently next time -->
