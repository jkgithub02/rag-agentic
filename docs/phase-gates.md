# Phase Gates and Code Smell Checklist

This runbook defines the required gate for every implementation phase.

## Required Gate for Every Phase

1. Implement only the approved phase scope.
2. Run phase-targeted tests first.
3. Run full checks:
   - `uv run ruff check .`
   - `uv run pytest`
4. Perform a code-smell review on changed files.
5. Do not start the next phase unless all checks are green.

## Code Smell Checklist

- Duplicated business logic introduced in new paths.
- Functions too large or doing more than one thing.
- Ambiguous naming for variables, methods, and states.
- Hidden side effects (state mutation outside explicit node boundaries).
- Weak error handling (swallowed errors or unclear failure paths).
- Unused imports, dead branches, or stale config fields.
- Magic constants not promoted to configuration.
- Trace payloads missing context for debugging.

## Phase Test Expectations

### Phase 1
- Contract tests for baseline invariants.
- Existing pipeline and upload tests stay green.

### Phase 2+
- Add focused tests for the phase behavior.
- Keep all previous tests green to prevent regressions.

## Commands

Run targeted tests:

```bash
uv run pytest tests/test_phase1_quality_contracts.py
```

Run full gate:

```bash
uv run ruff check .
uv run pytest
```
