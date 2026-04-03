## Agentic RAG Basics (From Scratch)

Minimal baseline implemented for review only.

Flow:

`Query -> Understand -> Retrieve (tool) -> Validate -> Retry (once) -> Generate -> Verify -> Answer`

## What Is Included

- Persistent local vector DB manager using Qdrant.
- Explicit agent tools (`search_chunks`, `fetch_chunks_by_ids`).
- Externalized prompts module.
- Minimal LangGraph pipeline.
- Thin FastAPI API (`/health`, `/ask`, `/trace/{trace_id}`).
- Basic tests for supported and safe-fail paths.

## Quick Start

```bash
uv sync
```

Copy `.env.example` to `.env` and set your Bedrock values.

Run API:

```bash
uv run uvicorn api.main:app --reload
```

Run tests:

```bash
uv run pytest
```

## Core Files

- `src/config.py`
- `src/db/vector_db.py`
- `src/agent/tools.py`
- `src/prompts.py`
- `src/orchestration/pipeline.py`
- `src/bootstrap.py`
- `api/main.py`
