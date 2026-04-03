## Agentic RAG Basics (From Scratch)

Minimal baseline implemented for review only.

Flow:

`Query -> Understand -> Retrieve (tool) -> Validate -> Retry (once) -> Generate -> Verify -> Answer`

## What Is Included

- Persistent local vector DB manager using Qdrant.
- Explicit agent tools (`search_chunks`, `fetch_chunks_by_ids`).
- Externalized prompts module.
- Minimal LangGraph pipeline.
- Thin FastAPI API (`/health`, `/ask`, `/upload`, `/trace/{trace_id}`).
- Single-file uploads (`pdf`, `txt`, `md`) with immediate indexing.
- Conflict flow with user choice: `replace` or `keep_both` using Windows-style names.
- Basic tests for supported and safe-fail paths.

## Quick Start

```bash
uv sync
```

Copy `.env.example` to `.env` and set your Bedrock chat values and Ollama embedding values.

Embeddings are configured to run through Ollama (`AGENTIC_RAG_EMBEDDING_PROVIDER=ollama`).

Run API:

```bash
uv run uvicorn api.main:app --reload
```

For local Qdrant stability, prefer single-process mode during demos:

```bash
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Run Streamlit UI:

```bash
uv run streamlit run ui/app.py
```

Run tests:

```bash
uv run pytest
```

Ollama setup (optional):

```bash
ollama pull nomic-embed-text
ollama serve
```

## Upload Behavior

- Uploads are single-file only.
- Files are stored directly under `documents/`.
- Allowed extensions: `.pdf`, `.txt`, `.md`.
- Max upload size is controlled by `AGENTIC_RAG_UPLOAD_MAX_FILE_SIZE_MB` (default `25`).

Example upload call:

```bash
curl -X POST "http://localhost:8000/upload" \
	-F "file=@documents/new-note.txt" \
	-F "conflict_policy=ask"
```

If the filename already exists and `conflict_policy=ask`, the API returns a conflict payload so the UI can prompt the user:

- `replace`: overwrite existing file and re-index it.
- `keep_both`: store as `name (1).ext`, `name (2).ext`, etc.

## Core Files

- `src/config.py`
- `src/db/vector_db.py`
- `src/agent/tools.py`
- `src/prompts.py`
- `src/orchestration/pipeline.py`
- `src/bootstrap.py`
- `src/upload_service.py`
- `api/main.py`
