## Agentic RAG Basics (From Scratch)

Minimal baseline implemented for review only.

Flow:

`Query -> Understand -> Retrieve (tool) -> Validate -> Retry (once) -> Hydrate Evidence (tool) -> Generate -> Verify -> Answer`

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

Run tests:

```bash
uv run pytest
```

## RAGAS Comprehensive Evaluation

The system includes a comprehensive RAGAS evaluation suite with 30 questions across 7 categories:

1. **Straightforward Factual** — Baseline questions that must always pass
2. **Precise Attribution** — Single-source citation accuracy verification
3. **Cross-Document Reasoning** — Multi-source attribution and disambiguation
4. **Inference Not Just Retrieval** — Faithfulness and synthesis quality
5. **Safe-Fail / Unanswerable** — System refusal to hallucinate
6. **Ambiguous Query Rewrite** — Orchestration reasoning visibility (traces)
7. **Semantic/Lexical Mismatch** — Hybrid retrieval validation

### Run Evaluation (CLI)

With the API running locally:

```bash
cd backend
uv run python scripts/evaluate.py --api-url http://localhost:8000 --output-dir reports
```

This will:
- Run all 30 questions
- Generate `reports/ragas_comprehensive_report.json` (with embedded traces)
- Generate `reports/RAGAS_EVALUATION.md` (markdown summary)
- Generate `reports/ragas_question_bank.json` (exact test bank used)

### Run Evaluation (pytest)

For CI/automated testing:

```bash
cd backend
uv run pytest tests/test_ragas_comprehensive.py -v
```

Or run specific category:

```bash
uv run pytest tests/test_ragas_comprehensive.py --collect-only
RUN_RAGAS_EVAL=1 uv run pytest tests/test_ragas_comprehensive.py::test_ragas_comprehensive_live_run -v
```

### Report Artifacts

Reports are saved to `backend/reports/`:

- `ragas_comprehensive_report.json` — Structured results with all metrics and embedded traces
- `RAGAS_EVALUATION.md` — Human-readable summary with category breakdown
- `ragas_question_bank.json` — Export of all 30 questions and expected assertions

Each question result includes:
- Query text and category
- System response (answer + citations)
- RAGAS metrics: faithfulness, context_recall, answer_relevancy
- Expected sources vs actual citations (pass/fail/warning)
- Full trace object (original query, rewritten query, events, citations)

## Phase Quality Gates

Implementation is executed in strict phases. Every phase must pass:

- phase-targeted tests,
- full lint (`uv run ruff check .`),
- full regression (`uv run pytest`),
- and a code-smell review of changed files.

See `docs/phase-gates.md` for the complete checklist and phase gate process.

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
