## Agentic RAG (Interview Build)

This repository implements a clean, maintainable Agentic RAG baseline with the exact flow:

`Query -> Understand -> Retrieve -> Validate -> Retry if needed -> Generate -> Verify grounding -> Answer`

The build uses `uv` for project and dependency management, a thin FastAPI service, a Streamlit demo, and RAGAS-based evaluation.

## Locked Corpus

The ingestion pipeline expects PDF sources in the local `documents/` directory:

- `attention is all you need.pdf`
- `bert.pdf`
- `RAG.pdf`

## Quick Start

### 1) Install dependencies

```bash
uv sync
```

### 2) Configure environment

Copy `.env.example` to `.env` and update Bedrock settings if needed.

### 3) Build the retrieval index

```bash
uv run python -m agentic_rag.ingestion
```

### 4) Start API

```bash
uv run uvicorn api.main:app --reload
```

API endpoints:

- `GET /health`
- `POST /ask`
- `GET /trace/{trace_id}`

### 5) Start Streamlit demo

```bash
uv run streamlit run ui/app.py
```

The UI includes a trace panel showing original vs rewritten query side-by-side, retry trigger state, and grounding verdict.

## Development Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run python scripts/evaluate_ragas.py
```

## Architecture Notes

- `src/agentic_rag/config.py`: settings and reproducibility controls.
- `src/agentic_rag/ingestion.py`: PDF extraction, chunking, and index persistence.
- `src/agentic_rag/retrieval.py`: deterministic lexical retrieval baseline.
- `src/agentic_rag/pipeline.py`: agentic control loop, retry policy, safe-fail behavior.
- `src/agentic_rag/trace_store.py`: in-memory trace storage.
- `api/main.py`: thin FastAPI layer.
- `ui/app.py`: Streamlit interface with traceability.
- `scripts/evaluate_ragas.py`: RAGAS evaluation runner.

## RAGAS Evaluation

Benchmark seed data is provided at `data/benchmark.json`.

Run:

```bash
uv run python scripts/evaluate_ragas.py
```

Output is written to `reports/ragas_report.json`.

## Scope and Maintainability

- Single retry cap to keep behavior deterministic during demos.
- Explicit safe-fail response when evidence is insufficient.
- Shared service path used by API and UI to avoid drift.
- Typed models and modular boundaries for maintainable extension.
