# Agentic RAG: Production-Ready Retrieval-Augmented Generation

**A high-performance, evaluation-driven RAG system combining LangGraph orchestration, hybrid vector retrieval, and strict grounding logic to achieve 91.19% RAGAS accuracy.**

## Table of Contents

- [Overview](#overview)
- [Performance Results](#performance-results)
- [System Architecture](#system-architecture)
- [Pipeline Design](#pipeline-design)
- [Components](#components)
- [Quick Start](#quick-start)
- [Advanced: RAGAS Evaluation](#advanced-ragas-evaluation)
- [Testing](#testing)

---

## Overview

This system implements an **agentic RAG pipeline** with multi-phase quality improvements:

- **Phase 1**: Removed unreliable pattern matching from grounding logic (Score: 64.92% → 75.84%)
- **Phase 2**: Strengthened answer generation and grounding prompts (Score: 75.84% → 91.19%)
- **Phase 3**: Fixed C5 (safe-fail) regressions by aligning prompt expectations (C5 pass rate: 50% → 100%)

**Final Score: 91.19%** — exceeding the 80% target across 7 evaluation categories.

### Key Features

✅ **Deterministic retrieval**: Hybrid dense/sparse search with configurable fusion
✅ **Explicit tool-calling**: Search and fetch operations with tracing
✅ **Grounding verification**: LLM-based validation of citations against evidence
✅ **Safe-fail mechanism**: Refuses to answer when evidence is insufficient
✅ **Full observability**: Every step traced with evidence, citations, and reasoning
✅ **Production API**: FastAPI with streaming responses and conflict resolution
✅ **Comprehensive evaluation**: 30-question RAGAS test suite across 7 categories
✅ **Standardized test suite**: 12 core test files with 100+ assertions

---

## Performance Results

### RAGAS Metrics (91.19% Overall)

| Metric | Score | Details |
|--------|-------|---------|
| **Overall Score** | **91.19%** | Across all 30 questions |
| Answer Relevancy | 81.89% | How well answers match queries |
| Faithfulness | 91.67% | Factual accuracy given evidence |
| Context Recall | 100% | Ability to retrieve relevant chunks |
| **C5 Safe-Fail** | **100%** (4/4) | Unanswerable questions properly refused |

### Category Breakdown

1. **Straightforward Factual (C1)**: Direct fact retrieval — Foundation category
2. **Precise Attribution (C2)**: Single-source citations — Grounding accuracy
3. **Cross-Document (C3)**: Multi-source reasoning — Synthesis capability
4. **Inference Quality (C4)**: Non-hallucination answers — Reasoning bounds
5. **Safe-Fail (C5)**: Refuse unanswerable — **Fixed in Phase 3** ✅
6. **Ambiguous Queries (C6)**: Rewrite management — Orchestration intelligence
7. **Semantic Mismatch (C7)**: Hybrid retrieval — Vector representation strength

**Phase 3 Fix**: Added new refusal pattern detection in grounding_prompt to recognize answer patterns like "The provided evidence does not contain..." that the strengthened answer_prompt now generates.

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query (Thread)                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                ┌────────────▼────────────┐
                │  Query Analysis        │
                │  (clarify if needed)   │
                └────────────┬────────────┘
                             │
                ┌────────────▼─────────────────────┐
                │  Vector Retrieval (Hybrid)      │
                │  Dense: LLM embeddings          │
                │  Sparse: BM25 keyword match     │
                │  Fusion: Weighted score         │
                └────────────┬─────────────────────┘
                             │
                ┌────────────▼──────────────┐
                │  Ambiguity Detection     │
                │  (score margin check)    │
                └────────────┬──────────────┘
                             │
                ┌────────────▼──────────────────────┐
                │  Evidence Hydration               │
                │  Fetch full chunk content         │
                │  Prepare for grounding            │
                └────────────┬──────────────────────┘
                             │
                ┌────────────▼──────────────┐
                │  Answer Generation       │
                │  (LLM synthesis)         │
                │  + Citation tracking     │
                └────────────┬──────────────┘
                             │
                ┌────────────▼──────────────────┐
                │  Grounding Verification       │
                │  Check if answer supported    │
                │  by cited evidence            │
                └────────────┬──────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
      ┌─────▼─────┐                    ┌─────▼──────┐
      │  Grounded │                    │ Ungrounded │
      │  (Answer) │                    │ (Safe-Fail)│
      └──────┬────┘                    └─────┬──────┘
             │                               │
   ┌─────────▼────────────────────────────────▼────────┐
   │  Return Response (citations + trace info)         │
   └────────────────────────────────────────────────────┘
```

### Component Relationship Diagram

```
┌──────────────────────────────────────────────────────┐
│                    FastAPI Server                    │
│  /ask  /ask/stream  /upload  /documents  /traces    │
└───────────────────────┬────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼──────┐  ┌────▼──────┐  ┌────▼──────┐
   │ Pipeline  │  │  Trace    │  │  Upload   │
   │ (Graph)   │  │  Store    │  │  Service  │
   └────┬──────┘  └───────────┘  └────┬──────┘
        │                              │
    ┌───▼──────────────────────────────▼───┐
    │         Vector DB Manager             │
    │  - Qdrant (vector store)             │
    │  - Chunk lookup                      │
    │  - Hybrid search (dense/sparse)      │
    └───────────────┬──────────────────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
   ┌────▼──────┐     ┌────────▼────┐
   │   LLM     │     │ Embeddings  │
   │ (Bedrock) │     │ (Ollama)    │
   └───────────┘     └─────────────┘
```

---

## Pipeline Design

### Node & Edge Structure

**Nodes** (7 processing stages):

1. **rewrite_query** — Clarify ambiguous queries using conversation history
2. **retrieve** — Search vector DB and fetch full evidence chunks
3. **ambiguity_check** — Detect conflicting sources (similar scores)
4. **hydrate_evidence** — Retrieve full chunk content by ID
5. **synthesize** — Generate answer with LLM using evidence
6. **grounding** — Verify answer is supported by evidence
7. **augment** — Format response with traces and metadata

**Conditional Edges**:

- `ambiguity_check` → `hydrate_evidence` (normal flow)
- `ambiguity_check` → Return `safe_fail=True` (ambiguous: abstain)
- `grounding` → Return `safe_fail=True` (unsupported: refuse)
- `grounding` → Return `safe_fail=False` (supported: answer)

### Error Handling & Retries

- **Query Analysis Failure**: Fall back to rule-based rewrite (fallback-rule source)
- **Synthesis Failure**: Log error, return trace, attempt grounding retry
- **Grounding Failure**: Return safe_fail=True with error context

All stages emit events to trace store for observability.

---

## Components

### Backend Services

| Component | Location | Purpose |
|-----------|----------|---------|
| **Pipeline** | `src/orchestration/pipeline.py` | LangGraph DAG orchestration (7-stage flow) |
| **Reasoner** | `src/services/reasoner.py` | Query/grounding/synthesis LLM interface |
| **Vector DB** | `src/db/vector_db.py` | Qdrant wrapper (hybrid search, indexing) |
| **Upload Service** | `src/services/upload_service.py` | File handling (validation, conflict resolution) |
| **Trace Store** | `src/services/trace_store.py` | Persistent event logging and trace retrieval |
| **LLM Client** | `src/services/llm_client.py` | Bedrock LLM with retry logic |
| **Config** | `src/core/config.py` | Settings & environment validation |
| **Prompts** | `src/core/prompts.py` | Externalized prompt templates (rewrite, grounding, synthesis) |
| **Tools** | `src/agent/tools.py` | Vector DB interface for agents |

### Frontend

| Component | Location | Purpose |
|-----------|----------|---------|
| **Chat Interface** | `frontend/src/features/chat/` | Query input, streaming responses |
| **Upload Tab** | `frontend/src/features/upload/` | File upload with conflict dialog |
| **Knowledge Base** | `frontend/src/features/knowledge/` | Document list & deletion |
| **API Client** | `frontend/src/lib/api-client.ts` | HTTP client for backend API |

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Service availability check |
| `POST` | `/ask` | Single synchronous query |
| `POST` | `/ask/stream` | Streaming response (SSE) |
| `POST` | `/upload` | Single file upload |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{filename}` | Delete document |
| `DELETE` | `/documents` | Delete all documents |
| `GET` | `/traces/{trace_id}` | Retrieve execution trace |

---

## Quick Start

### 1. Setup Python Environment

```bash
# Install dependencies
cd backend
uv sync

# Create .env from template
cp .env.example .env

# Configure environment:
# - AGENTIC_RAG_LLM_PROVIDER=bedrock (or openai, ollama)
# - AGENTIC_RAG_EMBEDDING_PROVIDER=ollama (or bedrock, openai)
# - AWS credentials for Bedrock (if used)
```

### 2. Start Local Services

```bash
# Terminal 1: Start Ollama (for embeddings)
ollama pull mxbai-embed-large
ollama serve

# Terminal 2: Start API server
cd backend
uv run python api/main.py
# Server runs on http://127.0.0.1:8000

# Terminal 3: Start Frontend (optional)
cd frontend
npm run dev
# UI runs on http://localhost:3000
```

### 3. Test Health

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

### 4. Upload Documents

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@documents/research-paper.pdf" \
  -F "conflict_policy=ask"
```

### 5. Query the System

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "thread_id": "session-123"
  }'
```

Response:
```json
{
  "answer": "...",
  "citations": ["doc-0001", "doc-0002"],
  "safe_fail": false,
  "trace_id": "trace-abc123"
}
```

---

## Advanced: RAGAS Evaluation

### Run 30-Question Test Suite

```bash
cd backend

# Start API (in separate terminal)
uv run python api/main.py

# Run evaluation (30 questions across 7 categories)
uv run python -m src.evaluation.evaluate \
  --api-url http://127.0.0.1:8000 \
  --output-dir reports

# Results:
# - reports/ragas_report.json (full metrics + traces)
# - Overall score, category breakdown, C5 safe-fail results
```

### Via pytest (CI/Automated)

```bash
# List evaluation tests
uv run pytest tests/test_evaluation_ragas.py --collect-only

# Run unit tests (contract validation)
uv run pytest tests/test_evaluation_ragas.py -v -k "not live"

# Run actual evaluation (requires running API)
uv run pytest tests/test_evaluation_ragas.py::test_question_bank_covers_all_requested_categories -v
```

### Understanding Report Output

```json
{
  "overall_score": 0.9119,
  "mean_scores": {
    "answer_relevancy": 0.8189,
    "faithfulness": 0.9167,
    "context_recall": 1.0
  },
  "questions": [
    {
      "id": "C1-001",
      "query": "What is BERT?",
      "category": "straightforward_factual",
      "answer_relevancy_assertion_pass": true,
      "faithfulness_assertion_pass": true,
      "context_recall_assertion_pass": true,
      "trace": { /* full execution trace */ }
    }
    // ... 29 more questions
  ]
}
```

---

## Testing

### Test Suite Organization

The test suite contains **12 standardized test files** with **100+ assertions**:

```
tests/
├── conftest.py                              # Shared fake fixtures
├── test_core_config_contracts.py            # Settings & model contracts
├── test_services_llm_client.py              # LLM retry logic
├── test_services_reasoner.py                # Query/grounding/synthesis
├── test_services_upload.py                  # File handling & validation
├── test_services_trace_store.py             # Trace persistence
├── test_db_vector_search.py                 # Hybrid retrieval
├── test_api_upload.py                       # Upload endpoints
├── test_api_observability.py                # Trace endpoints
├── test_orchestration_pipeline.py           # Pipeline orchestration
├── test_integration_document_access.py      # End-to-end access flow
└── test_evaluation_ragas.py                 # RAGAS evaluation suite
```

### Run All Tests

```bash
cd backend

# Quick smoke test (30s)
uv run pytest tests/test_core_config_contracts.py tests/test_services_llm_client.py -v

# Full test suite (60-120s depending on integration tests)
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_orchestration_pipeline.py -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Test Organization by Layer

| Layer | Files | Coverage |
|-------|-------|----------|
| Config | `test_core_config_contracts.py` | Settings, enums, validation |
| Services | `test_services_*.py` (5 files) | LLM, reasoning, uploads, traces |
| Database | `test_db_vector_search.py` | Hybrid search, indexing |
| API | `test_api_*.py` (2 files) | Endpoints, request/response |
| Orchestration | `test_orchestration_pipeline.py` | DAG flow, error handling |
| Integration | `test_integration_*.py` | End-to-end scenarios |
| Evaluation | `test_evaluation_ragas.py` | RAGAS suite, quality gates |

### Key Test Examples

```bash
# Verify grounding normalization
uv run pytest tests/test_services_reasoner.py::test_assess_grounding_normalizes_grounded_to_supported -v

# Verify safe-fail for unanswerable questions (C5)
uv run pytest tests/test_orchestration_pipeline.py::test_safe_fail_path -v

# Verify upload file size validation (5 new tests added)
uv run pytest tests/test_services_upload.py::test_upload_rejects_files_exceeding_max_size -v

# Verify malformed JSON handling
uv run pytest tests/test_services_reasoner.py::test_assess_grounding_handles_malformed_json -v
```

---

## Implementation Details

### Grounding Logic (Phase 3 Fix)

**Problem**: Strengthened answer_prompt told model "Don't use hedging phrases like 'I do not have sufficient evidence'", but grounding_prompt only recognized old phrases.

**Solution**: Added new refusal pattern detection:
- "The provided evidence does not contain"
- "does not provide information about"
- "evidence does not discuss"

**Result**: C5 safe-fail pass rate improved 50% → 100%

### Prompt Engineering Decisions

1. **Answer Prompt** (src/core/prompts.py): Encourages confident answers with specific citations
2. **Grounding Prompt**: Checks if answer actually relies on provided evidence
3. **Rewrite Prompt**: Clarifies vague queries using conversation history

All prompts are externalized for easy iteration and A/B testing.

### Hybrid Retrieval Strategy

- **Dense search**: LLM-generated embeddings (semantic meaning)
- **Sparse search**: BM25 keyword matching (lexical precision)
- **Fusion**: Weighted score combining both (configurable weights)
- **Ambiguity detection**: If top 2 chunks have similar scores, mark as ambiguous

### File Organization

```
agentic-RAG/
├── backend/
│   ├── api/                      # FastAPI handlers
│   ├── src/
│   │   ├── core/                 # Config, models, prompts
│   │   ├── db/                   # Vector DB manager
│   │   ├── agent/                # Tool definitions
│   │   ├── services/             # LLM, reasoner, upload, trace
│   │   └── orchestration/        # Pipeline DAG & nodes
│   ├── tests/                    # Pytest suite (12 files)
│   ├── reports/                  # RAGAS evaluation results
│   └── pyproject.toml            # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/                  # Next.js app layout
│   │   ├── components/           # UI components
│   │   ├── features/             # Chat, upload, knowledge features
│   │   └── lib/                  # API client, types
│   └── package.json              # Node.js dependencies
└── documents/                    # User-uploaded files
```

---

## Troubleshooting

### API won't start

```bash
# Check Qdrant availability
pip install qdrant-client
# Ensure embedding provider is running (Ollama)
ollama serve

# Clear corrupted DB
rm -rf backend/.qdrant
```

### Evaluation scores are low

1. Check evidence relevance: `curl http://localhost:8000/documents`
2. Verify LLM is responding: Check traces in reports
3. Review "faithfulness" failures: May indicate hallucination in generation
4. Review "context_recall": May indicate weak retrieval

### Upload file limits

```bash
# Increase max size (env var)
export AGENTIC_RAG_UPLOAD_MAX_FILE_SIZE_MB=50

# Restart API
uv run python api/main.py
```

---

## Future Improvements

- [ ] Multi-document summarization for large collections
- [ ] User feedback loop for reranking
- [ ] Streaming grounding verification
- [ ] A/B testing framework for prompt variations
- [ ] Semantic caching for repeated queries
- [ ] Advanced RAG patterns (Chain-of-Thought, ReAct)

---

## References

- **RAGAS**: [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)
- **LangGraph**: [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- **Qdrant**: [qdrant.tech](https://qdrant.tech)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

---

**Last Updated**: April 18, 2026  
