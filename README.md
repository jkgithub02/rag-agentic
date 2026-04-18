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

### What is This System?

Agentic RAG is a **production-ready Retrieval-Augmented Generation system** designed to answer questions grounded in uploaded documents with high accuracy and reliability. It combines:

- **Intelligent orchestration**: Multi-stage pipeline that clarifies ambiguous queries, detects conversation meta-questions, and routes intelligently
- **Hybrid retrieval**: Dense semantic search + sparse keyword matching with automatic fusion for best-in-class recall
- **Strict grounding**: Every answer is verified against retrieved evidence; uncertain answers are rejected rather than hallucinated
- **Conversation awareness**: Understands and answers questions about conversation history itself, not just documents
- **Full transparency**: Every decision is traced with evidence, citations, and reasoning for debugging and auditing

### How It Works

The system processes user queries through an 8-stage pipeline:

1. **Summarize conversation history** — Extract context from prior turns
2. **Analyze & rewrite queries** — Clarify vague questions using conversation context
3. **Detect query type** — Identify meta-questions about the conversation vs document searches
4. **Retrieve evidence** — Search vector DB with hybrid dense/sparse retrieval
5. **Validate retrieval** — Check quality and detect ambiguous results
6. **Generate answer** — Synthesize response with specific citations
7. **Verify grounding** — Check if answer is actually supported by evidence
8. **Return response** — Deliver answer with citations or mark as unanswerable

**Key principle**: If evidence is uncertain, refuse to answer rather than guess.

### How It's Evaluated

This system is evaluated using **RAGAS (Retrieval-Augmented Generation Assessment)**, a rigorous evaluation framework that tests:

**Test Coverage**: 30 questions across 7 categories:
- **C1 (Straightforward Factual)**: Basic fact retrieval from single source
- **C2 (Precise Attribution)**: Accurate citations and specific claims
- **C3 (Cross-Document)**: Multi-source synthesis and comparison
- **C4 (Inference Quality)**: Reasoning without hallucination
- **C5 (Safe-Fail)**: Correctly refusing unanswerable questions ✅ 100% pass rate
- **C6 (Ambiguous Queries)**: Handling vague or unclear questions
- **C7 (Semantic Mismatch)**: Finding answers despite wording differences

**Metrics Measured**:
| Metric | Score | Meaning |
|--------|-------|---------|
| **Overall Score** | **91.19%** | Weighted average across all categories |
| Answer Relevancy | 81.89% | How well answers match the query intent |
| Faithfulness | 91.67% | Factual accuracy given the retrieved evidence |
| Context Recall | 100% | Ability to find all relevant document sections |
| Safe-Fail Rate | 100% | Correctly refusing impossible questions |

**Continuous Validation**: Test suite is runnable at any time to verify quality hasn't degraded:
```bash
uv run python -m src.evaluation.evaluate --api-url http://127.0.0.1:8000 --output-dir reports
```

### Key Features

✅ **Deterministic retrieval**: Hybrid dense/sparse search with configurable fusion
✅ **Explicit tool-calling**: Search and fetch operations with complete tracing
✅ **Grounding verification**: LLM-based validation of citations against evidence
✅ **Safe-fail mechanism**: Refuses to answer when evidence is insufficient
✅ **Conversation awareness**: Detects and answers meta-queries about conversation history
✅ **Full observability**: Every step traced with evidence, citations, and reasoning
✅ **Production API**: FastAPI with streaming responses and conflict resolution
✅ **Comprehensive evaluation**: 30-question RAGAS test suite across 7 categories
✅ **Standardized test suite**: 12 core test files with 100+ assertions

### Production Readiness

This system has been optimized for production use through:
- **Strict grounding**: Rejects hallucinations with RAGAS-verified safe-fail mechanism
- **Full tracing**: Every decision logged with evidence and reasoning for audits
- **Error handling**: Graceful degradation with fallback responses
- **Streaming API**: Real-time token delivery for responsive UX
- **Conflict resolution**: Smart handling of duplicate documents and version conflicts
- **Conversation continuity**: Maintains context across turns with smart history management

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
5. **Safe-Fail (C5)**: Refuse unanswerable — Answers only when evidence exists
6. **Ambiguous Queries (C6)**: Rewrite management — Orchestration intelligence
7. **Semantic Mismatch (C7)**: Hybrid retrieval — Vector representation strength

**How Safe-Fail Works**: The grounding verifier uses LLM-based pattern detection to identify when models generate responses like "The provided evidence does not contain..." or admit uncertainty. These are converted to safe-fail rejections, ensuring users never receive answers without proper evidence.

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
                ┌────────────▼──────────────────────┐
                │  Detect Query Type               │
                │  (document vs conversation meta) │
                └────────────┬──────────────────────┘
                             │
            ┌────────────────┴─────────────────┐
            │                                  │
       ┌────▼─────┐             ┌─────────────▼───────────┐
       │Document  │             │ Conversation Meta-Query │
       │Query     │             │ (Answer from history)   │
       └────┬─────┘             └──────────────┬──────────┘
            │                                  │
            ├─────────┬──────────────────┬─────┤
            │                            │     │
   ┌────────▼────────────────────────────▼─┐  │ Answer → Finish
   │  Vector Retrieval (Hybrid)            │  │
   │  Dense: LLM embeddings                │  │
   │  Sparse: BM25 keyword match           │  │
   │  Fusion: Weighted score               │  │
   └────────┬──────────────────────────────┘  │
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

**Nodes** (8 processing stages):

1. **summarize_history** — Extract conversation summary for context
2. **rewrite_query** — Clarify ambiguous queries using conversation context
3. **detect_query_type** — Identify meta-queries about conversation (NEW)
4. **retrieve** — Search vector DB and fetch full evidence chunks
5. **validate** — Check retrieval results for quality/ambiguity
6. **generate** — Synthesize answer with LLM using evidence
7. **verify** — Check answer is supported by evidence (grounding)
8. **finish** — Format response with traces and metadata

**Conditional Edges**:

- `rewrite_query` → `detect_query_type` (always)
- `detect_query_type` → `finish` (conversation meta-query detected)
- `detect_query_type` → `retrieve` (document query: continue)
- `validate` → `generate` (results valid: proceed)
- `validate` → `fallback` (no results: safe-fail)
- `verify` → `finish` (success or safe-fail)

### Error Handling & Retries

- **Query Analysis Failure**: Fall back to rule-based rewrite (fallback-rule source)
- **Synthesis Failure**: Log error, return trace, attempt grounding retry
- **Grounding Failure**: Return safe_fail=True with error context

All stages emit events to trace store for observability.

### Conversation Meta-Query Detection (Phase 4)

**What is a conversation meta-query?**
- Questions about the **conversation itself**, not about documents
- Examples: "What have you been asking?", "Summarize our discussion", "What topics have we covered?"

**How it works:**

1. **Smart Detection**: An LLM-based detector (not hardcoded patterns) analyzes the rewritten query
   - Returns: `(is_conversation_query: bool, confidence: 0.0-1.0)`
   - Confidence threshold: 0.5 (lenient for UX)

2. **Early Exit**: If detected as a meta-query, the system skips document retrieval entirely
   - Avoids wasting time searching vector DB
   - Answers directly from conversation history

3. **Grounding as "SUPPORTED"**: Answers are marked as grounded since they come from your actual conversation
   - Won't be rejected by safe-fail checks

**Example Flow:**

```
User: "Can you summarise what I have been asking you?"
         ↓
[Query Analysis] → rewrite_query: "Can you summarise what I have been asking you?"
         ↓
[Type Detection] → is_conversation_query: true (confidence: 0.92)
         ↓
[Answer from History] → "Based on our conversation, you've asked about: 
                         1. Transformer architecture
                         2. BERT pretraining
                         3. RAG systems"
         ↓
[Done] → Return answer with grounding = SUPPORTED
```

**Benefits:**
- ✅ Faster responses for conversation questions (no retrieval latency)
- ✅ More natural feel (answers about conversation feel less "hallucinated")
- ✅ Reduces false document retrievals
- ✅ Extensible: LLM learns patterns, doesn't require code changes for new meta-query types

All stages emit events to trace store for observability.

---

## Components

### Backend Services

| Component | Location | Purpose |
|-----------|----------|---------|
| **Pipeline** | `src/orchestration/pipeline.py` | LangGraph DAG orchestration (8-stage flow with conversation detection) |
| **Reasoner** | `src/services/reasoner.py` | Query/grounding/synthesis/conversation-detection LLM interface |
| **Vector DB** | `src/db/vector_db.py` | Qdrant wrapper (hybrid search, indexing) |
| **Upload Service** | `src/services/upload_service.py` | File handling (validation, conflict resolution) |
| **Trace Store** | `src/services/trace_store.py` | Persistent event logging and trace retrieval |
| **LLM Client** | `src/services/llm_client.py` | Bedrock LLM with retry logic |
| **Config** | `src/core/config.py` | Settings & environment validation |
| **Prompts** | `src/core/prompts.py` | Externalized prompt templates (rewrite, grounding, synthesis, conversation-detection) |
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

### Conversation Meta-Query Detection (Phase 4)

**Implementation**:

1. **Detection Prompt** (`conversation_query_detection_prompt()`):
   - Provides clear examples of conversation queries vs document queries
   - Returns JSON: `{is_conversation_query: bool, confidence: 0.0-1.0}`
   - LLM-powered (learns semantic intent, not pattern-based)

2. **Early Pipeline Insertion**:
   - Runs in `detect_query_type` node after query rewriting
   - Confidence threshold: 0.5 (lenient to avoid false negatives)
   - If detected: synthesize answer from last 10 conversation turns, bypass retrieval

3. **Grounding Handling**:
   - Conversation answers marked as `SUPPORTED` (won't trigger safe-fail)
   - Gracefully handles: empty history, first interaction, synthesis errors
   - Rich answers: "Based on our conversation, you asked about..."

4. **Why LLM-based (not pattern-matching)**:
   - ✅ Handles varied phrasings: "What have we discussed?" vs "Recap our talk" vs "What topics did we cover?"
   - ✅ Adaptive: LLM learns patterns without code changes
   - ✅ Composable: Same prompt logic can detect other query types (helpfulness, clarification-needed, etc.)
   - ✅ UX-friendly: Faster responses, feels more "alive"

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


**Last Updated**: April 18, 2026  
