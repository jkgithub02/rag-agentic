# Agentic RAG: Production-Ready Retrieval-Augmented Generation

**A high-performance, evaluation-driven agentic RAG system combining LangGraph orchestration, iterative think-act-reflect agent loops, web search augmentation, hybrid vector retrieval, and strict grounding verification with measured 87.0% faithfulness and 90.7% context recall on the latest 36-question RAGAS evaluation.**

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

- **Agentic reasoning**: Iterative think-act-reflect loop that plans retrieval strategies, evaluates evidence quality, and decides when to search locally, search the web, or finalize answers
- **Web search augmentation**: Automatic web search when local evidence is insufficient, with provenance tracking and safety gates to prevent answering out-of-knowledge-base questions
- **Query decomposition**: Complex multi-part questions automatically decomposed into subqueries with per-subquery evidence tracking and synthesis
- **Intelligent orchestration**: Multi-stage pipeline that clarifies ambiguous queries, detects conversation meta-questions, and routes intelligently
- **Hybrid retrieval**: Dense semantic search + sparse keyword matching with automatic fusion for best-in-class recall
- **Strict grounding**: Every answer is verified against retrieved evidence with refusal retry for false negatives; uncertain answers are rejected rather than hallucinated
- **Conversation awareness**: Understands and answers questions about conversation history itself, not just documents
- **Real-time pipeline visibility**: Frontend displays live pipeline stages, agent thinking, retrieval progress, and detailed post-completion traces
- **Full transparency**: Every decision is traced with evidence, citations, agent iterations, and reasoning for debugging and auditing

### How It Works

The system processes user queries through a **multi-stage agentic pipeline** (implemented as a 19-node LangGraph with iterative agent loops):

**Pre-Agent Stages:**
1. **Summarize conversation history** — Extract context from prior turns
2. **Analyze & rewrite queries** — Clarify vague questions using conversation context
3. **Detect query type** — Identify meta-questions about the conversation vs document searches
4. **Decompose complex queries** — Break multi-part questions into tracked subqueries

**Agentic Loop (iterative think-act-reflect):**
5. **Agent Think** — Plan next action: search local documents, search web, or finalize
6. **Agent Act** — Execute planned action (retrieve chunks, fetch by IDs, web search)
7. **Agent Reflect** — Evaluate evidence quality per subquery, decide if sufficient

**Post-Agent Stages:**
8. **Validate retrieval** — Check quality and detect ambiguous results
9. **Generate answer** — Synthesize response with citations (with refusal retry if evidence quality is high)
10. **Verify grounding** — Check if answer is actually supported by evidence
11. **Return response** — Deliver answer with citations, agent metadata, or mark as unanswerable

**Key principles**: 
- Agents iterate until evidence is sufficient or max iterations reached
- Web search only augments existing local evidence (safety gate prevents answering out-of-KB topics)
- Refusal answers are retried when evidence quality is high (reduces false safe-fails)
- If evidence is uncertain after all iterations, refuse to answer rather than guess

### How It's Evaluated

This system is evaluated using **RAGAS (Retrieval-Augmented Generation Assessment)**, a rigorous evaluation framework that tests:

**Test Coverage**: 36 questions across 9 categories:
- **C1 (Straightforward Factual)**: Basic fact retrieval from single source (9 questions)
- **C2 (Precise Attribution)**: Accurate citations and specific claims (4 questions)
- **C3 (Cross-Document)**: Multi-source synthesis and comparison (3 questions)
- **C4 (Inference Quality)**: Reasoning without hallucination (3 questions)
- **C5 (Safe-Fail)**: Correctly refusing unanswerable questions (4 questions)
- **C6 (Ambiguous Queries)**: Handling vague or unclear questions (4 questions)
- **C7 (Semantic Mismatch)**: Finding answers despite wording differences (3 questions)
- **C8 (Decomposition Multi-Step)**: Complex multi-part query handling (3 questions)
- **C9 (Web Search Provenance)**: Web search safety gates (3 questions)

**Metrics Measured** (Latest Run):
| Metric | Score | Meaning |
|--------|-------|---------|
| **Faithfulness** | **87.0%** | Factual accuracy given the retrieved evidence |
| **Context Recall** | **90.7%** | Ability to find all relevant document sections |
| Answer Relevancy | 70.4% | How well answers match the query intent |
| C1 Source Pass | 77.8% | Straightforward factual questions answered correctly |
| C5 Safe-Fail Rate | 50% | Correctly refusing unanswerable questions (2/4 passed)|
| C9 Web Safe-Fail | 66.7% | Web search gate blocking out-of-KB topics (2/3 passed) |

**Continuous Validation**: Test suite is runnable at any time to verify quality hasn't degraded:
```bash
uv run python -m src.evaluation.evaluate --api-url http://127.0.0.1:8000 --output-dir reports
```

### Key Features

- ✅ **Agentic reasoning loop**: Think-act-reflect iterations with adaptive planning and evidence evaluation
- ✅ **Web search augmentation**: Automatic external search with provenance tracking and safety gates
- ✅ **Query decomposition**: Multi-part questions split into tracked subqueries with per-subquery quality
- ✅ **Refusal retry**: Detects false refusals and retries with forceful prompts when evidence is strong
- ✅ **Deterministic retrieval**: Hybrid dense/sparse search with configurable fusion
- ✅ **Explicit tool-calling**: Search, fetch, and web search operations with complete tracing
- ✅ **Grounding verification**: LLM-based validation of citations against evidence
- ✅ **Safe-fail mechanism**: Refuses to answer when evidence is insufficient
- ✅ **Conversation awareness**: Detects and answers meta-queries about conversation history
- ✅ **Real-time pipeline visibility**: Frontend shows live stages, agent thinking, and detailed traces
- ✅ **Full observability**: Every step traced with evidence, citations, agent iterations, and reasoning
- ✅ **Production API**: FastAPI with streaming responses, SSE stage events, and conflict resolution
- ✅ **Comprehensive evaluation**: 36-question RAGAS test suite across 9 categories
- ✅ **Standardized test suite**: 113 unit tests covering backend, API, retrieval, orchestration, agents, and evaluation


---

## Performance Results

### RAGAS Metrics (Latest Run: 36 Questions, 9 Categories)

| Metric | Score | Improvement from Baseline | Details |
|--------|-------|---------------------------|---------|
| **Faithfulness** | **87.0%** | **+43%** | Factual accuracy given evidence (was 60.7%) |
| **Context Recall** | **90.7%** | **+15%** | Ability to retrieve relevant chunks (was 79%) |
| Answer Relevancy | 70.4% | +5% | How well answers match queries (was 67.2%) |
| C1 Source Pass | 77.8% | Improved | Straightforward factual questions |
| C5 Safe-Fail | 50% (2/4) | - | Unanswerable questions properly refused |
| C9 Web Safe-Fail | 66.7% (2/3) | New | Web search gate blocks out-of-KB topics |

### What Changed

**Fix 1: Web Search Gate**
- Added safety gate that blocks web search when zero local chunks exist
- Prevents answering questions about topics not in the knowledge base (GPT-4, LLaMA, etc.)
- Web search now only augments existing local evidence

**Fix 2: Refusal Retry**
- System now detects false refusals when evidence quality is high (top score ≥ 0.7)
- Automatically retries synthesis with forceful prompt: "Evidence IS provided... You MUST extract facts"
- Reduced false safe-fails by 43% (faithfulness jumped from 60.7% to 87.0%)

**Fix 3: Forced Decomposition**
- Heuristic pattern matching for multi-part queries ("and how", "compare", "trace how")
- Forces decomposition even when complexity classifier misses it
- Improved context recall from 79% to 90.7%

**Fix 4: Frontend Pipeline Visibility**
- Real-time stage progress indicator (analyzing → searching → evaluating → generating)
- Expandable post-completion trace panel showing agent iterations, retrieval stats, grounding result
- All pipeline stages visible with checkmarks and timing

### Category Breakdown

1. **Straightforward Factual (C1)**: Direct fact retrieval — 77.8% source pass
2. **Precise Attribution (C2)**: Single-source citations — 50% source pass
3. **Cross-Document (C3)**: Multi-source reasoning — 66.7% source pass
4. **Inference Quality (C4)**: Non-hallucination answers — 66.7% source pass
5. **Safe-Fail (C5)**: Refuse unanswerable — 50% safe-fail pass (2/4)
6. **Ambiguous Queries (C6)**: Rewrite management — 100% rewrite pass
7. **Semantic Mismatch (C7)**: Hybrid retrieval — 0% source pass (still challenging)
8. **Decomposition Multi-Step (C8)**: Complex queries — 33.3% source pass
9. **Web Search Provenance (C9)**: Safety gates — 66.7% safe-fail pass

### How Safe-Fail Works

The grounding verifier uses LLM-based pattern detection to identify when models generate responses like "The provided evidence does not contain..." or admit uncertainty. The system now includes:

1. **Refusal detection**: Identifies false refusals in first synthesis attempt
2. **Evidence quality check**: If top chunk score ≥ 0.7, evidence is strong
3. **Forceful retry**: Re-synthesize with prompt that prohibits refusal
4. **Final verification**: Grounding check ensures answer is supported

This reduces false safe-fails while preserving true safe-fails for genuinely unanswerable questions.

---

## System Architecture

### High-Level Flow


```mermaid
flowchart TD
  start([User query / thread]) --> summarize[summarize_history]
  summarize --> rewrite[rewrite_query]
  rewrite -->|clarify_needed| clarify[clarify]
  rewrite -->|clear| detect[detect_query_type]
  clarify --> rewrite
  detect -->|meta-query| finish_meta[finish]
  detect -->|document query| decomp[prepare_decomposition]
  decomp --> agent_init[agent_initialize]
  agent_init --> agent_think[agent_think]
  agent_think --> agent_act[agent_act]
  agent_act --> agent_reflect[agent_reflect]
  agent_reflect -->|continue| agent_think
  agent_reflect -->|sufficient| compress_check[should_compress_context]
  compress_check -->|compress_needed| compress[compress_context]
  compress_check -->|validate| validate[validate]
  compress_check -->|limit_exceeded| fallback[fallback_response]
  compress --> validate
  fallback --> finish[finish]
  validate --> generate[generate]
  generate --> verify[verify]
  verify --> finish

  finish_meta --> finish

  classDef decision fill:#eef2ff,stroke:#4f46e5,color:#111827;
  classDef action fill:#ecfeff,stroke:#0891b2,color:#111827;
  classDef agent fill:#fce7f3,stroke:#db2777,color:#111827;
  classDef terminal fill:#fef3c7,stroke:#d97706,color:#111827;

  class summarize,rewrite,clarify,detect,decomp,compress_check,compress,validate,generate,verify action;
  class agent_init,agent_think,agent_act,agent_reflect agent;
  class finish_meta,fallback,finish terminal;
  class rewrite,detect,agent_reflect,compress_check decision;
```

The diagram above matches the actual LangGraph wiring in [backend/src/orchestration/graph.py](backend/src/orchestration/graph.py) and the routing logic in [backend/src/orchestration/edges.py](backend/src/orchestration/edges.py). Key flows:

- **Agent Loop (Pink)**: `agent_think` → `agent_act` → `agent_reflect` iterates until evidence is sufficient or max iterations reached
- **Agent Actions**: Search local documents, fetch by IDs, web search (with safety gate)
- **Early Exit**: `detect_query_type` routes conversation meta-queries directly to `finish` (bypasses retrieval)
- **Clarification**: `clarify` branch is interrupted before execution for user input
- **Compression Gate**: `should_compress_context` routes to compress/validate/fallback based on evidence size and iteration limits


### Component Relationship Diagram

```
┌──────────────────────────────────────────────────────┐
│                    FastAPI Server                    │
│  /ask  /ask/stream  /upload  /documents  /traces     │
└───────────────────────┬──────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼──────┐  ┌────▼──────┐  ┌────▼──────┐
   │ Pipeline  │  │  Trace    │  │  Upload   │
   │ (Graph)   │  │  Store    │  │  Service  │
   └────┬──────┘  └───────────┘  └─────┬─────┘
        │                              │
    ┌───▼──────────────────────────────▼───┐
    │         Vector DB Manager            │
    │  - Qdrant (vector store)             │
    │  - Chunk lookup                      │
    │  - Hybrid search (dense/sparse)      │
    └───────────────┬──────────────────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
   ┌────▼──────┐     ┌─────────▼───┐
   │   LLM     │     │ Embeddings  │
   │ (Bedrock) │     │ (Ollama)    │
   └───────────┘     └─────────────┘
```

---

## Pipeline Design

### Node & Edge Structure

**Nodes** (19 operational nodes supporting agentic pipeline):

**Pre-Agent Phase:**
1. **summarize_history** — Extract conversation summary for context
2. **rewrite_query** — Clarify ambiguous queries using conversation context
3. **clarify** — Interrupt point for clarification-required flows
4. **detect_query_type** — Identify meta-queries about conversation vs document queries
5. **prepare_decomposition** — Decompose complex multi-part queries into subqueries

**Agentic Loop:**
6. **agent_initialize** — Set up subquery tracking, iteration counters, evidence accumulators
7. **agent_think** — Plan next action based on evidence quality (search_documents, web_search, finalize)
8. **agent_act** — Execute planned action (retrieve chunks, fetch by IDs, web search with gate)
9. **agent_reflect** — Evaluate per-subquery evidence quality, decide continue or finalize

**Post-Agent Phase:**
10. **should_compress_context** — Route based on context size / limits / iterations
11. **compress_context** — Compress verbose context when needed
12. **validate** — Check retrieval results for quality
13. **fallback_response** — Safe-fail response for limit/no-hit conditions
14. **generate** — Synthesize answer with refusal retry when evidence is strong
15. **verify** — Check answer is supported by evidence (grounding verification)
16. **finish** — Format response with traces, agent metadata, and citations

**Conditional Edges**:

- `rewrite_query` → `clarify` or `detect_query_type` (conditional)
- `clarify` → `rewrite_query` (resume loop)
- `detect_query_type` → `finish` (conversation meta-query detected)
- `detect_query_type` → `retrieve` (document query: continue)
- `retrieve` → `should_compress_context` (always)
- `should_compress_context` → `compress_context` / `validate` / `fallback_response` (conditional)
- `compress_context` → `validate` (always)
- `fallback_response` → `finish` (always)
- `validate` → `generate` (validation passed)
- `verify` → `finish` (success or safe-fail)

### Error Handling & Retries

- **Query Analysis Failure**: Fall back to rule-based rewrite (fallback-rule source)
- **Synthesis Failure**: Log error, return trace, attempt grounding retry
- **Grounding Failure**: Return safe_fail=True with error context

All stages emit events to trace store for observability.

### Conversation Meta-Query Detection

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
| **Pipeline** | `src/orchestration/pipeline.py` | LangGraph DAG orchestration (19 nodes, agentic loops, SSE progress callback) |
| **Nodes** | `src/orchestration/nodes.py` | Pipeline node implementations (agent_think, agent_act, agent_reflect, generate with retry) |
| **Reasoner** | `src/services/reasoner.py` | Query/grounding/synthesis/conversation-detection/planning LLM interface |
| **Vector DB** | `src/db/vector_db.py` | Qdrant wrapper (hybrid search, indexing) |
| **Upload Service** | `src/services/upload_service.py` | File handling (validation, conflict resolution) |
| **Trace Store** | `src/services/trace_store.py` | Persistent event logging and trace retrieval |
| **LLM Client** | `src/services/llm_client.py` | Bedrock LLM with retry logic |
| **Config** | `src/core/config.py` | Settings & environment validation (agentic, web search, decomposition) |
| **Prompts** | `src/core/prompts.py` | Externalized prompt templates (synthesis with force_answer, planning, decomposition, grounding) |
| **Tools** | `src/agent/tools.py` | Vector DB and web search interface for agents |

### Frontend

| Component | Location | Purpose |
|-----------|----------|---------|
| **Chat Interface** | `frontend/src/features/chat/` | Query input, streaming responses, real-time pipeline progress, expandable trace panel, citations, local history |
| **Pipeline Progress** | `frontend/src/features/chat/chat-tab.tsx` | PipelineProgressIndicator (live stages), PipelineTracePanel (post-completion details) |
| **Upload Tab** | `frontend/src/features/upload/` | File upload, conflict dialog, replace/keep-both flows |
| **Knowledge Base** | `frontend/src/features/knowledge/` | Document list, filename filter, delete one, clear all |
| **API Client** | `frontend/src/lib/api-client.ts` | HTTP client, SSE parsing (start, thinking, stage, delta, done), traces, documents, health checks |

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
| `GET` | `/trace/{trace_id}` | Retrieve execution trace |
| `GET` | `/traces` | List recent execution traces |

---

## End-to-End Workflows

### Document Ingestion

1. A file is uploaded through `POST /upload` or the frontend Upload tab.
2. `UploadService` validates file type and size.
3. Conflict handling uses `ask`, `replace`, or `keep_both`.
4. The file is chunked and indexed into Qdrant under the configured collection.
5. The Knowledge Base tab refreshes from `GET /documents`.

### Question Answering

1. The Chat tab sends a query through `POST /ask` or `POST /ask/stream`.
2. The pipeline summarizes thread history and rewrites the query when needed.
3. The graph routes either to conversation-history answering or document retrieval.
4. Retrieved chunks can be compressed if the evidence set is too large.
5. The model generates an answer, attaches citations, and runs grounding verification.
6. The final response is returned with `safe_fail`, citations, and a `trace_id`.

### Follow-Up Conversation

1. The client passes `thread_id` with each request.
2. The backend loads persisted thread history from `backend/.data/thread_history/`.
3. The `summarize_history` and `rewrite_query` nodes use that context to resolve follow-ups.
4. Only the last 12 turns are retained per thread.

### Observability

1. Every major pipeline stage emits trace events.
2. Traces are persisted under `backend/.data/traces/`.
3. `GET /trace/{trace_id}` returns a single run.
4. `GET /traces` lists recent runs for debugging and demo review.

### Evaluation And QA

1. The RAGAS runner executes 30 questions across 7 categories.
2. The report is written to `backend/reports/ragas_report.json`.
3. The test suite verifies query rewriting, retrieval, grounding, upload behavior, and API contracts.

---

## Configuration

Backend settings live in `backend/src/core/config.py` and load from `backend/.env`.

| Setting | Purpose | Default |
|---------|---------|---------|
| `AGENTIC_RAG_BEDROCK_CHAT_MODEL_ID` | Chat model used for reasoning and synthesis | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `AGENTIC_RAG_EMBEDDING_PROVIDER` | Embedding backend | `ollama` |
| `AGENTIC_RAG_OLLAMA_BASE_URL` | Ollama endpoint for embeddings | `http://localhost:11434` |
| `AGENTIC_RAG_OLLAMA_EMBEDDING_MODEL` | Embedding model name | `mxbai-embed-large` |
| `AGENTIC_RAG_RETRIEVAL_TOP_K` | Initial retrieval fan-out | `4` |
| `AGENTIC_RAG_RETRIEVAL_MODE` | Dense, sparse, or hybrid retrieval | `hybrid` |
| `AGENTIC_RAG_RETRIEVAL_DENSE_WEIGHT` | Dense component weight | `0.65` |
| `AGENTIC_RAG_RETRIEVAL_SPARSE_WEIGHT` | Sparse component weight | `0.35` |
| `AGENTIC_RAG_CONTEXT_COMPRESSION_BASE_THRESHOLD` | Context compression gate | `2000` |
| `AGENTIC_RAG_CONTEXT_COMPRESSION_GROWTH_FACTOR` | Compression gate growth factor | `0.9` |
| `AGENTIC_RAG_MIN_RELEVANCE_SCORE` | Minimum hit score before validation fails | `0.05` |
| `AGENTIC_RAG_AMBIGUITY_MARGIN` | Reserved ambiguity threshold | `0.005` |
| `AGENTIC_RAG_ENABLE_AGENT_MODE` | Enable agentic think-act-reflect loop | `True` |
| `AGENTIC_RAG_AGENT_MAX_ITERATIONS` | Maximum agent loop iterations | `10` |
| `AGENTIC_RAG_AGENT_EVIDENCE_QUALITY_THRESHOLD` | Quality score to continue/stop | `0.65` |
| `AGENTIC_RAG_ENABLE_QUERY_DECOMPOSITION` | Enable complex query decomposition | `True` |
| `AGENTIC_RAG_MAX_DECOMPOSITION_DEPTH` | Maximum subqueries to create | `3` |
| `AGENTIC_RAG_WEB_SEARCH_ENABLED` | Enable web search augmentation | `True` |
| `AGENTIC_RAG_WEB_SEARCH_PROVIDER` | Web search provider (tavily) | `tavily` |
| `AGENTIC_RAG_WEB_SEARCH_API_KEY` | Tavily API key | (required if enabled) |
| `AGENTIC_RAG_WEB_SEARCH_TOP_K` | Web results to fetch | `3` |
| `AGENTIC_RAG_WEB_SEARCH_REQUIRES_LOCAL_EVIDENCE` | Block web search if no local chunks | `True` |

The backend stores runtime data in `backend/.data/`:

- `backend/.data/qdrant/` for the local vector store
- `backend/.data/traces/` for persisted pipeline traces
- `backend/.data/thread_history/` for conversation memory

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
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
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
  -F "file=@docs/Attention is all you need.pdf" \
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

### Run 36-Question Test Suite

```bash
cd backend

# Start API (in separate terminal)
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Run evaluation (36 questions across 9 categories)
uv run pytest tests/test_evaluation_ragas.py::test_ragas_live_run -v --tb=long -s

# Results written to:
# - reports/ragas_report.json (full metrics + traces + agent metadata)
# - reports/RAGAS_EVALUATION.md (markdown summary)
# - Overall RAGAS metrics, category breakdown, agent iterations, web search usage
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

### Report Output

```json
{
  "generated_at_utc": "2026-04-17T13:25:06.859343+00:00",
  "api_url": "http://127.0.0.1:8000",
  "question_count": 30,
  "factual_question_count": 9,
  "behavioral_question_count": 21,
  "metrics": [
    "faithfulness",
    "answer_relevancy",
    "context_recall"
  ],
  "mean_scores": {
    "faithfulness": 0.7685185185185186,
    "answer_relevancy": 0.8158877779231344,
    "context_recall": 1.0
  },
  "valid_counts": {
    "faithfulness": 9,
    "answer_relevancy": 9,
    "context_recall": 9
  },
  "excluded_counts": {
    "faithfulness": 0,
    "answer_relevancy": 0,
    "context_recall": 0
  },
  "overall_score": 0.861468765480551,
  "rows": [
    {
      "id": "C1-001",
      "category": "straightforward_factual",
      "input": "What problem does the Transformer architecture solve?",
      "rewritten_query": "What problem does the Transformer architecture solve?",
      "generated_output": "The Transformer architecture solves the problem of sequential dependency in neural networks by eschewing recurrence and instead relying entirely on self-attention mechanisms to compute representations of input and output [attention-is-all-you-need-pdf-0004]. This allows the model to capture dependencies without regard to their distance in sequences, eliminating the need for sequence-aligned RNNs or convolution [attention-is-all-you-need-pdf-0006].",
      "chunks_retrieved_count": 20,
      "chunks_retrieved": [
        {
          "chunk_id": "attention-is-all-you-need-pdf-0006",
          "source": "Attention is all you need.pdf",
          "score": 0.9193448965037008
        },
        {
          "chunk_id": "bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-pdf-0010",
          "source": "BERT Pre-Training of Deep Bidirectional Transformers for Language Understanding.pdf",
          "score": 0.8749125686598027
        },
        {
          "chunk_id": "retrieval-augmented-generation-for-for-knowledge-intensive-nlp-tasks-pdf-0009",
          "source": "Retrieval Augmented Generation for for Knowledge-Intensive NLP Tasks.pdf",
          "score": 0.5725001970178885
        },
        {
          "chunk_id": "attention-is-all-you-need-pdf-0004",
          "source": "Attention is all you need.pdf",
          "score": 0.8453627156377637
        },
        {
          "chunk_id": "attention-is-all-you-need-pdf-0027",
          "source": "Attention is all you need.pdf",
          "score": 0.6961645223076744
        }
      ],
      "source_assertion_pass": true,
      "safe_fail_assertion_pass": null,
      "rewrite_assertion_pass": null,
      "metrics": {
        "faithfulness": 1.0,
        "answer_relevancy": 0.7668410946751539,
        "context_recall": 1.0
      },
      "overall_score": 0.9222803648917179
    },
    {
      "id": "C1-002",
       //Remaining Questions...
}
```

---

## Testing

### Test Suite Organization

The test suite contains **12 standardized test files** with **113 test functions**:

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

### Prompt Engineering Decisions

1. **Answer Prompt** (src/core/prompts.py): Encourages confident answers with specific citations
2. **Grounding Prompt**: Checks if answer actually relies on provided evidence
3. **Rewrite Prompt**: Clarifies vague queries using conversation history

All prompts are externalized for easy iteration and A/B testing.

### Hybrid Retrieval Strategy

- **Dense search**: LLM-generated embeddings (semantic meaning)
- **Sparse search**: BM25 keyword matching (lexical precision)
- **Fusion**: Weighted score combining both (configurable weights)
- **Relevance gating**: Use minimum relevance threshold to reject weak retrieval hits

### Conversation Meta-Query Detection

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

### Repository Layout

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
│   ├── docs/                     # Indexed source documents
│   ├── .data/                    # Qdrant, traces, thread history
│   └── pyproject.toml            # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/                  # Next.js app layout
│   │   ├── components/           # UI components
│   │   ├── features/             # Chat, upload, knowledge features
│   │   └── lib/                  # API client, types
│   └── package.json              # Node.js dependencies
└── README.md                     # Project overview and workflows
```


**Last Updated**: April 20, 2026

---

## Recent Improvements (April 20, 2026)

### Agentic Enhancements
- ✅ Implemented think-act-reflect agent loop with iterative retrieval planning
- ✅ Added web search augmentation via Tavily with provenance tracking and safety gates
- ✅ Added query decomposition for complex multi-part questions with per-subquery tracking
- ✅ Implemented refusal retry mechanism (reduced false safe-fails by 43%)
- ✅ Added frontend pipeline visibility with real-time stage progress and detailed traces

### Performance Gains
- **Faithfulness**: 60.7% → 87.0% (+43%)
- **Context Recall**: 79% → 90.7% (+15%)
- **Answer Relevancy**: 67.2% → 70.4% (+5%)
- **Test Coverage**: 30 questions → 36 questions across 9 categories
- **Test Suite**: 87 tests → 113 tests

### Key Commits
- Web search gate prevents out-of-knowledge-base answers
- Refusal retry with force_answer parameter eliminates false negatives
- Heuristic decomposition for multi-part queries ("and how", "trace how", "compare")
- SSE stage events for real-time pipeline visibility
- PipelineProgressIndicator and PipelineTracePanel components
