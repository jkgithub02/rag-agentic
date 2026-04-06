from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_recall, faithfulness

TRANSFORMER_PDF = "Attention is all you need.pdf"
BERT_PDF = "BERT Pre-Training of Deep Bidirectional Transformers for Language Understanding.pdf"
RAG_PDF = "Retrieval Augmented Generation for for Knowledge-Intensive NLP Tasks.pdf"


@dataclass(frozen=True)
class RagasQuestion:
    id: str
    category: str
    query: str
    reference: str | None = None
    expected_sources: tuple[str, ...] = ()
    expect_safe_fail: bool = False
    expect_rewrite: bool = False


def build_question_bank() -> list[RagasQuestion]:
    """Canonical 7-category evaluation set for interview demo and regression checks."""

    return [
        # Category 1 - Straightforward Factual (Baseline)
        RagasQuestion(
            id="C1-001",
            category="straightforward_factual",
            query="What problem does the Transformer architecture solve?",
            reference=(
                "Transformer removes recurrence/convolutions from sequence modeling and "
                "uses attention to improve parallelization while modeling long-range dependencies."
            ),
            expected_sources=(TRANSFORMER_PDF,),
        ),
        RagasQuestion(
            id="C1-002",
            category="straightforward_factual",
            query="How many attention heads does the base Transformer model use?",
            reference="The base Transformer uses 8 attention heads.",
            expected_sources=(TRANSFORMER_PDF,),
        ),
        RagasQuestion(
            id="C1-003",
            category="straightforward_factual",
            query="What optimizer was used to train the Transformer?",
            reference=(
                "Transformer was trained with Adam optimizer with a custom "
                "learning-rate schedule."
            ),
            expected_sources=(TRANSFORMER_PDF,),
        ),
        RagasQuestion(
            id="C1-004",
            category="straightforward_factual",
            query="What are the two tasks BERT is pretrained on?",
            reference=(
                "BERT pretraining uses masked language modeling and next "
                "sentence prediction."
            ),
            expected_sources=(BERT_PDF,),
        ),
        RagasQuestion(
            id="C1-005",
            category="straightforward_factual",
            query="What is the difference between BERT Base and BERT Large?",
            reference=(
                "BERT Base has fewer layers/hidden size/heads than BERT Large "
                "(Base: 12 layers, 768 hidden, 12 heads; Large: 24 layers, 1024 hidden, 16 heads)."
            ),
            expected_sources=(BERT_PDF,),
        ),
        RagasQuestion(
            id="C1-006",
            category="straightforward_factual",
            query="What does the [CLS] token represent in BERT?",
            reference=(
                "[CLS] is the aggregate sequence representation token used "
                "for classification tasks."
            ),
            expected_sources=(BERT_PDF,),
        ),
        RagasQuestion(
            id="C1-007",
            category="straightforward_factual",
            query="What two components make up the RAG model?",
            reference="RAG combines a retriever and a generator (seq2seq language model).",
            expected_sources=(RAG_PDF,),
        ),
        RagasQuestion(
            id="C1-008",
            category="straightforward_factual",
            query="What dataset was used to evaluate RAG?",
            reference=(
                "RAG was evaluated on open-domain QA benchmarks including Natural Questions, "
                "WebQuestions, CuratedTREC, and TriviaQA."
            ),
            expected_sources=(RAG_PDF,),
        ),
        RagasQuestion(
            id="C1-009",
            category="straightforward_factual",
            query="What is the difference between RAG-Sequence and RAG-Token?",
            reference=(
                "RAG-Sequence conditions generation on one retrieved set for the whole sequence, "
                "while RAG-Token allows token-level changes in retrieved passages."
            ),
            expected_sources=(RAG_PDF,),
        ),
        # Category 2 - Precise Attribution
        RagasQuestion(
            id="C2-001",
            category="precise_attribution",
            query="What is the scaled dot-product formula used in attention?",
            expected_sources=(TRANSFORMER_PDF,),
        ),
        RagasQuestion(
            id="C2-002",
            category="precise_attribution",
            query="What is DPR and which paper uses it?",
            expected_sources=(RAG_PDF,),
        ),
        RagasQuestion(
            id="C2-003",
            category="precise_attribution",
            query="What is the next sentence prediction task?",
            expected_sources=(BERT_PDF,),
        ),
        RagasQuestion(
            id="C2-004",
            category="precise_attribution",
            query="What is the role of the decoder in the original sequence-to-sequence model?",
            expected_sources=(TRANSFORMER_PDF,),
        ),
        # Category 3 - Cross-Document Confusion
        RagasQuestion(
            id="C3-001",
            category="cross_document_confusion",
            query="How is attention used in BERT vs the original Transformer?",
            expected_sources=(
                TRANSFORMER_PDF,
                BERT_PDF,
            ),
        ),
        RagasQuestion(
            id="C3-002",
            category="cross_document_confusion",
            query="How does each paper handle tokenization?",
            expected_sources=(
                TRANSFORMER_PDF,
                BERT_PDF,
                RAG_PDF,
            ),
        ),
        RagasQuestion(
            id="C3-003",
            category="cross_document_confusion",
            query="What role does the encoder play across these three papers?",
            expected_sources=(
                TRANSFORMER_PDF,
                BERT_PDF,
                RAG_PDF,
            ),
        ),
        # Category 4 - Inference Not Just Retrieval
        RagasQuestion(
            id="C4-001",
            category="inference_grounding",
            query="Why did the authors of the Transformer paper remove recurrence?",
            expected_sources=(TRANSFORMER_PDF,),
        ),
        RagasQuestion(
            id="C4-002",
            category="inference_grounding",
            query="What limitation of traditional language models does BERT address?",
            expected_sources=(BERT_PDF,),
        ),
        RagasQuestion(
            id="C4-003",
            category="inference_grounding",
            query="In what scenario would RAG-Token outperform RAG-Sequence?",
            expected_sources=(RAG_PDF,),
        ),
        # Category 5 - Unanswerable / Safe-Fail
        RagasQuestion(
            id="C5-001",
            category="safe_fail_unanswerable",
            query="What GPU was used to train BERT?",
            expect_safe_fail=True,
        ),
        RagasQuestion(
            id="C5-002",
            category="safe_fail_unanswerable",
            query="Who peer-reviewed the Transformer paper?",
            expect_safe_fail=True,
        ),
        RagasQuestion(
            id="C5-003",
            category="safe_fail_unanswerable",
            query="What is GPT-4's architecture?",
            expect_safe_fail=True,
        ),
        RagasQuestion(
            id="C5-004",
            category="safe_fail_unanswerable",
            query="How does BERT perform on Malaysian language tasks?",
            expect_safe_fail=True,
        ),
        # Category 6 - Ambiguous Queries / Rewrite
        RagasQuestion(
            id="C6-001",
            category="ambiguous_rewrite",
            query="Tell me about attention",
            expect_rewrite=True,
        ),
        RagasQuestion(
            id="C6-002",
            category="ambiguous_rewrite",
            query="How does it handle context?",
            expect_rewrite=True,
        ),
        RagasQuestion(
            id="C6-003",
            category="ambiguous_rewrite",
            query="What's the training trick they use?",
            expect_rewrite=True,
        ),
        RagasQuestion(
            id="C6-004",
            category="ambiguous_rewrite",
            query="Compare the models",
            expect_rewrite=True,
        ),
        # Category 7 - Synonym / Lexical Mismatch
        RagasQuestion(
            id="C7-001",
            category="semantic_mismatch",
            query="How does the model learn word relationships?",
        ),
        RagasQuestion(
            id="C7-002",
            category="semantic_mismatch",
            query="What self-supervised objective is used?",
            expected_sources=(BERT_PDF,),
        ),
        RagasQuestion(
            id="C7-003",
            category="semantic_mismatch",
            query="How is context aggregated?",
            expected_sources=(TRANSFORMER_PDF,),
        ),
    ]


def _extract_sources(citations: list[str]) -> list[str]:
    sources: list[str] = []
    for citation in citations:
        source = citation.split("#", 1)[0].strip()
        if source and source not in sources:
            sources.append(source)
    return sources


def _extract_retrieved_contexts(trace: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    for event in trace.get("events", []):
        if event.get("stage") != "retrieve":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        hits = payload.get("hits")
        if not isinstance(hits, list):
            continue
        for hit in hits:
            if isinstance(hit, dict):
                text = hit.get("text")
                if isinstance(text, str) and text.strip():
                    contexts.append(text)
    return contexts


def _source_pass(case: RagasQuestion, actual_sources: list[str]) -> bool | None:
    if not case.expected_sources:
        return None
    expected = set(case.expected_sources)
    actual = set(actual_sources)
    # Precise-attribution style checks can be strict; other categories require coverage.
    if case.category == "precise_attribution":
        return actual == expected
    return expected.issubset(actual)


def _rewrite_detected(query: str, trace: dict[str, Any]) -> bool:
    original = trace.get("original_query")
    rewritten = trace.get("rewritten_query")
    if not isinstance(original, str) or not isinstance(rewritten, str):
        return False
    return " ".join(original.split()).lower() != " ".join(rewritten.split()).lower()


def _build_evaluator_models(
    base_url: str, embedding_model: str
) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    chat = ChatOllama(model="llama3.1", base_url=base_url, temperature=0)
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def run_comprehensive_evaluation(
    *,
    api_url: str,
    output_path: Path,
    ollama_base_url: str,
    ollama_embedding_model: str,
    request_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Run the 7-category evaluation against a live API and persist JSON report."""

    questions = build_question_bank()
    rows: list[dict[str, Any]] = []
    ragas_samples: list[SingleTurnSample] = []

    with httpx.Client(timeout=request_timeout_seconds) as client:
        for case in questions:
            thread_id = f"ragas-{case.id.lower()}"
            ask_response = client.post(
                f"{api_url.rstrip('/')}/ask",
                json={"query": case.query, "thread_id": thread_id},
            )
            ask_response.raise_for_status()
            payload = ask_response.json()

            trace_id = payload.get("trace_id")
            trace_response = client.get(f"{api_url.rstrip('/')}/trace/{trace_id}")
            trace_response.raise_for_status()
            trace = trace_response.json()

            citations = payload.get("citations", [])
            if not isinstance(citations, list):
                citations = []

            answer = payload.get("answer", "")
            if not isinstance(answer, str):
                answer = ""

            safe_fail = bool(payload.get("safe_fail", False))
            actual_sources = _extract_sources(citations)
            source_pass = _source_pass(case, actual_sources)
            rewrite_detected = _rewrite_detected(case.query, trace)
            contexts = _extract_retrieved_contexts(trace)

            rows.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "query": case.query,
                    "reference": case.reference,
                    "expected_sources": list(case.expected_sources),
                    "actual_sources": actual_sources,
                    "source_assertion_pass": source_pass,
                    "expect_safe_fail": case.expect_safe_fail,
                    "safe_fail": safe_fail,
                    "safe_fail_assertion_pass": (
                        None if not case.expect_safe_fail else safe_fail
                    ),
                    "expect_rewrite": case.expect_rewrite,
                    "rewrite_detected": rewrite_detected,
                    "rewrite_assertion_pass": (
                        None if not case.expect_rewrite else rewrite_detected
                    ),
                    "answer": answer,
                    "citations": citations,
                    "trace": trace,
                    "retrieved_contexts": contexts,
                }
            )

            ragas_samples.append(
                SingleTurnSample(
                    user_input=case.query,
                    response=answer,
                    reference=case.reference,
                    retrieved_contexts=contexts,
                )
            )

    dataset = EvaluationDataset(samples=ragas_samples)
    evaluator_llm, evaluator_embeddings = _build_evaluator_models(
        base_url=ollama_base_url,
        embedding_model=ollama_embedding_model,
    )

    metrics = [faithfulness, answer_relevancy, context_recall]
    ragas_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
        raise_exceptions=False,
    )
    ragas_rows = ragas_result.to_pandas().to_dict(orient="records")
    mean_scores = ragas_result._repr_dict  # noqa: SLF001 - best available stable export in ragas 0.4

    for index, ragas_row in enumerate(ragas_rows):
        rows[index]["ragas"] = ragas_row

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "api_url": api_url,
        "question_count": len(questions),
        "metrics": ["faithfulness", "answer_relevancy", "context_recall"],
        "mean_scores": mean_scores,
        "rows": rows,
        "categories": _category_summary(rows),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return report


def _category_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["category"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for category, items in sorted(grouped.items()):
        source_checks = [
            item["source_assertion_pass"]
            for item in items
            if item["source_assertion_pass"] is not None
        ]
        safe_fail_checks = [
            item["safe_fail_assertion_pass"]
            for item in items
            if item["safe_fail_assertion_pass"] is not None
        ]
        rewrite_checks = [
            item["rewrite_assertion_pass"]
            for item in items
            if item["rewrite_assertion_pass"] is not None
        ]

        summary.append(
            {
                "category": category,
                "count": len(items),
                "source_pass_rate": (
                    None
                    if not source_checks
                    else sum(1 for value in source_checks if value) / len(source_checks)
                ),
                "safe_fail_pass_rate": (
                    None
                    if not safe_fail_checks
                    else sum(1 for value in safe_fail_checks if value) / len(safe_fail_checks)
                ),
                "rewrite_pass_rate": (
                    None
                    if not rewrite_checks
                    else sum(1 for value in rewrite_checks if value) / len(rewrite_checks)
                ),
            }
        )
    return summary


def write_markdown_summary(report: dict[str, Any], *, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# RAGAS Evaluation Summary")
    lines.append("")
    lines.append(f"Generated: {report['generated_at_utc']}")
    lines.append(f"Questions: {report['question_count']}")
    lines.append("")
    lines.append("## Mean Metrics")
    mean_scores = report.get("mean_scores", {})
    if isinstance(mean_scores, dict):
        for key, value in mean_scores.items():
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Category Checks")
    for item in report.get("categories", []):
        category = item.get("category")
        count = item.get("count")
        lines.append(
            f"- {category}: count={count}, "
            f"source_pass={item.get('source_pass_rate')}, "
            f"safe_fail_pass={item.get('safe_fail_pass_rate')}, "
            f"rewrite_pass={item.get('rewrite_pass_rate')}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_question_bank(*, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(question) for question in build_question_bank()]
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
