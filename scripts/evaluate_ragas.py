from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_aws import BedrockEmbeddings, ChatBedrock
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    _AnswerRelevancy,
    _Faithfulness,
    _LLMContextPrecisionWithReference,
    _LLMContextRecall,
)

from agentic_rag.bootstrap import get_pipeline, get_trace_store
from agentic_rag.config import get_settings


def _load_benchmark(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark payload must be a list of objects")
    return payload


def _latest_contexts(trace_payload: dict[str, Any]) -> list[str]:
    events = trace_payload.get("events", [])
    for event in reversed(events):
        stage = event.get("stage")
        if stage not in {"retrieve", "retrieve_retry"}:
            continue

        hits = event.get("payload", {}).get("hits", [])
        return [hit.get("text", "") for hit in hits if hit.get("text")]
    return []


def _build_bedrock_wrappers() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    settings = get_settings()
    try:
        chat_model = ChatBedrock(
            model_id=settings.bedrock_chat_model_id,
            region_name=settings.aws_region,
            model_kwargs={"temperature": 0},
        )
        embedding_model = BedrockEmbeddings(
            model_id=settings.bedrock_embedding_model_id,
            region_name=settings.aws_region,
        )
    except Exception as exc:  # pragma: no cover - depends on machine credentials.
        raise RuntimeError(
            "Unable to initialize Bedrock for RAGAS evaluation. Configure AWS credentials "
            "and verify AGENTIC_RAG_AWS_REGION, AGENTIC_RAG_BEDROCK_CHAT_MODEL_ID, "
            "and AGENTIC_RAG_BEDROCK_EMBEDDING_MODEL_ID."
        ) from exc

    return LangchainLLMWrapper(chat_model), LangchainEmbeddingsWrapper(embedding_model)


def run(benchmark_path: Path, output_path: Path, limit: int | None) -> None:
    benchmark_rows = _load_benchmark(benchmark_path)
    if limit is not None:
        benchmark_rows = benchmark_rows[:limit]

    pipeline = get_pipeline()
    trace_store = get_trace_store()

    eval_rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        query = row["query"]
        ground_truth = row["ground_truth"]

        response = pipeline.ask(query)
        trace = trace_store.get(response.trace_id)
        trace_payload = trace.model_dump(mode="json") if trace is not None else {}

        eval_rows.append(
            {
                "question": query,
                "answer": response.answer,
                "contexts": _latest_contexts(trace_payload),
                "ground_truth": ground_truth,
            }
        )

    dataset = Dataset.from_list(eval_rows)
    ragas_llm, ragas_embeddings = _build_bedrock_wrappers()
    metrics = [
        _LLMContextPrecisionWithReference(llm=ragas_llm),
        _LLMContextRecall(llm=ragas_llm),
        _AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        _Faithfulness(llm=ragas_llm),
    ]

    try:
        result = evaluate(dataset=dataset, metrics=metrics)
    except Exception as exc:  # pragma: no cover - depends on remote model access.
        raise RuntimeError(
            "RAGAS evaluation failed during metric execution. Verify AWS credentials, "
            "Bedrock model access, and network connectivity."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "rows": result.to_pandas().to_dict(orient="records"),
                "mean_scores": result.to_pandas().mean(numeric_only=True).to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"RAGAS report written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Agentic RAG pipeline with RAGAS")
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/ragas_report.json"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.benchmark, args.output, args.limit)
