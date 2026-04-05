from __future__ import annotations

import json
from typing import TypeVar

from src.core.config import Settings
from src.core.models import (
    AnswerSynthesisOutput,
    EvidenceChunk,
    GroundingCheckOutput,
    GroundingResult,
    QueryAnalysisOutput,
)
from src.core.prompts import (
    answer_prompt,
    conversation_summary_prompt,
    grounding_prompt,
    query_analysis_prompt,
)
from src.services.llm_client import BedrockChatClient, LLMInvocationError

SchemaModelT = TypeVar("SchemaModelT")


class QueryReasoner:
    """Prompt-driven reasoner with strict LLM-only behavior."""

    def __init__(self, settings: Settings, llm_client: BedrockChatClient) -> None:
        self._settings = settings
        self._llm_client = llm_client

    def summarize_conversation(self, history: list[dict[str, str]]) -> str:
        if not history:
            return ""

        turns: list[str] = []
        for item in history[-8:]:
            role = item.get("role", "user").strip().lower()
            label = "User" if role == "user" else "Assistant"
            content = item.get("content", "").strip()
            if content:
                turns.append(f"{label}: {content}")

        if not turns:
            return ""

        prompt = conversation_summary_prompt(history="\n".join(turns))
        return self._llm_client.invoke_text(prompt).strip()

    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but analyze_query was invoked.")

        prompt = query_analysis_prompt(query=query, conversation_summary=conversation_summary)
        parsed = self._invoke_structured_raw(prompt)
        parsed = self._normalize_query_analysis_payload(parsed)
        output = QueryAnalysisOutput.model_validate(parsed)

        if output.is_clear:
            rewritten = (output.rewritten_query or query).strip()
            output.rewritten_query = self._normalize_rewritten_query(
                rewritten,
                operation="analyze_query",
            )
            output.questions = [output.rewritten_query]
            output.clarification_needed = None
        else:
            output.questions = []
            output.rewritten_query = None
            output.clarification_needed = (
                output.clarification_needed.strip()
                if isinstance(output.clarification_needed, str) and output.clarification_needed.strip()
                else "Please clarify what you want to know from the uploaded documents."
            )

        return output, "llm"

    @staticmethod
    def _normalize_query_analysis_payload(payload: dict[str, object]) -> dict[str, object]:
        normalized = dict(payload)

        raw_is_clear = normalized.get("is_clear")
        if isinstance(raw_is_clear, str):
            lowered = raw_is_clear.strip().lower()
            if lowered in {"true", "yes", "1"}:
                normalized["is_clear"] = True
            elif lowered in {"false", "no", "0"}:
                normalized["is_clear"] = False

        raw_questions = normalized.get("questions")
        if isinstance(raw_questions, str):
            normalized["questions"] = [raw_questions]
        elif not isinstance(raw_questions, list):
            normalized["questions"] = []

        cleaned_questions: list[str] = []
        for item in normalized.get("questions", []):
            if isinstance(item, str) and item.strip():
                cleaned_questions.append(item.strip())
        normalized["questions"] = cleaned_questions

        raw_clarification = normalized.get("clarification_needed")
        if isinstance(raw_clarification, bool):
            normalized["clarification_needed"] = None
        elif raw_clarification is not None and not isinstance(raw_clarification, str):
            normalized["clarification_needed"] = str(raw_clarification)

        raw_rewritten = normalized.get("rewritten_query")
        if raw_rewritten is not None and not isinstance(raw_rewritten, str):
            normalized["rewritten_query"] = str(raw_rewritten)

        return normalized

    def assess_grounding(
        self,
        *,
        answer: str,
        citations: list[str],
        evidence: list[str],
    ) -> tuple[GroundingResult, str, str | None]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but assess_grounding was invoked.")

        evidence_block = "\n".join(f"- {item}" for item in evidence if item.strip())
        if not evidence_block:
            evidence_block = "- No evidence snippets available."

        prompt = grounding_prompt(answer=answer, citations=citations, evidence=evidence_block)

        parsed = self._invoke_structured_raw(prompt)
        parsed["status"] = self._normalize_grounding_status(parsed.get("status"))
        output = GroundingCheckOutput.model_validate(parsed)
        reason = output.reason.strip() or "Reasoner grounding evaluation complete."
        return (
            GroundingResult(status=output.status, reason=reason),
            "llm",
            output.prompt_version,
        )

    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[str, list[str], str, str | None]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but synthesize_answer was invoked.")

        selected_chunks = self._select_chunks_for_synthesis(chunks)
        evidence_block = "\n".join(
            f"- [{chunk.chunk_id}] ({chunk.source}) {chunk.text[:300]}" for chunk in selected_chunks
        )
        if not evidence_block:
            evidence_block = "- No evidence snippets available."

        prompt = answer_prompt(query=query, evidence=evidence_block)

        output = self._invoke_structured(prompt, AnswerSynthesisOutput)
        answer = output.answer.strip()
        if not answer:
            raise LLMInvocationError("synthesize_answer produced empty answer.")

        allowed_ids = {chunk.chunk_id for chunk in chunks}
        citation_chunk_ids = [
            chunk_id for chunk_id in output.citation_chunk_ids if chunk_id in allowed_ids
        ]
        if not citation_chunk_ids:
            citation_chunk_ids = [chunk.chunk_id for chunk in chunks[:2] if chunk.chunk_id]
        if not citation_chunk_ids:
            raise LLMInvocationError("synthesize_answer produced invalid citation ids.")

        return answer, citation_chunk_ids, "llm", output.prompt_version

    @staticmethod
    def _select_chunks_for_synthesis(chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        if not chunks:
            return []

        # Prefer cross-source coverage before adding extra chunks from the same source.
        selected: list[EvidenceChunk] = []
        seen_ids: set[str] = set()
        source_counts: dict[str, int] = {}
        max_chunks = 8
        max_per_source = 2

        for chunk in chunks:
            if len(selected) >= max_chunks:
                break
            if chunk.chunk_id in seen_ids:
                continue
            if source_counts.get(chunk.source, 0) > 0:
                continue
            selected.append(chunk)
            seen_ids.add(chunk.chunk_id)
            source_counts[chunk.source] = 1

        for chunk in chunks:
            if len(selected) >= max_chunks:
                break
            if chunk.chunk_id in seen_ids:
                continue
            if source_counts.get(chunk.source, 0) >= max_per_source:
                continue
            selected.append(chunk)
            seen_ids.add(chunk.chunk_id)
            source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1

        return selected

    def _invoke_structured(self, prompt: str, model_type: type[SchemaModelT]) -> SchemaModelT:
        parsed = self._invoke_structured_raw(prompt)
        return model_type.model_validate(parsed)

    def _invoke_structured_raw(self, prompt: str) -> dict[str, object]:
        llm_text = self._llm_client.invoke_text(prompt)
        try:
            return self._parse_json_payload(llm_text)
        except Exception as first_exc:
            repair_prompt = (
                f"{prompt}\n\n"
                "Your previous output was not valid JSON. "
                "Return only a single JSON object with no prose, no markdown, and no code fences."
            )
            repaired_text = self._llm_client.invoke_text(repair_prompt)
            try:
                return self._parse_json_payload(repaired_text)
            except Exception as second_exc:
                raise LLMInvocationError(str(second_exc)) from first_exc

    @staticmethod
    def _normalize_rewritten_query(raw_query: str, *, operation: str) -> str:
        rewritten = raw_query.strip()
        if len(rewritten) < 2:
            raise LLMInvocationError(f"{operation} produced invalid output.")
        if not rewritten.endswith("?"):
            rewritten = rewritten + "?"
        return rewritten

    @staticmethod
    def _parse_json_payload(text: str) -> dict[str, object]:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response.")
        return json.loads(stripped[start : end + 1])

    @staticmethod
    def _normalize_grounding_status(value: object) -> object:
        if not isinstance(value, str):
            return value

        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        synonyms = {
            "grounded": "supported",
            "supported": "supported",
            "partially_grounded": "partial",
            "partially_supported": "partial",
            "partial": "partial",
            "ungrounded": "unsupported",
            "not_grounded": "unsupported",
            "unsupported": "unsupported",
        }
        return synonyms.get(normalized, normalized)
