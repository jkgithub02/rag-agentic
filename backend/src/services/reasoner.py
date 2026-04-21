from __future__ import annotations

import json
from typing import TypeVar

from src.core.config import Settings
from src.core.models import (
    AgentThought,
    AnswerSynthesisOutput,
    EvidenceChunk,
    GroundingCheckOutput,
    GroundingResult,
    QueryComplexity,
    QueryAnalysisOutput,
)
from src.core.prompts import (
    agent_step_planning_prompt,
    answer_prompt,
    conversation_summary_prompt,
    query_decomposition_prompt,
    query_complexity_prompt,
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
            GroundingResult(status=output.status, reason=reason, is_refusal=output.is_refusal),
            "llm",
            output.prompt_version,
        )

    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
        subqueries: list[str] | None = None,
        force_answer: bool = False,
    ) -> tuple[str, list[str], str, str | None]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but synthesize_answer was invoked.")

        selected_chunks = self._select_chunks_for_synthesis(chunks)
        evidence_block = "\n".join(
            f"- [{chunk.chunk_id}] ({chunk.source}) {chunk.text[:300]}" for chunk in selected_chunks
        )
        if not evidence_block:
            evidence_block = "- No evidence snippets available."

        # Add subquery context to prompt if decomposition was used
        if subqueries and len(subqueries) > 1:
            subquery_context = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(subqueries))
            query_text = f"{query}\n\nThis question was decomposed into:\n{subquery_context}\n\nAddress each part in your answer."
        else:
            query_text = query

        prompt = answer_prompt(query=query_text, evidence=evidence_block, force_answer=force_answer)

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

        # IMPORTANT: Prioritize web chunks (web search results) because they often contain
        # information missing from local documents (e.g., GPT info when only BERT is local)
        web_chunks = [c for c in chunks if getattr(c, "provenance", "local") == "web"]
        local_chunks = [c for c in chunks if getattr(c, "provenance", "local") == "local"]

        selected: list[EvidenceChunk] = []
        seen_ids: set[str] = set()
        source_counts: dict[str, int] = {}
        max_chunks = 8
        max_per_source = 2

        # First, include all web chunks (up to 3)
        for chunk in web_chunks[:3]:
            if chunk.chunk_id not in seen_ids:
                selected.append(chunk)
                seen_ids.add(chunk.chunk_id)
                source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1

        # Then fill remaining slots with local chunks (prioritize high scores)
        for chunk in local_chunks:
            if len(selected) >= max_chunks:
                break
            if chunk.chunk_id in seen_ids:
                continue
            # Take first 4 chunks by score, then enforce source diversity
            if len(selected) < 4 or source_counts.get(chunk.source, 0) == 0:
                selected.append(chunk)
                seen_ids.add(chunk.chunk_id)
                source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1
            elif source_counts.get(chunk.source, 0) < max_per_source:
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

    def detect_conversation_query(self, *, query: str) -> tuple[bool, float]:
        """
        Detect if a query is asking about the conversation history itself (meta-query)
        vs asking about document content.
        
        Returns: (is_conversation_query, confidence)
        """
        if not self._settings.reasoning_enabled:
            return False, 0.0
        
        from src.core.prompts import conversation_query_detection_prompt
        
        prompt = conversation_query_detection_prompt(query=query)
        try:
            parsed = self._invoke_structured_raw(prompt)
            is_conv = parsed.get("is_conversation_query", False)
            confidence = float(parsed.get("confidence", 0.5))
            return bool(is_conv), min(max(confidence, 0.0), 1.0)
        except Exception:
            # Default to document query on error
            return False, 0.0

    def detect_query_complexity(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> QueryComplexity:
        """LLM-driven complexity classifier used for smart routing."""
        if not self._settings.reasoning_enabled:
            return QueryComplexity.MODERATE

        prompt = query_complexity_prompt(
            query=query,
            conversation_summary=conversation_summary,
        )
        try:
            parsed = self._invoke_structured_raw(prompt)
            raw_value = parsed.get("query_complexity", parsed.get("complexity"))
            normalized = self._normalize_query_complexity(raw_value)
            if normalized is not None:
                return normalized
        except Exception:
            pass

        return QueryComplexity.MODERATE

    @staticmethod
    def _normalize_query_complexity(value: object) -> QueryComplexity | None:
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        mapping = {
            "simple": QueryComplexity.SIMPLE,
            "easy": QueryComplexity.SIMPLE,
            "moderate": QueryComplexity.MODERATE,
            "medium": QueryComplexity.MODERATE,
            "complex": QueryComplexity.COMPLEX,
            "hard": QueryComplexity.COMPLEX,
        }
        return mapping.get(normalized)

    def decompose_query_lightly(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> list[str]:
        """LLM-driven lightweight decomposition into searchable sub-queries."""
        max_subqueries = max(1, int(self._settings.max_decomposition_depth))

        if not self._settings.reasoning_enabled:
            return [query.strip()]

        prompt = query_decomposition_prompt(
            query=query,
            conversation_summary=conversation_summary,
            max_subqueries=max_subqueries,
        )
        try:
            parsed = self._invoke_structured_raw(prompt)
            raw_items = parsed.get("sub_queries")
            if not isinstance(raw_items, list):
                return [query.strip()]

            cleaned: list[str] = []
            seen_lower: set[str] = set()
            for item in raw_items:
                if not isinstance(item, str):
                    continue
                candidate = item.strip()
                if not candidate:
                    continue
                lower = candidate.lower()
                if lower in seen_lower:
                    continue
                seen_lower.add(lower)
                cleaned.append(candidate)
                if len(cleaned) >= max_subqueries:
                    break

            return cleaned or [query.strip()]
        except Exception:
            return [query.strip()]

    def plan_agent_step(
        self,
        *,
        query: str,
        conversation_summary: str | None,
        rewritten_queries: list[str],
        evidence_quality_score: float,
        chunk_count: int,
        agent_iterations: int,
        max_iterations: int,
        last_observation: str | None = None,
        subquery_statuses: list[dict[str, object]] | None = None,
    ) -> AgentThought:
        if not self._settings.reasoning_enabled:
            return AgentThought(
                reasoning="Reasoning disabled; defaulting to document search.",
                recommended_action="search_documents",
                confidence=0.0,
            )

        prompt = agent_step_planning_prompt(
            query=query,
            conversation_summary=conversation_summary,
            rewritten_queries=rewritten_queries,
            evidence_quality_score=evidence_quality_score,
            chunk_count=chunk_count,
            agent_iterations=agent_iterations,
            max_iterations=max_iterations,
            last_observation=last_observation,
            subquery_statuses=subquery_statuses,
        )
        try:
            parsed = self._invoke_structured_raw(prompt)
        except Exception:
            return AgentThought(
                reasoning="Planner unavailable; defaulting to document search.",
                recommended_action="search_documents",
                confidence=0.0,
            )

        raw_reasoning = parsed.get("reasoning")
        reasoning = raw_reasoning.strip() if isinstance(raw_reasoning, str) and raw_reasoning.strip() else "Plan next step using available evidence."

        raw_action = parsed.get("recommended_action")
        action = self._normalize_agent_action(raw_action)

        raw_confidence = parsed.get("confidence")
        try:
            confidence = float(raw_confidence)
        except Exception:
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        raw_target = parsed.get("target_subquery_index")
        target_subquery_index: int | None = None
        if raw_target is not None:
            try:
                target_subquery_index = int(raw_target)
            except (TypeError, ValueError):
                pass

        return AgentThought(
            reasoning=reasoning,
            recommended_action=action,
            confidence=confidence,
            target_subquery_index=target_subquery_index,
        )

    @staticmethod
    def _normalize_agent_action(value: object) -> str:
        if not isinstance(value, str):
            return "search_documents"

        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"search_documents", "search", "retrieve", "retrieval"}:
            return "search_documents"
        if normalized in {"web_search", "web", "internet", "external_search"}:
            return "web_search"
        if normalized in {"finalize", "finish", "complete", "done"}:
            return "finalize"
        return "search_documents"
