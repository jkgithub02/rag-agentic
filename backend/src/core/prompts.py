from __future__ import annotations

PROMPT_VERSION = "v1.0.0"


def rewrite_query_prompt(query: str) -> str:
    return (
        "Rewrite the query for retrieval clarity while preserving user intent. "
        "Return JSON: {\"rewritten_query\": \"...\", \"prompt_version\": \"...\"}.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Query: {query}"
    )


def query_clarity_prompt(*, query: str) -> str:
    return (
        "Decide whether this query needs clarification before retrieval. "
        "Return JSON with keys clarify_needed, reason, prompt_version.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Query: {query}"
    )


def retry_rewrite_prompt(*, original_query: str, retry_reason: str, evidence: str) -> str:
    return (
        "Rewrite the query for a retry retrieval pass. "
        "Return JSON: {\"rewritten_query\": \"...\", \"prompt_version\": \"...\"}.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Original query: {original_query}\n"
        f"Retry reason: {retry_reason}\n"
        f"Evidence:\n{evidence}"
    )


def answer_prompt(*, query: str, evidence: str) -> str:
    return (
        "Answer strictly from evidence. "
        "Return JSON with keys answer, citation_chunk_ids, prompt_version.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Query: {query}\n"
        f"Evidence:\n{evidence}"
    )


def grounding_prompt(*, answer: str, citations: list[str], evidence: str) -> str:
    return (
        "Check if answer is grounded in evidence. "
        "Return JSON with keys status, reason, prompt_version.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Answer: {answer}\n"
        f"Citations: {citations}\n"
        f"Evidence:\n{evidence}"
    )


def coverage_check_prompt(*, query: str, answer: str, evidence: str) -> str:
    return (
        "Decide if the answer should be marked unsupported "
        "because evidence coverage is insufficient. "
        "Use only the query, answer, and evidence. "
        "Return JSON with keys unsupported, missing_terms, reason, prompt_version.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Query: {query}\n"
        f"Answer: {answer}\n"
        f"Evidence:\n{evidence}"
    )


def naturalize_response_prompt(
    *,
    category: str,
    query: str,
    reason: str | None,
    evidence_count: int,
) -> str:
    return (
        "Write one short natural response for the user. "
        "Do not use JSON. Do not mention internal pipelines, retries, or validation states. "
        "Keep it direct and helpful.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Category: {category}\n"
        f"User query: {query}\n"
        f"Reason: {reason or 'N/A'}\n"
        f"Evidence count: {evidence_count}"
    )
