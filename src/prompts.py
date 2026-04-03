from __future__ import annotations

PROMPT_VERSION = "v1.0.0"


def rewrite_query_prompt(query: str) -> str:
    return (
        "Rewrite the query for retrieval clarity while preserving user intent. "
        "Return JSON: {\"rewritten_query\": \"...\", \"prompt_version\": \"...\"}.\n"
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
