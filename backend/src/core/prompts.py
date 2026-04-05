from __future__ import annotations

PROMPT_VERSION = "v1.0.0"


def conversation_summary_prompt(*, history: str) -> str:
    return (
        "Summarize the prior conversation in 1-2 concise sentences. "
        "Keep entities, unresolved intents, and document/topic references. "
        "Ignore greetings and off-topic chitchat. Return plain text only.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Conversation:\n{history}"
    )


def query_analysis_prompt(*, query: str, conversation_summary: str | None = None) -> str:
    summary_block = (
        f"Conversation summary:\n{conversation_summary}\n"
        if conversation_summary and conversation_summary.strip()
        else "Conversation summary:\n(none)\n"
    )
    return (
        "You are an expert query analyst and rewriter. "
        "Decide if the query is clear enough for document retrieval. "
        "If clear, rewrite it into a single self-contained retrieval query. "
        "If unclear, provide a short clarification question for the user. "
        "Use conversation summary only to resolve follow-up references (for example: him, it, that). "
        "If the query is a short confirmation follow-up (for example: yes, in available documents), "
        "inherit the referenced entity/topic from conversation summary and treat it as clear retrieval intent. "
        "If the user asks to search/retrieve from available documents, treat that as clear retrieval intent and set is_clear=true. "
        "For entity-plus-document requests (for example: summarize Jason Kong resume), prefer is_clear=true and produce a retrieval query. "
        "If user requests general knowledge outside uploaded/indexed documents, set is_clear=false and ask one concise clarification "
        "that this assistant can only answer from uploaded/indexed documents. "
        "Do not repeatedly ask the same clarification when user already asked to search available documents. "
        "Do not invent facts or broaden scope. "
        "Return JSON with keys: is_clear, rewritten_query, clarification_needed, prompt_version.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"{summary_block}"
        f"User query: {query}"
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
        "Set status to exactly one of: supported, partial, unsupported. "
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
