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
        "If clear, rewrite it into a single self-contained retrieval query that stays close to the user's wording. "
        "Also return a questions list containing one or more self-contained retrieval questions. "
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
        "Do not add new entities, documents, or assumptions during rewriting. "
        "For comparison questions, preserve all referenced documents and the comparison intent. "
        "Return JSON with keys: is_clear, questions, rewritten_query, clarification_needed, prompt_version.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"{summary_block}"
        f"User query: {query}"
    )


def answer_prompt(*, query: str, evidence: str) -> str:
    return (
        "Answer the query from the provided evidence in 1-3 concise sentences. "
        "Be helpful, direct, and confident. Only refuse if evidence is completely absent or irrelevant. "
        "When relevant evidence exists, provide an answer even if partial or requiring reasonable inference. "
        "Always cite specific source chunks: explicitly mention chunk_ids like [chunk_id_0012] or similar. "
        "Do NOT use hedging like 'I do not have sufficient evidence' if you have any relevant evidence. "
        "Do NOT apologize or express uncertainty in the answer text itself. "
        "Return JSON with keys answer, citation_chunk_ids, prompt_version.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Query: {query}\n"
        f"Evidence:\n{evidence}"
    )


def grounding_prompt(*, answer: str, citations: list[str], evidence: str) -> str:
    return (
        "Evaluate whether the answer is grounded in evidence. "
        "Set status to exactly one of: supported, partial, unsupported. "
        "Use 'supported' if answer is directly stated or clearly derivable from evidence. "
        "Use 'partial' if answer is grounded in relevant evidence with reasonable inference, multi-document synthesis, or acceptable gaps in detail (e.g., technical specifics, exact figures). "
        "Use 'unsupported' only if: (1) answer contradicts evidence, (2) answer contains factual hallucinations, (3) answer is an explicit refusal, or (4) evidence is completely empty/irrelevant/gibberish. "
        "A refusal explicitly states inability to answer or lack of information. Examples: 'I do not have sufficient evidence', 'No information about X is available', 'cannot answer this', 'not found in the evidence', 'The provided evidence does not contain', 'does not provide information about', 'evidence does not discuss'. "
        "Set is_refusal=true only if answer explicitly refuses to answer or states that information is unavailable/not provided/not discussed. If is_refusal=true, set status=unsupported. "
        "IMPORTANT GUIDELINES: "
        "- Do NOT mark unsupported for incomplete evidence on multi-document questions. "
        "- Do NOT mark unsupported if evidence exists but lacks perfect coverage. "
        "- DO mark partial/supported if at least some relevant evidence exists and is integrated into the answer. "
        "- Favor 'partial' status when evidence is present but incomplete. "
        "- Only mark unsupported if evidence is absent, irrelevant, or the answer explicitly refuses/states information unavailable. "
        "Return JSON with keys status, reason, is_refusal, prompt_version.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Answer: {answer}\n"
        f"Citations: {citations}\n"
        f"Evidence:\n{evidence}"
    )
