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
        "You are an expert query analyst and rewriter for document retrieval systems. "
        "ASSUME all queries are searchable in uploaded documents UNLESS they are clearly asking for external info. "
        "Your goal: accept queries as-is and optimize for retrieval rather than seeking clarification. "
        "\n"
        "REWRITE LOGIC (set is_clear=true by default):\n"
        "- Rewrite user queries into self-contained retrieval format staying close to original wording\n"
        "- Accept questions about topics, entities, concepts found in documents (example: 'tell me about attention' → search for 'attention')\n"
        "- Accept comparison requests, summaries, and entity-document lookups\n"
        "- Use conversation summary ONLY to resolve pronouns/references (him, it, that overview)\n"
        "- If query is vague (example: 'it'), inherit from summary and treat as clear\n"
        "\n"
        "CLARIFICATION LOGIC (set is_clear=false ONLY when absolutely necessary):\n"
        "- Only ask clarification if user explicitly requests EXTERNAL knowledge (weather, stock prices, current events NOT in documents)\n"
        "- Only ask if query is uninterpretable gibberish or has no document-relevant context\n"
        "- NEVER ask for clarification on document searches already requested\n"
        "\n"
        "CRITICAL: Bias toward is_clear=true. Assume documents contain relevant content. Rewrite and retrieve.\n"
        "\n"
        "Return JSON with keys: is_clear (boolean), questions (list of 1-5 retrieval questions), rewritten_query (single main query), clarification_needed (null if is_clear=true), prompt_version (string).\n"
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


def conversation_query_detection_prompt(*, query: str) -> str:
    return (
        "You are analyzing whether a user query is asking anything conversational OR asking about documents.\n"
        "\n"
        "CONVERSATION META-QUERIES (is_conversation_query=true): Ask anything casual or conversational, NOT FACTUAL\n"
        "- 'What topics have we discussed?'\n"
        "- 'Summarize what I asked you'\n"
        "- 'What have I been asking?'\n"
        "- 'Remind me of our conversation'\n"
        "- 'What documents did I upload?'\n"
        "- 'Recap what we talked about'\n"
        "\n"
        "DOCUMENT QUERIES (is_conversation_query=false): Ask about information IN the documents\n"
        "- 'What is [example term] in [example document]?'\n"
        "- 'Tell me about [term or document in knowledge base]'\n"
        "- 'Compare these [documents]'\n"
        "\n"
        "IMPORTANT: Meta-queries should have HIGH confidence (0.75-1.0).\n"
        "Document queries should have LOWER/MEDIUM confidence (0.1-0.5) since we always search documents.\n"
        "\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        "{\n"
        '  "is_conversation_query": boolean,\n'
        '  "confidence": number (0.0-1.0),\n'
        f'  "prompt_version": "{PROMPT_VERSION}"\n'
        "}\n"
        f"User query: {query}"
    )
