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


def answer_prompt(*, query: str, evidence: str, force_answer: bool = False) -> str:
    base = (
        "Answer the query from the provided evidence in 2-4 sentences. "
        "If the query has multiple parts, address each part explicitly. "
        "Directly address the specific question asked - do not provide tangential information. "
        "Be helpful, direct, and confident. "
        "When relevant evidence exists, provide an answer even if partial or requiring reasonable inference. "
        "When evidence spans multiple sources, cite every source used. "
        "Always cite specific source chunks: explicitly mention chunk_ids like [chunk_id_0012] or similar. "
        "Extract and synthesize factual claims directly from the evidence. Commit to answering based on what the evidence states. "
        "Do NOT hedge or qualify with phrases like 'does not specify', 'does not explicitly state', or 'is not detailed' if the answer is derivable from evidence. "
    )
    if force_answer:
        base += (
            "CRITICAL: Evidence IS provided below and DOES contain relevant information. "
            "You MUST extract facts from the evidence and present them. "
            "Do NOT refuse. Do NOT say evidence is insufficient. "
        )
    else:
        base += "Only refuse completely if evidence is truly absent or entirely irrelevant. "
    base += (
        "Return JSON with keys answer, citation_chunk_ids, prompt_version.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Query: {query}\n"
        f"Evidence:\n{evidence}"
    )
    return base


def grounding_prompt(*, answer: str, citations: list[str], evidence: str) -> str:
    return (
        "Evaluate whether the answer is grounded in evidence. "
        "Set status to exactly one of: supported, partial, unsupported. "
        "Use 'supported' if answer is directly stated or clearly derivable from evidence. "
        "Use 'partial' if answer is grounded in relevant evidence with reasonable inference, multi-document synthesis, or acceptable gaps in detail (e.g., technical specifics, exact figures). "
        "Use 'unsupported' only if: (1) answer contradicts evidence, (2) answer contains factual hallucinations, (3) answer is an explicit refusal, or (4) evidence is completely empty/irrelevant/gibberish. "
        "A refusal explicitly states inability to answer. Examples: 'I cannot answer this question', 'No information is available', 'The evidence does not contain any relevant information'. "
        "CRITICAL: Answers that only state 'does not specify' or 'does not explicitly state' WITHOUT providing factual claims are cop-outs and should be marked unsupported. "
        "Mark as PARTIAL only if the answer provides at least some factual information derived from evidence, even if noting gaps. "
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


def query_complexity_prompt(*, query: str, conversation_summary: str | None = None) -> str:
    summary_block = (
        f"Conversation summary:\n{conversation_summary}\n"
        if conversation_summary and conversation_summary.strip()
        else "Conversation summary:\n(none)\n"
    )
    return (
        "Classify the retrieval complexity of the user query for routing in an agentic RAG system. "
        "Return one label from: simple, moderate, complex.\n"
        "Label guidance:\n"
        "- simple: direct factual lookup likely solved in one retrieval pass.\n"
        "- moderate: explanation/summarization/why-how style queries that may need richer evidence.\n"
        "- complex: comparison, tradeoff, multi-part, or synthesis-heavy queries likely needing iterative steps. "
        "Queries with 'and how', 'and then', 'compare', 'trace how', 'evolved from', or referencing 3+ entities/papers are ALWAYS complex.\n"
        "Output only valid JSON with keys: query_complexity, confidence, prompt_version.\n"
        "confidence must be a number from 0.0 to 1.0.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"{summary_block}"
        f"User query: {query}"
    )


def query_decomposition_prompt(
    *,
    query: str,
    conversation_summary: str | None = None,
    max_subqueries: int = 3,
) -> str:
    summary_block = (
        f"Conversation summary:\n{conversation_summary}\n"
        if conversation_summary and conversation_summary.strip()
        else "Conversation summary:\n(none)\n"
    )
    return (
        "Decompose the user query into retrieval-focused sub-queries. "
        "Each sub-query should be concise, self-contained, and directly searchable in documents.\n"
        "Rules:\n"
        f"- Return at most {max_subqueries} sub-queries.\n"
        "- Preserve the original intent and key entities.\n"
        "- Do not invent external facts.\n"
        "- If decomposition is unnecessary, return exactly one query close to the original.\n"
        "Output only valid JSON with keys: sub_queries, prompt_version.\n"
        "sub_queries must be a JSON array of non-empty strings.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"{summary_block}"
        f"User query: {query}"
    )


def agent_step_planning_prompt(
    *,
    query: str,
    conversation_summary: str | None = None,
    rewritten_queries: list[str] | None = None,
    evidence_quality_score: float = 0.0,
    chunk_count: int = 0,
    agent_iterations: int = 0,
    max_iterations: int = 1,
    last_observation: str | None = None,
    subquery_statuses: list[dict[str, object]] | None = None,
) -> str:
    summary_block = (
        f"Conversation summary:\n{conversation_summary}\n"
        if conversation_summary and conversation_summary.strip()
        else "Conversation summary:\n(none)\n"
    )
    query_block = "\n".join(f"- {item}" for item in (rewritten_queries or []) if item.strip())
    if not query_block:
        query_block = f"- {query}"

    observation_block = last_observation.strip() if isinstance(last_observation, str) and last_observation.strip() else "(none)"

    subquery_block = "(none)"
    if subquery_statuses:
        lines: list[str] = []
        for i, sq in enumerate(subquery_statuses):
            status = sq.get("status", "pending")
            q = sq.get("query", "")
            quality = sq.get("quality_score", 0.0)
            lines.append(f"  [{i}] status={status} quality={quality:.4f} query=\"{q}\"")
        subquery_block = "\n".join(lines)

    return (
        "Plan the next step for an agentic RAG loop. "
        "Choose exactly one action from: search_documents, web_search, finalize.\n"
        "Action guidance:\n"
        "- search_documents: use when evidence is missing, weak, or incomplete for any subquery in local documents.\n"
        "- web_search: use when local document evidence is weak (quality < 0.5) or the query asks about current events, benchmarks, or external comparisons. Examples: 'latest research on X', 'current state of Y', 'what datasets were used' (when not in docs), 'how does A compare to B' (when B is not in docs). Web results are labeled separately from local evidence. "
        "IMPORTANT: web_search is for enriching topics already partially covered by local documents. If local retrieval returned 0 chunks, do NOT recommend web_search — the topic is likely outside the knowledge base.\n"
        "- finalize: use when evidence appears sufficient across all subqueries to proceed to validation/generation.\n"
        "If choosing search_documents or web_search, also return target_subquery_index (integer) indicating which subquery to target. "
        "Pick the subquery with the weakest evidence (lowest quality or still pending).\n"
        "Return only valid JSON with keys: reasoning, recommended_action, confidence, target_subquery_index, prompt_version.\n"
        "confidence must be a number from 0.0 to 1.0.\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"{summary_block}"
        f"Original query: {query}\n"
        f"Rewritten retrieval queries:\n{query_block}\n"
        f"Subquery execution status:\n{subquery_block}\n"
        f"Current evidence quality score: {evidence_quality_score:.4f}\n"
        f"Current chunk count: {chunk_count}\n"
        f"Current iteration: {agent_iterations}\n"
        f"Max iterations: {max_iterations}\n"
        f"Last observation: {observation_block}"
    )
