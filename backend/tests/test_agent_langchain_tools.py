from __future__ import annotations

import json

from src.agent.langchain_tools import create_langchain_tools
from src.agent.tools import AgentTools
from src.core.config import Settings
from src.core.models import EvidenceChunk


class StubVectorDb:
    def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        return [
            EvidenceChunk(
                chunk_id="c1",
                source="doc-a.md",
                text=f"chunk for {query}",
                score=0.8,
            )
        ][:top_k]

    def fetch_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        all_chunks = {
            "c1": EvidenceChunk(chunk_id="c1", source="doc-a.md", text="chunk one", score=0.8),
            "c2": EvidenceChunk(chunk_id="c2", source="doc-b.md", text="chunk two", score=0.6),
        }
        return [all_chunks[item] for item in chunk_ids if item in all_chunks]


class StubReasoner:
    def summarize_conversation(self, history: list[dict[str, str]]) -> str:
        return "summary:" + str(len(history))

    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
        subqueries: list[str] | None = None,
        force_answer: bool = False,
    ) -> tuple[str, list[str], str, str | None]:
        del query, subqueries, force_answer
        ids = [chunk.chunk_id for chunk in chunks[:1]]
        return "stub answer", ids, "llm", "v1.0.0"


def _tool_by_name(tools: list[object], name: str) -> object:
    for tool in tools:
        if getattr(tool, "name", None) == name:
            return tool
    raise AssertionError(f"Tool {name} not found")


def test_create_langchain_tools_contract() -> None:
    settings = Settings(_env_file=None)
    tools = create_langchain_tools(
        settings=settings,
        vector_db=StubVectorDb(),
        reasoner=StubReasoner(),  # type: ignore[arg-type]
    )
    names = [tool.name for tool in tools]
    assert names == [
        "search_documents",
        "fetch_chunks_by_ids",
        "compress_evidence",
        "verify_evidence_quality",
        "synthesize_answer",
    ]


def test_search_documents_returns_serialized_chunks() -> None:
    settings = Settings(_env_file=None)
    tools = create_langchain_tools(
        settings=settings,
        vector_db=StubVectorDb(),
        reasoner=StubReasoner(),  # type: ignore[arg-type]
    )
    search_tool = _tool_by_name(tools, "search_documents")
    payload = search_tool.invoke({"query": "bert", "top_k": 1})
    parsed = json.loads(payload)

    assert isinstance(parsed, list)
    assert parsed[0]["chunk_id"] == "c1"
    assert parsed[0]["source"] == "doc-a.md"


def test_verify_evidence_quality_returns_thresholded_score() -> None:
    settings = Settings(_env_file=None)
    settings.agent_evidence_quality_threshold = 0.65
    tools = create_langchain_tools(
        settings=settings,
        vector_db=StubVectorDb(),
        reasoner=StubReasoner(),  # type: ignore[arg-type]
    )
    verify_tool = _tool_by_name(tools, "verify_evidence_quality")

    chunks_json = json.dumps(
        [
            {
                "chunk_id": "c1",
                "source": "doc-a.md",
                "text": "evidence",
                "score": 0.8,
            },
            {
                "chunk_id": "c2",
                "source": "doc-b.md",
                "text": "more evidence",
                "score": 0.6,
            },
        ]
    )
    payload = verify_tool.invoke({"chunks_json": chunks_json})
    parsed = json.loads(payload)

    assert parsed["is_sufficient"] is True
    assert parsed["threshold"] == 0.65
    assert parsed["chunk_count"] == 2


def test_agent_tools_exposes_langchain_tools() -> None:
    settings = Settings(_env_file=None)
    agent_tools = AgentTools(StubVectorDb())  # type: ignore[arg-type]
    tools = agent_tools.get_langchain_tools(
        settings=settings,
        reasoner=StubReasoner(),  # type: ignore[arg-type]
    )
    assert [tool.name for tool in tools][:2] == ["search_documents", "fetch_chunks_by_ids"]
