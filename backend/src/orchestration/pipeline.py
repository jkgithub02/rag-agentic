from __future__ import annotations

import json
from pathlib import Path
import re
from uuid import uuid4

from src.agent.tools import AgentTools
from src.core.config import Settings
from src.core.models import AskResponse, GroundingResult, GroundingStatus, PipelineTrace, TraceEvent
from src.orchestration.edges import PipelineEdges
from src.orchestration.graph import build_pipeline_graph
from src.orchestration.nodes import PipelineNodes
from src.services.reasoner import QueryReasoner
from src.services.trace_store import TraceStore


class AgenticPipeline:
    """LangGraph pipeline composed from dedicated graph state, node, and edge modules."""

    def __init__(
        self,
        settings: Settings,
        tools: AgentTools,
        trace_store: TraceStore,
        reasoner: QueryReasoner | None = None,
    ) -> None:
        self._trace_store = trace_store
        self._thread_history: dict[str, list[dict[str, str]]] = {}
        self._thread_history_dir = Path(settings.vector_db_path).parent / "thread_history"
        self._thread_history_dir.mkdir(parents=True, exist_ok=True)
        self._nodes = PipelineNodes(
            settings=settings,
            tools=tools,
            reasoner=reasoner,
        )
        self._edges = PipelineEdges(settings=settings)
        self._graph = build_pipeline_graph(nodes=self._nodes, edges=self._edges)

    def ask(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        progress_callback: object | None = None,
    ) -> AskResponse:
        # Attach progress callback to nodes for SSE streaming
        if progress_callback is not None:
            self._nodes._progress_callback = progress_callback
        run_thread_id = thread_id or str(uuid4())
        history = list(self._load_thread_history(run_thread_id))
        try:
            state = self._graph.invoke(
                {
                    "query": query,
                    "history": history,
                },
                config={"configurable": {"thread_id": run_thread_id}},
            )
        finally:
            self._nodes._progress_callback = None
        state = self._coerce_interrupted_clarification_state(state)
        trace: PipelineTrace = state["trace"]
        self._trace_store.save(trace)

        next_history = [*history, {"role": "user", "content": query}]
        if self._should_persist_assistant_turn(state):
            next_history.append({"role": "assistant", "content": state["answer"]})
        trimmed = next_history[-12:]
        self._thread_history[run_thread_id] = trimmed
        self._save_thread_history(run_thread_id, trimmed)

        return AskResponse(
            answer=state["answer"],
            citations=state["citations"],
            safe_fail=state["safe_fail"],
            trace_id=trace.trace_id,
        )

    @staticmethod
    def _coerce_interrupted_clarification_state(state: dict[str, object]) -> dict[str, object]:
        if not state.get("clarify_needed") or state.get("retrieval_attempted"):
            return state

        clarify_message = state.get("clarify_message")
        if not isinstance(clarify_message, str) or not clarify_message.strip():
            clarify_message = "Please ask a more specific question about the uploaded documents."

        trace = state.get("trace")
        if isinstance(trace, PipelineTrace):
            if not any(event.stage == "clarify" for event in trace.events):
                trace.events.append(
                    TraceEvent(
                        stage="clarify",
                        payload={"message": clarify_message},
                    )
                )
            if not any(event.stage == "answer" for event in trace.events):
                trace.events.append(TraceEvent(stage="answer", payload={"safe_fail": True}))

        coerced = dict(state)
        coerced["answer"] = clarify_message
        coerced["citations"] = []
        coerced["safe_fail"] = True
        coerced["grounding"] = GroundingResult(
            status=GroundingStatus.UNSUPPORTED,
            reason="Query is too vague and requires clarification.",
        )
        if isinstance(trace, PipelineTrace):
            trace.final_grounding_status = GroundingStatus.UNSUPPORTED
            coerced["trace"] = trace
        return coerced

    def _load_thread_history(self, thread_id: str) -> list[dict[str, str]]:
        cached = self._thread_history.get(thread_id)
        if cached is not None:
            return cached

        path = self._thread_history_file(thread_id)
        if not path.exists():
            history: list[dict[str, str]] = []
            self._thread_history[thread_id] = history
            return history

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            history = []
            self._thread_history[thread_id] = history
            return history

        if not isinstance(payload, list):
            history = []
            self._thread_history[thread_id] = history
            return history

        history = []
        for item in payload[-12:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                history.append({"role": role, "content": content})

        self._thread_history[thread_id] = history
        return history

    def _save_thread_history(self, thread_id: str, history: list[dict[str, str]]) -> None:
        path = self._thread_history_file(thread_id)
        path.write_text(json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8")

    def _thread_history_file(self, thread_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", thread_id)
        return self._thread_history_dir / f"{safe}.json"

    @staticmethod
    def _should_persist_assistant_turn(state: dict[str, object]) -> bool:
        del state
        return True
