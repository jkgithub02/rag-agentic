from __future__ import annotations

from src.agent.tools import AgentTools
from src.core.config import Settings
from src.core.models import AskResponse, PipelineTrace
from src.orchestration.edges import PipelineEdges
from src.orchestration.graph import build_pipeline_graph
from src.orchestration.nodes import PipelineNodes
from src.services.reasoner import QueryReasoner
from src.services.response_policy import ResponsePolicy
from src.services.trace_store import TraceStore


class AgenticPipeline:
    """LangGraph pipeline composed from dedicated graph state, node, and edge modules."""

    def __init__(
        self,
        settings: Settings,
        tools: AgentTools,
        trace_store: TraceStore,
        reasoner: QueryReasoner | None = None,
        response_policy: ResponsePolicy | None = None,
    ) -> None:
        self._settings = settings
        self._tools = tools
        self._trace_store = trace_store
        self._nodes = PipelineNodes(
            settings=settings,
            tools=tools,
            reasoner=reasoner,
            response_policy=response_policy,
        )
        self._edges = PipelineEdges(settings)
        self._graph = build_pipeline_graph(nodes=self._nodes, edges=self._edges)

    def ask(self, query: str) -> AskResponse:
        state = self._graph.invoke({"query": query})
        trace: PipelineTrace = state["trace"]
        self._trace_store.save(trace)

        return AskResponse(
            answer=state["answer"],
            citations=state["citations"],
            safe_fail=state["safe_fail"],
            trace_id=trace.trace_id,
        )
