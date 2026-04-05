from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from src.core.models import (
    EvidenceChunk,
    GroundingResult,
    GroundingStatus,
    PipelineTrace,
    ResponseCategory,
    TraceEvent,
    ValidationResult,
    ValidationStatus,
)
from src.orchestration.edges import PipelineEdges
from src.orchestration.graph_state import PipelineState
from src.orchestration.nodes import PipelineNodes


def build_pipeline_graph(*, nodes: PipelineNodes, edges: PipelineEdges):
    graph = StateGraph(PipelineState)
    graph.add_node("summarize_history", nodes.summarize_history)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("validate", nodes.validate)
    graph.add_node("clarify", nodes.clarify)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify", nodes.verify)
    graph.add_node("finish", nodes.finish)

    graph.add_edge(START, "summarize_history")
    graph.add_edge("summarize_history", "rewrite_query")
    graph.add_conditional_edges(
        "rewrite_query",
        edges.route_after_rewrite,
        {"clarify": "clarify", "retrieve": "retrieve"},
    )
    graph.add_edge("clarify", "finish")
    graph.add_edge("retrieve", "validate")
    graph.add_conditional_edges(
        "validate",
        edges.route_after_validate,
        {"generate": "generate"},
    )
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "finish")
    graph.add_edge("finish", END)
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=(
            ValidationResult,
            ValidationStatus,
            GroundingResult,
            GroundingStatus,
            EvidenceChunk,
            PipelineTrace,
            ResponseCategory,
            TraceEvent,
        )
    )
    return graph.compile(checkpointer=InMemorySaver(serde=serializer))
