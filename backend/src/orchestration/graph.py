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
    graph.add_node("should_compress_context", nodes.should_compress_context)
    graph.add_node("compress_context", nodes.compress_context)
    graph.add_node("fallback_response", nodes.fallback_response)
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
    graph.add_edge("clarify", "rewrite_query")
    graph.add_edge("retrieve", "should_compress_context")
    graph.add_conditional_edges(
        "should_compress_context",
        edges.route_after_should_compress,
        {
            "compress_context": "compress_context",
            "validate": "validate",
            "fallback_response": "fallback_response",
        },
    )
    graph.add_edge("compress_context", "validate")
    graph.add_edge("fallback_response", "finish")
    graph.add_edge("validate", "generate")
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
    return graph.compile(
        checkpointer=InMemorySaver(serde=serializer),
        interrupt_before=["clarify"],
    )
