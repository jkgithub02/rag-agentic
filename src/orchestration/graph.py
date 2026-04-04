from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.orchestration.edges import PipelineEdges
from src.orchestration.graph_state import PipelineState
from src.orchestration.nodes import PipelineNodes


def build_pipeline_graph(*, nodes: PipelineNodes, edges: PipelineEdges):
    graph = StateGraph(PipelineState)
    graph.add_node("understand", nodes.understand)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("validate", nodes.validate)
    graph.add_node("retry", nodes.retry)
    graph.add_node("retry_exhausted", nodes.retry_exhausted)
    graph.add_node("clarify", nodes.clarify)
    graph.add_node("hydrate", nodes.hydrate)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify", nodes.verify)
    graph.add_node("finish", nodes.finish)

    graph.add_edge(START, "understand")
    graph.add_conditional_edges(
        "understand",
        edges.route_after_understand,
        {"clarify": "clarify", "retrieve": "retrieve"},
    )
    graph.add_edge("clarify", "finish")
    graph.add_edge("retrieve", "validate")
    graph.add_conditional_edges(
        "validate",
        edges.route_after_validate,
        {"retry": "retry", "retry_exhausted": "retry_exhausted", "hydrate": "hydrate"},
    )
    graph.add_edge("retry", "retrieve")
    graph.add_edge("retry_exhausted", "generate")
    graph.add_edge("hydrate", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "finish")
    graph.add_edge("finish", END)
    return graph.compile()
