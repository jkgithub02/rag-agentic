from __future__ import annotations

import streamlit as st

from agentic_rag.bootstrap import get_pipeline, get_trace_store

st.set_page_config(page_title="Agentic RAG Demo", layout="wide")
st.title("Agentic RAG Interview Demo")
st.caption(
    "Flow: Query -> Understand -> Retrieve -> Validate -> Retry if needed -> "
    "Generate -> Verify grounding -> Answer"
)

query = st.text_input("Ask a question across your indexed papers")
run_clicked = st.button("Run Agentic Query", type="primary")

if run_clicked and query.strip():
    with st.spinner("Running pipeline..."):
        response = get_pipeline().ask(query)
        trace = get_trace_store().get(response.trace_id)

    if response.safe_fail:
        st.warning(response.answer)
    else:
        st.success("Answer generated with evidence grounding.")
        st.write(response.answer)

    st.subheader("Citations")
    if response.citations:
        for citation in response.citations:
            st.write(f"- {citation}")
    else:
        st.write("No citations available for this response.")

    if trace is not None:
        st.subheader("Trace Panel")
        left, right = st.columns(2)
        left.text_area("Original query", value=trace.original_query, height=90, disabled=True)
        right.text_area("Rewritten query", value=trace.rewritten_query, height=90, disabled=True)

        if trace.retry_triggered:
            st.info(f"Retry triggered: {trace.retry_reason}")
        else:
            st.info("Retry not triggered.")

        st.write(f"Grounding verdict: {trace.final_grounding_status}")
        st.json([event.model_dump(mode="json") for event in trace.events])
