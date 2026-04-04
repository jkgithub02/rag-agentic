from __future__ import annotations

import httpx
import streamlit as st

st.set_page_config(page_title="Agentic RAG Upload + Ask", layout="wide")
st.title("Agentic RAG Upload Console")
st.caption("Upload PDF/TXT/MD, resolve filename conflicts, then query the indexed corpus.")

api_base = st.sidebar.text_input("API Base URL", value="http://localhost:8000").rstrip("/")

if "pending_upload" not in st.session_state:
    st.session_state.pending_upload = None
if "documents" not in st.session_state:
    st.session_state.documents = []


def send_upload(*, filename: str, content: bytes, policy: str) -> dict[str, object]:
    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{api_base}/upload",
            files={"file": (filename, content)},
            data={"conflict_policy": policy},
        )
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(str(detail))
    return response.json()


def fetch_documents() -> list[dict[str, object]]:
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{api_base}/documents")
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(str(detail))
    payload = response.json()
    docs = payload.get("documents", [])
    if not isinstance(docs, list):
        raise RuntimeError("Unexpected /documents response format.")
    return docs


def clear_all_documents() -> int:
    with httpx.Client(timeout=60) as client:
        response = client.delete(f"{api_base}/documents")
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(str(detail))
    payload = response.json()
    deleted_count = payload.get("deleted_count", 0)
    if not isinstance(deleted_count, int):
        raise RuntimeError("Unexpected delete-all response format.")
    return deleted_count


st.subheader("Upload")
uploaded = st.file_uploader("Select one file", type=["pdf", "txt", "md"])

if uploaded is not None and st.button("Upload", type="primary"):
    try:
        payload = send_upload(
            filename=uploaded.name,
            content=uploaded.getvalue(),
            policy="ask",
        )
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Upload failed: {exc}")
    else:
        if payload.get("status") == "conflict":
            st.session_state.pending_upload = {
                "filename": uploaded.name,
                "content": uploaded.getvalue(),
                "payload": payload,
            }
            st.warning(payload.get("message", "Filename conflict detected."))
        else:
            st.session_state.pending_upload = None
            st.success(
                "Stored as "
                f"{payload.get('stored_filename')} "
                f"with {payload.get('chunks_added')} chunks indexed."
            )
            try:
                st.session_state.documents = fetch_documents()
            except Exception:
                pass

pending = st.session_state.pending_upload
if pending is not None:
    conflict_payload = pending["payload"]
    st.info(
        "File already exists. "
        "Choose how to continue: replace the existing file or keep both versions."
    )
    st.write(f"Existing file: {conflict_payload.get('existing_filename')}")
    st.write(f"Suggested keep-both name: {conflict_payload.get('suggested_filename')}")

    left, right = st.columns(2)
    if left.button("Replace existing"):
        try:
            result = send_upload(
                filename=pending["filename"],
                content=pending["content"],
                policy="replace",
            )
        except Exception as exc:  # pragma: no cover - UI only
            st.error(f"Replace failed: {exc}")
        else:
            st.success(
                "Replaced with "
                f"{result.get('stored_filename')} "
                f"and indexed {result.get('chunks_added')} chunks."
            )
            st.session_state.pending_upload = None
            try:
                st.session_state.documents = fetch_documents()
            except Exception:
                pass

    if right.button("Keep both versions"):
        try:
            result = send_upload(
                filename=pending["filename"],
                content=pending["content"],
                policy="keep_both",
            )
        except Exception as exc:  # pragma: no cover - UI only
            st.error(f"Keep-both failed: {exc}")
        else:
            st.success(
                "Stored as "
                f"{result.get('stored_filename')} "
                f"and indexed {result.get('chunks_added')} chunks."
            )
            st.session_state.pending_upload = None
            try:
                st.session_state.documents = fetch_documents()
            except Exception:
                pass

st.divider()
st.subheader("Documents")
refresh_col, clear_col = st.columns(2)

if refresh_col.button("Refresh"):
    try:
        st.session_state.documents = fetch_documents()
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Refresh failed: {exc}")

if clear_col.button("Clear All", type="secondary"):
    try:
        deleted_count = clear_all_documents()
        st.session_state.documents = fetch_documents()
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Clear-all failed: {exc}")
    else:
        st.success(f"Cleared {deleted_count} document(s).")

documents = st.session_state.documents
if documents:
    st.dataframe(documents, use_container_width=True)
else:
    st.caption("No indexed documents found. Click Refresh to load the latest state.")

st.divider()
st.subheader("Ask")
query = st.text_input("Question")
thread_id = st.text_input("Thread ID (optional, for session continuity)")
if st.button("Run Query") and query.strip():
    try:
        with httpx.Client(timeout=120) as client:
            ask_payload: dict[str, str] = {"query": query}
            if thread_id.strip():
                ask_payload["thread_id"] = thread_id.strip()
            answer_response = client.post(f"{api_base}/ask", json=ask_payload)
        if answer_response.status_code >= 400:
            detail = answer_response.json().get("detail", answer_response.text)
            raise RuntimeError(str(detail))

        payload = answer_response.json()
        if payload.get("safe_fail"):
            st.warning(payload.get("answer", "No answer."))
        else:
            st.success("Answer generated.")
            st.write(payload.get("answer", ""))

        st.write("Citations")
        for citation in payload.get("citations", []):
            st.write(f"- {citation}")

        trace_id = payload.get("trace_id")
        if trace_id:
            st.caption(f"Trace ID: {trace_id}")
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Query failed: {exc}")
