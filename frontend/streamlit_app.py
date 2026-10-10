"""Streamlit frontend interface for MemoryGraph."""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MemoryGraph",
    page_icon="🧠",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        background-color: #1e293b;
        color: #94a3b8;
        margin-right: 6px;
    }
    .source-box {
        background-color: #1a1f2c;
        border-left: 3px solid #38bdf8;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 6px 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_headers(user_id: str) -> dict:
    """Return request headers including user ID."""
    return {"X-User-Id": user_id}


# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "session_default"


# --- Sidebar ---
with st.sidebar:
    st.title("🧠 MemoryGraph")
    st.caption("Phase 2: PDF RAG & Document Grounding")

    user_id = st.text_input("Active User ID", value="default_user", help="Identifies user for multi-tenant isolation.")
    headers = get_headers(user_id)

    st.markdown("---")
    st.subheader("📁 Documents")

    uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"], key="pdf_uploader")
    if uploaded_file is not None:
        if st.button("Upload & Index PDF", use_container_width=True):
            with st.spinner("Uploading and indexing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(
                        f"{API_URL}/documents/upload",
                        headers=headers,
                        files=files,
                        timeout=120,
                    )
                    if res.status_code == 201:
                        data = res.json()
                        if data.get("status") == "QUEUED":
                            st.info("📄 Uploaded & queued to Kafka for asynchronous indexing!")
                        else:
                            st.success(f"Indexed {data.get('chunk_count')} chunks successfully!")
                    else:
                        err = res.json().get("detail", res.text)
                        st.error(f"Upload failed: {err}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

    # List Existing Documents
    st.write("Your Indexed Documents:")
    try:
        doc_res = requests.get(f"{API_URL}/documents", headers=headers, timeout=10)
        if doc_res.status_code == 200:
            docs = doc_res.json()
            if not docs:
                st.info("No documents uploaded yet.")
            for doc in docs:
                with st.expander(f"📄 {doc['filename']}"):
                    st.write(f"**Chunks:** {doc['chunk_count']}")
                    st.write(f"**Size:** {doc['size_bytes'] / 1024:.1f} KB")
                    st.write(f"**Status:** `{doc['status']}`")
                    if st.button("🗑️ Delete", key=f"del_{doc['document_id']}", use_container_width=True):
                        del_res = requests.delete(
                            f"{API_URL}/documents/{doc['document_id']}",
                            headers=headers,
                            timeout=10,
                        )
                        if del_res.status_code == 200:
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")
        else:
            st.warning("Could not fetch document list.")
    except Exception:
        st.warning("API server not reachable.")

    st.markdown("---")
    st.subheader("🕸️ Knowledge Graph")
    try:
        g_res = requests.get(f"{API_URL}/documents/graph/stats", headers=headers, timeout=5)
        if g_res.status_code == 200:
            gdata = g_res.json()
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Entities", gdata.get("node_count", 0))
            with col_b:
                st.metric("Relations", gdata.get("edge_count", 0))

            if gdata.get("edge_count", 0) > 0:
                with st.expander("🔎 View Graph Relationships"):
                    t_res = requests.get(f"{API_URL}/documents/graph/triplets", headers=headers, timeout=5)
                    if t_res.status_code == 200:
                        triplets = t_res.json()
                        for t in triplets[:30]:
                            st.markdown(f"• `{t['subject']}` ➔ *{t['predicate']}* ➔ `{t['object']}`")
        else:
            st.info("No Knowledge Graph active yet.")
    except Exception:
        pass

    st.markdown("---")
    st.subheader("⚙️ System Status")
    try:
        health_res = requests.get(f"{API_URL}/health", timeout=5)
        if health_res.status_code == 200:
            hdata = health_res.json()
            llm_ok = hdata.get("llm", {}).get("healthy", False)
            emb_ok = hdata.get("embedder", {}).get("healthy", False)
            kafka_info = hdata.get("kafka", {})
            kafka_enabled = kafka_info.get("enabled", False)
            kafka_conn = kafka_info.get("connected", False)

            redis_info = hdata.get("redis", {})
            redis_enabled = redis_info.get("enabled", False)
            redis_conn = redis_info.get("connected", False)

            mysql_info = hdata.get("mysql", {})
            mysql_enabled = mysql_info.get("enabled", False)
            mysql_conn = mysql_info.get("connected", False)

            provider_name = hdata.get("llm", {}).get("provider", "LLM").upper()
            st.write(f"LLM ({provider_name}): {'🟢 Online' if llm_ok else '🟡 Fallback/Offline'}")
            st.write(f"Embedder (BGE): {'🟢 Loaded' if emb_ok else '🔴 Error'}")
            st.write(f"Graph Store: 🟢 Active (NetworkX)")
            st.write(f"Kafka: {'🟢 Connected' if kafka_conn else ('⚪ Disabled' if not kafka_enabled else '🔴 Disconnected')}")
            st.write(f"Redis: {'🟢 Connected' if redis_conn else ('⚪ Disabled' if not redis_enabled else '🔴 Disconnected')}")
            st.write(f"MySQL: {'🟢 Connected' if mysql_conn else ('⚪ Disabled' if not mysql_enabled else '🔴 Disconnected')}")
            st.write("Vector Store: 🟢 In-Memory")
        else:
            st.error("Health check returned status error.")
    except Exception:
        st.error("API backend is offline.")

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main Chat Area ---
st.header("🧠 GraphRAG Knowledge Intelligence")
st.write(
    "Ask questions across your documents. Responses synthesize **Multi-Hop Knowledge Graph Traversal** "
    "and **Semantic Vector Grounding**, citing exact relationship facts and document pages."
)

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "graph_triplets" in msg and msg["graph_triplets"]:
            with st.expander(f"🕸️ Knowledge Graph Evidence ({len(msg['graph_triplets'])} facts)"):
                for fact in msg["graph_triplets"]:
                    st.markdown(f"• **{fact}**")
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Document Sources ({len(msg['sources'])} chunks)"):
                for s in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>📄 {s['filename']}</strong> (Page {s['page_number']})<br>
                            <small><em>Chunk ID: {s['chunk_id']}</em></small><br>
                            <p style="margin-top: 5px; font-size: 13px;">"{s['snippet']}"</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        if "latency_ms" in msg:
            st.caption(f"⏱️ Latency: {msg['latency_ms']:.1f} ms | Request ID: `{msg.get('request_id', '')}`")

# Handle User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI Chat API
    with st.chat_message("assistant"):
        with st.spinner("Traversing Knowledge Graph & synthesizing response..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                }
                res = requests.post(
                    f"{API_URL}/chat",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

                if res.status_code == 200:
                    data = res.json()
                    response_text = data.get("response", "")
                    sources = data.get("sources", [])
                    graph_facts = data.get("graph_triplets", [])
                    latency_ms = data.get("latency_ms", 0.0)
                    request_id = data.get("request_id", "")

                    st.markdown(response_text)

                    if graph_facts:
                        with st.expander(f"🕸️ Knowledge Graph Evidence ({len(graph_facts)} facts)"):
                            for fact in graph_facts:
                                st.markdown(f"• **{fact}**")

                    if sources:
                        with st.expander(f"📚 Document Sources ({len(sources)} chunks)"):
                            for s in sources:
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <strong>📄 {s['filename']}</strong> (Page {s['page_number']})<br>
                                        <small><em>Chunk ID: {s['chunk_id']}</em></small><br>
                                        <p style="margin-top: 5px; font-size: 13px;">"{s['snippet']}"</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.caption(f"⏱️ Latency: {latency_ms:.1f} ms | Request ID: `{request_id}`")

                    # Save to session history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response_text,
                            "sources": sources,
                            "graph_triplets": graph_facts,
                            "latency_ms": latency_ms,
                            "request_id": request_id,
                        }
                    )
                else:
                    err_data = res.json()
                    err_msg = err_data.get("detail", "Error processing chat request.")
                    st.error(f"API Error ({res.status_code}): {err_msg}")
            except Exception as e:
                st.error(f"Failed to communicate with API server: {str(e)}")
