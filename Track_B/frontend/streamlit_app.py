import json
import os
import time
import requests
import streamlit as st

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="RAG Studio | ChatGPT Workbench",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. INJECT CHATGPT-INSPIRED DARK THEME CSS (#212121 Canvas, #171717 Sidebar, #2F2F2F Cards & Input)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Söhne:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* Main Canvas Styling - ChatGPT Dark Charcoal Palette */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #212121 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #ECECF1;
}

[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2F2F2F !important;
}

/* Hide Default Streamlit Branding */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 140px !important; /* Extra bottom padding so messages don't hide under fixed chat input */
    max-width: 900px !important; /* Centered ChatGPT width layout */
    margin: 0 auto;
}

/* Typography & Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #F3F4F6 !important;
    letter-spacing: -0.02em !important;
}

p, span, label {
    color: #D1D5DB;
}

/* Studio Header Bar (ChatGPT Charcoal Style) */
.studio-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: #171717;
    border: 1px solid #2F2F2F;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

.studio-title-group {
    display: flex;
    align-items: center;
    gap: 12px;
}

.studio-badge-green {
    background: rgba(16, 163, 127, 0.15);
    color: #10A37F;
    border: 1px solid rgba(16, 163, 127, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
}

.studio-badge-orange {
    background: rgba(249, 115, 22, 0.15);
    color: #F97316;
    border: 1px solid rgba(249, 115, 22, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
}

.status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #10A37F;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(16, 163, 127, 0.1);
    border: 1px solid rgba(16, 163, 127, 0.25);
    padding: 4px 12px;
    border-radius: 20px;
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10A37F;
    border-radius: 50%;
    box-shadow: 0 0 8px #10A37F;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.8; }
}

/* Custom Styled Tabs */
div[data-baseweb="tab-list"] {
    gap: 8px;
    background: #171717;
    padding: 6px;
    border-radius: 14px;
    border: 1px solid #2F2F2F;
}

div[data-baseweb="tab"] {
    height: 42px;
    border-radius: 10px !important;
    color: #9CA3AF !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0 20px !important;
    transition: all 0.2s ease;
    border: none !important;
}

div[data-baseweb="tab"]:hover {
    color: #ECECF1 !important;
    background: rgba(255, 255, 255, 0.05);
}

div[data-baseweb="tab"][aria-selected="true"] {
    background: #2F2F2F !important;
    color: #10A37F !important;
    border: 1px solid rgba(16, 163, 127, 0.4) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

/* ChatGPT Message Containers & Bubbles */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
}

/* User Chat Bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: transparent !important;
}

/* Claude-Style Backend Process Trace Box */
.trace-step-item {
    background: #171717;
    border-left: 3px solid #10A37F;
    padding: 10px 14px;
    margin-bottom: 6px;
    border-radius: 0 8px 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    border-top: 1px solid #2F2F2F;
    border-right: 1px solid #2F2F2F;
    border-bottom: 1px solid #2F2F2F;
}

.trace-step-title {
    color: #10A37F;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
}

.trace-step-desc {
    color: #9CA3AF;
    margin-top: 2px;
}

/* Citations Drawer */
.citation-box {
    background: #171717;
    border: 1px solid #2F2F2F;
    border-radius: 10px;
    padding: 12px;
    margin-top: 8px;
}

.citation-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #F97316;
    font-weight: 600;
}

/* Primary Action Buttons - ChatGPT Brand Green / Orange */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10A37F, #0D8A6C) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(16, 163, 127, 0.3) !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0D8A6C, #0A7259) !important;
    box-shadow: 0 6px 18px rgba(16, 163, 127, 0.4) !important;
    transform: translateY(-1px);
}

.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: 1px solid #3E3E3E !important;
    background: #2F2F2F !important;
    color: #ECECF1 !important;
}

.stButton > button:hover {
    border-color: #10A37F !important;
    color: #10A37F !important;
}

/* Input Fields */
.stTextInput > div > div > input {
    background-color: #2F2F2F !important;
    color: #ECECF1 !important;
    border: 1px solid #424242 !important;
    border-radius: 12px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #10A37F !important;
    box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.25) !important;
}

/* DOCKED BOTTOM CHAT INPUT (FIXED POSITION AT BOTTOM LIKE CHATGPT) */
div[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 24px !important;
    left: 60% !important;
    transform: translateX(-50%) !important;
    max-width: 800px !important;
    width: calc(100% - 340px) !important;
    z-index: 99999 !important;
    background-color: #2F2F2F !important;
    border: 1px solid #424242 !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
    padding: 4px 8px !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: #10A37F !important;
    box-shadow: 0 8px 36px rgba(16, 163, 127, 0.25) !important;
}

div[data-testid="stChatInput"] textarea {
    color: #ECECF1 !important;
    font-size: 15px !important;
}

/* Ingestion Dropzone Panel */
.upload-card {
    background: #2F2F2F;
    border: 2px dashed #424242;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: all 0.2s ease;
}

.upload-card:hover {
    border-color: #10A37F;
    background: #262626;
}

/* Benchmark Result Card */
.benchmark-card {
    background: #2F2F2F;
    border: 1px solid #3E3E3E;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.benchmark-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #3E3E3E;
    padding-bottom: 10px;
    margin-bottom: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. SIDEBAR CONFIGURATION
default_backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

with st.sidebar:
    st.markdown("### ⚙️ Studio Control Console")
    st.caption("Configure live pipeline parameters:")

    BACKEND_URL = st.text_input("Backend API Base URL", value=default_backend_url)

    st.markdown("---")
    st.markdown("#### 🧩 Retrieval Architecture")

    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        options=["recursive", "semantic"],
        index=0,
        help="Recursive character splitting vs Sentence embedding semantic splitting"
    )

    retrieval_mode = st.selectbox(
        "Retrieval Search Mode",
        options=["hybrid", "dense", "bm25"],
        index=0,
        help="Hybrid (Dense + BM25 RRF) vs Standalone Dense vs Standalone BM25"
    )

    reranker_enabled = st.checkbox(
        "Enable Cross-Encoder Reranker",
        value=True,
        help="Re-ranks candidate shortlist down to Top-5 final context chunks"
    )

    st.markdown("---")
    st.markdown("#### 🎚️ Candidate Parameters")

    fusion_top_k = st.slider("Candidate Shortlist Top-K (Fusion)", min_value=5, max_value=30, value=15)
    final_top_k = st.slider("Final Context Top-K (LLM)", min_value=1, max_value=10, value=5)


# 4. STUDIO NAVIGATION HEADER
st.markdown(f"""
<div class="studio-header">
    <div class="studio-title-group">
        <h2 style="margin: 0; font-size: 20px; color: #ECECF1;">⚡ RAG Assistant</h2>
        <span class="studio-badge-green">{retrieval_mode.upper()} SEARCH</span>
        <span class="studio-badge-orange">{chunking_strategy.upper()} CHUNKS</span>
    </div>
    <div class="status-pill">
        <div class="status-dot"></div>
        <span>READY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. INITIALIZE CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []


# 6. MAIN WORKSPACE TABS
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📄 Document Ingestion", "📊 Strategy Benchmark"])


# ----------------------------------------------------
# TAB 1: CHAT ASSISTANT (CHATGPT CHAT STREAM + FIXED BOTTOM INPUT)
# ----------------------------------------------------
with tab1:
    # Render All Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                # Claude-Style Collapsible Execution Trace
                trace_steps = message.get("trace_steps", [])
                latency = message.get("latency_ms", 0.0)
                provider = message.get("provider", "Unknown Provider")

                summary_str = f"⚡ Thought for {latency/1000:.2f}s · {len(trace_steps) if trace_steps else 4} backend steps completed"

                with st.expander(summary_str):
                    if trace_steps:
                        for step in trace_steps:
                            st.markdown(f"""
                            <div class="trace-step-item">
                                <div class="trace-step-title">
                                    <span>{step.get('name', 'Pipeline Step')}</span>
                                    <span>{step.get('duration_ms', 0):.1f} ms</span>
                                </div>
                                <div class="trace-step-desc">{step.get('summary', '')}</div>
                                <div style="color: #9CA3AF; font-size: 11px; margin-top: 3px;">{step.get('details', '')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption(f"Served by **{provider}** | Total Pipeline Latency: **{latency:.1f} ms**")

                # Citations Drawer
                if "citations" in message and message["citations"]:
                    with st.expander(f"📚 Context Citations ({len(message['citations'])})"):
                        for idx, cit in enumerate(message["citations"]):
                            page_str = f" · Page {cit['page_number']}" if cit.get('page_number') else ""
                            st.markdown(f"""
                            <div class="citation-box">
                                <div class="citation-header">[{idx+1}] {cit['filename']}{page_str}</div>
                                <div style="color: #D1D5DB; font-size: 12px; margin-top: 4px;">_{cit['snippet']}_</div>
                            </div>
                            """, unsafe_allow_html=True)

    # Chat Input Box (Fixed at Bottom of Viewport)
    if prompt := st.chat_input("Message RAG Assistant..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context & generating answer..."):
                payload = {
                    "message": prompt,
                    "config": {
                        "chunking": chunking_strategy,
                        "mode": retrieval_mode,
                        "reranker": reranker_enabled,
                        "fusion_top_k": fusion_top_k,
                        "final_top_k": final_top_k
                    }
                }

                try:
                    resp = requests.post(f"{BACKEND_URL}/api/chat", json=payload, timeout=180)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        provider = data.get("provider_used", "Unknown Provider")
                        citations = data.get("citations", [])
                        latency_ms = data.get("latency_ms", 0.0)
                        trace_steps = data.get("trace_steps", [])

                        # Stream output text
                        def text_stream():
                            for word in answer.split(" "):
                                yield word + " "
                                time.sleep(0.015)

                        st.write_stream(text_stream)

                        # Claude-Style Collapsible Backend Execution Trace
                        summary_str = f"⚡ Thought for {latency_ms/1000:.2f}s · {len(trace_steps) if trace_steps else 4} backend steps completed"

                        with st.expander(summary_str):
                            if trace_steps:
                                for step in trace_steps:
                                    st.markdown(f"""
                                    <div class="trace-step-item">
                                        <div class="trace-step-title">
                                            <span>{step.get('name', 'Pipeline Step')}</span>
                                            <span>{step.get('duration_ms', 0):.1f} ms</span>
                                        </div>
                                        <div class="trace-step-desc">{step.get('summary', '')}</div>
                                        <div style="color: #9CA3AF; font-size: 11px; margin-top: 3px;">{step.get('details', '')}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.caption(f"Served by **{provider}** | Total Latency: **{latency_ms:.1f} ms**")

                        # Citations Drawer
                        if citations:
                            with st.expander(f"📚 Context Citations ({len(citations)})"):
                                for idx, cit in enumerate(citations):
                                    page_str = f" · Page {cit['page_number']}" if cit.get('page_number') else ""
                                    st.markdown(f"""
                                    <div class="citation-box">
                                        <div class="citation-header">[{idx+1}] {cit['filename']}{page_str}</div>
                                        <div style="color: #D1D5DB; font-size: 12px; margin-top: 4px;">_{cit['snippet']}_</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Save message state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "provider": provider,
                            "citations": citations,
                            "latency_ms": latency_ms,
                            "trace_steps": trace_steps
                        })
                        st.rerun()
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend at {BACKEND_URL}: {str(e)}")


# ----------------------------------------------------
# TAB 2: DOCUMENT INGESTION WORKBENCH
# ----------------------------------------------------
with tab2:
    st.markdown("### 📄 Document Ingestion Pipeline")
    st.caption("Upload PDF, TXT, or Markdown files for dual-indexing into Chroma DB & Persistent BM25 Store.")

    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop document here or click to select", type=["pdf", "txt", "md"])

    if uploaded_file is not None:
        file_details = f"**File:** `{uploaded_file.name}` | **Size:** `{uploaded_file.size / 1024:.1f} KB`"
        st.markdown(file_details)

        if st.button("⚡ Index Document into Dual Storage", type="primary", use_container_width=True):
            with st.spinner("Executing document loading, page splitting, dual chunking & indexing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    res = requests.post(f"{BACKEND_URL}/api/ingest", files=files, timeout=120)
                    if res.status_code == 200:
                        ingest_data = res.json()
                        st.success(f"✅ Ingestion Complete! Status: **{ingest_data['status'].upper()}**")

                        # Summary Metric Cards
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Status", ingest_data['status'].upper())
                        m2.metric("Pages Processed", ingest_data.get('total_pages') or 1)
                        m3.metric("Recursive Chunks", ingest_data.get('recursive_chunks_created', 0))
                        m4.metric("Semantic Chunks", ingest_data.get('semantic_chunks_created', 0))

                        with st.expander("🔍 View Raw Indexing Metadata Payload"):
                            st.json(ingest_data)
                    else:
                        st.error(f"Ingestion failed ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Ingestion request error: {str(e)}")


# ----------------------------------------------------
# TAB 3: MULTI-STRATEGY RETRIEVAL BENCHMARK
# ----------------------------------------------------
with tab3:
    st.markdown("### 📊 Multi-Strategy Retrieval Benchmark")
    st.markdown("Execute a query side-by-side across all chunking and search mode variations.")

    compare_query = st.text_input("Comparison Query String", value="What are the core architectural principles?")

    if st.button("🚀 Execute Parallel Strategy Benchmark", type="primary", use_container_width=True):
        with st.spinner("Running parallel retrieval queries..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/api/compare",
                    json={"query": compare_query, "top_k": final_top_k},
                    timeout=30
                )
                if res.status_code == 200:
                    comp_data = res.json()
                    st.markdown(f"#### Results for query: *'{comp_data['query']}'*")

                    cols = st.columns(len(comp_data["results"]))
                    for idx, item in enumerate(comp_data["results"]):
                        cfg = item["config"]
                        with cols[idx]:
                            st.markdown(f"""
                            <div class="benchmark-card">
                                <div class="benchmark-card-header">
                                    <span style="font-weight: 700; color: #10A37F;">Strategy {idx+1}</span>
                                    <span style="font-family: 'JetBrains Mono'; font-size: 12px; color: #10A37F; font-weight: 600;">{item['latency_ms']:.1f} ms</span>
                                </div>
                                <div style="font-size: 13px; margin-bottom: 6px; color: #D1D5DB;"><b>Chunking:</b> <code>{cfg['chunking']}</code></div>
                                <div style="font-size: 13px; margin-bottom: 6px; color: #D1D5DB;"><b>Mode:</b> <code>{cfg['mode']}</code></div>
                                <div style="font-size: 13px; margin-bottom: 6px; color: #D1D5DB;"><b>Reranker:</b> <code>{cfg['reranker']}</code></div>
                                <div style="font-size: 13px; margin-bottom: 6px; color: #D1D5DB;"><b>Retrieved:</b> <code>{len(item['retrieved_chunks'])} chunks</code></div>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.expander("📄 View Retrieved Chunks"):
                                for chunk in item["retrieved_chunks"]:
                                    score = chunk.get('reranker_score') or chunk.get('rrf_score') or chunk.get('dense_score') or 0.0
                                    st.caption(f"**Chunk {chunk['chunk_id'][:8]}** · Score: `{score:.3f}`")
                                    st.code(chunk["text"][:160] + "...", language="text")
                else:
                    st.error(f"Comparison request failed ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Comparison error: {str(e)}")
