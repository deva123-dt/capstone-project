
import os
import sys
import time
import html
import numpy as np
import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from chunker import chunks


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL THEME
# =========================================================
st.markdown(
    """
    <style>
    /* ---------- GLOBAL ---------- */
    .stApp {
        background: #f5f7fb !important;
        color: #172033 !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #f5f7fb !important;
    }

    [data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e5e9f2 !important;
    }

    .main .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- TEXT ---------- */
    h1, h2, h3, h4, h5, h6,
    p, li, label, span, div {
        color: #172033;
    }

    h1 {
        font-size: 2.35rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.8px;
        margin-bottom: 0.2rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 750 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
    }

    /* ---------- SIDEBAR ---------- */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e1e6ef !important;
    }

    [data-testid="stSidebar"] * {
        color: #172033 !important;
    }

    /* ---------- SIDEBAR SELECTBOX ---------- */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #172033 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input {
        background: #ffffff !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
    }

    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg {
        fill: #475569 !important;
        color: #475569 !important;
    }

    /* ---------- SELECT DROPDOWN: FORCE LIGHT PROFESSIONAL MENU ---------- */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] > div > div,
    [data-baseweb="menu"],
    [data-baseweb="menu"] > div,
    [role="listbox"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #172033 !important;
        border-color: #dbe3ee !important;
    }

    [data-baseweb="popover"] *,
    [data-baseweb="menu"] *,
    [role="listbox"] *,
    [role="option"] {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
    }

    [data-baseweb="menu"] [role="option"],
    [role="listbox"] [role="option"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #172033 !important;
        min-height: 40px !important;
        padding: 9px 12px !important;
    }

    [data-baseweb="menu"] [role="option"]:hover,
    [role="listbox"] [role="option"]:hover {
        background: #eff6ff !important;
        background-color: #eff6ff !important;
        color: #1d4ed8 !important;
        -webkit-text-fill-color: #1d4ed8 !important;
    }

    [data-baseweb="menu"] [role="option"][aria-selected="true"],
    [role="listbox"] [role="option"][aria-selected="true"] {
        background: #dbeafe !important;
        background-color: #dbeafe !important;
        color: #1d4ed8 !important;
        -webkit-text-fill-color: #1d4ed8 !important;
        font-weight: 750 !important;
    }

    /* Remove any dark theme inherited by the dropdown popup */
    [data-baseweb="popover"] input,
    [data-baseweb="popover"] textarea {
        background: #ffffff !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
    }

    /* ---------- CHAT INPUT ---------- */
    [data-testid="stBottom"] {
        background: #f5f7fb !important;
        border-top: 1px solid #e1e6ef !important;
    }

    [data-testid="stBottom"] > div {
        background: #f5f7fb !important;
    }

    [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.08) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        caret-color: #2563eb !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9px !important;
    }

    [data-testid="stChatInput"] button svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        border-radius: 9px !important;
        border: 1px solid #d5dce8 !important;
        background: #ffffff !important;
        color: #172033 !important;
        font-weight: 650 !important;
    }

    .stButton > button:hover {
        border-color: #2563eb !important;
        color: #2563eb !important;
    }

    /* ---------- HERO ---------- */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        border-radius: 20px;
        padding: 34px 38px;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.14);
    }

    .hero * {
        color: #ffffff !important;
    }

    .hero-kicker {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        opacity: 0.85;
        margin-bottom: 9px;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 850;
        line-height: 1.1;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 1rem;
        line-height: 1.6;
        opacity: 0.88;
        max-width: 820px;
    }

    /* ---------- NATIVE METRICS ---------- */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e1e6ef !important;
        border-radius: 14px !important;
        padding: 17px 18px !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 650 !important;
    }

    [data-testid="stMetricValue"] {
        color: #172033 !important;
        font-weight: 800 !important;
    }

    /* ---------- CHAT ---------- */
    [data-testid="stChatMessage"] {
        border: 1px solid #e1e6ef !important;
        border-radius: 14px !important;
        margin-bottom: 12px !important;
        background: #ffffff !important;
    }

    [data-testid="stChatInput"] {
        border-color: #cfd7e6 !important;
    }

    /* ---------- CONTAINERS / CARDS ---------- */
    .card {
        background: #ffffff;
        border: 1px solid #e1e6ef;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    .source-number {
        color: #2563eb !important;
        font-weight: 800;
        font-size: 0.82rem;
    }

    .source-title {
        color: #172033 !important;
        font-weight: 750;
        font-size: 1rem;
    }

    .source-meta {
        color: #64748b !important;
        font-size: 0.86rem;
    }

    .score {
        color: #15803d !important;
        font-weight: 750;
    }

    /* ---------- BADGES ---------- */
    .badge {
        display: inline-block;
        padding: 5px 10px;
        margin-right: 6px;
        margin-bottom: 5px;
        border-radius: 999px;
        background: #eff6ff;
        color: #1d4ed8 !important;
        border: 1px solid #bfdbfe;
        font-size: 0.76rem;
        font-weight: 750;
    }

    /* ---------- PIPELINE ---------- */
    .pipeline-step {
        background: #ffffff;
        border: 1px solid #dfe5ef;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        min-height: 82px;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.035);
    }

    .pipeline-icon {
        font-size: 1.35rem;
    }

    .pipeline-name {
        font-size: 0.82rem;
        font-weight: 750;
        color: #172033 !important;
        margin-top: 4px;
    }

    /* ---------- INFO / WARNING ---------- */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* ---------- TABS ---------- */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1d4ed8 !important;
    }

    /* ---------- DIVIDER ---------- */
    hr {
        border-color: #e1e6ef !important;
    }

    /* ---------- MOBILE ---------- */
    @media (max-width: 800px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 1.75rem !important;
        }

        .hero-title {
            font-size: 1.65rem;
        }

        .hero {
            padding: 25px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CACHED BACKEND
# =========================================================
@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


@st.cache_resource
def load_vectorstore(_embeddings):
    return Chroma(
        collection_name="research_papers",
        embedding_function=_embeddings,
        persist_directory="vectorstore",
    )


@st.cache_resource
def load_llm():
    return ChatOllama(
        model="gemma3:1b",
        temperature=0,
        num_predict=80,
    )


@st.cache_resource
def load_bm25():
    corpus = [doc.page_content for doc in chunks]
    tokenized_corpus = [text.lower().split() for text in corpus]
    return BM25Okapi(tokenized_corpus)


embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
llm = load_llm()
bm25 = load_bm25()


# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_scores" not in st.session_state:
    st.session_state.last_scores = []


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.success("● System Ready")

    st.divider()

    st.markdown("### 🔎 Retrieval Method")

    retrieval_method = st.selectbox(
        "Choose retrieval",
        ["Hybrid", "Dense", "MMR"],
        label_visibility="collapsed",
    )

    descriptions = {
        "Dense": "Semantic vector retrieval using ChromaDB.",
        "MMR": "Semantic retrieval with relevance + diversity.",
        "Hybrid": "Combines semantic retrieval with BM25 keyword search.",
    }

    st.caption(descriptions[retrieval_method])

    st.divider()

    st.markdown("### 🧠 AI Stack")
    st.write("**LLM:** Gemma 3 1B")
    st.write("**Embedding:** Nomic Embed Text")
    st.write("**Vector DB:** ChromaDB")
    st.write("**Embedding Size:** 768D")

    st.divider()

    st.markdown("### 📚 Knowledge Base")
    st.write("**4** research papers")
    st.write("**86** indexed pages")
    st.write("**649** chunks")

    with st.expander("View papers"):
        st.write("• Attention Is All You Need")
        st.write("• LoRA")
        st.write("• QLoRA")
        st.write("• Retrieval-Augmented Generation")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_scores = []
        st.session_state.last_response_time = None
        st.rerun()


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">🤖 AI Research Assistant</div>
        <div class="hero-title">Research Paper Answer Bot</div>
        <div class="hero-subtitle">
            Ask questions from curated Generative AI research papers
            using Retrieval-Augmented Generation with local AI.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Badges use normal Streamlit markdown, not raw HTML.
badge_text = (
    "**RAG**  •  **ChromaDB**  •  **BM25**  •  "
    "**Cosine Similarity**  •  **Gemma 3**  •  **Local AI**"
)
st.markdown(badge_text)


# =========================================================
# PROJECT STATS
# =========================================================
response_value = (
    f"{st.session_state.last_response_time:.2f}s"
    if st.session_state.last_response_time is not None
    else "—"
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("📚 Research Papers", "4")

with c2:
    st.metric("🧩 Indexed Chunks", "649")

with c3:
    st.metric("🧠 Embedding Dimension", "768D")

with c4:
    st.metric("⚡ Response Time", response_value)


# =========================================================
# PIPELINE
# =========================================================
st.markdown("### 🧠 RAG Pipeline")

pipeline = [
    ("📥", "Query"),
    ("🔎", "Retrieve"),
    ("🎯", "Cosine"),
    ("📚", "Top 3"),
    ("🧾", "Context"),
    ("🤖", "Gemma 3"),
    ("💬", "Answer"),
]

cols = st.columns(len(pipeline))

for col, (icon, name) in zip(cols, pipeline):
    with col:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <div class="pipeline-icon">{icon}</div>
                <div class="pipeline-name">{name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# RETRIEVAL COMPARISON
# =========================================================
with st.expander("📊 Retrieval Method Comparison", expanded=False):
    st.markdown(
        """
        | Method | How it works | Project use |
        |---|---|---|
        | **Dense** | Semantic vector similarity | Finds meaning-related chunks |
        | **MMR** | Relevance + diversity | Reduces repetitive results |
        | **BM25** | Keyword-based lexical retrieval | Finds exact technical terms |
        | **Hybrid** | Dense + BM25 | Combines semantic + keyword retrieval |
        | **Cosine Similarity** | Measures vector direction similarity | Final reranking of candidates |
        """
    )


# =========================================================
# CHAT HISTORY
# =========================================================
if st.session_state.messages:
    st.markdown("### 💬 Conversation")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================
query = st.chat_input(
    "Ask something about LoRA, QLoRA, RAG, or Transformers..."
)


# =========================================================
# QUESTION PROCESSING
# =========================================================
if query:
    start_time = time.perf_counter()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    try:
        # -------------------------------------------------
        # RETRIEVAL
        # -------------------------------------------------
        if retrieval_method == "Dense":
            results = vectorstore.similarity_search(query, k=3)

        elif retrieval_method == "MMR":
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 3,
                    "fetch_k": 6,
                },
            )
            results = retriever.invoke(query)

        else:
            dense_results = vectorstore.similarity_search(query, k=3)

            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)

            top_indices = np.argsort(bm25_scores)[::-1][:3]
            bm25_results = [chunks[i] for i in top_indices]

            combined = dense_results + bm25_results

            unique_results = []
            seen = set()

            for doc in combined:
                key = (
                    doc.metadata.get("paper_title"),
                    doc.metadata.get("page_number"),
                    doc.page_content[:120],
                )

                if key not in seen:
                    seen.add(key)
                    unique_results.append(doc)

            results = unique_results[:3]

        # -------------------------------------------------
        # COSINE SIMILARITY RERANKING
        # -------------------------------------------------
        query_vector = np.array(
            embeddings.embed_query(query),
            dtype=float,
        )

        scored_results = []

        for doc in results:
            doc_vector = np.array(
                embeddings.embed_query(doc.page_content),
                dtype=float,
            )

            denominator = (
                np.linalg.norm(query_vector)
                * np.linalg.norm(doc_vector)
            )

            similarity = (
                float(np.dot(query_vector, doc_vector) / denominator)
                if denominator
                else 0.0
            )

            scored_results.append((doc, similarity))

        scored_results.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        final_results = scored_results[:3]

        # -------------------------------------------------
        # CONTEXT
        # -------------------------------------------------
        context = "\n\n".join(
            f"Paper: {doc.metadata.get('paper_title')}\n"
            f"Page: {doc.metadata.get('page_number')}\n"
            f"Content: {doc.page_content}"
            for doc, _ in final_results
        )

        history = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in st.session_state.messages[-4:]
        )

        prompt = f"""
You are a research paper question-answering assistant.

Conversation history:
{history}

Answer the current question using ONLY the research paper context below.

Current question:
{query}

Research Paper Context:
{context}

Rules:
1. Use only the provided research paper context.
2. Do not add information from your own knowledge.
3. If the answer is not available in the context, say:
"I could not find the answer in the provided research papers."
4. Give a clear and concise answer.
5. Do not invent paper titles or page numbers.
"""

        # -------------------------------------------------
        # STREAMING RESPONSE
        # -------------------------------------------------
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in llm.stream(prompt):
                text = getattr(chunk, "content", "")

                if text:
                    full_response += text
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        elapsed = time.perf_counter() - start_time

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )

        st.session_state.last_response_time = elapsed
        st.session_state.last_sources = [
            doc for doc, _ in final_results
        ]
        st.session_state.last_scores = [
            score for _, score in final_results
        ]

        # -------------------------------------------------
        # RESPONSE INFO
        # -------------------------------------------------
        st.divider()

        info1, info2, info3 = st.columns(3)

        with info1:
            st.info(f"⚡ **Response:** {elapsed:.2f}s")

        with info2:
            st.info(f"🔎 **Retrieval:** {retrieval_method}")

        with info3:
            st.info(f"📚 **Sources:** {len(final_results)}")


        # -------------------------------------------------
        # TOP 3 SOURCES
        # -------------------------------------------------
        st.markdown("### 📄 Top 3 Sources")

        for i, (doc, score) in enumerate(final_results, 1):
            title = html.escape(
                str(doc.metadata.get("paper_title", "Unknown Paper"))
            )
            page = doc.metadata.get("page_number", "?")

            st.markdown(
                f"""
                <div class="card">
                    <div class="source-number">SOURCE {i}</div>
                    <div class="source-title">{title}</div>
                    <div class="source-meta">
                        📄 Page {page}
                        &nbsp;&nbsp;•&nbsp;&nbsp;
                        <span class="score">
                            Cosine Similarity: {score:.3f}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"View retrieved content — Source {i}"):
                st.write(doc.page_content[:1200])


    except Exception as e:
        st.error(
            "The request could not be completed. "
            "Please check that Ollama is running and the required models are installed."
        )
        st.caption(str(e))


# =========================================================
# PROJECT INFORMATION
# =========================================================
st.divider()

tab1, tab2, tab3 = st.tabs(
    ["🏗️ Architecture", "📋 Project Details", "🧪 Experiments"]
)

with tab1:
    st.markdown("### End-to-End Architecture")
    st.write(
        "PDF Research Papers → PyPDFLoader → Recursive Chunking → "
        "Nomic Embeddings → ChromaDB → Dense/MMR/Hybrid Retrieval → "
        "Cosine Similarity Reranking → Top-3 Context → Gemma 3 1B → Answer"
    )

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("**Retrieval Layer**")
        st.write("• Dense semantic search")
        st.write("• MMR retrieval")
        st.write("• BM25 keyword search")
        st.write("• Hybrid retrieval")
        st.write("• Cosine similarity reranking")

    with a2:
        st.markdown("**Generation Layer**")
        st.write("• Local Gemma 3 1B")
        st.write("• Context-only prompting")
        st.write("• Conversational session memory")
        st.write("• Streaming output")
        st.write("• Source attribution")

with tab2:
    st.markdown("### Project Statistics")

    st.write("**Domain:** Generative AI / RAG / NLP")
    st.write("**Papers:** 4")
    st.write("**Pages:** 86")
    st.write("**Chunks:** 649")
    st.write("**Vector Database:** ChromaDB")
    st.write("**Embedding Model:** Nomic Embed Text")
    st.write("**Embedding Dimension:** 768")
    st.write("**LLM:** Gemma 3 1B")
    st.write("**Framework:** LangChain")
    st.write("**UI:** Streamlit")

    st.warning(
        "Commercial embedding API was not evaluated because an external "
        "commercial API key was not available in the development environment."
    )

with tab3:
    st.markdown("### Implemented Experiments")

    experiments = [
        "Two local embedding models tested: Nomic Embed Text and mxbai-embed-large",
        "Dense vector retrieval",
        "MMR retrieval",
        "BM25 keyword retrieval",
        "Hybrid Dense + BM25 retrieval",
        "Cosine similarity reranking",
        "10-question RAG evaluation",
        "Top-3 source attribution",
        "Failure-case testing",
        "Conversational memory",
    ]

    for item in experiments:
        st.write(f"✅ {item}")

st.caption(
    "Research Paper Answer Bot • Local RAG System • "
    "ChromaDB + LangChain + Ollama"
)
