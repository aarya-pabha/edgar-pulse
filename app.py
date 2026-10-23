import streamlit as st
import os
import re
import ast
import time
import html
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any, Callable

from rag_guardrails import (
    load_env,
    get_llm,
    check_query_budget,
    check_prompt_injection,
    resolve_tickers,
    route_query,
    build_synthesis_chat_template,
    execute_parallel_tools,
    get_market_data,
    get_financial_rag,
    assess_retrieval,
    verify_numbers,
    render_audit_trail,
    CRAG_LOW_THRESHOLD,
    FinalAnswer,
    AgentResponse,
)

# Load local environment variables from .env if present
load_env()

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EDGAR Pulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional dark theme, typography, buttons & avatars
st.markdown("""
    <style>
    /* Geist & Geist Mono Font Stack */
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

    /* Base Container & Typography */
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"],
    .stMarkdown,
    p, span, label, input {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #131217;
        color: #ededed;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] {
        background-color: #1c1b20;
        border-right: 1px solid #2b2b30;
    }

    /* Item 39: Constrained Reading Measure (1088px max, 74ch prose measure) */
    .block-container,
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1088px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Constrain chat message prose measure to prevent ultrawide eye fatigue */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        max-width: 74ch;
        line-height: 1.6;
        letter-spacing: -0.005em;
    }

    /* Tabular Numerics Font Stack */
    code, pre,
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    [data-testid="stMetric"] *,
    [data-testid="stTable"],
    [data-testid="stDataFrame"],
    table, th, td,
    .tabular-nums,
    .ep-tabular,
    .ep-badge,
    .ep-metric,
    [data-testid="stBadge"],
    .stCaption {
        font-family: 'Geist Mono', monospace;
        font-variant-numeric: tabular-nums;
    }

    /* Scoped Button Hierarchy - Secondary Defaults */
    .stButton > button,
    .stButton > button[data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-secondary"],
    .ep-btn-secondary {
        border-radius: 6px;
        min-height: 36px;
        background-color: #1c1b20;
        color: #ededed;
        border: 1px solid #2b2b30;
        font-weight: 500;
        padding: 0.375rem 0.875rem;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    .ep-btn-secondary:hover {
        background-color: #25242b;
        color: #ffffff;
        border-color: #3e3d47;
    }

    /* Scoped Button Hierarchy - Primary Action Buttons */
    .stButton > button[data-testid="stBaseButton-primary"],
    .stButton > [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"],
    .stButton > button[kind="primary"],
    .ep-btn-primary {
        border-radius: 6px;
        min-height: 36px;
        background-color: #141414 !important;
        color: #ffffff !important;
        border: 1px solid #3e3d47 !important;
        font-weight: 600;
        padding: 0.375rem 0.875rem;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button[data-testid="stBaseButton-primary"]:hover,
    .stButton > [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover,
    .ep-btn-primary:hover {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
        border-color: #55555c !important;
    }

    /* Item 40: High-Contrast Focus Visible Rings across all interactive inputs */
    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible,
    [tabindex="0"]:focus-visible,
    .stButton > button:focus-visible,
    [data-testid="stBaseButton-secondary"]:focus-visible,
    [data-testid="stBaseButton-primary"]:focus-visible,
    [data-testid="stDownloadButton"] > button:focus-visible,
    [data-testid="stExpander"] summary:focus-visible,
    [data-testid="stChatInput"] textarea:focus-visible {
        outline: 2px solid #1d5fbf !important;
        outline-offset: 2px !important;
    }

    /* Scoped Download Button Hierarchy */
    [data-testid="stDownloadButton"] > button {
        border-radius: 6px;
        min-height: 36px;
        background-color: #1c1b20;
        color: #ededed;
        border: 1px solid #2b2b30;
        font-weight: 500;
        padding: 0.375rem 0.875rem;
        transition: all 0.15s ease-in-out;
    }

    [data-testid="stDownloadButton"] > button:hover {
        background-color: #25242b;
        color: #ffffff;
        border-color: #3e3d47;
    }

    /* Item 27: Grounded & Unverified Numerical Annotations */
    .ep-grounded {
        border-bottom: 1.5px dotted #44a282 !important;
        font-variant-numeric: tabular-nums;
        cursor: help;
        transition: background-color 0.15s ease;
    }
    .ep-grounded:hover {
        background-color: rgba(68, 162, 130, 0.12);
    }
    .ep-unverified {
        border-bottom: 1.5px dotted #e5566f !important;
        font-variant-numeric: tabular-nums;
        cursor: help;
        transition: background-color 0.15s ease;
    }
    .ep-unverified:hover {
        background-color: rgba(229, 86, 111, 0.12);
    }

    /* Custom Chat Avatars: YOU / EP Typographic Markers */
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #1c1b20 !important;
        border: 1px solid #2b2b30 !important;
        border-radius: 6px !important;
        color: #ededed !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 2rem !important;
        height: 2rem !important;
        overflow: hidden !important;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #1c1b20 !important;
        border: 1px solid #1d5fbf !important;
        border-radius: 6px !important;
        color: #1d5fbf !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 2rem !important;
        height: 2rem !important;
        overflow: hidden !important;
    }

    /* Suppress default Streamlit robot / user icons (SVG, Material Icon span, img) */
    [data-testid="stChatMessageAvatarUser"] > *,
    [data-testid="stChatMessageAvatarAssistant"] > *,
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] > *,
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] > *,
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        display: none !important;
    }

    [data-testid="stChatMessageAvatarUser"]::after,
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"]::after {
        content: "YOU";
        font-family: 'Geist Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        line-height: 1;
        color: #ededed;
        letter-spacing: 0.02em;
    }

    [data-testid="stChatMessageAvatarAssistant"]::after,
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"]::after {
        content: "EP";
        font-family: 'Geist Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        line-height: 1;
        color: #1d5fbf;
        letter-spacing: 0.02em;
    }

    /* Live Telemetry Pulsing Pip */
    .ep-live-pip {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #44a282;
        box-shadow: 0 0 6px rgba(68, 162, 130, 0.6);
        animation: ep-pulse 2s infinite ease-in-out;
    }

    @keyframes ep-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(0.85); }
    }

    /* Precision Icon Weight & Micro-Bezel Hierarchy (high-end-visual-design & shadcn) */
    .material-symbols-rounded,
    .material-symbols-outlined,
    [data-testid="stIconMaterial"] {
        font-variation-settings: 'FILL' 0, 'wght' 200, 'GRAD' 0, 'opsz' 18 !important;
        font-size: 14px !important;
        vertical-align: middle !important;
    }

    .ep-icon-shell-blue {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 4px;
        background: rgba(29, 95, 191, 0.1);
        border: 1px solid rgba(29, 95, 191, 0.25);
        flex-shrink: 0;
    }

    .ep-icon-shell-emerald {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 4px;
        background: rgba(68, 162, 130, 0.1);
        border: 1px solid rgba(68, 162, 130, 0.25);
        flex-shrink: 0;
    }
    </style>
""", unsafe_allow_html=True)

if "OPENAI_API_KEY" not in os.environ:
    st.markdown("""
    <div style="background: #1c1b20; border: 1px solid #e5566f; border-radius: 8px; padding: 24px; max-width: 600px; margin: 2rem auto; font-family: 'Geist', sans-serif;">
        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 1.1rem; color: #ededed; margin-bottom: 8px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e5566f" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
            Configuration Required: OpenAI API Key Missing
        </div>
        <div style="font-size: 0.85rem; color: #88888e; line-height: 1.5; margin-bottom: 16px;">
            EDGAR Pulse requires an <code style="color: #1d5fbf; font-family: 'Geist Mono', monospace;">OPENAI_API_KEY</code> environment variable to perform structured routing, RAG synthesis, and embeddings generation.
        </div>
        <div style="background: #131217; border: 1px solid #2b2b30; border-radius: 6px; padding: 12px; font-family: 'Geist Mono', monospace; font-size: 0.75rem; color: #ededed; margin-bottom: 12px;">
            # Add to your .env file or environment:<br>
            OPENAI_API_KEY=sk-...<br>
            PINECONE_API_KEY=pcsk_... # Optional (Local BM25 fallback active without Pinecone)
        </div>
        <div style="font-size: 0.75rem; color: #88888e;">
            After configuring, refresh this page to connect to the EDGAR Pulse terminal.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 2. DATA MODELS (WITH REQUIRED DOCSTRINGS) ---
class TickerExtraction(BaseModel):
    """
    Extracts a list of stock tickers or company names mentioned in the user's query.
    Used to identify which companies the user wants to research.
    """
    symbols: List[str] = Field(description="List of stock tickers or company names.")

class RoutePrediction(BaseModel):
    """
    Determines which tools to use based on the user's query.
    Can select multiple tools if the query requires both financial RAG and market data.
    """
    tools: List[Literal["financial_rag", "market_data", "general_chat"]] = Field(description="List of selected tools.")

# --- 3. CACHED INITIALIZATION ---
@st.cache_data(show_spinner=False)
def load_nasdaq_data() -> pd.DataFrame:
    """
    Caches the NASDAQ symbols DataFrame using st.cache_data (Context7: serializable tabular data).
    """
    csv_path = Path(__file__).resolve().parent / "nasdaq-listed.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        return df
    logger.warning("nasdaq-listed.csv not found at %s", csv_path)
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def initialize_pinecone_index():
    """
    Caches the Pinecone VectorStoreIndex connection using st.cache_resource (Context7: shared singleton).
    """
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-6-luna")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    Settings.llm = get_llm(LLM_MODEL, timeout=30.0, max_retries=2)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)

    try:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone Key Missing")
        pc = Pinecone(api_key=api_key)
        return VectorStoreIndex.from_vector_store(
            vector_store=PineconeVectorStore(pinecone_index=pc.Index("financial-rag-agent"))
        )
    except Exception as e:
        logger.warning("Pinecone vector store unavailable: %s", e)
        return None

def initialize_resources():
    """Backward-compatible helper returning (nasdaq_df, pinecone_index)."""
    return load_nasdaq_data(), initialize_pinecone_index()

@st.cache_resource(show_spinner=False)
def load_bm25_retrievers() -> dict:
    """
    Loads pre-exported 10-K document nodes from data/ and initializes BM25 retrievers (P2.1).
    Enables sub-millisecond local hybrid search and offline resilience.
    """
    retrievers = {}
    data_dir = Path(__file__).resolve().parent / "data"
    for ticker in ["AAPL", "TSLA", "NVDA"]:
        file_path = data_dir / f"nodes_{ticker}.json"
        if file_path.exists():
            try:
                import json
                from llama_index.core.schema import TextNode
                from llama_index.retrievers.bm25 import BM25Retriever
                with open(file_path, "r", encoding="utf-8") as f:
                    node_dicts = json.load(f)
                nodes = [TextNode.from_dict(d) for d in node_dicts]
                retrievers[ticker] = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=8)
            except Exception as e:
                logger.warning("Failed to load BM25 retriever for %s: %s", ticker, e)
    return retrievers

# Silently load resources
nasdaq_df = load_nasdaq_data()
pinecone_index = initialize_pinecone_index()
bm25_retrievers = load_bm25_retrievers()

# --- 4. AGENT LOGIC ---
def run_agent(
    user_query: str,
    index: Any,
    df: pd.DataFrame,
    bm25_retrievers: Optional[dict] = None,
    status_cb: Optional[Callable[[str], None]] = None
) -> AgentResponse:
    """
    Orchestrates the financial agent pipeline:
    1. First-line prompt injection guard (P2.4).
    2. Rules-first router with LLM fallback (P1.2).
    3. Deterministic ticker resolution once per query, never guessing (P1.1).
    4. Concurrent execution of market_data and financial_rag with hybrid BM25 (P1.3, P2.1).
    5. CRAG confidence gate evaluating dense retrieval quality (P2.2).
    6. Single structured synthesis call using FinalAnswer (P1.4).
    7. Deterministic numerical grounding verification & telemetry (P2.3, P2.5).
    """
    t_start = time.perf_counter()

    # P2.4 Prompt-injection First-Line Guard
    is_safe, injection_msg = check_prompt_injection(user_query)
    if not is_safe:
        return AgentResponse(
            answer="Suspicious prompt injection detected. Your query was blocked by the security guardrail.",
            sources=[],
            context_used=[],
            scratchpad="Security guardrail blocked adversarial prompt injection pattern.",
            cited_tickers=[],
            telemetry={"blocked": True, "llm_calls": 0}
        )

    router_prompt = (
        "Route the user query to the correct tool based on these strict definitions:\n"
        "1. 'financial_rag': Company internal details (Revenue, Risks, Strategy, CEO).\n"
        "2. 'market_data': Real-Time Trading Metrics (Price, PE, Volume) ONLY.\n"
        "3. 'general_chat': Non-business questions.\n"
        "Query: {query_str}"
    )

    def router_fallback(q: str) -> List[str]:
        try:
            res = Settings.llm.structured_predict(
                RoutePrediction,
                PromptTemplate(router_prompt),
                query_str=q
            )
            return res.tools
        except Exception as e:
            logger.debug("Router LLM fallback failed: %s", e)
            return ["financial_rag"]

    t_router_start = time.perf_counter()
    tools = route_query(user_query, llm_fallback=router_fallback)
    t_router = time.perf_counter() - t_router_start

    if tools == ["general_chat"]:
        if status_cb:
            status_cb("Phase 1/4 · Routing: General conversational query identified.")
        chat_response = Settings.llm.chat([
            ChatMessage(
                role="system",
                content="You are EDGAR Pulse, a financial AI analyst specializing in SEC Form 10-K filings and real-time market data. Politely answer general questions."
            ),
            ChatMessage(role="user", content=user_query)
        ])
        return AgentResponse(
            answer=chat_response.message.content,
            sources=["General Knowledge"],
            context_used=[],
            scratchpad="General inquiry - retrieval skipped.",
            cited_tickers=[],
            telemetry={
                "tools": tools,
                "router_ms": round(t_router * 1000, 1),
                "retrieval_ms": 0.0,
                "synthesis_ms": round((time.perf_counter() - t_start - t_router) * 1000, 1),
                "llm_calls": 1
            }
        )

    def llm_ticker_fallback(q: str) -> List[str]:
        try:
            res = Settings.llm.structured_predict(
                TickerExtraction,
                PromptTemplate("Identify all company names or stock symbols in: {query_str}. Return list of symbols."),
                query_str=q
            )
            return res.symbols
        except Exception as e:
            logger.debug("LLM ticker fallback failed: %s", e)
            return []

    tickers = resolve_tickers(user_query, df, llm_fallback=llm_ticker_fallback)
    if status_cb:
        if tickers:
            status_cb(f"Phase 1/4 · Routing & Resolution: Tools: `{', '.join(tools)}` · Target equities: `{', '.join(tickers)}`")
        else:
            status_cb(f"Phase 1/4 · Routing: Tools: `{', '.join(tools)}` · No verified equities identified.")

    if not tickers:
        return AgentResponse(
            answer="Which company do you mean? Please specify a company name or ticker symbol (e.g., Apple, TSLA, NVDA).",
            sources=[],
            context_used=[],
            scratchpad="No verified company tickers identified in query.",
            cited_tickers=[],
            telemetry={
                "tools": tools,
                "router_ms": round(t_router * 1000, 1),
                "retrieval_ms": 0.0,
                "synthesis_ms": 0.0,
                "llm_calls": 0
            }
        )

    need_pe = bool(re.search(r"\b(pe|p/e|valuation|earnings)\b", user_query, re.IGNORECASE))

    if status_cb:
        status_cb("Phase 2/4 · Retrieval: Executing concurrent hybrid retrieval (Dense Pinecone + Local BM25)...")
    t_retrieval_start = time.perf_counter()
    tool_results = execute_parallel_tools(
        tools=tools,
        query=user_query,
        tickers=tickers,
        market_fn=lambda t, pe: get_market_data(t, need_pe=pe),
        rag_fn=lambda q, t: get_financial_rag(q, t, index, bm25_retrievers=bm25_retrievers),
        need_pe=need_pe
    )
    t_retrieval = time.perf_counter() - t_retrieval_start

    # P2.2 CRAG Confidence Gate
    dense_nodes = tool_results.get("dense_nodes", [])
    crag_eval = assess_retrieval(user_query, dense_nodes)

    has_market_data = bool(
        tool_results.get("market_data") and
        tool_results.get("market_data") != "No companies found." and
        "unavailable" not in tool_results.get("market_data", "").lower()
    )

    if crag_eval["status"] == "INSUFFICIENT" and not has_market_data:
        if status_cb:
            status_cb(f"Phase 3/4 · CRAG Gate: Context quality below threshold ({crag_eval['top_score']:.2f} < {CRAG_LOW_THRESHOLD}). Initiating honest refusal.")
        return AgentResponse(
            answer="Data not available in the context.",
            sources=[],
            context_used=[],
            scratchpad=f"CRAG gate INSUFFICIENT (top score {crag_eval['top_score']:.4f} < {CRAG_LOW_THRESHOLD}). Immediate refusal with 0 LLM calls.",
            cited_tickers=[],
            telemetry={
                "tools": tools,
                "tickers": tickers,
                "router_ms": round(t_router * 1000, 1),
                "retrieval_ms": round(t_retrieval * 1000, 1),
                "synthesis_ms": 0.0,
                "llm_calls": 0,
                "crag_status": crag_eval["status"],
                "crag_top_score": crag_eval["top_score"],
                "grounding": {"ratio": 1.0, "grounded": [], "unverified": []},
                "grounding_ratio": 1.0
            }
        )

    if status_cb:
        if "financial_rag" in tools and dense_nodes:
            status_cb(f"Phase 3/4 · CRAG Gate: Context quality verified ({crag_eval['status']}, Score: {crag_eval['top_score']:.2f}). Proceeding to synthesis.")
        elif has_market_data:
            status_cb("Phase 3/4 · CRAG Gate: Real-time market feed verified. Proceeding to synthesis.")
        else:
            status_cb("Phase 3/4 · Context Validation: Audited context verified for synthesis.")

    chat_tmpl = build_synthesis_chat_template(
        user_query=user_query,
        market_data=tool_results.get("market_data", ""),
        rag_content=tool_results.get("financial_rag", "")
    )

    if status_cb:
        status_cb("Phase 4/4 · Synthesis: Synthesizing grounded financial analysis...")
    t_synth_start = time.perf_counter()
    raw_answer = Settings.llm.structured_predict(
        FinalAnswer,
        chat_tmpl
    )
    t_synth = time.perf_counter() - t_synth_start

    if isinstance(raw_answer, FinalAnswer):
        final_answer = raw_answer
    elif hasattr(raw_answer, "answer"):
        final_answer = FinalAnswer(
            scratchpad=getattr(raw_answer, "scratchpad", "") or "",
            answer=getattr(raw_answer, "answer", "") or str(raw_answer),
            cited_tickers=getattr(raw_answer, "cited_tickers", []) or tickers
        )
    elif isinstance(raw_answer, dict):
        final_answer = FinalAnswer(
            scratchpad=raw_answer.get("scratchpad", "") or "",
            answer=raw_answer.get("answer", str(raw_answer)),
            cited_tickers=raw_answer.get("cited_tickers", []) or tickers
        )
    else:
        final_answer = FinalAnswer(
            scratchpad="Synthesized directly from context.",
            answer=str(raw_answer),
            cited_tickers=tickers
        )

    # P2.3 Deterministic Numerical Grounding Check
    if status_cb:
        status_cb("Phase 4/4 · Verification: Validating numerical figures against audited context...")
    grounding = verify_numbers(
        answer=final_answer.answer,
        chunks=tool_results.get("context_used", []),
        market_data=tool_results.get("market_data", "")
    )

    telemetry = {
        "tools": tools,
        "tickers": tickers,
        "router_ms": round(t_router * 1000, 1),
        "retrieval_ms": round(t_retrieval * 1000, 1),
        "synthesis_ms": round(t_synth * 1000, 1),
        "llm_calls": 1,
        "rag_skipped": "financial_rag" not in tools,
        "prompt_cache_eligible": len(tool_results.get("financial_rag", "")) > 1000,
        "crag_status": crag_eval.get("status", "N/A"),
        "crag_top_score": crag_eval.get("top_score", 0.0),
        "grounding": grounding,
        "grounding_ratio": grounding.get("ratio", 1.0),
        "market_data": tool_results.get("market_data", "")
    }

    return AgentResponse(
        answer=final_answer.answer,
        sources=list(dict.fromkeys(tool_results.get("sources", []))),
        context_used=tool_results.get("context_used", []),
        scratchpad=final_answer.scratchpad,
        cited_tickers=final_answer.cited_tickers,
        telemetry=telemetry
    )

# --- 4.5 FINANCIAL UTILITIES & FORMATTING (ITEMS 25, 27, 28, 30, 31) ---

def parse_market_metrics(raw: Any) -> List[dict]:
    """
    Extracts structured financial market dictionaries from market_data string or context list (Item 25).
    """
    if not raw:
        return []
    items = []
    targets = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [str(raw)])
    for t in targets:
        if not isinstance(t, str):
            continue
        for match in re.finditer(r"\{[^{}]*['\"]Ticker['\"][^{}]*\}", t):
            try:
                d = ast.literal_eval(match.group(0))
                if isinstance(d, dict) and "Ticker" in d:
                    if not any(item["Ticker"] == d["Ticker"] for item in items):
                        items.append(d)
            except Exception:
                pass
    return items

def format_metric_val(val: Any, prefix: str = "", suffix: str = "", is_currency: bool = False) -> str:
    """
    Formats financial numbers with institutional notation (T, B, M, tabular).
    """
    if val is None or val == "N/A" or str(val).strip() == "":
        return "N/A"
    try:
        num = float(val)
        if is_currency:
            if abs(num) >= 1e12:
                return f"${num/1e12:,.2f}T"
            elif abs(num) >= 1e9:
                return f"${num/1e9:,.2f}B"
            elif abs(num) >= 1e6:
                return f"${num/1e6:,.2f}M"
            else:
                return f"${num:,.2f}"
        if abs(num) >= 1e9:
            return f"{num/1e9:,.2f}B{suffix}"
        elif abs(num) >= 1e6:
            return f"{num/1e6:,.1f}M{suffix}"
        elif suffix == "x":
            return f"{num:.1f}x"
        return f"{prefix}{num:,.2f}{suffix}"
    except (ValueError, TypeError):
        return f"{prefix}{val}{suffix}"

def render_market_metrics_strip(metrics: List[dict]):
    """
    Renders structured quantitative financial metric cards before narrative prose (Item 25).
    Safely escapes all dynamic metrics to prevent HTML injection / XSS.
    """
    if not metrics:
        return
    for m in metrics:
        ticker = html.escape(str(m.get("Ticker", "UNKNOWN")))
        price = html.escape(format_metric_val(m.get("Price"), is_currency=True))
        mcap = html.escape(format_metric_val(m.get("Market Cap"), is_currency=True))
        pe = html.escape(format_metric_val(m.get("PE Ratio"), suffix="x"))
        high52 = html.escape(format_metric_val(m.get("52w High"), is_currency=True))
        vol = html.escape(format_metric_val(m.get("Volume"), suffix=""))

        st.markdown(f"""
        <div style="background: #131217; border: 1px solid #2b2b30; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 18px; align-items: center;">
            <div style="border-right: 1px solid #2b2b30; padding-right: 14px;">
                <div style="font-family: 'Geist', sans-serif; font-size: 0.95rem; font-weight: 700; color: #1d5fbf; line-height: 1;">{ticker}</div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.65rem; color: #88888e; margin-top: 2px;">NASDAQ</div>
            </div>
            <div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.62rem; color: #88888e; text-transform: uppercase;">Price</div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.92rem; font-weight: 600; color: #ededed;">{price}</div>
            </div>
            <div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.62rem; color: #88888e; text-transform: uppercase;">P/E Ratio</div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.92rem; font-weight: 600; color: #ededed;">{pe}</div>
            </div>
            <div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.62rem; color: #88888e; text-transform: uppercase;">Market Cap</div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.92rem; font-weight: 600; color: #ededed;">{mcap}</div>
            </div>
            <div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.62rem; color: #88888e; text-transform: uppercase;">52W High</div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.92rem; font-weight: 600; color: #ededed;">{high52}</div>
            </div>
            <div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.62rem; color: #88888e; text-transform: uppercase;">Volume</div>
                <div style="font-family: 'Geist Mono', monospace; font-size: 0.92rem; font-weight: 600; color: #ededed;">{vol}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def sanitize_untrusted_markdown(text: str) -> str:
    """
    Sanitizes untrusted text by neutralizing HTML tags (< and >) and ampersands
    to prevent XSS / HTML injection when rendering with st.markdown(..., unsafe_allow_html=True).
    Preserves Markdown syntax while defanging HTML tags and script injections.
    """
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def annotate_grounded_numbers(answer: str, grounding_info: Optional[dict]) -> str:
    """
    Subtly annotates verified numbers with dotted underlines in the displayed response (Item 27).
    Pure text in state/history is preserved unmutated for export and evaluation integrity.
    Neutralizes untrusted HTML in the source answer before applying annotation spans.
    Prevents KaTeX math mode parser collisions on currency figures (e.g. $320 ... $324).
    """
    if not answer:
        return ""

    annotated = sanitize_untrusted_markdown(answer)
    if not grounding_info:
        return annotated

    grounded = grounding_info.get("grounded", [])
    unverified = grounding_info.get("unverified", [])
    if not grounded and not unverified:
        return annotated

    seen = set()
    for g in sorted(set(grounded), key=len, reverse=True):
        if not g or len(g.strip()) <= 1:
            continue
        clean_g = g.lstrip("$").strip()
        if not clean_g or clean_g in seen:
            continue
        # Avoid false positive matching on SEC form names like 10-K or standalone small year numbers
        if clean_g in ("10", "10-K", "10-Q", "2023", "2024", "2025"):
            continue
        seen.add(clean_g)
        # Capture optional leading dollar sign so currency figures remain unified inside the span
        pattern = re.compile(rf"(?<![\d\w>])(?<!10-)(\$)?({re.escape(clean_g)})(?!-K|-Q)(?![\d\w<])", re.IGNORECASE)
        replacement = r'<span class="ep-grounded" title="Audited: Grounded in SEC 10-K / Market Data" style="border-bottom: 1.5px dotted #44a282; font-variant-numeric: tabular-nums;">\1\2</span>'
        annotated = pattern.sub(replacement, annotated)

    for u in sorted(set(unverified), key=len, reverse=True):
        if not u or len(u.strip()) <= 1:
            continue
        clean_u = u.lstrip("$").strip()
        if not clean_u or clean_u in seen:
            continue
        if clean_u in ("10", "10-K", "10-Q", "2023", "2024", "2025"):
            continue
        seen.add(clean_u)
        pattern = re.compile(rf"(?<![\d\w>])(?<!10-)(\$)?({re.escape(clean_u)})(?!-K|-Q)(?![\d\w<])", re.IGNORECASE)
        replacement = r'<span class="ep-unverified" title="Unverified: Outside audited tolerance" style="border-bottom: 1.5px dotted #e5566f; font-variant-numeric: tabular-nums;">\1\2</span>'
        annotated = pattern.sub(replacement, annotated)

    return annotated

def render_footnote_anchors(sources: List[str]):
    """
    Renders clickable footnote citation badges linking to audit data (Item 28).
    Safely escapes source labels to prevent HTML injection / XSS.
    """
    if not sources:
        return
    badges = []
    for idx, s in enumerate(sources, 1):
        s_safe = html.escape(str(s))
        if "10-K" in s or "Apple" in s or "Tesla" in s or "Nvidia" in s:
            badge_text = f"[{idx}] {s_safe}"
        elif "Market" in s:
            badge_text = f"[{idx}] Real-Time NASDAQ"
        else:
            badge_text = f"[{idx}] {s_safe}"
        badges.append(f'<span style="font-family: \'Geist Mono\', monospace; font-size: 0.68rem; color: #88888e; background: #131217; border: 1px solid #2b2b30; border-radius: 4px; padding: 2px 7px; margin-right: 6px;">{badge_text}</span>')

    st.markdown(f"""
    <div style="margin-top: 8px; margin-bottom: 4px; display: flex; flex-wrap: wrap; gap: 4px; align-items: center;">
        <span style="font-family: 'Geist Mono', monospace; font-size: 0.65rem; color: #1d5fbf; font-weight: 600; margin-right: 6px; letter-spacing: 0.04em;">AUDITED SOURCES</span>
        {''.join(badges)}
    </div>
    """, unsafe_allow_html=True)

def render_assistant_payload(message: dict):
    """
    Unified presentation helper for assistant responses in both history replay and live execution (Finding 6).
    Renders market metrics cards, grounding annotations, refusal notes, footnotes, and audit trail.
    """
    # 1. Quantitative market metric card strip (Item 25)
    m_data = message.get("telemetry", {}).get("market_data") or message.get("context", [])
    m_metrics = parse_market_metrics(m_data)
    if m_metrics:
        render_market_metrics_strip(m_metrics)

    # 2. Numerical grounding underline annotations (Item 27)
    g_info = message.get("telemetry", {}).get("grounding")
    annotated_content = annotate_grounded_numbers(message.get("content", ""), g_info)
    # Neutralize KaTeX math-mode delimiter collisions on currency figures ($...$)
    safe_markdown = re.sub(r"(?<!\\)\$", "&#36;", annotated_content)
    st.markdown(safe_markdown, unsafe_allow_html=True)

    # 3. Contextual Refusal Guidance (Item 41, Finding 5)
    if "Data not available in the context." in message.get("content", ""):
        st.markdown("""
        <div style="background: #1c1b20; border: 1px solid #2b2b30; border-radius: 6px; padding: 10px 14px; margin-top: 8px; margin-bottom: 8px; font-family: 'Geist', sans-serif; font-size: 0.78rem; color: #88888e; line-height: 1.45;">
            <div style="display: flex; align-items: center; gap: 6px; color: #1d5fbf; font-weight: 600; font-family: 'Geist Mono', monospace; font-size: 0.7rem; margin-bottom: 4px;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#1d5fbf" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                SEC FILING COVERAGE NOTICE
            </div>
            Deep SEC Form 10-K RAG retrieval is currently indexed for <strong style="color: #ededed;">AAPL</strong> (Apple), <strong style="color: #ededed;">TSLA</strong> (Tesla), and <strong style="color: #ededed;">NVDA</strong> (NVIDIA). Real-time market metrics remain active across all 5,295+ NASDAQ equities.
        </div>
        """, unsafe_allow_html=True)

    # 4. Footnote citation anchors (Item 28)
    if message.get("sources"):
        render_footnote_anchors(message["sources"])

    # 5. Full audit trail (Item 32-38)
    if message.get("sources") or message.get("context") or message.get("scratchpad") or message.get("telemetry"):
        render_audit_trail(message)

def get_contextual_followups(answer: str, cited_tickers: List[str], query: str = "") -> List[str]:
    """
    Generates 2 smart contextual follow-up query suggestions (Item 31).
    """
    tickers = [t.upper() for t in (cited_tickers or [])]
    q_lower = query.lower()

    if len(tickers) >= 2 or ("compare" in q_lower or "vs" in q_lower):
        t1 = tickers[0] if tickers else "AAPL"
        t2 = tickers[1] if len(tickers) > 1 else "TSLA"
        return [
            f"Compare {t1} vs {t2} valuation metrics (P/E and Market Cap)",
            f"Analyze key competitive risk differences between {t1} and {t2}"
        ]

    if "AAPL" in tickers:
        if "risk" in q_lower or "supply" in q_lower:
            return [
                "What are Apple's revenue trends by geographic segment in the 10-K?",
                "What is Apple's current P/E ratio and 52-week trading range?"
            ]
        else:
            return [
                "What are Apple's primary supply chain risks from the 10-K?",
                "Compare Apple's valuation metrics with Tesla"
            ]
    elif "TSLA" in tickers:
        if "risk" in q_lower or "threat" in q_lower:
            return [
                "What is Tesla's current P/E ratio, market cap, and trading volume?",
                "How does Tesla describe its autonomous vehicle regulatory risks?"
            ]
        else:
            return [
                "What are the major supply chain and manufacturing risks for Tesla?",
                "Compare Tesla's valuation and growth metrics with Apple"
            ]
    elif "NVDA" in tickers:
        if "strategy" in q_lower or "ceo" in q_lower:
            return [
                "What are NVIDIA's main data center concentration risks in the 10-K?",
                "Fetch NVIDIA's real-time market cap, volume, and P/E ratio"
            ]
        else:
            return [
                "Who is NVIDIA's CEO and what is their AI enterprise strategy?",
                "Compare NVIDIA's valuation and risks with Apple and Tesla"
            ]
    elif tickers:
        t = tickers[0]
        return [
            f"Fetch {t} real-time stock price, P/E ratio, and 52-week high",
            f"Compare {t} with peer valuation and industry market cap"
        ]

    return [
        "Compare supply chain risks between Apple and Tesla",
        "Fetch real-time valuation metrics for TSLA and NVDA"
    ]

def generate_research_brief(messages: List[dict]) -> str:
    """
    Generates a formal, audited Markdown research brief from the current session turns (Item 30).
    """
    if not messages:
        return "# EDGAR Pulse - Research Brief\n\nNo active research inquiries in this session."

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    user_queries = [m for m in messages if m.get("role") == "user"]
    assistant_responses = [m for m in messages if m.get("role") == "assistant"]

    all_tickers = set()
    for m in assistant_responses:
        for t in m.get("cited_tickers", []):
            all_tickers.add(t)
    tickers_str = ", ".join(sorted(all_tickers)) if all_tickers else "N/A"

    lines = [
        "# EDGAR Pulse - Research Brief & Verification Audit",
        f"**Generated:** {now_utc}  ",
        f"**Session Scope:** {len(user_queries)} inquiries analyzed | **Tickers Covered:** `{tickers_str}`  ",
        "**Verification Engine:** EDGAR Pulse (Dense Pinecone + BM25 RRF + yfinance + CRAG Gate + Numerical Grounding)  ",
        "",
        "---",
        ""
    ]

    q_idx = 1
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "user":
            user_text = msg.get("content", "").strip()
            asst_msg = None
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                asst_msg = messages[i + 1]
                i += 1

            lines.append(f"## {q_idx}. Inquiry: {user_text}")
            lines.append("")

            if asst_msg:
                lines.append("### Executive Synthesis")
                lines.append(asst_msg.get("content", "").strip())
                lines.append("")

                lines.append("### Provenance & Audit Metadata")
                telem = asst_msg.get("telemetry") or {}
                cited_t = asst_msg.get("cited_tickers") or []
                sources = asst_msg.get("sources") or []
                grounding = telem.get("grounding") or {}
                ratio = telem.get("grounding_ratio", grounding.get("ratio", 1.0))

                lines.append(f"- **Cited Tickers:** `{', '.join(cited_t)}`" if cited_t else "- **Cited Tickers:** None")
                lines.append(f"- **Data Sources:** {', '.join(sources)}" if sources else "- **Data Sources:** General Knowledge")
                lines.append(f"- **Numerical Grounding Ratio:** {ratio * 100:.1f}% verified against audited context")
                if grounding.get("grounded"):
                    lines.append(f"  - *Verified Figures:* {', '.join(grounding.get('grounded'))}")
                if grounding.get("unverified"):
                    lines.append(f"  - *Unverified Figures:* {', '.join(grounding.get('unverified'))}")

                r_ms = telem.get("router_ms", 0.0)
                ret_ms = telem.get("retrieval_ms", 0.0)
                s_ms = telem.get("synthesis_ms", 0.0)
                lines.append(f"- **Latency Breakdown:** Router: `{r_ms}ms` | Retrieval: `{ret_ms}ms` | Synthesis: `{s_ms}ms`")

                if asst_msg.get("scratchpad"):
                    lines.append(f"- **Reasoning Scratchpad:** *\"{asst_msg.get('scratchpad').strip()}\"*")

            lines.append("")
            lines.append("---")
            lines.append("")
            q_idx += 1
        i += 1

    lines.append("*End of Audited Research Brief - Generated by EDGAR Pulse*")
    return "\n".join(lines)

# --- 5. UI LOGIC ---
with st.sidebar:
    # Local Vector Monogram: EP geometric lockup in #1d5fbf
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.25rem;">
        <svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="38" height="38" rx="8" fill="#1c1b20" stroke="#2b2b30" stroke-width="1.5"/>
            <!-- E glyph in #1d5fbf -->
            <path d="M10 11.5H19.5V14.2H13.2V17.2H18.5V19.8H13.2V22.8H19.5V25.5H10V11.5Z" fill="#1d5fbf"/>
            <!-- P glyph in #1d5fbf -->
            <path d="M21.5 11.5H27C28.933 11.5 30.5 12.8 30.5 14.8V16.8C30.5 18.8 28.933 20.2 27 20.2H24.4V25.5H21.5V11.5ZM24.4 13.9V17.8H26.7C27.4732 17.8 28.1 17.2 28.1 16.4V15.2C28.1 14.4 27.4732 13.9 26.7 13.9H24.4Z" fill="#1d5fbf"/>
            <!-- Delta market accent glyph -->
            <path d="M28.5 7.5L31.5 11.5H25.5L28.5 7.5Z" fill="#1d5fbf"/>
        </svg>
        <div>
            <div style="font-family: 'Geist', sans-serif; font-weight: 700; font-size: 1.05rem; color: #ededed; letter-spacing: -0.01em; line-height: 1.2;">EDGAR PULSE</div>
            <div style="display: flex; align-items: center; gap: 5px; font-family: 'Geist Mono', monospace; font-size: 0.65rem; color: #44a282; letter-spacing: 0.04em;">
                <span class="ep-live-pip"></span> SYSTEM OPERATIONAL
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Ingested 10-K Corpora Panel
    st.markdown("""
    <div style="background: #131217; border: 1px solid #2b2b30; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px; font-family: 'Geist', sans-serif; font-size: 0.75rem; font-weight: 600; color: #ededed; letter-spacing: 0.02em; margin-bottom: 8px;">
            <span class="ep-icon-shell-blue">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#1d5fbf" stroke-width="1.35"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
            </span>
            Ingested 10-K Corpora
        </div>
        <table style="width: 100%; border-collapse: collapse; font-family: 'Geist Mono', monospace; font-size: 0.72rem; color: #ededed;">
            <tr style="border-bottom: 1px solid #222226; color: #88888e;">
                <th style="text-align: left; padding: 3px 0;">Ticker</th>
                <th style="text-align: left; padding: 3px 4px;">Filing</th>
                <th style="text-align: right; padding: 3px 0;">Nodes</th>
            </tr>
            <tr style="border-bottom: 1px solid #1c1b20;">
                <td style="padding: 4px 0; font-weight: 600; color: #1d5fbf;">AAPL</td>
                <td style="padding: 4px 4px; color: #c4c4c8;">Apple Inc. FY23</td>
                <td style="padding: 4px 0; text-align: right; color: #88888e;">145</td>
            </tr>
            <tr style="border-bottom: 1px solid #1c1b20;">
                <td style="padding: 4px 0; font-weight: 600; color: #1d5fbf;">NVDA</td>
                <td style="padding: 4px 4px; color: #c4c4c8;">NVIDIA Corp. FY24</td>
                <td style="padding: 4px 0; text-align: right; color: #88888e;">215</td>
            </tr>
            <tr>
                <td style="padding: 4px 0; font-weight: 600; color: #1d5fbf;">TSLA</td>
                <td style="padding: 4px 4px; color: #c4c4c8;">Tesla, Inc. FY23</td>
                <td style="padding: 4px 0; text-align: right; color: #88888e;">251</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    nasdaq_count = len(nasdaq_df) if not nasdaq_df.empty else 4000
    st.markdown(f"""
    <div style="background: #131217; border: 1px solid #2b2b30; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px; font-family: 'Geist', sans-serif; font-size: 0.75rem; font-weight: 600; color: #ededed; letter-spacing: 0.02em; margin-bottom: 6px;">
            <span class="ep-icon-shell-emerald">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#44a282" stroke-width="1.35"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            </span>
            Real-Time Market Feed
        </div>
        <div style="font-family: 'Geist', sans-serif; font-size: 0.78rem; color: #c4c4c8; line-height: 1.4;">
            Active coverage across <span style="font-family: 'Geist Mono', monospace; font-weight: 600; color: #ededed;">{nasdaq_count:,}</span> NASDAQ-listed equities via <code style="font-size: 0.72rem; color: #1d5fbf; background: rgba(29, 95, 191, 0.1); padding: 1px 4px; border-radius: 3px;">yfinance</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Runtime Architecture Drawer
    with st.expander("Runtime Architecture & Specs", expanded=False):
        pinecone_status = "Online (Connected)" if pinecone_index is not None else "Fallback (Local BM25)"
        st.markdown(f"""
        <div style="font-family: 'Geist Mono', monospace; font-size: 0.72rem; color: #88888e; line-height: 1.6;">
            <div><strong style="color: #ededed;">Model:</strong> gpt-6-luna (OpenAI)</div>
            <div><strong style="color: #ededed;">Embeddings:</strong> text-embedding-3-small</div>
            <div><strong style="color: #ededed;">Retriever:</strong> Dense + BM25 RRF (k=60)</div>
            <div><strong style="color: #ededed;">CRAG Gate:</strong> 0.70 high / 0.40 low</div>
            <div><strong style="color: #ededed;">Grounding:</strong> ±0.5% tolerance</div>
            <div><strong style="color: #ededed;">Vector DB:</strong> {pinecone_status}</div>
        </div>
        """, unsafe_allow_html=True)

    if pinecone_index is None:
        st.warning("SEC 10-K vector search is offline (Pinecone index connection not established). Market data remains active.")

    # Finding 1: Reactive Session Budget Meter via placeholder
    quota_placeholder = st.empty()
    def render_quota_meter(holder, current_q: int, max_q: int = 30):
        r = min(1.0, current_q / max_q)
        p = round(r * 100, 1)
        holder.markdown(f"""
        <div style="background: #131217; border: 1px solid #2b2b30; border-radius: 6px; padding: 10px 12px; margin-top: 14px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: #88888e; margin-bottom: 6px;">
                <span>SESSION QUOTA</span>
                <span style="color: #ededed; font-weight: 600;">{current_q} / {max_q}</span>
            </div>
            <div style="width: 100%; height: 4px; background: #222226; border-radius: 2px; overflow: hidden;">
                <div style="width: {p}%; height: 100%; background: {'#e5566f' if r > 0.8 else '#1d5fbf'}; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_quota_meter(quota_placeholder, st.session_state.get("query_count", 0))

    # Two-Step Clear History Confirmation
    with st.popover("Clear Conversation", use_container_width=True):
        st.markdown("<div style='font-family: Geist; font-size: 0.8rem; color: #ededed; margin-bottom: 0.5rem;'>Reset conversation history and session query quota?</div>", unsafe_allow_html=True)
        if st.button("Confirm Reset", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state["query_count"] = 0
            st.rerun()

    # One-Click Research Brief Export
    if st.session_state.get("messages"):
        brief_md = generate_research_brief(st.session_state.messages)
        st.download_button(
            label="Export Research Brief (.md)",
            data=brief_md,
            file_name=f"edgar_pulse_brief_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

# Main Header & Scope Bar (Compact Noun Header)
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #2b2b30; padding-bottom: 0.75rem; margin-bottom: 1rem;">
    <div>
        <span style="font-family: 'Geist', sans-serif; font-size: 1.5rem; font-weight: 700; color: #ededed; letter-spacing: -0.02em;">EDGAR Pulse</span>
        <span style="font-family: 'Geist', sans-serif; font-size: 0.85rem; color: #88888e; margin-left: 0.75rem;">SEC Form 10-K & Real-Time Market Terminal</span>
    </div>
    <div style="display: flex; align-items: center; gap: 6px; font-family: 'Geist Mono', monospace; font-size: 0.68rem; color: #44a282; background: rgba(68, 162, 130, 0.08); border: 1px solid rgba(68, 162, 130, 0.25); padding: 3px 8px; border-radius: 4px;">
        <span class="ep-live-pip"></span> PIPELINE ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)

# Chat State & Abuse Guard State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0

if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    prompt = st.session_state.pop("pending_prompt")
else:
    prompt = None

# Auto-collapsing Onboarding Matrix: expanded on first run, collapsed once chat starts
is_chat_active = len(st.session_state.messages) > 0
with st.expander("Sample Financial Inquiries & Capability Matrix", expanded=not is_chat_active):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 7px; margin-bottom: 0.45rem;">
            <span class="ep-icon-shell-blue">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#1d5fbf" stroke-width="1.35"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
            </span>
            <span style="font-family: 'Geist Mono', monospace; font-size: 0.68rem; font-weight: 600; color: #1d5fbf; letter-spacing: 0.05em;">SEC 10-K FILINGS</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Apple Supply Chain Risks", key="btn_q1", use_container_width=True):
            st.session_state["pending_prompt"] = "What are the primary supply chain risks mentioned in Apple's latest 10-K?"
            st.rerun()
        if st.button("NVIDIA CEO & AI Strategy", key="btn_q2", use_container_width=True):
            st.session_state["pending_prompt"] = "Who is the CEO of Nvidia and what is their strategy?"
            st.rerun()

    with col2:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 7px; margin-bottom: 0.45rem;">
            <span class="ep-icon-shell-emerald">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#44a282" stroke-width="1.35"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            </span>
            <span style="font-family: 'Geist Mono', monospace; font-size: 0.68rem; font-weight: 600; color: #44a282; letter-spacing: 0.05em;">NASDAQ METRICS</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Tesla Valuation & P/E Ratio", key="btn_q3", use_container_width=True):
            st.session_state["pending_prompt"] = "What is the current PE ratio and market cap of Tesla?"
            st.rerun()
        if st.button("Microsoft 52-Week Range", key="btn_q4", use_container_width=True):
            st.session_state["pending_prompt"] = "Fetch the trading volume and 52-week high for Microsoft."
            st.rerun()

    with col3:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 7px; margin-bottom: 0.45rem;">
            <span class="ep-icon-shell-blue">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#1d5fbf" stroke-width="1.35"><circle cx="7" cy="12" r="3"></circle><circle cx="17" cy="12" r="3"></circle><line x1="10" y1="12" x2="14" y2="12"></line></svg>
            </span>
            <span style="font-family: 'Geist Mono', monospace; font-size: 0.68rem; font-weight: 600; color: #1d5fbf; letter-spacing: 0.05em;">HYBRID RAG</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Compare Apple vs Tesla Risks", key="btn_q5", type="primary", use_container_width=True):
            st.session_state["pending_prompt"] = "Compare the supply chain risks of Apple and Tesla."
            st.rerun()
        if st.button("Tesla Stock vs Threats", key="btn_q6", use_container_width=True):
            st.session_state["pending_prompt"] = "Compare the competitive threats facing Tesla with its current stock price."
            st.rerun()

# Display History (Finding 6: Unified presentation helper + Interactive Actions on Latest Turn)
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_payload(message)

            is_latest = (idx == len(st.session_state.messages) - 1)
            # Interactive Ticker Disambiguation on Latest Turn
            if is_latest and message.get("disambiguation"):
                q_text = message.get("query", "")
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 6px; margin: 0.5rem 0 0.35rem 0;">
                    <span class="ep-icon-shell-blue">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#1d5fbf" stroke-width="1.35"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>
                    </span>
                    <span style="font-family: 'Geist Mono', monospace; font-size: 0.68rem; font-weight: 600; color: #1d5fbf; letter-spacing: 0.05em;">SELECT AN INDEXED COMPANY TO PROCEED:</span>
                </div>
                """, unsafe_allow_html=True)
                d_col1, d_col2, d_col3 = st.columns(3)
                with d_col1:
                    if st.button("Apple Inc. (AAPL)", key=f"dis_aapl_{idx}", use_container_width=True):
                        st.session_state["pending_prompt"] = f"{q_text} Apple".strip()
                        st.rerun()
                with d_col2:
                    if st.button("Tesla, Inc. (TSLA)", key=f"dis_tsla_{idx}", use_container_width=True):
                        st.session_state["pending_prompt"] = f"{q_text} Tesla".strip()
                        st.rerun()
                with d_col3:
                    if st.button("NVIDIA Corp. (NVDA)", key=f"dis_nvda_{idx}", use_container_width=True):
                        st.session_state["pending_prompt"] = f"{q_text} NVIDIA".strip()
                        st.rerun()

            # Contextual Follow-Up Suggestions on Latest Turn
            elif is_latest and message.get("followups"):
                followups = message["followups"]
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 6px; margin: 0.65rem 0 0.35rem 0;">
                    <span class="ep-icon-shell-blue">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#1d5fbf" stroke-width="1.35"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    </span>
                    <span style="font-family: 'Geist Mono', monospace; font-size: 0.68rem; font-weight: 600; color: #1d5fbf; letter-spacing: 0.05em;">RECOMMENDED FOLLOW-UP INQUIRIES:</span>
                </div>
                """, unsafe_allow_html=True)
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    if st.button(followups[0], key=f"btn_f1_{idx}", use_container_width=True):
                        st.session_state["pending_prompt"] = followups[0]
                        st.rerun()
                with f_col2:
                    if st.button(followups[1], key=f"btn_f2_{idx}", use_container_width=True):
                        st.session_state["pending_prompt"] = followups[1]
                        st.rerun()
        else:
            st.markdown(re.sub(r"(?<!\\)\$", r"\$", message["content"]))

# Live Query Budget Caption via placeholder for reactive updates
budget_caption_holder = st.empty()
def render_budget_caption(holder, q_cnt: int):
    rem = 30 - q_cnt
    holder.caption(f"Max 2,000 characters · {rem} queries remaining this session")

render_budget_caption(budget_caption_holder, st.session_state.get("query_count", 0))

# Handle Input (Button or Text)
if user_input := st.chat_input("Ask a financial question...") or prompt:
    final_query = prompt if prompt else user_input

    # OWASP LLM10 Cost & Abuse Guard
    is_valid, error_msg = check_query_budget(final_query, st.session_state["query_count"])
    if not is_valid:
        st.error(f"Budget Exceeded: {error_msg}")
        st.stop()
    
    # Finding 1: Increment query count and immediately update reactive sidebar & caption placeholders
    st.session_state["query_count"] += 1
    render_quota_meter(quota_placeholder, st.session_state["query_count"])
    render_budget_caption(budget_caption_holder, st.session_state["query_count"])
    
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(re.sub(r"(?<!\\)\$", r"\$", final_query))

    with st.chat_message("assistant"):
        t_pipeline_start = time.perf_counter()
        with st.status("Executing Financial Intelligence Engine...", expanded=True) as status:
            try:
                response = run_agent(
                    final_query,
                    pinecone_index,
                    nasdaq_df,
                    bm25_retrievers=bm25_retrievers,
                    status_cb=lambda msg: status.write(f"`{time.strftime('%H:%M:%S')}` {msg}")
                )
                t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000
                status.update(
                    label=f"Analysis Finished ({t_total_ms:.0f} ms)",
                    state="complete",
                    expanded=False
                )
            except Exception as e:
                logger.error("Agent execution failed: %s", e)
                error_msg = f"Analysis pipeline encountered an exception: {e}"
                st.error(error_msg)
                status.update(label="Execution Terminated", state="error", expanded=False)
                # Prevent orphaned user message by appending error response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "context": [],
                    "scratchpad": "Execution terminated prematurely due to an unhandled exception.",
                    "cited_tickers": [],
                    "telemetry": {"error": True}
                })
                st.stop()

        followups = get_contextual_followups(response.answer, response.cited_tickers, final_query)
        is_disambig = "Which company do you mean?" in response.answer

        # Construct assistant payload dict
        asst_payload = {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "context": response.context_used,
            "scratchpad": response.scratchpad,
            "cited_tickers": response.cited_tickers,
            "telemetry": response.telemetry,
            "followups": followups if not is_disambig and "Data not available" not in response.answer else [],
            "disambiguation": is_disambig,
            "query": final_query,
        }

        # Save to history and rerun to render through persistent message loop with interactive listeners
        st.session_state.messages.append(asst_payload)
        st.rerun()
