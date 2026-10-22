import streamlit as st
import os
import re
import time
import logging
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any

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
    page_title="Wall St. AI Analyst",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    /* Default Button State */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #f0f2f6; /* Light gray background */
        color: #0f172a;            /* Dark slate text - THIS FIXES THE INVISIBILITY */
        border: 1px solid #d1d5db; /* Light gray border */
        font-weight: 600;          /* Makes the text slightly bolder for readability */
        transition: all 0.2s ease-in-out; /* Smooth hover transition */
    }
    
    /* Hover State */
    .stButton>button:hover {
        background-color: #e2e8f0; /* Slightly darker gray on hover */
        color: #000000;            /* Pure black text on hover */
        border-color: #94a3b8;     /* Darker border on hover */
    }
    
    .reportview-container {
        background: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

if "OPENAI_API_KEY" not in os.environ:
    st.error("❌ OPENAI_API_KEY missing. Please check Space Settings.")
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
    bm25_retrievers: Optional[dict] = None
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
        chat_response = Settings.llm.chat([
            ChatMessage(
                role="system",
                content="You are a helpful Wall Street Financial Analyst AI. Politely answer general questions."
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

    chat_tmpl = build_synthesis_chat_template(
        user_query=user_query,
        market_data=tool_results.get("market_data", ""),
        rag_content=tool_results.get("financial_rag", "")
    )

    t_synth_start = time.perf_counter()
    final_answer = Settings.llm.structured_predict(
        FinalAnswer,
        chat_tmpl
    )
    t_synth = time.perf_counter() - t_synth_start

    # P2.3 Deterministic Numerical Grounding Check
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
        "crag_status": crag_eval.get("status", "N/A"),
        "crag_top_score": crag_eval.get("top_score", 0.0),
        "grounding": grounding,
        "grounding_ratio": grounding.get("ratio", 1.0)
    }

    return AgentResponse(
        answer=final_answer.answer,
        sources=list(dict.fromkeys(tool_results.get("sources", []))),
        context_used=tool_results.get("context_used", []),
        scratchpad=final_answer.scratchpad,
        cited_tickers=final_answer.cited_tickers,
        telemetry=telemetry
    )

# --- 5. UI LOGIC ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bullish.png", width=80)
    
    st.markdown("### 🧠 Agent Capabilities")
    
    st.info("**Deep Dive (10-K Reports)**")
    st.markdown("I have ingested the full SEC 10-K filings for the following companies:")
    st.markdown("- 🍎 **Apple (AAPL)**\n- 🚗 **Tesla (TSLA)**\n- 🎮 **Nvidia (NVDA)**")
    
    st.success("**Live Market Data**")
    st.markdown("I can fetch real-time trading metrics for **all companies listed on the NASDAQ**.")

    if pinecone_index is None:
        st.warning("⚠️ 10-K RAG search is offline (Pinecone index connection not established). Market data remains active.")
    
    st.markdown("---")
    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main Hero Section
st.title("🏛️ Wall St. AI Analyst")
st.markdown("""
Welcome! This hybrid AI agent bridges the gap between **Real-Time Market Data** and **Deep 10-K Analysis**. 
It utilizes a dynamic routing engine to fetch real-time quantitative metrics via `yfinance` and qualitative insights from a Pinecone Vector Database.
""")

# Sample Questions Section
with st.expander("💡 View Sample Questions", expanded=True):
    st.markdown("""
    **Try asking about Qualitative 10-K Data:**
    * *"What are the primary supply chain risks mentioned in Apple's latest 10-K?"*
    * *"Who is the CEO of Nvidia and what is their strategy?"*
    
    **Try asking for Real-Time Quantitative Data:**
    * *"What is the current PE ratio and market cap of Tesla?"*
    * *"Fetch the trading volume and 52-week high for Microsoft."*
    
    **Try a Hybrid Search (Live Data + RAG):**
    * *"Compare the competitive threats facing Tesla with its current stock price."*
    """)
    
    # Single Automated Action Button
    if st.button("🚀 Auto-Run a Complex Query: Compare Apple & Tesla Risks"):
        prompt = "Compare the supply chain risks of Apple and Tesla."
    else:
        prompt = None

# Chat State & Abuse Guard State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") or message.get("context") or message.get("scratchpad") or message.get("telemetry"):
            render_audit_trail(message)

# Handle Input (Button or Text)
if user_input := st.chat_input("Ask a financial question...") or prompt:
    final_query = prompt if prompt else user_input

    # OWASP LLM10 Cost & Abuse Guard
    is_valid, error_msg = check_query_budget(final_query, st.session_state["query_count"])
    if not is_valid:
        st.error(f"⚠️ {error_msg}")
        st.stop()
    st.session_state["query_count"] += 1
    
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)

    with st.chat_message("assistant"):
        with st.status("🧠 Analyzing 10-Ks and Market Data...", expanded=True) as status:
            try:
                response = run_agent(
                    final_query,
                    pinecone_index,
                    nasdaq_df,
                    bm25_retrievers=bm25_retrievers
                )
                status.update(label="✅ Analysis Complete", state="complete", expanded=False)
            except Exception as e:
                logger.error("Agent execution failed: %s", e)
                st.error(f"Error: {e}")
                status.update(label="❌ Error", state="error")
                st.stop()

        # The answer prints outside the status block so it is immediately visible!
        st.markdown(response.answer)
        render_audit_trail(response)
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.answer,
            "sources": response.sources,
            "context": response.context_used,
            "scratchpad": response.scratchpad,
            "cited_tickers": response.cited_tickers,
            "telemetry": response.telemetry,
        })
