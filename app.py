import streamlit as st
import os
import logging
from pathlib import Path
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

from rag_guardrails import (
    check_query_budget,
    get_symbol_from_csv,
    safe_extract_market_data,
    build_synthesis_messages,
)

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
class AgentResponse(BaseModel):
    """
    Structured output for the financial agent. 
    Contains the synthesized natural language answer, the list of cited sources, 
    and the raw context chunks used to formulate the answer.
    """
    answer: str
    sources: List[str]
    context_used: List[str]

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
@st.cache_resource(show_spinner=False)
def initialize_resources():
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=0, timeout=30.0, max_retries=2)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)

    # Locate CSV directly anchored to app location
    csv_path = Path(__file__).resolve().parent / "nasdaq-listed.csv"
    if csv_path.exists():
        nasdaq_df = pd.read_csv(csv_path)
        nasdaq_df.columns = [c.strip() for c in nasdaq_df.columns]
    else:
        logger.warning("nasdaq-listed.csv not found at %s", csv_path)
        nasdaq_df = pd.DataFrame()

    # Connect to Pinecone
    try:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone Key Missing")
        pc = Pinecone(api_key=api_key)
        index = VectorStoreIndex.from_vector_store(
            vector_store=PineconeVectorStore(pinecone_index=pc.Index("financial-rag-agent"))
        )
    except Exception as e:
        logger.warning("Pinecone vector store unavailable: %s", e)
        index = None
    
    return nasdaq_df, index

# Silently load resources
nasdaq_df, pinecone_index = initialize_resources()

# --- 4. HELPER FUNCTIONS ---
def get_tickers_from_query(query: str, index, df) -> List[str]:
    program = OpenAIPydanticProgram.from_defaults(
        output_cls=TickerExtraction,
        prompt_template_str="Identify all companies in query: {query_str}. Return list.",
        llm=Settings.llm
    )
    raw_entities = program(query_str=query).symbols
    valid_tickers = []
    for entity in raw_entities:
        ticker = get_symbol_from_csv(entity, df)
        if not ticker and len(entity) <= 5:
            ticker = entity.upper()
        if ticker:
            valid_tickers.append(ticker)
    
    if not valid_tickers and index:
        try:
            nodes = index.as_retriever(similarity_top_k=1).retrieve(query)
            if nodes and nodes[0].metadata.get("ticker"):
                valid_tickers.append(nodes[0].metadata.get("ticker"))
        except Exception as e:
            logger.debug("Index ticker fallback retrieval failed: %s", e)
    return list(set(valid_tickers))

# --- 5. TOOLS ---
def get_market_data(query: str, index, df) -> str:
    tickers = get_tickers_from_query(query, index, df)
    if not tickers:
        return "No companies found."
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            results.append(safe_extract_market_data(ticker, stock))
        except Exception as e:
            logger.warning("Failed to retrieve market data for %s: %s", ticker, e)
            results.append(f"{ticker}: market data currently unavailable")
    return "\n".join(results)

def get_financial_rag(query: str, index, df) -> dict:
    target_tickers = get_tickers_from_query(query, index, df)
    SUPPORTED = ["AAPL", "TSLA", "NVDA"]
    payload = {"content": "", "sources": [], "raw_nodes": []}
    
    if index is None:
        payload["content"] = "[NOTICE: 10-K vector database is currently offline. Deep report analysis is unavailable.]\n"
        return payload

    for ticker in target_tickers:
        if ticker not in SUPPORTED:
            payload["content"] += f"\n[NOTE: No 10-K report available for {ticker}.]\n"
            continue
            
        filters = MetadataFilters(filters=[ExactMatchFilter(key="ticker", value=ticker)])
        engine = index.as_query_engine(similarity_top_k=3, filters=filters)
        resp = engine.query(query)
        
        payload["content"] += f"\n--- {ticker} 10-K Data ---\n{resp.response}\n"
        for n in resp.source_nodes:
            payload["sources"].append(f"{n.metadata.get('company')} 10-K")
            payload["raw_nodes"].append(n.node.get_text())
            
    return payload

# --- 6. AGENT LOGIC ---
def run_agent(user_query: str, index, df) -> AgentResponse:
    router_prompt = """
    Route the user query to the correct tool based on these strict definitions:
    1. "financial_rag": Company internal details (Revenue, Risks, Strategy, CEO).
    2. "market_data": Real-Time Trading Metrics (Price, PE, Volume) ONLY.
    3. "general_chat": Non-business questions.
    Query: {query_str}
    """
    router = OpenAIPydanticProgram.from_defaults(
        output_cls=RoutePrediction,
        prompt_template_str=router_prompt,
        llm=Settings.llm
    )
    tools = router(query_str=user_query).tools
    
    results = {}
    sources = []
    context_used = []
    
    if "market_data" in tools:
        res = get_market_data(user_query, index, df)
        results["market_data"] = res
        context_used.append(res)
        sources.append("Real-time Market Data")
        
    if "financial_rag" in tools:
        res = get_financial_rag(user_query, index, df)
        results["financial_rag"] = res["content"]
        sources.extend(res["sources"])
        context_used.extend(res["raw_nodes"])
        
    messages = build_synthesis_messages(user_query, results)
    chat_response = Settings.llm.chat(messages)
    response_text = chat_response.message.content
    return AgentResponse(answer=response_text, sources=list(set(sources)), context_used=context_used)

def render_audit_trail(sources: List[str], context: List[str]):
    """
    Unified audit trail rendering across both historical conversation and live agent outputs.
    """
    with st.expander("🔍 Audit Trail (Read the Source Data)"):
        st.markdown("### 📚 Cited Sources")
        st.write(sources)
        st.divider()
        st.markdown("### 📄 Raw Context Snippets")
        for i, ctx in enumerate(context):
            st.caption(f"**Context Fragment {i+1}:**")
            st.text(str(ctx))

# --- 7. UI LOGIC ---
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
        if "sources" in message and "context" in message:
            render_audit_trail(message["sources"], message["context"])

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
                response = run_agent(final_query, pinecone_index, nasdaq_df)
                status.update(label="✅ Analysis Complete", state="complete", expanded=False)
            except Exception as e:
                logger.error("Agent execution failed: %s", e)
                st.error(f"Error: {e}")
                status.update(label="❌ Error", state="error")
                st.stop()

        # The answer prints outside the status block so it is immediately visible!
        st.markdown(response.answer)
        render_audit_trail(response.sources, response.context_used)
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.answer,
            "sources": response.sources,
            "context": response.context_used
        })
