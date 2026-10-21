# 🏛️ Wall St. AI Analyst: Hybrid RAG & Live Market Data Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-0f172a?style=flat-square&logo=python&logoColor=3776AB)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-0f172a?style=flat-square&logo=llamaindex&logoColor=white)](https://www.llamaindex.ai/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-0f172a?style=flat-square&logo=pinecone&logoColor=000000)](https://www.pinecone.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o_Mini-0f172a?style=flat-square&logo=openai&logoColor=white)](https://openai.com/)
[![Ragas Evaluation](https://img.shields.io/badge/Ragas_Faithfulness-83.5%25-0f172a?style=flat-square&logo=target&logoColor=white)](https://github.com/explodinggradients/ragas)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-0f172a?style=flat-square&logo=streamlit&logoColor=FF4B4B)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Live_Space-0f172a?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/spaces/Aarya003/Financial-Analyst-Agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-0f172a?style=flat-square&logo=open-source-initiative&logoColor=white)](LICENSE)

## 🚀 Live Interactive Demo
**Interact with the deployed agent here:** [Wall St. AI Analyst on Hugging Face Spaces](https://huggingface.co/spaces/Aarya003/Financial-Analyst-Agent)

> **Note on Cold Starts:** Free-tier Hugging Face Spaces spin down into a sleep state after a period of inactivity. If the app is sleeping, waking the container takes approximately 30–60 seconds on first launch.

## 📌 Project Overview
Standard Retrieval-Augmented Generation (RAG) applications fail in the financial sector because they attempt to extract highly volatile, real-time numbers (like stock prices or PE ratios) from static, outdated PDF documents. 

The **Wall St. AI Analyst** solves this by implementing a **Semantic Routing Architecture**. It dynamically intercepts user queries and routes them to the appropriate specialized tool: a Vector Database for deep, qualitative 10-K analysis, or a live financial API for quantitative, real-time trading metrics.

## 📂 Repository Structure
```text
financial-ai-agent/
│
├── app.py                                 # Streamlit UI & agent routing application
├── rag_guardrails.py                      # Headless guardrails, symbol lookup, & safe metrics
├── nasdaq-listed.csv                      # Universe of supported tickers for Market Data
├── requirements.txt                       # Production deployment dependencies
├── requirements-dev.txt                   # Developer and testing dependencies (pytest)
│
├── evaluation/                            
│   ├── phase3_validation_set.csv          # Synthetic evaluation questions & contexts
│   └── phase3_ragas_scores_adjusted.csv   # Audited Ragas scores reflecting honest refusals
│
├── notebooks/                             
│   └── Financial_RAG.ipynb                # Ingestion, architecture, & Ragas evaluation flow
│
├── tests/                                 # Unit & regression test suite (100% headless)
│   ├── test_rag_guardrails.py             # Quota, regex, and error-isolation tests
│   └── test_evaluation_baseline.py        # Benchmark score integrity tests
│
└── README.md                              # Project documentation
```

## 🏗️ System Architecture & Engineering Decisions

This agent was built with a decoupled architecture prioritizing strict boundary enforcement to eliminate LLM hallucinations.

### 1. The Ingestion Pipeline (Handling Massive Financial Tables)

Financial SEC 10-K reports contain "Consolidated Financial Statements"—massive grids of data. Standard character-based text splitters destroy these tables by cutting them in half, causing the LLM to lose column headers and hallucinate numbers.

* **Solution:** Utilizes LlamaIndex's `MarkdownElementNodeParser`. This structural parser treats Markdown tables as atomic elements, refusing to split them.
* **Trade-off & Fix:** This resulted in massive context chunks (up to 3,000 tokens per table). To ensure the LLM found the exact data without overflowing context windows, the retrieval engine fetches an expanded window (`similarity_top_k=8`) coupled with strict Metadata Filtering (e.g., `ticker='AAPL'`) to prevent cross-company data contamination.

### 2. The Pydantic Semantic Router

The core decision engine is powered by an `OpenAIPydanticProgram`. Instead of letting the LLM guess how to answer, the router enforces strict tool selection:

* **Tool 1 (`financial_rag`):** Triggered for internal company details, historical revenue, risks, and strategy. Queries Pinecone vector index.
* **Tool 2 (`market_data`):** Triggered for real-time trading metrics (Price, Market Cap, Volume). Queries `yfinance.fast_info` (<150ms latency) with graceful fallback.
* **Hybrid Mode:** If a user asks *"Compare Tesla's current price to its supply chain risks"*, the router triggers both tools and synthesizes the final comparative answer.
* **Cost & Abuse Guard:** Protected against OWASP LLM10 unbounded consumption with per-query length limits (1,000 chars) and per-session quotas (25 queries).

## 📊 Rigorous Evaluation Pipeline (Ragas)

An AI agent is only as good as its validation. The system was evaluated against a custom synthetic benchmark of 20 high-complexity financial questions using the **Ragas** framework, utilizing `gpt-4o` as the judge.

### Overcoming the "Honest Refusal" Penalty

To prevent hallucinations, the agent was strictly prompted: *"If the data is missing, simply output: 'Data not available in the context.'"* While the agent successfully followed this rule, standard Ragas algorithms penalized these honest refusals with a 0.0 Answer Relevancy score because the evaluator LLM could not reverse-engineer the original question from a refusal.

Our audited evaluation pipeline adjusts for these programmatic refusals while preserving genuine scores on partial answers:

* **Faithfulness: 0.8349 (83.5%)** — 83.5% of generated claims are grounded in the retrieved 10-K context (evaluated across 18 non-null rows; 2 rows are unanswerable edge-cases yielding NaNs).
* **Answer Relevancy: 0.8983 (89.8%)** — 89.8% direct alignment with user intent, accounting for honest partial refusals.

## 💻 Tech Stack

* **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `gpt-4o`, `text-embedding-3-small`)
* **Orchestration:** LlamaIndex (`v0.10+`)
* **Vector Database:** Pinecone
* **Market Data API:** yfinance (`fast_info`)
* **Evaluation Framework:** Ragas, Pandas, AsyncIO
* **Frontend/Deployment:** Streamlit, Hugging Face Spaces
* **Testing:** Pytest

## 🛠️ Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/aarya-pabha/financial-ai-agent.git
   cd financial-ai-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt   # for tests
   ```

3. Run the headless test suite:
   ```bash
   pytest
   ```

4. Configure environment variables:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export PINECONE_API_KEY="pcsk_..."
   ```

5. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
