"""
Calibrate CRAG Confidence Gate thresholds (P2.2).
Runs questions from evaluation/phase3_validation_set.csv across Pinecone dense retrieval,
computes top-score and coverage distributions, and identifies separation thresholds
between answerable and trap (unanswerable) questions.
"""
import os
import re
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.openai import OpenAIEmbedding

from rag_guardrails import resolve_tickers, route_query, load_env

def calibrate():
    load_env()
    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        print("Missing API keys in environment.")
        return

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    pc = Pinecone(api_key=pinecone_key)
    index = VectorStoreIndex.from_vector_store(
        vector_store=PineconeVectorStore(pinecone_index=pc.Index("financial-rag-agent"))
    )

    df_path = PROJECT_ROOT / "evaluation" / "phase3_validation_set.csv"
    if not df_path.exists():
        print(f"Validation set not found at {df_path}")
        return

    val_df = pd.read_csv(df_path)
    csv_path = PROJECT_ROOT / "nasdaq-listed.csv"
    nasdaq_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    results = []

    print("\n--- Running CRAG Calibration over 20 Validation Questions ---")
    for i, row in val_df.iterrows():
        query = row["user_input"]
        expected_resp = str(row["response"]).strip()
        is_trap = (expected_resp.lower() == "data not available in the context.")

        tickers = resolve_tickers(query, nasdaq_df)
        if not tickers:
            tickers = ["AAPL", "TSLA", "NVDA"]

        top_scores = []
        all_scores = []

        for ticker in tickers:
            if ticker in {"AAPL", "TSLA", "NVDA"}:
                try:
                    filters = MetadataFilters(filters=[ExactMatchFilter(key="ticker", value=ticker)])
                    retriever = index.as_retriever(similarity_top_k=8, filters=filters)
                    nodes = retriever.retrieve(query)
                    scores = [n.score for n in nodes if n.score is not None]
                    if scores:
                        top_scores.append(max(scores))
                        all_scores.extend(scores)
                except Exception as e:
                    print(f"Retrieval error for {ticker}: {e}")

        best_score = max(top_scores) if top_scores else 0.0
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        results.append({
            "idx": i + 1,
            "query": query,
            "is_trap": is_trap,
            "top_score": round(best_score, 4),
            "avg_score": round(avg_score, 4),
            "num_scores": len(all_scores),
        })

    res_df = pd.DataFrame(results)
    print("\nDetailed Question Results:")
    print(res_df[["idx", "is_trap", "top_score", "avg_score", "query"]].to_string())

    traps = res_df[res_df["is_trap"]]
    answerable = res_df[~res_df["is_trap"]]

    print("\n--- Score Distribution Summary ---")
    print("Trap Questions (N={}):".format(len(traps)))
    print("  Top score min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
        traps["top_score"].min(), traps["top_score"].max(), traps["top_score"].mean()
    ))
    print("Answerable Questions (N={}):".format(len(answerable)))
    print("  Top score min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
        answerable["top_score"].min(), answerable["top_score"].max(), answerable["top_score"].mean()
    ))

if __name__ == "__main__":
    calibrate()
