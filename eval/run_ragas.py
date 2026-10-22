"""
Phase 3 Ragas Evaluation Runner (P3.1, P3.2).
Evaluates the gpt-6-luna agent against the 20 benchmark questions using official Ragas,
testing against the agent's actual retrieved contexts and applying the Strict Honest Refusal Rule.
"""

import sys
import types
import logging
from typing import Any
from pathlib import Path
import pandas as pd
import numpy as np

# Runtime compatibility shim for LangChain 0.4 / Ragas 0.4
mod1 = types.ModuleType("langchain_community.chat_models.vertexai")
mod1.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = mod1

import langchain_community.llms as llms
if not hasattr(llms, "VertexAI"):
    llms.VertexAI = type("VertexAI", (), {})

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rag_guardrails import load_env
load_env()

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app import run_agent, initialize_resources, load_bm25_retrievers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ragas_eval")

def run_ragas_benchmark():
    val_path = project_root / "evaluation" / "phase3_validation_set.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation CSV not found at {val_path}")

    df_val = pd.read_csv(val_path)
    logger.info("Initializing financial agent resources for Ragas evaluation...")
    nasdaq_df, pinecone_index = initialize_resources()
    bm25_retrievers = load_bm25_retrievers()

    logger.info("Evaluating %d benchmark questions with gpt-6-luna agent...", len(df_val))

    agent_records = []
    samples_to_eval = []
    eval_indices = []

    ragas_faithfulness = {}
    ragas_relevancy = {}

    for idx, row in df_val.iterrows():
        query = str(row["user_input"]).strip()
        logger.info("[%d/%d] Running agent: %s", idx + 1, len(df_val), query[:60])
        res = run_agent(query, pinecone_index, nasdaq_df, bm25_retrievers)

        response_text = (res.answer or "").strip()
        contexts = res.context_used or []
        is_clarification = ("which company" in response_text.lower() or "are you interested in" in response_text.lower())
        is_exact_refusal = (response_text.lower() == "data not available in the context.")

        agent_records.append({
            "question_idx": idx,
            "user_input": query,
            "response": response_text,
            "scratchpad": res.scratchpad or "",
            "cited_tickers": ",".join(res.cited_tickers or []),
            "crag_status": res.telemetry.get("crag_status", "UNKNOWN") if res.telemetry else "UNKNOWN",
            "grounding_ratio": res.telemetry.get("grounding_ratio", 0.0) if res.telemetry else 0.0,
            "is_exact_refusal": is_exact_refusal,
            "is_clarification": is_clarification,
        })

        # Strict Honest Refusal Rule:
        # Programmatic honest refusals and zero-guessing clarifications receive 1.0 Relevancy.
        # Faithfulness is NaN because no factual context claims are made.
        if is_exact_refusal or is_clarification:
            ragas_faithfulness[idx] = np.nan
            ragas_relevancy[idx] = 1.0
            logger.info("Row %d: Honest refusal / clarification -> Faithfulness: NaN, Relevancy: 1.0", idx)
        else:
            if not contexts:
                contexts = ["No external filing context."]
            sample = SingleTurnSample(
                user_input=query,
                response=response_text,
                retrieved_contexts=contexts
            )
            samples_to_eval.append(sample)
            eval_indices.append(idx)

    logger.info("Submitting %d answerable queries to Ragas evaluator (gpt-4o judge)...", len(samples_to_eval))

    if samples_to_eval:
        dataset = EvaluationDataset(samples=samples_to_eval)
        evaluator_llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)
        evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        run_config = RunConfig(max_workers=4, timeout=120, max_retries=5)

        res = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config
        )
        df_ragas_out = res.to_pandas()

        for i, original_idx in enumerate(eval_indices):
            f_score = float(df_ragas_out.loc[i, "faithfulness"])
            r_score = float(df_ragas_out.loc[i, "answer_relevancy"])
            ragas_faithfulness[original_idx] = f_score
            ragas_relevancy[original_idx] = r_score

    df_scored = pd.DataFrame(agent_records)
    df_scored["ragas_faithfulness"] = [ragas_faithfulness.get(i, np.nan) for i in range(len(df_scored))]
    df_scored["ragas_answer_relevancy"] = [ragas_relevancy.get(i, np.nan) for i in range(len(df_scored))]

    out_csv = project_root / "evaluation" / "phase3_ragas_scores_gpt6_luna.csv"
    df_scored.to_csv(out_csv, index=False)
    logger.info("Saved audited Ragas scores to %s", out_csv)

    valid_f = df_scored["ragas_faithfulness"].dropna()
    valid_f_mean = valid_f.mean() if not valid_f.empty else 0.0
    conservative_f_mean = df_scored["ragas_faithfulness"].fillna(0.0).mean()
    nan_count = int(df_scored["ragas_faithfulness"].isna().sum())
    mean_relevancy = df_scored["ragas_answer_relevancy"].mean()

    print("\n" + "="*65)
    print(" RAGAS EVALUATION COMPARISON REPORT (P3.1 / P3.2)")
    print("="*65)
    print(f" Total Evaluated Rows        : {len(df_scored)}")
    print(f" Faithfulness (Valid-row)    : {valid_f_mean * 100:.2f}%  (Baseline: 83.49%)")
    print(f" Faithfulness (fillna 0.0)   : {conservative_f_mean * 100:.2f}%")
    print(f" Faithfulness NaN Count      : {nan_count}  (Baseline: 2 NaNs at Rows 14 & 19)")
    print(f" Answer Relevancy Mean       : {mean_relevancy * 100:.2f}%  (Baseline: 89.83%)")
    print("="*65 + "\n")

    return df_scored

if __name__ == "__main__":
    run_ragas_benchmark()
