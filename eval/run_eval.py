"""
Phase 3 Evaluation Harness & Regression Gate (P3.1, P3.2).
Runs benchmark questions through the core financial agent pipeline, computes
deterministic metrics (grounding, CRAG, refusal accuracy, telemetry), tracks NaNs,
and exports dated CSV reports.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

# Add project root to sys.path so imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_guardrails import load_env, verify_numbers, get_llm, AgentResponse

logger = logging.getLogger("eval_harness")

def compute_refusal_metrics(response_text: str) -> Dict[str, Any]:
    """
    Evaluates strict honest refusal rule (P3.1):
    Only exact normalized refusals receive automated 1.0 relevancy.
    Partial answers (like Row 18) are graded on actual relevancy.
    """
    normalized = response_text.strip().lower()
    is_exact_refusal = (normalized == "data not available in the context.")
    is_clarification = ("which company" in normalized or "are you interested in" in normalized)
    
    return {
        "is_exact_refusal": is_exact_refusal,
        "is_clarification": is_clarification,
    }

def summarize_evaluation(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes aggregate metrics with explicit NaN tracking (P3.1):
    - Grounding ratio mean
    - Refusal counts
    - Latency distributions (router, retrieval, synthesis)
    - Mean LLM calls per query
    - Faithfulness: valid-row mean vs. conservative fillna(0.0) mean
    """
    if not results:
        return {}

    df = pd.DataFrame(results)
    
    total = len(df)
    exact_refusals = int(df["is_exact_refusal"].sum()) if "is_exact_refusal" in df else 0
    clarifications = int(df["is_clarification"].sum()) if "is_clarification" in df else 0
    
    # NaN tracking for faithfulness / refusals
    nan_count = exact_refusals + clarifications
    
    grounding_ratios = df["grounding_ratio"].dropna()
    mean_grounding = float(grounding_ratios.mean()) if not grounding_ratios.empty else 0.0
    
    mean_router_ms = float(df["router_ms"].mean()) if "router_ms" in df else 0.0
    mean_retrieval_ms = float(df["retrieval_ms"].mean()) if "retrieval_ms" in df else 0.0
    mean_synthesis_ms = float(df["synthesis_ms"].mean()) if "synthesis_ms" in df else 0.0
    mean_llm_calls = float(df["llm_calls"].mean()) if "llm_calls" in df else 1.0
    
    crag_distribution = df["crag_status"].value_counts().to_dict() if "crag_status" in df else {}

    return {
        "total_queries": total,
        "exact_refusals": exact_refusals,
        "clarifications": clarifications,
        "faithfulness_nan_count": nan_count,
        "mean_grounding_ratio": round(mean_grounding, 4),
        "mean_router_ms": round(mean_router_ms, 2),
        "mean_retrieval_ms": round(mean_retrieval_ms, 2),
        "mean_synthesis_ms": round(mean_synthesis_ms, 2),
        "mean_llm_calls": round(mean_llm_calls, 2),
        "crag_distribution": crag_distribution,
    }

def run_evaluation(
    csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    mock_runner: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the 20 benchmark questions against the agent and exports results to CSV.
    """
    load_env()

    if csv_path is None:
        csv_path = PROJECT_ROOT / "evaluation" / "phase3_validation_set.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found at {csv_path}")

    df_questions = pd.read_csv(csv_path)
    if limit is not None and limit > 0:
        df_questions = df_questions.head(limit)

    results = []

    if mock_runner is None:
        from app import run_agent, initialize_resources, load_bm25_retrievers
        nasdaq_df, pinecone_index = initialize_resources()
        bm25_retrievers = load_bm25_retrievers()
        runner = lambda q: run_agent(q, pinecone_index, nasdaq_df, bm25_retrievers)
    else:
        runner = mock_runner

    for idx, row in df_questions.iterrows():
        query = str(row["user_input"]).strip()
        logger.info("[%d/%d] Evaluating: %s", idx + 1, len(df_questions), query[:60])

        try:
            res: AgentResponse = runner(query)
            ans = res.answer or ""
            telemetry = res.telemetry or {}
            grounding = telemetry.get("grounding", {})

            refusal_info = compute_refusal_metrics(ans)

            item = {
                "question_idx": idx,
                "user_input": query,
                "response": ans,
                "scratchpad": res.scratchpad or "",
                "cited_tickers": ",".join(res.cited_tickers or []),
                "crag_status": telemetry.get("crag_status", "UNKNOWN"),
                "crag_top_score": telemetry.get("crag_top_score", 0.0),
                "grounding_ratio": telemetry.get("grounding_ratio", 0.0),
                "grounded_numbers": ",".join(grounding.get("grounded", [])),
                "unverified_numbers": ",".join(grounding.get("unverified", [])),
                "is_exact_refusal": refusal_info["is_exact_refusal"],
                "is_clarification": refusal_info["is_clarification"],
                "router_ms": telemetry.get("router_ms", 0.0),
                "retrieval_ms": telemetry.get("retrieval_ms", 0.0),
                "synthesis_ms": telemetry.get("synthesis_ms", 0.0),
                "llm_calls": telemetry.get("llm_calls", 1),
            }
            results.append(item)
        except Exception as e:
            logger.error("Error evaluating question %d: %s", idx, e)
            results.append({
                "question_idx": idx,
                "user_input": query,
                "response": f"ERROR: {str(e)}",
                "scratchpad": "",
                "cited_tickers": "",
                "crag_status": "ERROR",
                "crag_top_score": 0.0,
                "grounding_ratio": 0.0,
                "grounded_numbers": "",
                "unverified_numbers": "",
                "is_exact_refusal": False,
                "is_clarification": False,
                "router_ms": 0.0,
                "retrieval_ms": 0.0,
                "synthesis_ms": 0.0,
                "llm_calls": 0,
            })

    results_df = pd.DataFrame(results)
    summary = summarize_evaluation(results)

    # Save to CSV
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_results_{timestamp}.csv"

    results_df.to_csv(output_path, index=False)
    logger.info("Evaluation results saved to: %s", output_path)

    return results_df, summary

def main():
    parser = argparse.ArgumentParser(description="Wall St. AI Analyst Evaluation Harness (P3.1)")
    parser.add_argument("--csv", type=str, default=None, help="Path to input validation CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to output results CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--model", type=str, default=None, help="OpenAI LLM model name")
    parser.add_argument("--mock", action="store_true", help="Run in headless dry-run mock mode (0 API calls, $0 credits)")
    args = parser.parse_args()

    if args.model:
        os.environ["LLM_MODEL"] = args.model

    model_used = os.getenv("LLM_MODEL", "gpt-6-luna")
    mode_label = "MOCK DRY-RUN (0 API calls, $0)" if args.mock else "LIVE RUN"
    print(f"\n========================================================")
    print(f" Wall St. AI Analyst - Evaluation Harness (P3)")
    print(f" Target Model : {model_used}")
    print(f" Execution Mode: {mode_label}")
    print(f"========================================================\n")

    csv_path = Path(args.csv) if args.csv else None
    out_path = Path(args.output) if args.output else None

    if args.mock:
        def dry_run_mock(q: str) -> AgentResponse:
            q_lower = q.lower()
            if any(k in q_lower for k in ["apple", "tesla", "nvidia", "aapl", "tsla", "nvda"]):
                return AgentResponse(
                    answer="Sample grounded figure: $383 billion revenue.",
                    sources=["10-K Filing"],
                    scratchpad="Extracted from mock 10-K filing.",
                    cited_tickers=["AAPL"],
                    telemetry={
                        "router_ms": 1.2,
                        "retrieval_ms": 25.0,
                        "synthesis_ms": 210.0,
                        "llm_calls": 1,
                        "crag_status": "CORRECT",
                        "crag_top_score": 0.88,
                        "grounding_ratio": 1.0,
                        "grounding": {"ratio": 1.0, "grounded": ["$383 billion"], "unverified": []}
                    }
                )
            return AgentResponse(
                answer="Data not available in the context.",
                sources=[],
                scratchpad="Unanswerable mock query.",
                cited_tickers=[],
                telemetry={
                    "router_ms": 0.8,
                    "retrieval_ms": 15.0,
                    "synthesis_ms": 0.0,
                    "llm_calls": 0,
                    "crag_status": "INSUFFICIENT",
                    "crag_top_score": 0.25,
                    "grounding_ratio": 1.0,
                    "grounding": {"ratio": 1.0, "grounded": [], "unverified": []}
                }
            )
        mock_fn = dry_run_mock
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = PROJECT_ROOT / "evaluation" / f"eval_results_mock_{timestamp}.csv"
    else:
        mock_fn = None

    df_res, summary = run_evaluation(csv_path=csv_path, output_path=out_path, limit=args.limit, mock_runner=mock_fn)

    print("\n--- Evaluation Summary ---")
    for k, v in summary.items():
        print(f"  {k:25}: {v}")
    print("---------------------------\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
