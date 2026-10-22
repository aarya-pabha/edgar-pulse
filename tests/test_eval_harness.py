"""
Unit tests for Phase 3 Evaluation Harness (evaluation/run_eval.py).
Tests strict refusal contracts, NaN tracking, aggregation logic,
and mock headless execution with zero API keys.
"""

import pytest
import pandas as pd
from pathlib import Path
from evaluation.run_eval import compute_refusal_metrics, summarize_evaluation, run_evaluation
from rag_guardrails import AgentResponse

def test_compute_refusal_metrics():
    # 1. Exact normalized refusal (Strict Honest Refusal Rule)
    exact = "Data not available in the context."
    metrics = compute_refusal_metrics(exact)
    assert metrics["is_exact_refusal"] is True
    assert metrics["is_clarification"] is False

    # Normalized with whitespace or casing
    exact_whitespace = "  data not available in the context. \n"
    assert compute_refusal_metrics(exact_whitespace)["is_exact_refusal"] is True

    # 2. Partial refusal (Row 18 scenario: Apple answered, MSFT/GOOGL refused)
    partial = "Apple faces legal proceedings risks. Data not available in the context for MSFT."
    p_metrics = compute_refusal_metrics(partial)
    assert p_metrics["is_exact_refusal"] is False
    assert p_metrics["is_clarification"] is False

    # 3. Clarification request (Zero-guessing scenario)
    clarification = "Which company or stock ticker are you interested in? (e.g., Apple / AAPL)"
    c_metrics = compute_refusal_metrics(clarification)
    assert c_metrics["is_exact_refusal"] is False
    assert c_metrics["is_clarification"] is True

def test_summarize_evaluation():
    sample_results = [
        {
            "question_idx": 0,
            "user_input": "What is Tesla revenue?",
            "response": "Tesla revenue is $96.8B.",
            "is_exact_refusal": False,
            "is_clarification": False,
            "grounding_ratio": 1.0,
            "crag_status": "CORRECT",
            "router_ms": 1.2,
            "retrieval_ms": 105.0,
            "synthesis_ms": 450.0,
            "llm_calls": 1,
        },
        {
            "question_idx": 1,
            "user_input": "What is Microsoft revenue?",
            "response": "Data not available in the context.",
            "is_exact_refusal": True,
            "is_clarification": False,
            "grounding_ratio": 1.0,
            "crag_status": "INSUFFICIENT",
            "router_ms": 0.8,
            "retrieval_ms": 50.0,
            "synthesis_ms": 0.0,
            "llm_calls": 0,
        },
        {
            "question_idx": 2,
            "user_input": "Compare risks for all companies.",
            "response": "Which company do you mean?",
            "is_exact_refusal": False,
            "is_clarification": True,
            "grounding_ratio": 0.0,
            "crag_status": "UNKNOWN",
            "router_ms": 1.0,
            "retrieval_ms": 0.0,
            "synthesis_ms": 0.0,
            "llm_calls": 0,
        },
    ]

    summary = summarize_evaluation(sample_results)
    assert summary["total_queries"] == 3
    assert summary["exact_refusals"] == 1
    assert summary["clarifications"] == 1
    assert summary["faithfulness_nan_count"] == 2
    assert summary["mean_grounding_ratio"] == round((1.0 + 1.0 + 0.0) / 3, 4)
    assert summary["mean_llm_calls"] == round((1 + 0 + 0) / 3, 2)
    assert summary["crag_distribution"]["CORRECT"] == 1
    assert summary["crag_distribution"]["INSUFFICIENT"] == 1

def test_run_evaluation_headless_mock(tmp_path):
    # Create a small dummy validation CSV
    input_csv = tmp_path / "test_val.csv"
    input_csv.write_text("user_input\nWhat is Apple's revenue?\nWhere is Microsoft located?\n", encoding="utf-8")

    out_csv = tmp_path / "output_eval.csv"

    def mock_runner(query: str) -> AgentResponse:
        if "Apple" in query:
            return AgentResponse(
                answer="Apple revenue is $383 billion.",
                sources=["AAPL_10K"],
                scratchpad="Apple 2023: 383B",
                cited_tickers=["AAPL"],
                telemetry={
                    "router_ms": 1.5,
                    "retrieval_ms": 20.0,
                    "synthesis_ms": 300.0,
                    "llm_calls": 1,
                    "crag_status": "CORRECT",
                    "crag_top_score": 0.85,
                    "grounding_ratio": 1.0,
                    "grounding": {"ratio": 1.0, "grounded": ["$383 billion"], "unverified": []},
                }
            )
        return AgentResponse(
            answer="Data not available in the context.",
            sources=[],
            scratchpad="",
            cited_tickers=[],
            telemetry={
                "router_ms": 1.0,
                "retrieval_ms": 10.0,
                "synthesis_ms": 0.0,
                "llm_calls": 0,
                "crag_status": "INSUFFICIENT",
                "crag_top_score": 0.22,
                "grounding_ratio": 1.0,
                "grounding": {"ratio": 1.0, "grounded": [], "unverified": []},
            }
        )

    df_res, summary = run_evaluation(
        csv_path=input_csv,
        output_path=out_csv,
        mock_runner=mock_runner
    )

    assert len(df_res) == 2
    assert out_csv.exists()
    assert summary["total_queries"] == 2
    assert summary["exact_refusals"] == 1
    assert summary["faithfulness_nan_count"] == 1
    assert summary["mean_llm_calls"] == 0.5
    assert df_res.loc[0, "is_exact_refusal"] == False
    assert df_res.loc[1, "is_exact_refusal"] == True
