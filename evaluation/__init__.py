"""
Evaluation and benchmark package for EDGAR Pulse.
Provides deterministic regression harnesses, Ragas evaluation runners, and validation datasets.
"""
from evaluation.run_eval import compute_refusal_metrics, summarize_evaluation, run_evaluation

__all__ = ["compute_refusal_metrics", "summarize_evaluation", "run_evaluation"]
