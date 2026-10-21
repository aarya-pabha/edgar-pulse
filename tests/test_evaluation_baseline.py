import pandas as pd
from pathlib import Path

def test_evaluation_csv_integrity():
    csv_path = Path(__file__).resolve().parent.parent / "evaluation" / "phase3_ragas_scores_adjusted.csv"
    assert csv_path.exists(), "Evaluation adjusted scores CSV must exist"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 20, "Evaluation set must contain exactly 20 test questions"
    
    # Audit check: Row 18 is a partial refusal (Apple answer + refusal for MSFT/GOOGL)
    # Relevancy must be 0.61065, not artificially boosted to 1.0
    row_18 = df.loc[18]
    assert "Summarize the legal proceedings risks" in row_18["user_input"]
    assert abs(row_18["answer_relevancy"] - 0.61065) < 1e-4
    
    # Audit check: Exactly 2 rows have NaN faithfulness (rows 14 and 19)
    nan_indices = df[df["faithfulness"].isna()].index.tolist()
    assert nan_indices == [14, 19], f"Expected NaNs at indices [14, 19], got {nan_indices}"
    
    # Non-null faithfulness mean must match published 83.5%
    faithfulness_mean = df["faithfulness"].mean()
    assert abs(faithfulness_mean - 0.8349) < 0.001
