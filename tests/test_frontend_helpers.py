import pytest
from app import (
    format_metric_val,
    parse_market_metrics,
    sanitize_untrusted_markdown,
    annotate_grounded_numbers,
    generate_research_brief,
)

def test_format_metric_val_currency():
    assert format_metric_val(182.50, is_currency=True) == "$182.50"
    assert format_metric_val("250.75", is_currency=True) == "$250.75"
    assert format_metric_val(None, is_currency=True) == "N/A"
    assert format_metric_val("N/A", is_currency=True) == "N/A"

def test_format_metric_val_billions_and_millions():
    # Trillions scaling
    assert format_metric_val(2800000000000, is_currency=True) == "$2.80T"
    # Billions scaling
    assert format_metric_val(2800000000, is_currency=True) == "$2.80B"
    # Millions scaling with custom suffix
    assert format_metric_val(45000000, suffix="") == "45.0M"
    # Multiples like P/E
    assert format_metric_val(31.2, suffix="x") == "31.2x"

def test_parse_market_metrics_from_dict():
    raw_dict = {"Ticker": "AAPL", "Price": 182.5, "Market Cap": 2800000000000}
    metrics = parse_market_metrics(raw_dict)
    assert len(metrics) == 1
    assert metrics[0]["Ticker"] == "AAPL"
    assert metrics[0]["Price"] == 182.5

def test_parse_market_metrics_from_string():
    raw_str = "{'Ticker': 'TSLA', 'Price': 220.0, 'PE Ratio': 65.4}, {'Ticker': 'NVDA', 'Price': 480.0}"
    metrics = parse_market_metrics(raw_str)
    assert len(metrics) == 2
    assert metrics[0]["Ticker"] == "TSLA"
    assert metrics[1]["Ticker"] == "NVDA"

def test_parse_market_metrics_empty_or_invalid():
    assert parse_market_metrics("") == []
    assert parse_market_metrics("No companies found.") == []
    assert parse_market_metrics(None) == []

def test_annotate_grounded_numbers_success():
    answer = "Apple reported $383.29B in revenue with an operating margin of 29.8%."
    grounding = {
        "grounded": ["$383.29B", "29.8%"],
        "unverified": []
    }
    annotated = annotate_grounded_numbers(answer, grounding)
    assert '<span class="ep-grounded"' in annotated
    assert '$383.29B' in annotated
    assert '29.8%' in annotated

def test_annotate_grounded_numbers_unverified():
    answer = "The projected market growth is 85%."
    grounding = {
        "grounded": [],
        "unverified": ["85%"]
    }
    annotated = annotate_grounded_numbers(answer, grounding)
    assert '<span class="ep-unverified"' in annotated
    assert '85%' in annotated

def test_annotate_grounded_numbers_immune_to_sec_forms():
    # Finding 4 regression test: "10-K" and "10-Q" must never be split into underlined "10" + "-K"
    answer = "According to Apple's latest 10-K and 10-Q filings, revenue was $383.29B."
    grounding = {
        "grounded": ["10", "10-K", "10-Q", "$383.29B"],
        "unverified": []
    }
    annotated = annotate_grounded_numbers(answer, grounding)
    # "$383.29B" should be grounded
    assert '<span class="ep-grounded" title="Audited: Grounded in SEC 10-K / Market Data" style="border-bottom: 1.5px dotted #44a282; font-variant-numeric: tabular-nums;">$383.29B</span>' in annotated
    # "10-K" and "10-Q" must remain completely intact without internal span tags
    assert "10-K" in annotated
    assert "10-Q" in annotated
    assert '<span class="ep-grounded">10</span>-K' not in annotated
    assert '10</span>-K' not in annotated
    assert '10</span>-Q' not in annotated

def test_generate_research_brief_structure():
    messages = [
        {"role": "user", "content": "What are Apple's supply chain risks?"},
        {
            "role": "assistant",
            "content": "Apple depends on single-source suppliers in Asia.",
            "sources": ["10-K Apple Inc. FY23"],
            "cited_tickers": ["AAPL"],
            "telemetry": {
                "router_ms": 12.5,
                "retrieval_ms": 145.2,
                "synthesis_ms": 320.1,
                "grounding_ratio": 1.0,
                "grounding": {"ratio": 1.0, "grounded": ["single-source"], "unverified": []}
            },
            "scratchpad": "Direct 10-K risk factor citation."
        }
    ]
    brief = generate_research_brief(messages)
    assert "# EDGAR Pulse - Research Brief" in brief
    assert "## 1. Inquiry: What are Apple's supply chain risks?" in brief
    assert "AAPL" in brief
    assert "10-K Apple Inc. FY23" in brief
    assert "Numerical Grounding Ratio:" in brief
    assert "100.0% verified" in brief
    assert "Router: `12.5ms`" in brief

def test_generate_research_brief_empty():
    brief = generate_research_brief([])
    assert "No active research inquiries" in brief

def test_sanitize_untrusted_markdown():
    # Defangs dangerous script and HTML tags
    malicious = "<script>alert('XSS')</script><img src=x onerror=alert(1)>"
    sanitized = sanitize_untrusted_markdown(malicious)
    assert "<script>" not in sanitized
    assert "<img" not in sanitized
    assert "&lt;script&gt;alert('XSS')&lt;/script&gt;" in sanitized
    assert "&lt;img src=x onerror=alert(1)&gt;" in sanitized

def test_annotate_grounded_numbers_xss_protection():
    # Verifies that malicious HTML in LLM output is defanged while numerical grounding spans are preserved
    answer = "Revenue reached $383.29B. <script>stealCookies();</script> <iframe src='evil.com'></iframe>"
    grounding = {
        "grounded": ["$383.29B"],
        "unverified": []
    }
    annotated = annotate_grounded_numbers(answer, grounding)
    assert '<span class="ep-grounded"' in annotated
    assert '$383.29B' in annotated
    # Dangerous tags must be escaped
    assert "<script>" not in annotated
    assert "<iframe" not in annotated
    assert "&lt;script&gt;stealCookies();&lt;/script&gt;" in annotated
    assert "&lt;iframe src='evil.com'&gt;&lt;/iframe&gt;" in annotated

def test_annotate_grounded_numbers_empty_grounding_xss_protection():
    # Verifies that general chat or responses with empty grounding are still neutralized
    answer = "Hello <script>alert(1)</script>"
    annotated = annotate_grounded_numbers(answer, None)
    assert "<script>" not in annotated
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in annotated

def test_annotate_grounded_numbers_currency_capture():
    # Verifies that raw numbers in grounded list capture preceding dollar signs into the span
    answer = "Alphabet shares trade at **$320.76** for GOOG and **$324.22** for GOOGL."
    grounding = {
        "grounded": ["320.76", "324.22"],
        "unverified": []
    }
    annotated = annotate_grounded_numbers(answer, grounding)
    assert '**<span class="ep-grounded"' in annotated
    assert '>$320.76</span>**' in annotated
    assert '>$324.22</span>**' in annotated
    # Ensure markdown asterisks are not torn apart by the span
    assert '**$' not in annotated

def test_annotate_grounded_numbers_deduplication():
    # Verifies that duplicate currency and raw number tokens do not create nested spans
    answer = "Prices are $320.76 and 320.76."
    grounding = {
        "grounded": ["$320.76", "320.76"],
        "unverified": []
    }
    annotated = annotate_grounded_numbers(answer, grounding)
    assert '<span class="ep-grounded">' not in annotated
    assert '<<span' not in annotated
    assert '</span></span>' not in annotated
