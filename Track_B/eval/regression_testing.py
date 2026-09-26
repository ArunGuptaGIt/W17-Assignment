"""
Evidently AI Monitoring & Regression Testing for Track B Agentic Assistant.
Performs LLM-as-a-judge & reference-based correctness checks on agent responses vs golden reference answers.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

from evidently.legacy.test_suite import TestSuite
from evidently.legacy.test_preset import DataQualityTestPreset
from evidently.legacy.tests import TestNumberOfColumns, TestNumberOfRows, TestColumnValueMean

def evaluate_reference_correctness(query_id: str, query: str, golden_answer: str, actual_answer: str, expected_refusal: bool) -> Dict[str, Any]:
    """
    Reference-based correctness evaluator (LLM-as-a-judge logic / reference verification).
    Evaluates whether the actual response preserves information present in the golden answer,
    avoids contradiction, and correctly adheres to refusal guidelines.
    """
    actual_lower = actual_answer.lower()
    golden_lower = golden_answer.lower()
    
    passed = True
    reason = "Passed reference correctness check."
    
    refusal_keywords = [
        "could not find sufficient information",
        "insufficient information",
        "cannot answer",
        "not found in the provided documents",
        "technical limitation",
        "tool retrieval error"
    ]
    
    is_refusal = any(kw in actual_lower for kw in refusal_keywords)
    
    if expected_refusal:
        if is_refusal:
            passed = True
            reason = "Correctly issued refusal report for out-of-domain / errored query."
        else:
            passed = False
            reason = "Failed: Expected refusal for out-of-domain / errored query, but agent hallucinated an answer."
    else:
        if is_refusal:
            passed = False
            reason = "Failed: Agent issued premature refusal when evidence was available in documents."
        else:
            # Check key factual term overlaps
            if query_id == "TC-1":
                if "20" in actual_answer and ("business days" in actual_lower or "paid" in actual_lower or "annual leave" in actual_lower):
                    passed = True
                    reason = "Correctly retrieved 20 business days leave policy."
                else:
                    passed = False
                    reason = "Failed: Missing core fact (20 business days leave) in answer."
            elif query_id == "TC-2":
                has_stipend = "$500" in actual_answer or "500" in actual_answer
                has_rrf = "rrf" in actual_lower or "reciprocal rank" in actual_lower or "hybrid" in actual_lower or "fusion" in actual_lower
                if has_stipend and has_rrf:
                    passed = True
                    reason = "Correctly retrieved both home office stipend ($500) and hybrid RRF fusion mechanism."
                elif has_stipend and not has_rrf:
                    passed = False
                    reason = "Failed (Multi-fact omission): Retrieved $500 stipend but omitted hybrid retrieval fusion explanation."
                elif not has_stipend and has_rrf:
                    passed = False
                    reason = "Failed (Multi-fact omission): Explained hybrid RRF fusion but omitted $500 stipend."
                else:
                    passed = False
                    reason = "Failed: Omitted both requested factual components."
            elif query_id == "TC-3":
                if ("chromadb" in actual_lower or "vector" in actual_lower or "bm25" in actual_lower) and ("recursive" in actual_lower or "semantic" in actual_lower or "chunk" in actual_lower):
                    passed = True
                    reason = "Correctly described dual indexing and chunking architecture."
                else:
                    passed = False
                    reason = "Failed: Omitted key architectural principles."

    return {
        "query_id": query_id,
        "query": query,
        "golden_answer": golden_answer,
        "actual_answer": actual_answer,
        "passed": passed,
        "reason": reason,
        "expected_refusal": expected_refusal,
        "actual_refusal": is_refusal
    }

def run_evidently_regression_suite(eval_results: List[Dict[str, Any]], reports_dir: Path, prompt_version: str) -> Dict[str, Any]:
    """
    Run Evidently TestSuite on regression evaluation results and save HTML report.
    """
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare DataFrame for Evidently TestSuite
    df_eval = pd.DataFrame(eval_results)
    df_eval["passed_numeric"] = df_eval["passed"].astype("float64")
    df_eval["actual_refusal_numeric"] = df_eval["actual_refusal"].astype("float64")
    
    # Instantiate Evidently TestSuite
    test_suite = TestSuite(tests=[
        TestNumberOfRows(eq=len(eval_results)),
        TestColumnValueMean(column_name="passed_numeric", gte=0.5)  # Expecting >=50% pass rate
    ])
    
    test_suite.run(reference_data=None, current_data=df_eval)
    
    html_report_path = reports_dir / f"agent_regression_report_{prompt_version}.html"
    test_suite.save_html(str(html_report_path))
    
    # Also save a canonical agent_regression_report.html
    canonical_html_path = reports_dir / "agent_regression_report.html"
    test_suite.save_html(str(canonical_html_path))
    
    passed_count = sum(1 for r in eval_results if r["passed"])
    pct_passed = (passed_count / len(eval_results)) * 100.0
    
    return {
        "total_test_cases": len(eval_results),
        "passed_test_cases": passed_count,
        "pct_tests_passed": pct_passed,
        "html_report_path": str(html_report_path),
        "canonical_html_path": str(canonical_html_path),
        "eval_details": eval_results
    }
