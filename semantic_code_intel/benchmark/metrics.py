"""
Evaluation metrics for Code RAG retrieval: Recall@K, MRR, Precision@K, and Latency percentiles.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    total_queries: int = 0
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0


class LatencyStats(BaseModel):
    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    sub_second_percentage: float = 100.0


def compute_latency_stats(latencies_ms: List[float]) -> LatencyStats:
    """Compute percentiles and descriptive statistics for latency measurements."""
    if not latencies_ms:
        return LatencyStats()

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    def percentile(p: float) -> float:
        idx = int(p * n)
        return sorted_lats[min(idx, n - 1)]

    sub_sec_count = sum(1 for lat in sorted_lats if lat < 1000.0)

    return LatencyStats(
        count=n,
        min_ms=round(sorted_lats[0], 2),
        max_ms=round(sorted_lats[-1], 2),
        mean_ms=round(statistics.mean(sorted_lats), 2),
        p50_ms=round(percentile(0.50), 2),
        p90_ms=round(percentile(0.90), 2),
        p95_ms=round(percentile(0.95), 2),
        p99_ms=round(percentile(0.99), 2),
        sub_second_percentage=round((sub_sec_count / n) * 100.0, 2)
    )


def evaluate_retrieval(
    query_eval_records: List[Dict[str, Any]]
) -> EvaluationMetrics:
    """
    Compute standard information retrieval quality metrics:
    Hit@1, Hit@3, Hit@5, MRR, Precision@K.
    """
    total = len(query_eval_records)
    if total == 0:
        return EvaluationMetrics()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: List[float] = []

    for item in query_eval_records:
        retrieved_files = item.get("retrieved_files", [])
        expected_file = item.get("expected_file", "")
        target_symbol = item.get("target_symbol", "")
        retrieved_symbols = item.get("retrieved_symbols", [])

        # Check match by file path or symbol name
        match_found = False
        rr = 0.0

        for rank, (r_file, r_sym) in enumerate(zip(retrieved_files, retrieved_symbols), start=1):
            is_match = (expected_file in r_file) or (target_symbol and target_symbol in r_sym)
            if is_match:
                if not match_found:
                    match_found = True
                    rr = 1.0 / rank
                    if rank <= 1:
                        hits_at_1 += 1
                    if rank <= 3:
                        hits_at_3 += 1
                    if rank <= 5:
                        hits_at_5 += 1

        reciprocal_ranks.append(rr)

    return EvaluationMetrics(
        total_queries=total,
        hit_rate_at_1=round(hits_at_1 / total, 4),
        hit_rate_at_3=round(hits_at_3 / total, 4),
        hit_rate_at_5=round(hits_at_5 / total, 4),
        mrr=round(statistics.mean(reciprocal_ranks), 4),
        precision_at_1=round(hits_at_1 / total, 4),
        precision_at_3=round(hits_at_3 / (total * 3), 4),
        precision_at_5=round(hits_at_5 / (total * 5), 4)
    )
