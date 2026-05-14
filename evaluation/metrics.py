"""
Evaluation metrics:
  - Schema compliance
  - Recall@K
  - URL integrity
  - Turn cap compliance
  - Hallucination rate
  - Behavior probe pass rate
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraceResult:
    trace_id: str
    turns: int
    schema_valid: bool
    urls_valid: bool
    recommended_names: list[str]
    expected_names: list[str]
    end_of_conversation_reached: bool
    behavior_probes_passed: list[str] = field(default_factory=list)
    behavior_probes_failed: list[str] = field(default_factory=list)


def recall_at_k(expected: list[str], retrieved: list[str], k: int = 10) -> float:
    """
    Compute Recall@K.
    Recall@K(q) = |Relevant(q) ∩ TopK(q)| / |Relevant(q)|
    """
    if not expected:
        return 0.0
    top_k = set(retrieved[:k])
    relevant = set(expected)
    return len(relevant & top_k) / len(relevant)


def mean_recall_at_k(results: list[TraceResult], k: int = 10) -> float:
    """Macro-average Recall@K across all traces."""
    if not results:
        return 0.0
    scores = [recall_at_k(r.expected_names, r.recommended_names, k) for r in results]
    return sum(scores) / len(scores)


def schema_compliance_rate(results: list[TraceResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.schema_valid) / len(results)


def url_integrity_rate(results: list[TraceResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.urls_valid) / len(results)


def turn_cap_compliance_rate(results: list[TraceResult], max_turns: int = 8) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.turns <= max_turns) / len(results)


def hallucination_rate(results: list[TraceResult], all_catalog_names: set[str]) -> float:
    """% of recommended assessments not found in the catalog."""
    total = 0
    hallucinated = 0
    for r in results:
        for name in r.recommended_names:
            total += 1
            if name not in all_catalog_names:
                hallucinated += 1
    return hallucinated / total if total > 0 else 0.0


def behavior_probe_pass_rate(results: list[TraceResult]) -> float:
    total_probes = sum(
        len(r.behavior_probes_passed) + len(r.behavior_probes_failed) for r in results
    )
    passed = sum(len(r.behavior_probes_passed) for r in results)
    return passed / total_probes if total_probes > 0 else 0.0


def aggregate_report(results: list[TraceResult], catalog_names: set[str]) -> dict:
    """Build a full evaluation report dict."""
    return {
        "num_traces": len(results),
        "schema_compliance": round(schema_compliance_rate(results), 4),
        "url_integrity": round(url_integrity_rate(results), 4),
        "mean_recall_at_10": round(mean_recall_at_k(results, k=10), 4),
        "turn_cap_compliance": round(turn_cap_compliance_rate(results), 4),
        "hallucination_rate": round(hallucination_rate(results, catalog_names), 4),
        "behavior_probe_pass_rate": round(behavior_probe_pass_rate(results), 4),
    }
