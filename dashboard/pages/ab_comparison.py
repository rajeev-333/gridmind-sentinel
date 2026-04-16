"""
A/B Comparison Tab — dashboard/pages/ab_comparison.py

Compares v1 (no reranker) vs v2 (with reranker) benchmark results.

Demo Mode: When api_url is None, uses pre-computed results from demo_data.py
           (matching the README benchmark table exactly).
Live Mode: Runs 5 scenarios via the live API using run_ab_benchmark().
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PLOTLY_DARK = {
    "template":      "plotly_dark",
    "paper_bgcolor": "rgba(15,15,35,0)",
    "plot_bgcolor":  "rgba(15,15,35,0)",
    "font":          {"color": "#c0c0ff"},
}


def render(api_url: str | None):
    from dashboard.demo_data import DEMO_AB_RESULTS

    st.subheader("⚖️ A/B Comparison: v1 vs v2")
    st.markdown(
        "Compares **Agent v1** (FAISS + BM25 raw, no reranker) against "
        "**Agent v2** (FAISS + BM25 + Cross-encoder reranker) across 5 "
        "benchmark scenarios."
    )

    st.info(
        "ℹ️ v1 is a simulated baseline: fault detection is identical (wavelet-based), "
        "but v1 has slightly lower RAG confidence (−0.04) and lower latency (−50ms) "
        "since it skips the cross-encoder reranking step."
    )

    is_demo = api_url is None

    if is_demo:
        # In demo mode, show pre-computed results immediately
        results = DEMO_AB_RESULTS
        st.success("📊 Showing pre-computed benchmark results from the README §A/B Comparison table.")
    else:
        if st.button("▶️ Run A/B Benchmark (5 scenarios)", key="run_ab", use_container_width=True):
            with st.spinner("Running 5 scenarios via live API..."):
                import sys, os
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
                from src.evaluation.benchmark import run_ab_benchmark
                results = run_ab_benchmark(api_base=api_url)
                st.session_state["ab_results"] = results

        results = st.session_state.get("ab_results", [])
        if not results:
            st.caption("Results will appear here after running the benchmark.")
            return

    # ── Summary table ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Results Table")

    rows = []
    for r in results:
        rows.append({
            "Scenario":         r["scenario"],
            "Expected Fault":   r["expected_fault"].replace("_", " ").title(),
            "v1 Fault":         r["v1_fault_type"].replace("_", " ").title(),
            "v2 Fault":         r["v2_fault_type"].replace("_", " ").title(),
            "v1 Conf":          r["v1_confidence"],
            "v2 Conf":          r["v2_confidence"],
            "Δ Conf":           r["confidence_delta"],
            "v1 Lat (ms)":      r["v1_latency_ms"],
            "v2 Lat (ms)":      r["v2_latency_ms"],
            "v1 Guard":         r["v1_guardrail"],
            "v2 Guard":         r["v2_guardrail"],
            "v1 ✓":             "✅" if r["v1_correct"] else "❌",
            "v2 ✓":             "✅" if r["v2_correct"] else "❌",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Confidence comparison chart ────────────────────────────────────────
    st.markdown("### Confidence: v1 vs v2")
    scenarios = [r["scenario"] for r in results]
    v1_confs  = [r["v1_confidence"] for r in results]
    v2_confs  = [r["v2_confidence"] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="v1 (No Reranker)",
        x=scenarios, y=v1_confs,
        marker_color="#60a5fa",
        text=[f"{c:.3f}" for c in v1_confs],
        textposition="outside",
        textfont={"color": "#c0c0ff"},
    ))
    fig.add_trace(go.Bar(
        name="v2 (With Reranker)",
        x=scenarios, y=v2_confs,
        marker_color="#6c63ff",
        text=[f"{c:.3f}" for c in v2_confs],
        textposition="outside",
        textfont={"color": "#c0c0ff"},
    ))
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, 1.1], "title": "Confidence"},
        xaxis_tickangle=-20,
        height=320,
        legend={"font": {"color": "#c0c0ff"}},
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Guardrail comparison ──────────────────────────────────────────────
    st.markdown("### Guardrail Outcomes")
    gc1, gc2 = st.columns(2)

    def _count_guardrails(results, version):
        counts = {"PASS": 0, "WARN": 0, "BLOCK": 0}
        for r in results:
            gs = r.get(f"{version}_guardrail", "PASS")
            counts[gs] = counts.get(gs, 0) + 1
        return counts

    for col, version, label in [(gc1, "v1", "v1 — No Reranker"), (gc2, "v2", "v2 — With Reranker")]:
        counts = _count_guardrails(results, version)
        with col:
            fig_g = go.Figure(go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.4,
                marker_colors=["#4ade80", "#fbbf24", "#f87171"],
                textinfo="label+value",
            ))
            fig_g.update_layout(
                title={"text": label, "x": 0.5, "font": {"color": "#c0c0ff"}},
                height=250,
                **PLOTLY_DARK,
            )
            st.plotly_chart(fig_g, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────────────
    st.markdown("### Summary Statistics")
    v1_correct = sum(1 for r in results if r["v1_correct"])
    v2_correct = sum(1 for r in results if r["v2_correct"])
    v1_avg_lat = sum(r["v1_latency_ms"] for r in results) / len(results)
    v2_avg_lat = sum(r["v2_latency_ms"] for r in results) / len(results)
    v1_avg_conf = sum(r["v1_confidence"] for r in results) / len(results)
    v2_avg_conf = sum(r["v2_confidence"] for r in results) / len(results)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("v1 Accuracy", f"{v1_correct}/{len(results)}", help="Scenarios correctly classified")
    sc2.metric("v2 Accuracy", f"{v2_correct}/{len(results)}", delta=str(v2_correct - v1_correct))
    sc3.metric("v1 Avg Conf", f"{v1_avg_conf:.3f}")
    sc4.metric("v2 Avg Conf", f"{v2_avg_conf:.3f}", delta=f"+{v2_avg_conf - v1_avg_conf:.3f}")
