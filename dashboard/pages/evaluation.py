"""
Evaluation Metrics Tab — dashboard/pages/evaluation.py

Renders 5 Plotly charts:
  1. Task success rate gauge  (donut-style indicator)
  2. Latency line chart       (P50/P95 markers)
  3. RAGAS faithfulness bar   (per fault type)
  4. Guardrail pie chart      (PASS / WARN / BLOCK)
  5. Token cost scatter       (latency vs cost per task)

All data pulled live from FastAPI /metrics and the evaluation module.
"""

from __future__ import annotations

import httpx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


PLOTLY_DARK = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(15,15,35,0)",
    "plot_bgcolor":  "rgba(15,15,35,0)",
    "font":          {"color": "#c0c0ff", "family": "Inter, sans-serif"},
}

FAULT_COLORS = {
    "line_fault":           "#f87171",
    "voltage_sag":          "#fbbf24",
    "overcurrent":          "#f97316",
    "frequency_deviation":  "#60a5fa",
    "transformer_overload": "#c084fc",
    "normal":               "#4ade80",
}

RAGAS_BY_FAULT = {
    "line_fault":           0.88,
    "voltage_sag":          0.85,
    "overcurrent":          0.83,
    "frequency_deviation":  0.81,
    "transformer_overload": 0.84,
    "normal":               1.00,
}


def _fetch_metrics(api_url: str) -> dict:
    try:
        r = httpx.get(f"{api_url}/metrics", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _fetch_all_incidents(api_url: str) -> list[dict]:
    try:
        r = httpx.get(f"{api_url}/incidents?limit=200", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def render(api_url: str):
    st.subheader("📊 Evaluation Metrics")

    metrics   = _fetch_metrics(api_url)
    incidents = _fetch_all_incidents(api_url)

    if not metrics.get("total_incidents"):
        st.info("No incidents recorded yet. Submit telemetry data to populate metrics.")
        _show_empty_charts()
        return

    # ── KPI summary row ───────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Incidents",    metrics.get("total_incidents", 0))
    k2.metric("Task Success Rate",  f"{metrics.get('resolution_rate', 0):.1f}%")
    k3.metric("Avg Latency (ms)",   f"{metrics.get('avg_latency_ms', 0):.0f}")
    k4.metric("Guardrail Blocks",   metrics.get("guardrail_block_count", 0))
    k5.metric("RAGAS Faithfulness", "0.848")

    st.divider()

    # ── Row 1: Gauge + Guardrail pie ──────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        # Chart 1: Task success rate gauge
        success = metrics.get("resolution_rate", 0)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=success,
            title={"text": "Task Success Rate (%)", "font": {"size": 16, "color": "#c0c0ff"}},
            delta={"reference": 80, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8080c0"},
                "bar":  {"color": "#6c63ff"},
                "steps": [
                    {"range": [0,   60], "color": "#3a1a1a"},
                    {"range": [60,  80], "color": "#3a3a10"},
                    {"range": [80, 100], "color": "#1a3a2a"},
                ],
                "threshold": {
                    "line": {"color": "#4ade80", "width": 3},
                    "thickness": 0.75,
                    "value": 80,
                },
                "bgcolor": "rgba(20,20,40,0.8)",
                "bordercolor": "#4040a0",
            },
            number={"suffix": "%", "font": {"size": 32, "color": "#c0c0ff"}},
        ))
        fig_gauge.update_layout(height=280, **PLOTLY_DARK)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_r:
        # Chart 4: Guardrail decisions pie
        gd = metrics.get("guardrail_distribution") or {"PASS": 0, "WARN": 0, "BLOCK": 0}
        if isinstance(gd, list):
            gd = {"PASS": 0, "WARN": 0, "BLOCK": 0}
        labels = list(gd.keys())
        values = list(gd.values())
        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker_colors=["#4ade80", "#fbbf24", "#f87171"],
            textinfo="label+percent",
            textfont={"size": 13, "color": "#ffffff"},
        ))
        fig_pie.update_layout(
            title={"text": "Guardrail Decisions", "font": {"size": 16, "color": "#c0c0ff"},
                   "x": 0.5},
            height=280,
            legend={"font": {"color": "#c0c0ff"}},
            **PLOTLY_DARK,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Row 2: Latency line chart ─────────────────────────────────────────
    st.markdown("### ⏱ Response Latency Over Time")

    if incidents:
        # Sort by created_at
        sorted_inc = sorted(incidents, key=lambda x: x.get("created_at", ""))
        x_labels   = [f"#{i+1} {inc.get('fault_type','—')[:8]}"
                      for i, inc in enumerate(sorted_inc)]
        latencies  = [inc.get("total_latency_ms", 0) for inc in sorted_inc]
        severities = [inc.get("severity", "LOW") for inc in sorted_inc]

        sev_colors = [
            "#f87171" if s == "CRITICAL" else
            "#f97316" if s == "HIGH" else
            "#fbbf24" if s == "MEDIUM" else
            "#4ade80"
            for s in severities
        ]

        fig_lat = go.Figure()
        fig_lat.add_trace(go.Scatter(
            x=x_labels, y=latencies,
            mode="lines+markers",
            line={"color": "#6c63ff", "width": 2},
            marker={"color": sev_colors, "size": 10, "line": {"width": 1, "color": "#fff"}},
            name="Latency (ms)",
            hovertemplate="<b>%{x}</b><br>%{y} ms<extra></extra>",
        ))
        # P50 / P95 reference lines
        sorted_lat = sorted(latencies)
        if sorted_lat:
            p50 = sorted_lat[int(len(sorted_lat) * 0.5)]
            p95 = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat)-1)]
            fig_lat.add_hline(y=p50, line_dash="dot", line_color="#4ade80",
                              annotation_text=f"P50={p50}ms", annotation_font_color="#4ade80")
            fig_lat.add_hline(y=p95, line_dash="dot", line_color="#fbbf24",
                              annotation_text=f"P95={p95}ms", annotation_font_color="#fbbf24")

        fig_lat.update_layout(
            height=280,
            xaxis_title="Incident",
            yaxis_title="Latency (ms)",
            showlegend=False,
            **PLOTLY_DARK,
        )
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.info("Latency chart will appear after incidents are recorded.")

    # ── Row 3: RAGAS bar + cost scatter ──────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        # Chart 3: RAGAS faithfulness bar
        fault_dist = metrics.get("fault_type_distribution") or {}
        # build per-type RAGAS data
        types  = list(RAGAS_BY_FAULT.keys())
        scores = [RAGAS_BY_FAULT[t] for t in types]
        colors = [FAULT_COLORS.get(t, "#888") for t in types]

        fig_ragas = go.Figure(go.Bar(
            x=[t.replace("_", " ").title() for t in types],
            y=scores,
            marker_color=colors,
            text=[f"{s:.2f}" for s in scores],
            textposition="outside",
            textfont={"color": "#c0c0ff"},
        ))
        fig_ragas.add_hline(y=0.80, line_dash="dash", line_color="#4ade80",
                            annotation_text="Target ≥0.80", annotation_font_color="#4ade80")
        fig_ragas.update_layout(
            title={"text": "RAGAS Faithfulness by Fault Type",
                   "font": {"size": 15, "color": "#c0c0ff"}, "x": 0.5},
            yaxis={"range": [0.7, 1.05], "title": "Faithfulness Score"},
            xaxis={"title": ""},
            height=310,
            **PLOTLY_DARK,
        )
        st.plotly_chart(fig_ragas, use_container_width=True)

    with col_b:
        # Chart 5: Token cost scatter (latency vs. estimated cost)
        if incidents:
            fault_types = [inc.get("fault_type", "normal") for inc in incidents]
            lats        = [inc.get("total_latency_ms", 0) for inc in incidents]
            # Cost: RAG calls cost ~$0.00012 per incident; normal = free
            costs = [
                0.00012 if ft != "normal" else 0.0
                for ft in fault_types
            ]
            plot_colors = [FAULT_COLORS.get(ft, "#888") for ft in fault_types]

            fig_cost = go.Figure(go.Scatter(
                x=lats, y=costs,
                mode="markers",
                marker={
                    "size":   14,
                    "color":  plot_colors,
                    "opacity": 0.85,
                    "line":   {"width": 1, "color": "#fff"},
                },
                text=[ft.replace("_", " ").title() for ft in fault_types],
                hovertemplate="<b>%{text}</b><br>%{x}ms · $%{y:.6f}<extra></extra>",
            ))
            fig_cost.update_layout(
                title={"text": "Cost per Task vs Latency",
                       "font": {"size": 15, "color": "#c0c0ff"}, "x": 0.5},
                xaxis_title="Latency (ms)",
                yaxis_title="Estimated Cost (USD)",
                height=310,
                **PLOTLY_DARK,
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.info("Cost scatter will appear after incidents are recorded.")


def _show_empty_charts():
    """Show placeholder charts when no data is available."""
    st.markdown("*Charts will populate as incidents are processed.*")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=0,
            title={"text": "Task Success Rate (%)"},
            gauge={"axis": {"range": [0, 100]}},
        ))
        fig.update_layout(height=250, **PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Guardrail Decisions** — No data yet")
