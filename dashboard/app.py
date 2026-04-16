"""
GridMind Sentinel Dashboard — dashboard/app.py

Main Streamlit application with 5 tabs.

Modes:
    - Demo Mode (default): Uses pre-loaded benchmark data — no backend required.
      Perfect for Streamlit Community Cloud deployment.
    - Live Mode: Connects to a running FastAPI backend at the configured URL.

Tabs:
    1. Live Monitor   — Auto-refresh, active incidents, agent status
    2. Eval Metrics   — 5 Plotly charts (gauge, latency, RAGAS, guardrail, cost)
    3. Agent Traces   — Expandable trace viewer per incident
    4. Incidents      — Searchable/filterable full incident table
    5. A/B Comparison — v1 (no reranker) vs v2 (with reranker) on 5 scenarios

Run:
    streamlit run dashboard/app.py
"""

import sys
import os

# Ensure the project root is on PYTHONPATH so src.* imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridMind Sentinel",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark glassmorphism sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
}
[data-testid="stSidebar"] * { color: #e0e0ff !important; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1e3f 0%, #2a2a5c 100%);
    border: 1px solid #4040a0;
    border-radius: 12px;
    padding: 16px;
}

/* Status badges */
.badge-pass  { background:#1a4a2e; color:#4ade80; padding:3px 10px;
               border-radius:12px; font-size:0.8em; font-weight:700; }
.badge-warn  { background:#4a3a10; color:#fbbf24; padding:3px 10px;
               border-radius:12px; font-size:0.8em; font-weight:700; }
.badge-block { background:#4a1a1a; color:#f87171; padding:3px 10px;
               border-radius:12px; font-size:0.8em; font-weight:700; }
.badge-crit  { background:#5a1a1a; color:#ff6060; padding:3px 10px;
               border-radius:12px; font-size:0.8em; font-weight:700; }

/* Header gradient */
.hero { background: linear-gradient(135deg, #0f0f23 0%, #1a0a3e 50%, #0a1a3e 100%);
        border-radius:16px; padding:28px 36px; margin-bottom:24px;
        border:1px solid #4040a0; }
.hero h1 { font-size:2.4em; margin:0; color:#c0c0ff; }
.hero p  { font-size:1.1em; color:#8080c0; margin:8px 0 0 0; }

/* Tab styling */
button[data-baseweb="tab"] { font-size:0.95em; font-weight:600; }

/* Demo mode banner */
.demo-banner {
    background: linear-gradient(90deg, #1a2a1a 0%, #1e3a1e 100%);
    border: 1px solid #4ade80;
    border-radius: 10px;
    padding: 10px 18px;
    margin-bottom: 16px;
    font-size: 0.9em;
    color: #4ade80;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ GridMind Sentinel")
    st.markdown("**v0.4.0** · Phase 4 Complete")
    st.divider()

    # ── Demo Mode toggle ──────────────────────────────────────────────────
    demo_mode = st.toggle(
        "📊 Demo Mode",
        value=True,
        help="Use pre-loaded sample data (no backend required). Turn off to connect to a live FastAPI instance.",
    )
    st.session_state["demo_mode"] = demo_mode

    if demo_mode:
        st.success("Demo data active — no backend needed!")
        api_url = None
    else:
        api_url = st.text_input(
            "FastAPI URL",
            value="http://localhost:8000",
            help="Base URL of the running FastAPI backend",
        )
        # Health check
        try:
            import httpx
            r = httpx.get(f"{api_url}/health", timeout=2)
            if r.status_code == 200:
                st.success("🟢 API Online")
            else:
                st.warning("🟡 API Degraded")
        except Exception:
            st.error("🔴 API Offline")

    st.session_state["api_url"] = api_url

    st.divider()
    st.markdown("""
**Navigation**
1. 📡 Live Monitor
2. 📊 Eval Metrics
3. 🔬 Agent Traces
4. 📋 Incidents
5. ⚖️ A/B Comparison
""")
    st.divider()
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/rajeev-333/gridmind-sentinel)"
        "\n\n[![Tests](https://img.shields.io/badge/Tests-47%2F47-brightgreen)](https://github.com/rajeev-333/gridmind-sentinel)"
    )
    st.caption("GridMind Sentinel — M.Tech Portfolio Project · NIT Warangal")

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚡ GridMind Sentinel</h1>
  <p>Multi-Agent AI System · Power Grid Fault Detection · RAG Remediation · Safety Guardrails</p>
</div>
""", unsafe_allow_html=True)

if demo_mode:
    st.markdown("""
<div class="demo-banner">
  📊 <strong>Demo Mode</strong> — Showing pre-loaded benchmark data (5 incidents, 3 scenarios).
  Toggle <em>Demo Mode</em> off in the sidebar to connect to a live FastAPI backend.
</div>
""", unsafe_allow_html=True)

# ── Import tab pages ──────────────────────────────────────────────────────────
from dashboard.pages.live_monitor  import render as render_live
from dashboard.pages.evaluation    import render as render_eval
from dashboard.pages.traces        import render as render_traces
from dashboard.pages.incidents     import render as render_incidents
from dashboard.pages.ab_comparison import render as render_ab

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Monitor",
    "📊 Eval Metrics",
    "🔬 Agent Traces",
    "📋 Incidents",
    "⚖️ A/B Comparison",
])

with tab1:
    render_live(api_url)

with tab2:
    render_eval(api_url)

with tab3:
    render_traces(api_url)

with tab4:
    render_incidents(api_url)

with tab5:
    render_ab(api_url)
