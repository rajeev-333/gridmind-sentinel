"""
Live Monitor Tab — dashboard/pages/live_monitor.py

Auto-refreshes every 10 seconds. Shows:
  - Active incidents (last 5, highlighted by severity)
  - Summary KPI metrics
  - A manual "Trigger Telemetry" form for demo

Demo Mode: When api_url is None, uses pre-seeded data from demo_data.py.
"""

from __future__ import annotations

import httpx
import streamlit as st
from datetime import datetime


API_SEVERITY_COLORS = {
    "LOW":      "#4ade80",
    "MEDIUM":   "#fbbf24",
    "HIGH":     "#f97316",
    "CRITICAL": "#f87171",
}


def _fetch_incidents(api_url: str, limit: int = 10) -> list[dict]:
    try:
        r = httpx.get(f"{api_url}/incidents?limit={limit}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _fetch_metrics(api_url: str) -> dict:
    try:
        r = httpx.get(f"{api_url}/metrics", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def render(api_url: str | None):
    """Render the Live Monitor tab."""
    from dashboard.demo_data import DEMO_INCIDENTS, DEMO_METRICS

    st.subheader("📡 Live Grid Monitor")

    # ── Demo vs Live mode ─────────────────────────────────────────────────
    is_demo = api_url is None

    # ── Auto-refresh control ──────────────────────────────────────────────
    col_r, col_s = st.columns([3, 1])
    with col_r:
        st.caption("Auto-refreshes every 10 seconds. Click to force refresh.")
    with col_s:
        force_refresh = st.button("🔄 Refresh Now", key="live_refresh")

    if not is_demo:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=10_000, key="live_autorefresh")
        except ImportError:
            st.info("ℹ️ Install `streamlit-autorefresh` for automatic 10-second refresh.")

    # ── Key metrics row ───────────────────────────────────────────────────
    metrics = DEMO_METRICS if is_demo else _fetch_metrics(api_url)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Incidents", metrics.get("total_incidents", 0))
    with m2:
        rate = metrics.get("resolution_rate", 0)
        st.metric("Resolution Rate", f"{rate:.1f}%")
    with m3:
        st.metric("Guardrail BLOCKs", metrics.get("guardrail_block_count", 0))
    with m4:
        st.metric("Avg Latency (ms)", f"{metrics.get('avg_latency_ms', 0):.0f}")

    st.divider()

    # ── Recent incidents ──────────────────────────────────────────────────
    st.markdown("### 🚨 Recent Incidents")
    incidents = (DEMO_INCIDENTS[:5] if is_demo else _fetch_incidents(api_url, limit=5))

    if not incidents:
        st.info("No incidents yet. Submit telemetry data to begin monitoring.")
    else:
        for inc in incidents:
            sev   = inc.get("severity", "LOW")
            color = API_SEVERITY_COLORS.get(sev, "#888")
            gs    = inc.get("guardrail_status", "PASS")
            gs_badge = {
                "PASS":  '<span class="badge-pass">PASS</span>',
                "WARN":  '<span class="badge-warn">WARN</span>',
                "BLOCK": '<span class="badge-block">BLOCK</span>',
            }.get(gs, gs)

            with st.expander(
                f"{'🔴' if sev == 'CRITICAL' else '🟡' if sev == 'HIGH' else '🟢'} "
                f"**{inc.get('fault_type','—').replace('_',' ').title()}** — "
                f"{sev}  ·  {inc.get('incident_id','')[:8]}...",
                expanded=(sev == "CRITICAL"),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Severity:** <span style='color:{color}'>{sev}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {inc.get('confidence', 0):.2%}")
                    st.markdown(f"**Guardrail:** {gs_badge}", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**Resolved:** {'✅ Yes' if inc.get('resolved') else '🔒 Needs Approval'}")
                    st.markdown(f"**Latency:** {inc.get('total_latency_ms', 0)} ms")
                    ts = inc.get("created_at", "")
                    st.markdown(f"**Time:** {ts[:19].replace('T', ' ') if ts else '—'}")

                if inc.get("action_steps"):
                    st.markdown("**Actions:**")
                    for step in inc["action_steps"][:3]:
                        st.markdown(f"  • {step}")

    st.divider()

    # ── Demo telemetry form / Live telemetry form ─────────────────────────
    st.markdown("### ⚡ Submit Telemetry")

    if is_demo:
        st.info(
            "🔌 **Demo Mode** — Telemetry submission requires a live FastAPI backend. "
            "Toggle off Demo Mode in the sidebar and start `uvicorn src.api.main:app --port 8000` locally to enable."
        )
    else:
        with st.form("telemetry_form", clear_on_submit=False):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                v_pu = st.number_input("Voltage (pu)", min_value=0.0, max_value=1.5,
                                       value=1.0, step=0.01, format="%.3f")
            with fc2:
                i_pu = st.number_input("Current (pu)", min_value=0.0, max_value=3.0,
                                       value=0.5, step=0.01, format="%.3f")
            with fc3:
                f_hz = st.number_input("Frequency (Hz)", min_value=47.0, max_value=53.0,
                                       value=50.0, step=0.1, format="%.1f")

            preset = st.selectbox("Quick Preset", [
                "Custom",
                "Scenario A — Moderate Fault (PASS)",
                "Scenario B — Severe Fault (BLOCK)",
                "Scenario C — Normal Operation",
            ])

            submitted = st.form_submit_button("🚀 Submit to API", use_container_width=True)

        if submitted:
            if preset == "Scenario A — Moderate Fault (PASS)":
                v_pu, i_pu, f_hz = 0.78, 1.4, 49.5
            elif preset == "Scenario B — Severe Fault (BLOCK)":
                v_pu, i_pu, f_hz = 0.30, 2.5, 48.8
            elif preset == "Scenario C — Normal Operation":
                v_pu, i_pu, f_hz = 1.01, 0.45, 50.0

            with st.spinner("Running multi-agent pipeline..."):
                try:
                    resp = httpx.post(
                        f"{api_url}/telemetry",
                        json={"voltage_pu": v_pu, "current_pu": i_pu, "frequency_hz": f_hz},
                        timeout=120,
                    )
                    data = resp.json()
                    sev = data.get("severity", "LOW")
                    gs  = data.get("guardrail_status", "PASS")
                    st.success(
                        f"✅ **{data.get('fault_type','—').replace('_',' ').title()}** · "
                        f"{sev} · Guardrail: {gs} · "
                        f"{data.get('total_latency_ms', 0)} ms"
                    )
                    with st.expander("Full Response JSON"):
                        st.json(data)
                except Exception as e:
                    st.error(f"API Error: {e}")
