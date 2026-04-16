"""
Incidents Tab — dashboard/pages/incidents.py

Searchable/filterable table of all past incidents.
Click any row to expand the full incident JSON.

Demo Mode: When api_url is None, uses pre-seeded data from demo_data.py.
"""

from __future__ import annotations

import json
import httpx
import pandas as pd
import streamlit as st


def _fetch_incidents(api_url: str, severity: str = "", fault_type: str = "") -> list[dict]:
    params = "?limit=200"
    if severity:
        params += f"&severity={severity}"
    if fault_type:
        params += f"&fault_type={fault_type}"
    try:
        r = httpx.get(f"{api_url}/incidents{params}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def render(api_url: str | None):
    from dashboard.demo_data import DEMO_INCIDENTS

    st.subheader("📋 Incident History")

    is_demo = api_url is None

    # ── Filter row ────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    with fc1:
        sev_filter = st.selectbox(
            "Filter by Severity",
            ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
            key="inc_sev_filter",
        )
    with fc2:
        ft_filter = st.selectbox(
            "Filter by Fault Type",
            ["All", "normal", "voltage_sag", "overcurrent",
             "frequency_deviation", "line_fault", "transformer_overload"],
            key="inc_ft_filter",
        )
    with fc3:
        search = st.text_input(
            "🔍 Search (incident ID, fault type, steps)",
            placeholder="Type to filter...",
            key="inc_search",
        )

    # ── Fetch / load data ─────────────────────────────────────────────────
    if is_demo:
        incidents = list(DEMO_INCIDENTS)
        # Apply filters client-side for demo data
        if sev_filter != "All":
            incidents = [i for i in incidents if i.get("severity") == sev_filter]
        if ft_filter != "All":
            incidents = [i for i in incidents if i.get("fault_type") == ft_filter]
    else:
        sev_q = sev_filter if sev_filter != "All" else ""
        ft_q  = ft_filter  if ft_filter  != "All" else ""
        incidents = _fetch_incidents(api_url, severity=sev_q, fault_type=ft_q)

    # Client-side text search
    if search:
        search_lower = search.lower()
        incidents = [
            inc for inc in incidents
            if search_lower in json.dumps(inc).lower()
        ]

    st.caption(f"Showing **{len(incidents)}** incidents")

    if not incidents:
        st.info("No incidents match the current filters.")
        return

    # ── Build display DataFrame ───────────────────────────────────────────
    rows = []
    for inc in incidents:
        gs  = inc.get("guardrail_status", "PASS")
        ts  = (inc.get("created_at") or "")[:19].replace("T", " ")
        rows.append({
            "ID":           inc.get("incident_id", "")[:8] + "...",
            "Fault Type":   inc.get("fault_type", "—").replace("_", " ").title(),
            "Severity":     inc.get("severity", "—"),
            "Confidence":   inc.get("confidence", 0.0),
            "Guardrail":    gs,
            "Latency (ms)": inc.get("total_latency_ms", 0),
            "Resolved":     "✅ Yes" if inc.get("resolved") else "🔒 No",
            "Timestamp":    ts,
            "Steps":        len(inc.get("action_steps") or []),
            "_raw":         inc,
        })

    df = pd.DataFrame(rows)
    display_df = df.drop(columns=["_raw"])

    # Color-coded severity
    def sev_color(val):
        colors = {
            "CRITICAL": "background-color: rgba(248,113,113,0.15)",
            "HIGH":     "background-color: rgba(249,115,22,0.12)",
            "MEDIUM":   "background-color: rgba(251,191,36,0.10)",
            "LOW":      "background-color: rgba(74,222,128,0.08)",
        }
        return colors.get(val, "")

    styled = display_df.style.applymap(sev_color, subset=["Severity"])

    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "Confidence":    st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            "Latency (ms)":  st.column_config.NumberColumn("Latency (ms)", format="%d"),
        },
        hide_index=True,
    )

    st.divider()

    # ── JSON viewer for selected incident ─────────────────────────────────
    st.markdown("### 🔎 Full JSON Viewer")
    short_ids = [f"{i.get('incident_id','')[:8]}... — {i.get('fault_type','')}" for i in incidents]
    selected = st.selectbox("Select incident to inspect:", short_ids, key="json_viewer_sel")

    if selected:
        idx = short_ids.index(selected)
        inc = incidents[idx]
        with st.expander("📄 Raw Incident JSON", expanded=True):
            st.json(inc)

        # Human approval for blocked — only in live mode
        if inc.get("guardrail_status") == "BLOCK" and not inc.get("resolved"):
            st.warning("⛔ This incident is currently BLOCKED — awaiting human approval.")
            if is_demo:
                st.info("🔌 Connect to a live API to approve blocked incidents.")
            else:
                iid = inc.get("incident_id", "")
                if st.button(f"✅ Approve {iid[:8]}...", key="json_approve"):
                    try:
                        resp = httpx.post(f"{api_url}/approve/{iid}", timeout=10)
                        if resp.status_code == 200:
                            st.success("Approved! Refresh the page to see updated status.")
                        else:
                            st.error(f"Failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
