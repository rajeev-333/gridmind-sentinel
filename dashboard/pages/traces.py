"""
Agent Traces Tab — dashboard/pages/traces.py

Shows a per-incident trace viewer. Incidents listed in a table;
clicking expands the full agent_trace with step timings and I/O.

Demo Mode: When api_url is None, uses pre-seeded data from demo_data.py.
"""

from __future__ import annotations

import json
import httpx
import streamlit as st


def _fetch_incidents(api_url: str) -> list[dict]:
    try:
        r = httpx.get(f"{api_url}/incidents?limit=100", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _fetch_incident(api_url: str, incident_id: str) -> dict:
    try:
        r = httpx.get(f"{api_url}/incidents/{incident_id}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


NODE_ICONS = {
    "supervisor":        "🎯",
    "fault_analyzer":    "🔬",
    "remediation_agent": "🛠️",
    "guardrail_engine":  "🛡️",
    "report_generator":  "📝",
}

SEVERITY_COLORS = {
    "LOW":      "#4ade80",
    "MEDIUM":   "#fbbf24",
    "HIGH":     "#f97316",
    "CRITICAL": "#f87171",
}


def render(api_url: str | None):
    from dashboard.demo_data import DEMO_INCIDENTS

    st.subheader("🔬 Agent Traces")
    st.caption("Expand any incident to see the full multi-agent execution trace.")

    is_demo = api_url is None
    incidents = DEMO_INCIDENTS if is_demo else _fetch_incidents(api_url)

    if not incidents:
        st.info("No incidents yet — submit telemetry to see agent traces.")
        return

    # ── Incident selector table ───────────────────────────────────────────
    st.markdown("### Select an Incident")

    # Build display rows
    rows = []
    for inc in incidents:
        sev  = inc.get("severity", "LOW")
        gs   = inc.get("guardrail_status", "PASS")
        ts   = (inc.get("created_at") or "")[:19].replace("T", " ")
        rows.append({
            "ID (short)":     inc.get("incident_id", "")[:8] + "...",
            "Fault Type":     inc.get("fault_type", "—").replace("_", " ").title(),
            "Severity":       sev,
            "Confidence":     f"{inc.get('confidence', 0):.1%}",
            "Guardrail":      gs,
            "Latency (ms)":   inc.get("total_latency_ms", 0),
            "Resolved":       "✅" if inc.get("resolved") else "🔒",
            "Timestamp":      ts,
            "_id":            inc.get("incident_id", ""),
        })

    # Show summary table (without hidden _id column display)
    import pandas as pd
    df = pd.DataFrame(rows)
    display_df = df.drop(columns=["_id"])

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Severity":    st.column_config.TextColumn("Severity"),
            "Guardrail":   st.column_config.TextColumn("Guardrail"),
            "Resolved":    st.column_config.TextColumn("Resolved"),
            "Latency (ms)": st.column_config.NumberColumn("Latency (ms)", format="%d"),
        },
        hide_index=True,
    )

    st.divider()

    # ── Trace expander for each incident ──────────────────────────────────
    st.markdown("### Trace Details")

    for i, inc in enumerate(incidents[:20]):  # cap at 20 to avoid overload
        sev   = inc.get("severity", "LOW")
        color = SEVERITY_COLORS.get(sev, "#888")
        gs    = inc.get("guardrail_status", "PASS")
        iid   = inc.get("incident_id", "")

        with st.expander(
            f"{NODE_ICONS.get('fault_analyzer','🔬')} "
            f"**{inc.get('fault_type','—').replace('_',' ').title()}** · "
            f"{sev} · {gs} · {iid[:8]}...",
        ):
            # In demo mode use embedded agent_trace; in live mode re-fetch
            if is_demo:
                detailed = inc
            else:
                detailed = _fetch_incident(api_url, iid)

            steps = detailed.get("action_steps") or inc.get("action_steps", [])
            refs  = detailed.get("references") or []
            trace = detailed.get("agent_trace", {})

            # Summary row
            rc1, rc2, rc3 = st.columns(3)
            rc1.markdown(f"**Fault:** {inc.get('fault_type','—').replace('_',' ').title()}")
            rc2.markdown(f"**Severity:** <span style='color:{color}'>{sev}</span>",
                         unsafe_allow_html=True)
            rc3.markdown(f"**Latency:** {inc.get('total_latency_ms',0)} ms")

            # Synthetic trace display (nodes in order)
            st.markdown("**Execution Trace:**")
            nodes_in_order = ["supervisor", "fault_analyzer"]
            if sev != "LOW":
                nodes_in_order += ["remediation_agent", "guardrail_engine"]
            nodes_in_order.append("report_generator")

            for j, node in enumerate(nodes_in_order):
                icon = NODE_ICONS.get(node, "⚙️")
                node_data = trace.get(node, {})
                lat_str = f" `{node_data.get('latency_ms', '~')}ms`" if node_data else ""
                status_note = ""
                if node == "guardrail_engine":
                    gstatus = node_data.get("status", gs)
                    kw = node_data.get("triggered_keywords", [])
                    kw_str = f" · keywords: {kw}" if kw else " · no dangerous ops"
                    status_note = f" → **{gstatus}**{kw_str}"
                elif node == "report_generator":
                    resolved = "✅ Resolved" if inc.get("resolved") else "🔒 Awaiting Approval"
                    status_note = f" → {resolved}"
                elif node == "remediation_agent":
                    docs = node_data.get("rag_docs", 5)
                    status_note = f" → {docs} docs retrieved"

                st.markdown(
                    f"&nbsp;&nbsp;{'└─' if j == len(nodes_in_order)-1 else '├─'} "
                    f"{icon} `{node}`{lat_str}{status_note}",
                    unsafe_allow_html=True,
                )

            if steps:
                st.markdown("**Action Plan:**")
                for step in steps:
                    st.markdown(f"  • {step}")

            if refs:
                st.markdown(f"**References:** {', '.join(refs)}")

            if gs == "BLOCK":
                st.warning(
                    "⛔ **BLOCKED**: This incident requires human approval before "
                    "proceeding with switching operations."
                )
                if not is_demo:
                    if st.button(f"✅ Approve Incident {iid[:8]}...", key=f"approve_{iid}"):
                        try:
                            resp = httpx.post(f"{api_url}/approve/{iid}", timeout=10)
                            if resp.status_code == 200:
                                st.success("Incident approved! Refresh to see updated status.")
                            else:
                                st.error(f"Approval failed: {resp.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.info("🔌 Connect to a live API to approve blocked incidents.")
