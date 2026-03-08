"""
app.py - Streamlit HMI Dashboard for AI-MPC Datacenter Cooling

Changes in this version:
  - Temperature & reservoir gauges replaced with time-series line plots
    with upper/lower safety-bound reference lines
  - Zone 2 shutdown displays a full-width emergency banner
  - Water % calculation uses flow_rates from MPC result (no hardcoded values)
  - Active constraints only shown when genuinely triggered (no false positives)
  - Audit log stores reservoir_a/b/c for history plots
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="AI-MPC Datacenter Cooling",
    page_icon="🌡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding: 1rem 2rem; }
    h1, h2, h3 { color: #58a6ff; }
    .constraint-critical { background-color:#3d1a1a; border:1px solid #f85149; border-radius:6px;
        padding:8px 12px; color:#f85149; font-weight:600; margin:3px 0; }
    .constraint-warning  { background-color:#2d2208; border:1px solid #e3b341; border-radius:6px;
        padding:8px 12px; color:#e3b341; font-weight:600; margin:3px 0; }
    .constraint-ok       { background-color:#0d2318; border:1px solid #3fb950; border-radius:6px;
        padding:8px 12px; color:#3fb950; margin:3px 0; }
    .shutdown-banner { background-color:#3d1a1a; border:2px solid #f85149; border-radius:8px;
        padding:16px 20px; color:#f85149; font-weight:700; font-size:18px;
        text-align:center; margin:8px 0 16px 0; }
    .explanation-box { background-color:#161b22; border:1px solid #388bfd; border-radius:12px;
        padding:18px 20px; font-size:15px; line-height:1.9; color:#e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .alert-critical { color:#f85149; font-weight:700; font-size:16px; }
    .alert-warning  { color:#e3b341; font-weight:700; font-size:16px; }
    .alert-info     { color:#3fb950; font-size:15px; }
    div[data-testid="metric-container"] { background-color:#161b22;
        border:1px solid #30363d; border-radius:8px; padding:8px; }
</style>
""", unsafe_allow_html=True)

STATE_PATH     = "data/current_state.json"
AUDIT_PATH     = "data/audit_log.json"
OVERRIDES_PATH = "data/overrides.json"

_DARK_BG    = "#161b22"
_PLOT_BG    = "#0d1117"
_FONT_COLOR = "#e6edf3"
_GRID_COLOR = "#30363d"

_T_MAX = [35.0, 33.0, 34.0]
_T_MIN = [18.0, 18.0, 18.0]
_RES_WARN = [20.0, 20.0, 15.0, 20.0]
_RES_CRIT = [10.0, 10.0,  8.0, 10.0]


def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
        font=dict(color=_FONT_COLOR, family="monospace"),
        margin=dict(l=40, r=20, t=40, b=30),
        **kwargs,
    )


# ===========================================================================
# History helpers
# ===========================================================================

def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_audit_log(n: int = 60, current_step: int = None):
    try:
        with open(AUDIT_PATH) as f:
            log = json.load(f)

        # Find the last restart: last index where step goes backwards
        last_restart = 0
        for idx in range(1, len(log)):
            if log[idx].get("step", 0) < log[idx - 1].get("step", 0):
                last_restart = idx
        log = log[last_restart:]

        # If we know the current step, drop any entries that are AHEAD of it
        # (stale entries from a previous longer run that haven't been overwritten yet)
        if current_step is not None:
            log = [e for e in log if e.get("step", 0) <= current_step]

        # Deduplicate by step within this run (keep last occurrence)
        seen = {}
        for entry in log:
            seen[entry.get("step", -1)] = entry
        deduped = sorted(seen.values(), key=lambda e: e.get("step", 0))
        return deduped[-n:]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_overrides(enabled, valve_positions):
    os.makedirs("data", exist_ok=True)
    with open(OVERRIDES_PATH, "w") as f:
        json.dump({"enabled": enabled, "valve_positions": valve_positions}, f, indent=2)


# ===========================================================================
# Chart builders
# ===========================================================================

def make_temp_history(audit_log: list) -> go.Figure:
    """Line plot: zone temperatures over time with T_max reference lines.
    The final point on each trace is highlighted so it's clear which value
    matches the metric cards below (current reading).
    """
    if not audit_log:
        return go.Figure()

    steps = [e.get("step", i) for i, e in enumerate(audit_log)]
    colors = ["#3fb950", "#e3b341", "#58a6ff"]
    tmax_colors = ["rgba(248,81,73,0.5)", "rgba(248,81,73,0.7)", "rgba(248,81,73,0.5)"]
    zone_labels = ["Zone 1", "Zone 2", "Zone 3"]

    fig = go.Figure()
    for i, (label, color, t_max, tc) in enumerate(
            zip(zone_labels, colors, _T_MAX, tmax_colors), start=1):
        temps = [e.get("temperatures", [0, 0, 0])[i-1]
                 if isinstance(e.get("temperatures"), list) else 0
                 for e in audit_log]

        # Historical line (no markers to keep it clean)
        fig.add_trace(go.Scatter(
            x=steps, y=temps, name=label,
            line=dict(color=color, width=2),
            mode="lines",
            showlegend=True,
        ))

        # Highlighted current (last) point — this is what the metric card shows
        if steps and temps:
            fig.add_trace(go.Scatter(
                x=[steps[-1]], y=[temps[-1]],
                mode="markers+text",
                marker=dict(color=color, size=10, line=dict(color="white", width=2)),
                text=[f"  {temps[-1]:.1f}°C"],
                textfont=dict(color=color, size=11),
                textposition="middle right",
                showlegend=False,
                hoverinfo="skip",
            ))

        # T_max reference line
        fig.add_hline(
            y=t_max, line=dict(color=tc, width=1.5, dash="dash"),
            annotation_text=f"max {label} ({t_max:.0f}°C)",
            annotation_position="right",
            annotation_font=dict(color=tc, size=10),
        )

    # "NOW" vertical line at the latest step
    if steps:
        fig.add_vline(
            x=steps[-1],
            line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
            annotation_text="NOW",
            annotation_position="top",
            annotation_font=dict(color="rgba(255,255,255,0.5)", size=9),
        )

    fig.update_layout(
        **_base_layout(height=280, title=dict(text="Zone Temperatures (°C) — history + current reading", font=dict(size=14))),
        xaxis=dict(title="Step", gridcolor=_GRID_COLOR),
        yaxis=dict(title="°C", gridcolor=_GRID_COLOR, range=[18, 42]),
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
        hovermode="x unified",
    )
    return fig


def make_reservoir_gauges(sensor: dict) -> go.Figure:
    """4 gauge dials — one per reservoir — with warning (20%) and critical (10%) thresholds."""
    from plotly.subplots import make_subplots

    labels  = ["Source A",    "Source B",    "Source C",    "Main"]
    keys    = ["reservoir_a", "reservoir_b", "reservoir_c", "reservoir_main"]
    warns   = [20.0,          20.0,          15.0,          20.0]
    crits   = [10.0,          10.0,           8.0,          10.0]
    # Per-source accent colours (used for needle + value text when healthy)
    accents = ["#f85149",     "#e3b341",     "#58a6ff",     "#3fb950"]

    values = [float(sensor.get(k, 50.0)) for k in keys]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=labels,
        specs=[[{"type": "indicator"}] * 4],
    )

    for col_idx, (label, val, warn, crit, accent) in enumerate(
            zip(labels, values, warns, crits, accents), start=1):

        # Needle colour: red if critical, amber if warning, else accent
        if val <= crit:
            bar_color  = "#f85149"
        elif val <= warn:
            bar_color  = "#e3b341"
        else:
            bar_color  = accent

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                number=dict(
                    suffix="%",
                    font=dict(size=22, color=bar_color),
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickwidth=1,
                        tickcolor=_FONT_COLOR,
                        tickfont=dict(size=9, color=_FONT_COLOR),
                        nticks=6,
                    ),
                    bar=dict(color=bar_color, thickness=0.25),
                    bgcolor=_PLOT_BG,
                    borderwidth=1,
                    bordercolor=_GRID_COLOR,
                    steps=[
                        dict(range=[0,    crit], color="rgba(248,81,73,0.25)"),
                        dict(range=[crit, warn], color="rgba(227,179,65,0.15)"),
                        dict(range=[warn, 100],  color="rgba(63,185,80,0.07)"),
                    ],
                    threshold=dict(
                        line=dict(color="#f85149", width=3),
                        thickness=0.85,
                        value=crit,
                    ),
                ),
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        **_base_layout(height=230),
        title=dict(text="Reservoir Levels — Current Capacity", font=dict(size=14)),
    )
    # Fix subplot title font
    for ann in fig.layout.annotations:
        ann.font = dict(color=_FONT_COLOR, size=12)
    return fig


def make_valve_chart(u_optimal: list, overrides: dict = None) -> go.Figure:
    labels = ["V1 (Z1/SrcA)", "V2 (Z1/SrcB)", "V3 (Z2/SrcA)", "V4 (Z3/SrcC)"]
    values = [v * 100 for v in u_optimal]
    colors = ["#388bfd", "#58a6ff", "#388bfd", "#58a6ff"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color=_GRID_COLOR, width=1)),
        name="MPC Optimal",
        text=[f"{v:.1f}%" for v in values],
        textposition="inside", insidetextfont=dict(color="white", size=13),
    ))
    # Minimum reference line at 35%
    fig.add_vline(x=35, line=dict(color="rgba(227,179,65,0.4)", width=1, dash="dot"),
                  annotation_text="min 35%", annotation_font=dict(color="#e3b341", size=9))

    if overrides and overrides.get("enabled"):
        ov = [float(overrides["valve_positions"].get(str(i), u_optimal[i])) * 100 for i in range(4)]
        fig.add_trace(go.Scatter(
            x=ov, y=labels, mode="markers",
            marker=dict(color="#e3b341", size=14, symbol="diamond"), name="Manual Override",
        ))

    fig.update_layout(
        **_base_layout(height=220, title=dict(text="Valve Positions (%)", font=dict(size=14))),
        xaxis=dict(range=[0, 110], gridcolor=_GRID_COLOR, title="Opening %"),
        yaxis=dict(gridcolor=_GRID_COLOR),
        legend=dict(orientation="h", y=-0.2),
        barmode="overlay",
    )
    return fig


def make_shap_chart(ranked_features: list) -> go.Figure:
    if not ranked_features:
        return go.Figure()
    features = [f["description"].replace(" (°C)", "").replace(" (%)", "") for f in ranked_features]
    shap_vals = [f["shap_value"] for f in ranked_features]
    colors    = ["#f85149" if v > 0 else "#3fb950" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals, y=features, orientation="h",
        marker=dict(color=colors, line=dict(color=_GRID_COLOR, width=0.5)),
        text=[f"{v:+.4f}" for v in shap_vals],
        textposition="outside", textfont=dict(size=11, color=_FONT_COLOR),
    ))
    fig.add_vline(x=0, line_color=_GRID_COLOR, line_width=1)
    fig.update_layout(
        **_base_layout(height=280),
        title=dict(text="SHAP Feature Attribution (red=increases cooling)", font=dict(size=13)),
        xaxis=dict(title="SHAP Value", gridcolor=_GRID_COLOR),
        yaxis=dict(gridcolor=_GRID_COLOR, autorange="reversed"),
    )
    return fig


def make_airflow_chart(sensor: dict) -> go.Figure:
    zones = ["Zone 1", "Zone 2", "Zone 3"]
    flows = [sensor.get(f"airflow_zone{i}", 0) for i in range(1, 4)]
    fig = go.Figure(go.Bar(
        x=zones, y=flows,
        marker=dict(color=["#58a6ff", "#388bfd", "#1f6feb"], line=dict(color=_GRID_COLOR, width=1)),
        text=[f"{v:.2f} m³/s" for v in flows],
        textposition="outside", textfont=dict(color=_FONT_COLOR, size=11),
    ))
    fig.update_layout(
        **_base_layout(height=200),
        title=dict(text="Airflow Rates (m³/s)", font=dict(size=13)),
        yaxis=dict(gridcolor=_GRID_COLOR, title="m³/s"),
        xaxis=dict(gridcolor=_GRID_COLOR),
    )
    return fig


# ===========================================================================
# Constraint renderer
# ===========================================================================

def render_constraints(active_constraints: list):
    if not active_constraints:
        st.markdown('<div class="constraint-ok">✓ All constraints satisfied — normal operation</div>',
                    unsafe_allow_html=True)
        return
    icons   = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}
    classes = {"CRITICAL": "constraint-critical", "WARNING": "constraint-warning", "INFO": "constraint-ok"}
    for c in active_constraints:
        sev   = c.get("severity", "INFO")
        label = c.get("label", "Unknown")
        util  = c.get("utilization_pct")
        util_str = f" [{util:.1f}% of limit]" if util is not None else ""
        st.markdown(
            f'<div class="{classes.get(sev,"constraint-ok")}">'
            f'{icons.get(sev,"⚪")} {label}{util_str}</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# Main dashboard
# ===========================================================================

def render_dashboard(state: dict, audit_log: list):
    sensor      = state.get("sensor", {})
    mpc         = state.get("mpc", {})
    shap        = state.get("shap", {})
    active      = mpc.get("active_constraints", [])
    u_opt       = mpc.get("u_optimal", [0.5] * 4)
    explanation = state.get("explanation", "")
    alert       = state.get("alert", "")
    zone2_sd    = mpc.get("zone2_shutdown", {})

    # --- Header ---
    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
    with h1: st.title("🌡 AI-MPC Datacenter Cooling")
    with h2: st.metric("Step", state.get("step", 0))
    with h3: st.metric("Cycle", f"{state.get('cycle_ms', 0):.0f} ms")
    with h4:
        n_crit = sum(1 for c in active if c.get("severity") == "CRITICAL")
        st.metric("Constraints", len(active), delta=n_crit or None, delta_color="inverse")

    phase = sensor.get("phase", "normal")
    phase_labels = {
        "normal":       "🟢 Phase: Normal Operation",
        "water_shortage": "🟡 Phase: Water Shortage — MPC compensating",
        "critical":     "🔴 Phase: Critical Water Crisis",
        "zone_overload": "🔴 Phase: Zone Overload — Load Migration Required",
    }
    st.caption(
        f"{phase_labels.get(phase, phase)} | "
        f"Last updated: {state.get('timestamp_str', 'N/A')} | "
        f"Solver: {mpc.get('solver', 'N/A')}"
    )

    # --- Zone 2 shutdown banner ---
    if zone2_sd.get("active"):
        st.markdown(
            f'<div class="shutdown-banner">'
            f'⚠️ ZONE 2 THERMAL SHUTDOWN — Zone 2 at {zone2_sd.get("zone2_temp", 0):.1f}°C | '
            f'Source A at {zone2_sd.get("srca_level", 0):.1f}% — COOLING IMPOSSIBLE<br>'
            f'ACTION REQUIRED: Migrate Zone 2 workloads to Zone 1 and Zone 3 immediately!'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Alert banner ---
    if alert and not zone2_sd.get("active"):
        css = ("alert-critical" if "CRITICAL" in alert.upper() or "🔴" in alert
               else "alert-warning" if "WARNING" in alert.upper() or "🟡" in alert
               else "alert-info")
        st.markdown(f'<div class="{css}">{alert}</div>', unsafe_allow_html=True)

    st.divider()

    # ---- ROW 1: Temperature history + Reservoir history ---------------
    tc, rc = st.columns(2)
    with tc:
        st.subheader("Zone Temperatures")
        if audit_log:
            st.plotly_chart(make_temp_history(audit_log), use_container_width=True, key="temp_hist")
        # Current values
        cur_temps = [sensor.get(f"temp_zone{i}", 0) for i in range(1, 4)]
        t_cols = st.columns(3)
        for i, (col, t, tm) in enumerate(zip(t_cols, cur_temps, _T_MAX), start=1):
            pct   = t / tm * 100
            color = "🔴" if pct > 95 else ("🟡" if pct > 85 else "🟢")
            col.metric(f"{color} Zone {i}", f"{t:.1f}°C", f"limit {tm:.0f}°C")

    with rc:
        st.subheader("Reservoir Levels")
        st.plotly_chart(make_reservoir_gauges(sensor), use_container_width=True, key="res_gauges")
        # Current values beneath gauges
        cur_res = [sensor.get(k, 0) for k in ("reservoir_a","reservoir_b","reservoir_c","reservoir_main")]
        r_cols  = st.columns(4)
        for col, lvl, label, warn in zip(r_cols, cur_res,
                                          ["SrcA","SrcB","SrcC","Main"], _RES_WARN):
            icon = "🔴" if lvl < 10 else ("🟡" if lvl < warn else "💧")
            col.metric(f"{icon} {label}", f"{lvl:.1f}%")

    st.divider()

    # ---- ROW 2: Valves + Constraints + Airflow ------------------------
    vc, cc, ac = st.columns([2, 2, 1.5])

    with vc:
        st.subheader("MPC Valve Positions")
        overrides = {}
        if os.path.exists(OVERRIDES_PATH):
            try:
                with open(OVERRIDES_PATH) as f:
                    overrides = json.load(f)
            except Exception:
                pass
        st.plotly_chart(make_valve_chart(u_opt, overrides), use_container_width=True, key="valves")

        # Water usage — use flow_rates from MPC result (correct source)
        fr      = mpc.get("flow_rates", [150, 120, 180, 100])
        reg     = mpc.get("regulatory_limit", 312.0)
        w_used  = sum(u * r for u, r in zip(u_opt, fr))
        reg_pct = w_used / reg * 100 if reg > 0 else 0
        color   = "🔴" if reg_pct >= 100 else ("🟡" if reg_pct >= 92 else "🟢")
        st.caption(
            f"Water this step: {w_used:.0f} L | "
            f"Policy limit: {reg:.0f} L | "
            f"{color} {reg_pct:.1f}% of limit"
        )

    with cc:
        st.subheader("Active Constraints")
        render_constraints(active)
        if shap.get("constraint_summary"):
            st.caption(shap["constraint_summary"])

    with ac:
        st.subheader("Airflow")
        st.plotly_chart(make_airflow_chart(sensor), use_container_width=True, key="airflow")
        st.metric("Ambient", f"{sensor.get('ambient_temp', 0):.1f}°C")

    st.divider()

    # ---- ROW 3: SHAP + LLM explanation --------------------------------
    sc, lc = st.columns([1.5, 2])

    with sc:
        st.subheader("SHAP Feature Attribution")
        ranked = shap.get("ranked_features", [])
        if ranked:
            st.plotly_chart(make_shap_chart(ranked), use_container_width=True, key="shap")
            prim = shap.get("primary_driver", {})
            st.caption(
                f"Primary driver: **{prim.get('description','N/A')}** "
                f"= {prim.get('value',0):.2f} | SHAP {prim.get('shap_value',0):+.4f}"
            )
        else:
            st.info("Accumulating history for SHAP analysis (~10 steps) …")

    with lc:
        st.subheader("🤖 AI Operator Assistant")
        if explanation:
            # Render each line as a paragraph for conversational feel
            paragraphs = "".join(
                f"<p style='margin:0 0 10px 0'>{line}</p>"
                for line in explanation.split("\n") if line.strip()
            )
            st.markdown(
                f'<div class="explanation-box">'
                f'<div style="color:#58a6ff;font-size:12px;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px">'
                f'🗣 System Intelligence — Live Briefing</div>'
                f'{paragraphs}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="explanation-box">Generating explanation…</div>',
                        unsafe_allow_html=True)
        cost    = mpc.get("cost", float("inf"))
        st.caption(f"Optimisation cost: {'∞ (fallback)' if cost > 1e9 else f'{cost:.5f}'}")

    st.divider()

    # ---- ROW 4: Manual override ---------------------------------------
    with st.expander("Manual Valve Override (Human-in-the-Loop)", expanded=False):
        st.warning(
            "Manual overrides bypass MPC. The system logs override decisions separately. "
            "Disable to restore autonomous control."
        )
        ov_enabled   = st.checkbox("Enable Manual Override", value=overrides.get("enabled", False))
        ov_cols      = st.columns(4)
        ov_positions = {}
        valve_names  = ["V1 (Z1/SrcA)", "V2 (Z1/SrcB)", "V3 (Z2/SrcA)", "V4 (Z3/SrcC)"]
        for i, (col, vname) in enumerate(zip(ov_cols, valve_names)):
            with col:
                default = overrides.get("valve_positions", {}).get(str(i), u_opt[i])
                ov_positions[str(i)] = st.slider(
                    vname, 0.0, 1.0, float(default), 0.05,
                    key=f"ov_{i}", disabled=not ov_enabled,
                )
        if st.button("Apply Override", type="primary", disabled=not ov_enabled):
            save_overrides(ov_enabled, ov_positions)
            st.success("Override saved — effective on next control cycle.")
        elif not ov_enabled and overrides.get("enabled"):
            save_overrides(False, {})
            st.info("Override disabled — autonomous MPC control restored.")

    st.divider()

    # ---- ROW 5: Audit log --------------------------------------------
    st.subheader("Audit Log (Last 10 Decisions)")
    if audit_log:
        df = pd.DataFrame(audit_log[-10:][::-1])
        display_cols = {
            "step": "Step", "timestamp": "Time",
            "temperatures": "Temps (°C)", "u_optimal": "Valves",
            "reservoir_a": "SrcA%", "reservoir_b": "SrcB%",
            "mpc_cost": "Cost", "n_active_constraints": "Constraints",
            "primary_driver": "Primary Driver", "alert": "Alert",
        }
        if "temperatures" in df.columns:
            df["temperatures"] = df["temperatures"].apply(
                lambda t: f"[{t[0]:.1f},{t[1]:.1f},{t[2]:.1f}]" if isinstance(t, list) else str(t)
            )
        if "u_optimal" in df.columns:
            df["u_optimal"] = df["u_optimal"].apply(
                lambda u: "[" + ",".join(f"{v:.2f}" for v in u) + "]" if isinstance(u, list) else str(u)
            )
        if "mpc_cost" in df.columns:
            df["mpc_cost"] = df["mpc_cost"].apply(lambda c: f"{c:.4f}" if c < 1e9 else "∞")

        df2 = df.rename(columns=display_cols)
        avail = [v for v in display_cols.values() if v in df2.columns]
        st.dataframe(df2[avail], use_container_width=True, hide_index=True)
    else:
        st.info("Audit log empty — start the control loop: `python src/main.py`")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    state = load_state()
    if state is None:
        st.title("AI-MPC Datacenter Cooling")
        st.warning(
            "No control data found. Start the loop:\n\n```bash\npython src/main.py\n```"
        )
        time.sleep(5)
        st.rerun()
        return

    audit_log = load_audit_log(n=60, current_step=state.get("step"))
    render_dashboard(state, audit_log)
    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    main()
