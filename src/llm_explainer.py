"""
llm_explainer.py — Natural-Language Explanation Layer

Generates a structured, operator-readable explanation for each MPC decision.
Format:
  CONSTRAINTS: [what is violated/near-violated]
  SHAP DRIVER: [which feature is pushing the decision]
  MPC ACTIONS: [valve-by-valve reasoning]
  FORECAST: [what happens next / load migration if needed]
"""

import logging
import os
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

_FLOW_RATES  = [150, 120, 180, 100]
_VALVE_NAMES = [
    "V1 (Zone1/SrcA — municipal)",
    "V2 (Zone1/SrcB — reclaimed)",
    "V3 (Zone2/SrcA — municipal)",
    "V4 (Zone3/SrcC — groundwater)",
]


class LLMExplainer:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=key) if key else None

    def generate_explanation(
        self,
        sensor: Dict[str, Any],
        mpc_result: Dict[str, Any],
        shap_result: Dict[str, Any],
    ) -> tuple:
        ctx = self._build_context(sensor, mpc_result, shap_result)
        if self._client:
            try:
                return self._call_llm(ctx)
            except Exception as exc:
                logger.warning("LLM API failed (%s) — rule-based fallback", exc)
        return self._rule_based_explanation(ctx)

    def _build_context(self, sensor, mpc_result, shap_result) -> Dict[str, Any]:
        u   = mpc_result.get("u_optimal", [0.5] * 4)
        fr  = mpc_result.get("flow_rates", _FLOW_RATES)
        reg = mpc_result.get("regulatory_limit", 312.0)

        water_used = sum(u_i * r for u_i, r in zip(u, fr))
        reg_pct    = water_used / reg * 100.0 if reg > 0 else 0.0

        ranked  = shap_result.get("ranked_features", [])
        primary = shap_result.get("primary_driver", {})

        return {
            "temps":          [sensor.get(f"temp_zone{i}", 0.0) for i in range(1, 4)],
            "reservoirs":     [sensor.get("reservoir_a", 0), sensor.get("reservoir_b", 0),
                               sensor.get("reservoir_c", 0), sensor.get("reservoir_main", 0)],
            "ambient":        sensor.get("ambient_temp", 22.0),
            "valves":         [round(v * 100, 1) for v in u],
            "water_used":     round(water_used, 0),
            "reg_pct":        round(reg_pct, 1),
            "constraints":    mpc_result.get("active_constraints", []),
            "primary_driver": primary.get("description", "unknown"),
            "primary_shap":   primary.get("shap_value", 0.0),
            "top_features":   ranked[:3],
            "zone2_shutdown": mpc_result.get("zone2_shutdown", {}),
            "phase":          sensor.get("phase", "normal"),
        }

    def _call_llm(self, ctx: Dict[str, Any]) -> tuple:
        z2sd = ctx["zone2_shutdown"]
        shutdown_note = (
            "\n\nCRITICAL SYSTEM EVENT: Zone 2 thermal limit reached AND Source A depleted. "
            "Zone 2 CANNOT be cooled. Load migration to Zone 1 and Zone 3 is REQUIRED."
        ) if z2sd.get("active") else ""

        constraint_summary = "; ".join(
            c.get("label", "") for c in ctx["constraints"]
            if c.get("severity") in ("CRITICAL", "WARNING")
        ) or "None"

        shap_summary = ", ".join(
            f"{f['description']} (φ={f['shap_value']:+.3f})"
            for f in ctx["top_features"]
        ) or "No SHAP data yet"

        valve_str = " | ".join(
            f"{name}→{v}%" for name, v in zip(_VALVE_NAMES, ctx["valves"])
        )

        response = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=320,
            system=(
                "You are an AI operator assistant giving a live verbal briefing on datacenter cooling decisions. "
                "Speak naturally and conversationally, like a knowledgeable engineer talking to a colleague — "
                "not a structured report. Use plain English sentences, no bullet headers. "
                "Cover in 3–4 flowing sentences: what constraint or risk triggered action, "
                "which physical factor is driving the decision (explain the SHAP driver in plain English), "
                "what the MPC has done with the valves and why, "
                "and what to expect in the next 30 seconds. "
                "If Zone 2 shutdown is active, open with a clear migration warning. "
                "Be specific with numbers. Sound human, clear, and direct."
            ),
            messages=[{"role": "user", "content": (
                f"Temperatures: Z1={ctx['temps'][0]:.1f}°C(max35), "
                f"Z2={ctx['temps'][1]:.1f}°C(max33), Z3={ctx['temps'][2]:.1f}°C(max34)\n"
                f"Reservoirs: A={ctx['reservoirs'][0]:.1f}%, B={ctx['reservoirs'][1]:.1f}%, "
                f"C={ctx['reservoirs'][2]:.1f}%\n"
                f"Water: {ctx['water_used']:.0f}L ({ctx['reg_pct']:.1f}% of limit)\n"
                f"Active constraints: {constraint_summary}\n"
                f"SHAP top-3: {shap_summary}\n"
                f"Valves: {valve_str}"
                f"{shutdown_note}"
            )}],
        )
        explanation = response.content[0].text.strip()
        return explanation, self._derive_alert(ctx)

    def _rule_based_explanation(self, ctx: Dict[str, Any]) -> tuple:
        temps, res, valves = ctx["temps"], ctx["reservoirs"], ctx["valves"]
        constr = ctx["constraints"]
        z2sd   = ctx["zone2_shutdown"]

        crits = [c["label"] for c in constr if c.get("severity") == "CRITICAL"]
        warns = [c["label"] for c in constr if c.get("severity") == "WARNING"]
        drv   = ctx["primary_driver"]
        shap  = ctx["primary_shap"]
        shap_dir = "pushing the system to demand more cooling" if shap > 0 else "actually giving the MPC some room to ease off"

        # Sentence 1 — what triggered action
        if crits:
            s1 = f"We've got a critical situation here — {crits[0].lower().rstrip('.')}."
        elif warns:
            s1 = f"The system is flagging a warning: {warns[0].lower().rstrip('.')}."
        else:
            s1 = f"Everything is looking healthy right now — all thermal and water constraints are within safe bounds."

        # Sentence 2 — SHAP driver in plain English
        s2 = (
            f"The biggest factor driving the current decision is {drv} (sitting at {ctx['temps'][0]:.1f}°C for Zone 1, "
            f"and the SHAP score is {shap:+.3f}), which is {shap_dir}."
        )

        # Sentence 3 — what MPC did
        v_summary = []
        if res[0] < 15:
            v_summary.append(f"V1 and V3 — the Source A municipal valves — have been throttled back to {valves[0]}% and {valves[2]}% respectively to protect the nearly-depleted reservoir")
            v_summary.append(f"V2 (reclaimed water) has been opened up to {valves[1]}% to compensate")
        else:
            v_summary.append(f"the MPC is running V1 at {valves[0]}%, V2 at {valves[1]}%, V3 at {valves[2]}%, and V4 at {valves[3]}% to balance cooling load against water cost")
        s3 = f"In response, {'; '.join(v_summary)}."

        # Sentence 4 — forecast
        if z2sd.get("active"):
            s4 = (
                f"⚠️ Heads up — Zone 2 is now at {temps[1]:.1f}°C and Source A is completely gone, "
                "so we cannot cool it any further. Workloads need to be migrated to Zone 1 and Zone 3 right now to prevent hardware damage."
            )
        elif res[0] < 8:
            steps_left = max(1, int(res[0] / 0.4))
            s4 = (
                f"Source A is down to {res[0]:.1f}% — we're about {steps_left} control steps away from it running dry. "
                "Get ready to shift Zone 2 workloads before cooling becomes impossible."
            )
        elif temps[1] > 30:
            s4 = (
                f"Zone 2 is already at {temps[1]:.1f}°C, which is creeping toward its 33°C limit. "
                "If Source A keeps dropping, we'll lose the ability to cool Zone 2 and will need to act fast."
            )
        else:
            s4 = (
                f"We're currently using {ctx['reg_pct']:.1f}% of the regulatory water budget, so there's still headroom. "
                "No immediate risk — the MPC is just holding the minimum-cost baseline for now."
            )

        return "\n\n".join([s1, s2, s3, s4]), self._derive_alert(ctx)

    def _derive_alert(self, ctx: Dict[str, Any]) -> str:
        z2sd = ctx["zone2_shutdown"]
        if z2sd.get("active"):
            return (
                f"🔴 ZONE SHUTDOWN: Zone 2 at {ctx['temps'][1]:.1f}°C — "
                "SrcA depleted, cooling impossible. Migrate workloads NOW."
            )
        crits = [c for c in ctx["constraints"] if c.get("severity") == "CRITICAL"]
        warns = [c for c in ctx["constraints"] if c.get("severity") == "WARNING"]
        if crits:
            return f"🔴 CRITICAL: {crits[0]['label']}"
        if warns:
            return f"🟡 WARNING: {warns[0]['label']}"
        return "✅ Normal: All thermal and water constraints satisfied."
