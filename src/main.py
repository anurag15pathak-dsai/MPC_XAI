"""
main.py - Control Loop Orchestrator for AI-MPC Datacenter Cooling
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_simulator import SensorDataSimulator
from mpc_engine import MPCEngine
from xai_layer import XAILayer, make_mpc_predict_fn
from llm_explainer import LLMExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

CONTROL_INTERVAL_SEC = 5.0
DATA_DIR             = Path("data")
STATE_PATH           = DATA_DIR / "current_state.json"
AUDIT_PATH           = DATA_DIR / "audit_log.json"
OVERRIDES_PATH       = DATA_DIR / "overrides.json"
MAX_AUDIT_ENTRIES    = 200


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


class ControlLoop:
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        if not Path("config/policy_constraints.json").exists():
            SensorDataSimulator.generate_policy_constraints()
            logger.info("Generated config/policy_constraints.json")

        self.simulator = SensorDataSimulator(seed=42)
        self.mpc       = MPCEngine(constraints_path="config/policy_constraints.json")
        self.xai       = XAILayer(mpc_predict_fn=make_mpc_predict_fn(self.mpc))
        self.llm       = LLMExplainer()

        self.audit_log: list = []
        if AUDIT_PATH.exists():
            try:
                with open(AUDIT_PATH) as f:
                    self.audit_log = json.load(f)
                logger.info("Loaded %d existing audit entries", len(self.audit_log))
            except Exception:
                pass

        self._running = True
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *_):
        logger.info("Shutdown signal %s received — stopping after current step …", _[0])
        self._running = False

    def run(self):
        logger.info("Control loop started (interval=%ds)", int(CONTROL_INTERVAL_SEC))
        step = 0
        while self._running:
            t0 = time.monotonic()
            try:
                self._run_step(step)
            except Exception as exc:
                logger.exception("Unhandled error at step %d: %s", step, exc)
            elapsed   = time.monotonic() - t0
            sleep_for = max(0.0, CONTROL_INTERVAL_SEC - elapsed)
            time.sleep(sleep_for)
            step += 1
        logger.info("Control loop stopped.")

    def _run_step(self, step: int) -> None:
        t0 = time.monotonic()

        # 1. Sensor data
        sensor           = self.simulator.simulate_step()
        T_current        = np.array([sensor["temp_zone1"], sensor["temp_zone2"], sensor["temp_zone3"]])
        reservoir_levels = np.array([sensor["reservoir_a"], sensor["reservoir_b"],
                                     sensor["reservoir_c"], sensor["reservoir_main"]])
        ambient_seq      = np.full(10, sensor["ambient_temp"])

        # 2. MPC solve
        mpc_result = self.mpc.solve(
            T_current=T_current,
            ambient_sequence=ambient_seq,
            reservoir_levels=reservoir_levels,
        )
        mpc_result = self._apply_overrides(mpc_result)

        # Feed valve positions back to simulator for airflow coupling
        self.simulator._last_valve_positions = mpc_result["u_optimal"]

        logger.info(
            "Step %3d | solver=%-10s | V=[%s] | T=[%.1f,%.1f,%.1f]°C | Res=[%.0f%%,%.0f%%,%.0f%%] | phase=%s",
            sensor["step"], mpc_result.get("solver", "?"),
            ",".join(f"{v*100:.0f}" for v in mpc_result["u_optimal"]),
            T_current[0], T_current[1], T_current[2],
            reservoir_levels[0], reservoir_levels[1], reservoir_levels[2],
            sensor.get("phase", "?"),
        )

        # Zone 2 shutdown alert
        z2sd = mpc_result.get("zone2_shutdown", {})
        if z2sd.get("active"):
            logger.warning("ZONE 2 SHUTDOWN ACTIVE — %s", z2sd.get("recommendation", ""))

        # 3. SHAP attribution
        shap_result = self.xai.explain(sensor, mpc_result)
        if shap_result.get("primary_driver"):
            pd = shap_result["primary_driver"]
            logger.info("SHAP | primary driver: %s (φ=%.3f)",
                        pd.get("description"), pd.get("shap_value", 0))

        # 4. LLM explanation
        explanation, alert = self.llm.generate_explanation(sensor, mpc_result, shap_result)
        logger.info("LLM alert: %s", alert)

        # 5. Persist
        cycle_ms = (time.monotonic() - t0) * 1000.0
        ts       = datetime.now(timezone.utc)

        state = {
            "step":          sensor["step"],
            "timestamp":     ts.isoformat(),
            "timestamp_str": ts.strftime("%H:%M:%S UTC"),
            "cycle_ms":      cycle_ms,
            "sensor":        sensor,
            "mpc":           mpc_result,
            "shap":          shap_result,
            "explanation":   explanation,
            "alert":         alert,
        }
        _write_json_atomic(STATE_PATH, state)

        audit_entry = {
            "step":                 sensor["step"],
            "timestamp":            ts.strftime("%H:%M:%S"),
            "temperatures":         T_current.tolist(),
            "u_optimal":            mpc_result["u_optimal"],
            # Reservoir levels stored individually for history plots in app.py
            "reservoir_a":          round(float(reservoir_levels[0]), 1),
            "reservoir_b":          round(float(reservoir_levels[1]), 1),
            "reservoir_c":          round(float(reservoir_levels[2]), 1),
            "reservoir_main":       round(float(reservoir_levels[3]), 1),
            "mpc_cost":             mpc_result.get("cost", float("inf")),
            "n_active_constraints": len(mpc_result.get("active_constraints", [])),
            "primary_driver":       (shap_result.get("primary_driver") or {}).get("description", ""),
            "alert":                alert,
            "cycle_ms":             round(cycle_ms, 1),
            "zone2_shutdown":       z2sd.get("active", False),
            "phase":                sensor.get("phase", "normal"),
        }
        self.audit_log.append(audit_entry)
        if len(self.audit_log) > MAX_AUDIT_ENTRIES:
            self.audit_log = self.audit_log[-MAX_AUDIT_ENTRIES:]
        _write_json_atomic(AUDIT_PATH, self.audit_log)

    def _apply_overrides(self, mpc_result: dict) -> dict:
        if not OVERRIDES_PATH.exists():
            return mpc_result
        try:
            with open(OVERRIDES_PATH) as f:
                overrides = json.load(f)
            if overrides.get("enabled"):
                u_new = list(mpc_result["u_optimal"])
                for i, v in overrides.get("valve_positions", {}).items():
                    u_new[int(i)] = float(v)
                mpc_result          = dict(mpc_result)
                mpc_result["u_optimal"] = u_new
                mpc_result["status"]    = "manual_override"
        except Exception as exc:
            logger.warning("Could not apply overrides: %s", exc)
        return mpc_result


if __name__ == "__main__":
    ControlLoop().run()
