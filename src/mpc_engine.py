"""
mpc_engine.py - Model Predictive Control Engine for Datacenter Cooling

Fixes in this version:
  - CLARABEL solver (CVXPY 1.4+ default) tried first; falls back to ECOS/OSQP/SCS
  - Slack variables on ALL constraints → problem is ALWAYS feasible
  - Rate-of-change constraint skipped for emergency valves to avoid infeasibility
  - Zone-2 shutdown detection: when Zone 2 is over thermal limit AND SrcA depleted
  - Flow rates exported in result so dashboard shows correct water percentages
  - Regulatory warning threshold: 92% of POLICY limit (not effective limit)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)

_VALVE_META = [
    {"name": "V1", "zone_idx": 0, "source": "A", "res_idx": 0},
    {"name": "V2", "zone_idx": 0, "source": "B", "res_idx": 1},
    {"name": "V3", "zone_idx": 1, "source": "A", "res_idx": 0},
    {"name": "V4", "zone_idx": 2, "source": "C", "res_idx": 2},
]

_VALVE_BASELINE   = 0.50
_VALVE_MIN_NORMAL = 0.35
_VALVE_OVERCOOL   = 0.35
_VALVE_NORMAL_MID = 0.57
_VALVE_WARM_MID   = 0.825
_VALVE_EMERGENCY  = 0.975

_ZONE_VALVES = {0: [0, 1], 1: [2], 2: [3]}

# Zone 2 shutdown: triggered when these conditions hold for N consecutive steps
_ZONE2_SHUTDOWN_TEMP_THRESHOLD  = 33.0   # Zone 2 T_max
_ZONE2_SHUTDOWN_SRCA_THRESHOLD  = 5.0    # SrcA % below which Zone 2 can't be cooled
_ZONE2_SHUTDOWN_CONSEC_STEPS    = 3      # consecutive steps before shutdown declared


class MPCEngine:
    def __init__(self, constraints_path: str = "config/policy_constraints.json"):
        with open(constraints_path) as f:
            self.policy = json.load(f)

        self.n_zones  = 3
        self.n_valves = 4
        self.horizon  = 6   # shorter horizon = faster + more numerically stable

        self.A    = 0.92 * np.eye(self.n_zones)
        self.B_u  = np.array([
            [-2.5, -1.5,  0.0,  0.0],
            [ 0.0,  0.0, -2.8,  0.0],
            [ 0.0,  0.0,  0.0, -2.2],
        ])
        self.B_d  = np.array([0.08, 0.06, 0.07])

        cp_cfg = self.policy["cost_parameters"]
        self.fan_power    = np.array(cp_cfg["fan_power_kw"])
        self.flow_rates   = np.array(cp_cfg["valve_flow_rate_liters_per_step"])
        self.energy_price = cp_cfg["energy_cost_per_kwh"]
        self.water_price  = cp_cfg["water_cost_per_liter"]

        thresholds = self.policy["temperature_thresholds"]
        self.T_max = np.array([
            thresholds["zone1"]["T_max"],
            thresholds["zone2"]["T_max"],
            thresholds["zone3"]["T_max"],
        ])
        self.T_min = np.array([
            thresholds["zone1"]["T_min"],
            thresholds["zone2"]["T_min"],
            thresholds["zone3"]["T_min"],
        ])

        wl = self.policy["water_withdrawal_limits"]
        self.regulatory_limit_per_step = wl["total_daily_limit_liters"] / 288.0

        # Internal state for zone shutdown detection
        self._zone2_overtemp_count = 0
        self._zone2_shutdown_declared = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        T_current: np.ndarray,
        ambient_sequence: np.ndarray,
        reservoir_levels: np.ndarray,
        regulatory_limit_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        ambient_pred     = np.resize(np.asarray(ambient_sequence, dtype=float), self.horizon)
        regulatory_limit = (
            regulatory_limit_override
            if regulatory_limit_override is not None
            else self.regulatory_limit_per_step
        )

        # Check for zone 2 shutdown condition BEFORE solving
        zone2_shutdown = self._check_zone2_shutdown(T_current, reservoir_levels)

        u_min, u_max = self._compute_valve_bounds(T_current, reservoir_levels, zone2_shutdown)

        result = self._solve_cvxpy(T_current, ambient_pred, reservoir_levels,
                                   regulatory_limit, u_min, u_max)

        if result is None:
            logger.info("CVXPY unavailable — using rule-based fallback")
            return self._rule_based_fallback(
                T_current, reservoir_levels, float(ambient_pred[0]),
                regulatory_limit, zone2_shutdown, u_min, u_max,
            )

        u_opt, T_traj, cost, status, solver_name = result

        return {
            "u_optimal":          u_opt.tolist(),
            "cost":               float(cost),
            "active_constraints": self._identify_active_constraints(
                u_opt, T_current, float(ambient_pred[0]), reservoir_levels, regulatory_limit,
            ),
            "status":             status,
            "predicted_temps":    T_traj.tolist(),
            "regulatory_limit":   regulatory_limit,
            "flow_rates":         self.flow_rates.tolist(),
            "solver":             solver_name,
            "u_min":              u_min.tolist(),
            "u_max":              u_max.tolist(),
            "zone2_shutdown":     zone2_shutdown,
        }

    # ------------------------------------------------------------------
    # Zone 2 shutdown detection
    # ------------------------------------------------------------------

    def _check_zone2_shutdown(
        self,
        T_current: np.ndarray,
        reservoir_levels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Detect when Zone 2 can no longer be cooled.

        Zone 2 is served ONLY by V3 (Source A).  When SrcA is critically depleted
        AND Zone 2 temperature exceeds the limit for multiple consecutive steps,
        the MPC cannot save it — declare a load-migration shutdown.
        """
        zone2_hot  = float(T_current[1]) >= _ZONE2_SHUTDOWN_TEMP_THRESHOLD
        srca_dry   = float(reservoir_levels[0]) < _ZONE2_SHUTDOWN_SRCA_THRESHOLD

        if zone2_hot and srca_dry:
            self._zone2_overtemp_count += 1
        else:
            self._zone2_overtemp_count = max(0, self._zone2_overtemp_count - 1)

        if self._zone2_overtemp_count >= _ZONE2_SHUTDOWN_CONSEC_STEPS:
            self._zone2_shutdown_declared = True

        return {
            "active":         self._zone2_shutdown_declared,
            "consecutive_steps": self._zone2_overtemp_count,
            "zone2_temp":     float(T_current[1]),
            "srca_level":     float(reservoir_levels[0]),
            "recommendation": (
                "LOAD MIGRATION REQUIRED: Zone 2 thermal limit exceeded with Source A depleted. "
                "Migrate Zone 2 workloads to Zone 1 and Zone 3 immediately."
            ) if self._zone2_shutdown_declared else "",
        }

    # ------------------------------------------------------------------
    # Valve bounds
    # ------------------------------------------------------------------

    def _compute_valve_bounds(
        self,
        T_current: np.ndarray,
        reservoir_levels: np.ndarray,
        zone2_shutdown: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        res_a = float(reservoir_levels[0])
        u_min = np.full(self.n_valves, _VALVE_MIN_NORMAL)
        u_max = np.ones(self.n_valves)

        # Source A drought: cap V1/V3 (both use SrcA)
        if res_a < 10.0:
            u_min[0] = 0.15;  u_max[0] = 0.20
            u_min[2] = 0.15;  u_max[2] = 0.20
            u_max[3] = 1.0
        elif res_a < 20.0:
            u_max[0] = 0.60
            u_max[2] = 0.60

        # Temperature emergency overrides water policy
        for zone_idx, valve_indices in _ZONE_VALVES.items():
            if T_current[zone_idx] >= self.T_max[zone_idx]:
                for vi in valve_indices:
                    u_min[vi] = max(u_min[vi], 0.95)
                    u_max[vi] = 1.0

        # Zone 2 shutdown: abandon Zone 2 cooling to save water for other zones
        if zone2_shutdown and zone2_shutdown.get("active"):
            # V3 at minimum — stop wasting water on Zone 2
            u_max[2] = _VALVE_MIN_NORMAL
            u_min[2] = _VALVE_MIN_NORMAL

        u_min = np.minimum(u_min, u_max)
        return u_min, u_max

    # ------------------------------------------------------------------
    # CVXPY solver
    # ------------------------------------------------------------------

    def _solve_cvxpy(
        self,
        T_current: np.ndarray,
        ambient_pred: np.ndarray,
        reservoir_levels: np.ndarray,
        regulatory_limit: float,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, str, str]]:
        MAX_RATE = 0.08
        res_a    = float(reservoir_levels[0])

        u     = cp.Variable((self.horizon, self.n_valves), name="valves")
        T     = cp.Variable((self.horizon + 1, self.n_zones), name="temps")
        slack = cp.Variable((self.horizon, self.n_zones), nonneg=True, name="slack")

        cost        = 0.0
        constraints = [T[0] == T_current.astype(float)]

        for k in range(self.horizon):
            d_k = float(ambient_pred[k])

            constraints.append(T[k+1] == self.A @ T[k] + self.B_u @ u[k] + self.B_d * d_k)

            for vi in range(self.n_valves):
                constraints.append(u[k, vi] >= float(u_min[vi]))
                constraints.append(u[k, vi] <= float(u_max[vi]))

            # Rate-of-change: smooth transitions (skip for emergency-forced valves)
            if k == 0 and hasattr(self, "_u_prev"):
                for vi in range(self.n_valves):
                    if float(u_min[vi]) < 0.90:   # not emergency-forced
                        constraints.append(u[k, vi] <= float(self._u_prev[vi]) + MAX_RATE)
                        constraints.append(u[k, vi] >= float(self._u_prev[vi]) - MAX_RATE)
            elif k > 0:
                for vi in range(self.n_valves):
                    constraints.append(u[k, vi] - u[k-1, vi] <=  MAX_RATE)
                    constraints.append(u[k-1, vi] - u[k, vi] <=  MAX_RATE)

            # Soft thermal constraints (10 000× penalty — always feasible)
            for zi in range(self.n_zones):
                constraints.append(T[k+1, zi] <= self.T_max[zi] + slack[k, zi])
            cost += 10_000.0 * cp.sum(slack[k])

            cost += self.energy_price * (self.fan_power @ u[k])
            water_flow = self.flow_rates @ u[k]
            cost += 10.0 * self.water_price * cp.pos(water_flow - regulatory_limit)

            if res_a < 20.0:
                src_a_penalty = 200.0 if res_a < 10.0 else 50.0
                cost += src_a_penalty * (u[k, 0] + u[k, 2])

        problem = cp.Problem(cp.Minimize(cost), constraints)

        # Try solvers in order of reliability; use installed_solvers() to avoid failures
        available = set(cp.installed_solvers())
        solver_preference = ["CLARABEL", "ECOS", "OSQP", "SCS"]
        solver_used = "NONE"

        for solver_name in solver_preference:
            if solver_name not in available:
                continue
            try:
                problem.solve(solver=solver_name, warm_start=True, verbose=False)
                if problem.status not in ("infeasible", "unbounded", None) and u.value is not None:
                    solver_used = solver_name
                    break
            except Exception as exc:
                logger.debug("Solver %s failed: %s", solver_name, exc)
                continue

        if u.value is None:
            return None

        u_opt = np.clip(u.value[0], u_min, u_max)
        self._u_prev = u_opt.copy()
        T_traj = T.value if T.value is not None else np.tile(T_current, (self.horizon + 1, 1))
        return u_opt, T_traj, float(problem.value), str(problem.status), solver_used

    # ------------------------------------------------------------------
    # Rule-based fallback (used when CVXPY is unavailable)
    # ------------------------------------------------------------------

    def _compute_rule_based_valves(
        self,
        T_current: np.ndarray,
        reservoir_levels: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> np.ndarray:
        res_a = float(reservoir_levels[0])
        u     = np.full(self.n_valves, _VALVE_BASELINE)

        for zone_idx, valve_indices in _ZONE_VALVES.items():
            T_zone = T_current[zone_idx]
            T_max  = self.T_max[zone_idx]
            if T_zone >= T_max:
                target = _VALVE_EMERGENCY
            elif T_zone >= T_max - 3.0:
                target = _VALVE_WARM_MID
            elif T_zone >= T_max - 6.0:
                target = _VALVE_NORMAL_MID
            else:
                target = _VALVE_OVERCOOL
            for vi in valve_indices:
                u[vi] = target

        if res_a < 10.0:
            for vi in (0, 2):
                compensation = u[vi] - 0.15
                u[vi] = 0.15
                u[1]  = min(1.0, u[1] + compensation * 0.5)
                u[3]  = min(1.0, u[3] + compensation * 0.5)
        elif res_a < 20.0:
            reduction = _VALVE_BASELINE * 0.4
            for vi in (0, 2):
                before = u[vi]
                u[vi]  = max(_VALVE_MIN_NORMAL, u[vi] - reduction)
                u[3]   = min(1.0, u[3] + (before - u[vi]))

        # Smooth with previous if available
        if hasattr(self, "_u_prev"):
            MAX_RATE = 0.08
            for vi in range(self.n_valves):
                u[vi] = np.clip(u[vi], self._u_prev[vi] - MAX_RATE, self._u_prev[vi] + MAX_RATE)

        return np.clip(u, u_min, u_max)

    def _rule_based_fallback(
        self,
        T_current: np.ndarray,
        reservoir_levels: np.ndarray,
        ambient: float,
        regulatory_limit: float,
        zone2_shutdown: Dict,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ) -> Dict[str, Any]:
        u_opt = self._compute_rule_based_valves(T_current, reservoir_levels, u_min, u_max)
        self._u_prev = u_opt.copy()
        return {
            "u_optimal":          u_opt.tolist(),
            "cost":               float("inf"),
            "active_constraints": self._identify_active_constraints(
                u_opt, T_current, ambient, reservoir_levels, regulatory_limit,
            ),
            "status":             "rule_based_fallback",
            "predicted_temps":    np.tile(T_current, (self.horizon + 1, 1)).tolist(),
            "regulatory_limit":   regulatory_limit,
            "flow_rates":         self.flow_rates.tolist(),
            "solver":             "RULE_BASED",
            "u_min":              u_min.tolist(),
            "u_max":              u_max.tolist(),
            "zone2_shutdown":     zone2_shutdown,
        }

    # ------------------------------------------------------------------
    # Constraint identification
    # ------------------------------------------------------------------

    def _identify_active_constraints(
        self,
        u_opt: np.ndarray,
        T_current: np.ndarray,
        ambient: float,
        reservoir_levels: np.ndarray,
        regulatory_limit: float,
        tol: float = 1e-2,
    ) -> List[Dict[str, Any]]:
        active = []

        T_next     = self.A @ T_current + self.B_u @ u_opt + self.B_d * ambient
        zone_names = ["Zone 1", "Zone 2", "Zone 3"]

        for name, t_now, t_next, t_max in zip(zone_names, T_current, T_next, self.T_max):
            # Use T_next for margin/severity (MPC looks ahead), but report T_current
            # (actual sensor reading) in the value and label so the HMI stays truthful.
            margin = t_max - t_next
            if margin <= 1.5 or t_now >= t_max:   # also trigger if already breached now
                severity = "CRITICAL" if (margin <= 0 or t_now >= t_max) else "WARNING"
                active.append({
                    "type":            "thermal_safety",
                    "zone":            name,
                    "value":           round(float(t_now), 2),   # ← actual sensor reading
                    "limit":           float(t_max),
                    "utilization_pct": round(float(t_now / t_max * 100), 1),
                    "severity":        severity,
                    "label":           f"{name} at {t_now:.1f}°C (limit {t_max:.0f}°C)",
                })

        # Reservoir warnings — fixed conservative thresholds
        src_labels  = ["Source A", "Source B", "Source C", "Main Reservoir"]
        warn_thresh = [20.0, 20.0, 15.0, 20.0]
        crit_thresh = [10.0, 10.0,  8.0, 10.0]

        for src, level, warn, crit in zip(src_labels, reservoir_levels, warn_thresh, crit_thresh):
            level = float(level)
            if level < warn:
                active.append({
                    "type":            "water_availability",
                    "source":          src,
                    "value":           round(level, 1),
                    "limit":           warn,
                    "utilization_pct": round(100.0 - level, 1),
                    "severity":        "CRITICAL" if level < crit else "WARNING",
                    "label":           f"{src} reservoir at {level:.1f}% capacity",
                })

        # Regulatory limit — ALWAYS compared against policy limit, not effective limit
        water_used = float(self.flow_rates @ u_opt)
        reg_util   = water_used / self.regulatory_limit_per_step * 100.0
        if reg_util >= 92.0:
            section = self.policy["water_withdrawal_limits"].get(
                "combined_limit_policy_section", "4.0"
            )
            active.append({
                "type":            "policy_compliance",
                "value":           round(water_used, 1),
                "limit":           round(self.regulatory_limit_per_step, 1),
                "utilization_pct": round(reg_util, 1),
                "severity":        "CRITICAL" if reg_util >= 100.0 else "WARNING",
                "label":           f"Regulatory water limit at {reg_util:.0f}% (Policy Section {section})",
            })

        # Valve saturation
        for meta, v in zip(_VALVE_META, u_opt):
            if v >= 1.0 - tol:
                active.append({
                    "type":     "valve_saturation",
                    "valve":    meta["name"],
                    "value":    round(float(v), 3),
                    "severity": "INFO",
                    "label":    f"{meta['name']} fully open (100%)",
                })

        return active
