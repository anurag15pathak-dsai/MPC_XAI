"""
xai_layer.py - SHAP-based Explainability Layer for MPC Decisions

Uses SHAP KernelExplainer to attribute each MPC control decision to its
input features. To keep explanations within the 5-second control budget,
the KernelExplainer is applied to the actual MPC black-box with nsamples=12
and a compact k-means background dataset, rebuilt every 25 steps.

Features explained:
    temp_zone1            – Zone 1 server temperature (°C)
    temp_zone2            – Zone 2 server temperature (°C)
    temp_zone3            – Zone 3 server temperature (°C)
    ambient_temp          – External ambient temperature (°C)
    water_level_min       – Minimum reservoir level across all sources (%)
    regulatory_utilization – Current water usage as % of regulatory limit
    cost_weight_ratio     – Alpha / beta cost weighting ratio
"""

import logging
from typing import Any, Callable, Dict, List

import numpy as np
import shap

logger = logging.getLogger(__name__)

# Ordered feature list (must match the order in _features_to_array)
FEATURE_NAMES: List[str] = [
    "temp_zone1",
    "temp_zone2",
    "temp_zone3",
    "ambient_temp",
    "water_level_min",
    "regulatory_utilization",
    "cost_weight_ratio",
]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "temp_zone1": "Zone 1 Server Temperature (°C)",
    "temp_zone2": "Zone 2 Server Temperature (°C)",
    "temp_zone3": "Zone 3 Server Temperature (°C)",
    "ambient_temp": "External Ambient Temperature (°C)",
    "water_level_min": "Minimum Reservoir Level (%)",
    "regulatory_utilization": "Policy Water Limit Utilization (%)",
    "cost_weight_ratio": "Energy-to-Water Cost Weight Ratio",
}

# Plausible ranges used to seed background dataset before history accumulates
_FEATURE_RANGES: Dict[str, tuple] = {
    "temp_zone1": (20.0, 42.0),
    "temp_zone2": (20.0, 40.0),
    "temp_zone3": (20.0, 41.0),
    "ambient_temp": (15.0, 40.0),
    "water_level_min": (5.0, 100.0),
    "regulatory_utilization": (0.0, 100.0),
    "cost_weight_ratio": (0.5, 2.5),
}


class XAILayer:
    """
    SHAP KernelExplainer wrapper for the MPC black-box function.

    Maintains a rolling history of observed (feature, output) pairs.
    Rebuilds the KernelExplainer every `rebuild_interval` steps using a
    k-means summary of the accumulated background dataset.
    """

    def __init__(self, mpc_predict_fn: Callable, rebuild_interval: int = 25):
        """
        Args:
            mpc_predict_fn:   Black-box function: (N, n_features) → (N,) floats.
                              Returns the total valve opening (sum of u*) for each row.
            rebuild_interval: Re-fit KernelExplainer every N explain() calls.
        """
        self.mpc_predict_fn = mpc_predict_fn
        self.rebuild_interval = rebuild_interval

        self._step = 0
        self._explainer: shap.KernelExplainer = None  # type: ignore[assignment]
        self._history: List[np.ndarray] = []          # accumulated feature vectors
        self._max_history = 200                        # rolling window cap

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain(
        self,
        sensor_reading: Dict[str, Any],
        mpc_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute SHAP feature attributions for the current MPC decision.

        Args:
            sensor_reading: Current sensor data dict (from data_simulator).
            mpc_result:     MPC output dict (from mpc_engine.solve()).

        Returns:
            Attribution dict with keys:
              features          – Input feature values
              shap_values       – Dict mapping feature name → SHAP value
              ranked_features   – Features ranked by attribution magnitude
              primary_driver    – Top-ranked feature dict
              active_constraints – Forwarded from mpc_result
              constraint_summary – Human-readable constraint summary string
        """
        features = self._extract_features(sensor_reading, mpc_result)
        x = self._features_to_array(features)

        # Accumulate history for background dataset
        self._history.append(x.copy())
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Rebuild KernelExplainer on schedule
        if self._explainer is None or (self._step % self.rebuild_interval == 0):
            self._rebuild_explainer(x)

        # Compute SHAP values (KernelExplainer expects 2-D input)
        shap_vals = self._compute_shap(x)

        ranked = self._rank_features(features, shap_vals)
        active = mpc_result.get("active_constraints", [])

        self._step += 1

        return {
            "features": features,
            "shap_values": {f: float(v) for f, v in zip(FEATURE_NAMES, shap_vals)},
            "ranked_features": ranked,
            "primary_driver": ranked[0] if ranked else {},
            "active_constraints": active,
            "constraint_summary": _summarize_active_constraints(active),
            "expected_value": float(
                self._explainer.expected_value
                if self._explainer is not None
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        sensor_reading: Dict[str, Any],
        mpc_result: Dict[str, Any],
    ) -> Dict[str, float]:
        """Derive the 7 explanation features from sensor + MPC outputs."""
        reservoir_vals = [
            sensor_reading.get("reservoir_a", 50.0),
            sensor_reading.get("reservoir_b", 50.0),
            sensor_reading.get("reservoir_c", 50.0),
            sensor_reading.get("reservoir_main", 50.0),
        ]

        # Water usage as % of regulatory limit
        u_opt = mpc_result.get("u_optimal", [0.5] * 4)
        flow_rates = [150.0, 120.0, 180.0, 100.0]  # L/step (matches mpc_engine defaults)
        water_used = sum(u * r for u, r in zip(u_opt, flow_rates))
        reg_limit = mpc_result.get("regulatory_limit", 1.0)
        reg_util = (water_used / reg_limit * 100.0) if reg_limit > 0 else 0.0

        alpha = 0.6
        beta = 0.4
        cost_weight_ratio = alpha / beta if beta > 0 else 1.5

        return {
            "temp_zone1": float(sensor_reading["temp_zone1"]),
            "temp_zone2": float(sensor_reading["temp_zone2"]),
            "temp_zone3": float(sensor_reading["temp_zone3"]),
            "ambient_temp": float(sensor_reading["ambient_temp"]),
            "water_level_min": float(min(reservoir_vals)),
            "regulatory_utilization": float(reg_util),
            "cost_weight_ratio": float(cost_weight_ratio),
        }

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([features[f] for f in FEATURE_NAMES], dtype=float)

    def _build_background(self, current_x: np.ndarray) -> np.ndarray:
        """
        Construct background dataset for KernelSHAP.
        Uses accumulated history when available; otherwise generates
        uniform samples across plausible feature ranges.
        """
        n_background = 30

        if len(self._history) >= n_background:
            raw = np.array(self._history[-n_background:])
        else:
            # Synthetic background from plausible ranges
            rng = np.random.default_rng(0)
            raw = np.column_stack([
                rng.uniform(lo, hi, n_background)
                for lo, hi in [_FEATURE_RANGES[f] for f in FEATURE_NAMES]
            ])

        return raw

    def _rebuild_explainer(self, current_x: np.ndarray) -> None:
        """Fit a new KernelExplainer using a k-means summary of the background."""
        logger.debug("Rebuilding SHAP KernelExplainer (step %d)", self._step)
        background_raw = self._build_background(current_x)

        # k-means summary reduces background to 10 representative points
        n_clusters = min(10, len(background_raw))
        background_summary = shap.kmeans(background_raw, n_clusters)

        self._explainer = shap.KernelExplainer(
            self.mpc_predict_fn,
            background_summary,
        )
        logger.debug("KernelExplainer rebuilt with %d background clusters", n_clusters)

    def _compute_shap(self, x: np.ndarray) -> np.ndarray:
        """Compute SHAP values for a single feature vector."""
        try:
            raw = self._explainer.shap_values(
                x.reshape(1, -1),
                nsamples=12,   # Low sample count for real-time use
                silent=True,
            )
            # KernelExplainer may return list of arrays for multi-output
            if isinstance(raw, list):
                raw = raw[0]
            return np.asarray(raw).flatten()
        except Exception as e:
            logger.warning("SHAP computation failed (%s) — using uniform attribution", e)
            return np.zeros(len(FEATURE_NAMES))

    def _rank_features(
        self,
        features: Dict[str, float],
        shap_vals: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Rank features by absolute SHAP magnitude, descending."""
        abs_shap = np.abs(shap_vals)
        ranked_idx = np.argsort(abs_shap)[::-1]

        ranked = []
        for rank, idx in enumerate(ranked_idx):
            fname = FEATURE_NAMES[idx]
            sv = float(shap_vals[idx])
            ranked.append({
                "rank": rank + 1,
                "feature": fname,
                "description": FEATURE_DESCRIPTIONS[fname],
                "value": float(features[fname]),
                "shap_value": sv,
                "abs_shap": float(abs_shap[idx]),
                "direction": "increases_cooling" if sv > 0 else "reduces_cooling",
            })
        return ranked


# ------------------------------------------------------------------
# Module-level utility
# ------------------------------------------------------------------

def _summarize_active_constraints(active_constraints: List[Dict]) -> str:
    """Produce a terse human-readable summary of binding constraints."""
    if not active_constraints:
        return "No active constraints — system operating within normal bounds."

    criticals = [c for c in active_constraints if c.get("severity") == "CRITICAL"]
    warnings = [c for c in active_constraints if c.get("severity") == "WARNING"]

    parts = []
    if criticals:
        parts.append("CRITICAL: " + "; ".join(c["label"] for c in criticals[:2]))
    if warnings:
        parts.append("WARNING: " + "; ".join(c["label"] for c in warnings[:2]))

    return " | ".join(parts)


def make_mpc_predict_fn(mpc_engine) -> Callable:  # type: ignore[type-arg]
    """
    Factory: wraps an MPCEngine instance as a black-box callable compatible
    with SHAP KernelExplainer.

    The output scalar is the total valve opening (sum of u*), which
    serves as a proxy for overall cooling intensity.

    Args:
        mpc_engine: An initialised MPCEngine instance.

    Returns:
        predict(X: ndarray shape (N, 7)) -> ndarray shape (N,)
    """
    horizon = mpc_engine.horizon

    def predict(X: np.ndarray) -> np.ndarray:
        results = []
        for row in X:
            t1, t2, t3, t_amb, w_min, _reg_util, _ratio = row
            T_current = np.array([t1, t2, t3])
            reservoir_levels = np.full(4, max(w_min, 5.0))
            ambient_seq = np.full(horizon, t_amb)

            result = mpc_engine.solve(T_current, ambient_seq, reservoir_levels)
            results.append(float(sum(result["u_optimal"])))

        return np.array(results)

    return predict


# Fix forward-reference annotation
from typing import Callable  # noqa: E402 (placed after use for readability)
