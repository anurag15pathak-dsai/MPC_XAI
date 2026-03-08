"""
data_simulator.py - Simulated IoT Sensor Data Generator for Datacenter Cooling

Simulation narrative (5-second steps):
  Phase 0 — Steps  0-11  (~60 s)  : All healthy. Normal operation.
  Phase 1 — Steps 12-23  (~60 s)  : SrcA 90→28%, SrcB 80→22%. MPC must compensate.
  Phase 2 — Steps 24-35  (~60 s)  : SrcA 5-8%, SrcB 8-14%. Zone 2 temp rising.
  Phase 3 — Steps 36+             : Zone 2 over limit; MPC declares load migration.
"""

import json
import os
from typing import Dict, Any

import numpy as np

ZONE_NAMES      = ["zone1", "zone2", "zone3"]
RESERVOIR_NAMES = ["reservoir_a", "reservoir_b", "reservoir_c", "reservoir_main"]


class SensorDataSimulator:
    def __init__(self, seed: int = 42):
        self.rng  = np.random.default_rng(seed)
        self.step = 0
        self._reservoir_levels = np.array([90.0, 80.0, 90.0, 95.0])

    def simulate_step(self, t: float = None) -> Dict[str, Any]:
        if t is None:
            t = self.step

        # Phase boundary injections
        if self.step == 12:
            self._reservoir_levels = np.array([90.0, 80.0, 90.0, 95.0])
        elif self.step == 24:
            self._reservoir_levels = np.array([8.0, 14.0, 55.0, 35.0])

        # Phase-specific parameters
        if self.step < 12:
            temp_baselines    = np.array([26.0, 27.0, 26.0])
            ambient_base      = 22.0
            consumption_rates = np.array([0.12, 0.15, 0.08, 0.10])
        elif self.step < 24:
            phase_frac        = (self.step - 12) / 12.0
            temp_baselines    = np.array([26.0, 27.0 + 1.0 * phase_frac, 26.0])
            ambient_base      = 22.0 + 1.5 * phase_frac
            consumption_rates = np.array([5.2, 4.8, 0.08, 0.10])
        elif self.step < 36:
            phase_frac        = (self.step - 24) / 12.0
            temp_baselines    = np.array([26.0, 28.0 + 4.5 * phase_frac, 26.0])
            ambient_base      = 24.0 + 2.0 * phase_frac
            consumption_rates = np.array([0.15, 0.20, 0.08, 0.10])
        else:
            temp_baselines    = np.array([26.0, 34.0, 26.0])
            ambient_base      = 26.0
            consumption_rates = np.array([0.08, 0.10, 0.05, 0.08])

        # Temperatures (small noise only)
        temp_noise   = self.rng.normal(0.0, 0.25, 3)
        zone_offsets = np.array([0.0, 2.0, 1.0])
        temps = np.clip(temp_baselines + zone_offsets + temp_noise, 18.0, 45.0)

        # Ambient
        ambient_temp = float(ambient_base + self.rng.normal(0.0, 0.1))

        # Reservoirs
        noise = self.rng.normal(0.0, 0.05, 4)
        self._reservoir_levels -= consumption_rates + noise
        refill_mask    = self.rng.random(4) < 0.02
        refill_amounts = self.rng.uniform(3.0, 10.0, 4)
        self._reservoir_levels += refill_amounts * refill_mask
        self._reservoir_levels  = np.clip(self._reservoir_levels, 0.0, 100.0)

        # Airflow coupled to valve positions injected by main.py
        valve_positions = getattr(self, "_last_valve_positions", [0.50, 0.50, 0.55, 0.45])
        zone_valve_avg = [
            (valve_positions[0] + valve_positions[1]) / 2.0,
            valve_positions[2],
            valve_positions[3],
        ]
        airflow_noise = self.rng.normal(0.0, 0.03, 3)
        airflows = np.clip(
            [1.5 + avg * 3.0 + airflow_noise[i] for i, avg in enumerate(zone_valve_avg)],
            0.5, 5.0
        )

        reading = {
            "step":           self.step,
            "timestamp":      float(t),
            "temp_zone1":     float(temps[0]),
            "temp_zone2":     float(temps[1]),
            "temp_zone3":     float(temps[2]),
            "airflow_zone1":  float(airflows[0]),
            "airflow_zone2":  float(airflows[1]),
            "airflow_zone3":  float(airflows[2]),
            "ambient_temp":   ambient_temp,
            "reservoir_a":    float(self._reservoir_levels[0]),
            "reservoir_b":    float(self._reservoir_levels[1]),
            "reservoir_c":    float(self._reservoir_levels[2]),
            "reservoir_main": float(self._reservoir_levels[3]),
            "phase":          self._current_phase(),
        }
        self.step += 1
        return reading

    def _current_phase(self) -> str:
        if self.step <= 12:  return "normal"
        if self.step <= 24:  return "water_shortage"
        if self.step <= 36:  return "critical"
        return "zone_overload"

    @staticmethod
    def generate_policy_constraints(output_path: str = "config/policy_constraints.json") -> Dict[str, Any]:
        constraints = {
            "version": "1.0",
            "effective_date": "2026-01-01",
            "description": "Water withdrawal and thermal safety policy for datacenter cooling under constrained water availability.",
            "temperature_thresholds": {
                "zone1": {"T_max": 35.0, "T_min": 18.0, "unit": "celsius"},
                "zone2": {"T_max": 33.0, "T_min": 18.0, "unit": "celsius"},
                "zone3": {"T_max": 34.0, "T_min": 18.0, "unit": "celsius"},
                "critical_alert_celsius": 38.0,
            },
            "water_withdrawal_limits": {
                "source_a": {"daily_limit_liters": 50000, "hourly_limit_liters": 3000,
                             "description": "Municipal grid water — primary supply", "policy_section": "4.1"},
                "source_b": {"daily_limit_liters": 30000, "hourly_limit_liters": 2000,
                             "description": "Reclaimed water — secondary supply", "policy_section": "4.2"},
                "source_c": {"daily_limit_liters": 20000, "hourly_limit_liters": 1500,
                             "description": "Groundwater — emergency use only", "policy_section": "4.3"},
                "total_daily_limit_liters": 90000,
                "combined_limit_policy_section": "4.0",
            },
            "valve_constraints": {
                "valve1": {"min_position": 0.0, "max_position": 1.0, "zone": "zone1", "source": "source_a"},
                "valve2": {"min_position": 0.0, "max_position": 1.0, "zone": "zone1", "source": "source_b"},
                "valve3": {"min_position": 0.0, "max_position": 1.0, "zone": "zone2", "source": "source_a"},
                "valve4": {"min_position": 0.0, "max_position": 1.0, "zone": "zone3", "source": "source_c"},
            },
            "cost_parameters": {
                "energy_cost_per_kwh": 0.12,
                "water_cost_per_liter": 0.003,
                "alpha_energy_weight": 0.6,
                "beta_water_weight": 0.4,
                "fan_power_kw": [2.5, 2.5, 3.0, 2.0],
                "valve_flow_rate_liters_per_step": [150, 120, 180, 100],
            },
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(constraints, f, indent=2)
        return constraints
