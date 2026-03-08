# System Design Document
## AI-MPC Datacenter Cooling — SAS (Smart Autonomous Systems)

---

## 1. Problem Statement

Modern datacenters face a dual constraint crisis: **thermal safety** (server overheating causes hardware failure) and **water scarcity** (cooling systems consume large volumes of water under drought and regulatory restrictions). Existing rule-based cooling systems cannot simultaneously optimise across multiple conflicting constraints in real time, and provide no explainability to operators.

---

## 2. Solution Overview

An autonomous AI-powered datacenter cooling system that:
- Uses **Model Predictive Control (MPC)** to solve a constrained optimisation problem every 5 seconds
- Applies **SHAP-based explainability (XAI)** to identify why each decision was made
- Uses a **Large Language Model (Claude)** to translate control decisions into plain-English operator briefings
- Provides a live **Streamlit HMI dashboard** for human oversight

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Control Loop (main.py)                  │
│                     Cycle: every 5 seconds                  │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │
           ▼              ▼              ▼
   ┌───────────┐  ┌──────────────┐  ┌──────────────┐
   │  Sensor   │  │  MPC Engine  │  │  XAI Layer   │
   │Simulator  │→ │  (CVXPY +    │→ │  (SHAP       │
   │           │  │  CLARABEL)   │  │  Kernel      │
   └───────────┘  └──────────────┘  │  Explainer)  │
                                    └──────┬───────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │  LLM Explainer   │
                                  │  (Claude Haiku)  │
                                  └──────┬───────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   data/ (JSON files) │
                              │  current_state.json  │
                              │  audit_log.json      │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Streamlit HMI       │
                              │  (app.py)            │
                              │  Port 8501           │
                              └─────────────────────┘
```

---

## 4. Component Design

### 4.1 Data Simulator (`data_simulator.py`)

Generates synthetic IoT sensor readings every 5 seconds across 4 simulation phases:

| Phase | Steps | Description |
|-------|-------|-------------|
| Normal | 0–11 | Healthy operation, all reservoirs full |
| Water Shortage | 12–23 | Source A (5.2/step) and Source B (4.8/step) drain rapidly |
| Critical | 24–35 | Zone 2 temperature rises toward 33°C limit |
| Zone Overload | 36+ | Zone 2 exceeds limit, load migration required |

**Sensor outputs per step:**
- Zone temperatures: `temp_zone1`, `temp_zone2`, `temp_zone3`
- Airflow rates: `airflow_zone1/2/3`
- Reservoir levels: `reservoir_a/b/c/reservoir_main`
- Ambient temperature
- Current phase label

---

### 4.2 MPC Engine (`mpc_engine.py`)

Solves a quadratic program (QP) every 5 seconds using **CVXPY** with the **CLARABEL** solver.

**State-space model:**
```
T[k+1] = A·T[k] + B_u·u[k] + B_d·d[k]
```
Where:
- `T[k]` = zone temperatures (3×1)
- `u[k]` = valve positions (4×1), range [0, 1]
- `d[k]` = ambient disturbance

**Objective function:**
```
min  α · Σ(fan_power · u²)  +  β · Σ(flow_rate · u)
```
- α = 0.6 (energy cost weight)
- β = 0.4 (water cost weight)

**Constraints enforced:**
- Zone temperature bounds: T_min ≤ T ≤ T_max per zone
- Valve bounds: u_min ≤ u ≤ u_max (tightened dynamically based on reservoir levels)
- Regulatory water limit: Σ(flow_rate · u) ≤ 312.5 L/step (Policy Section 4.0)
- Reservoir safety: valves throttled when source < 20% (warning) or < 10% (critical)

**Fallback:** Rule-based valve control when CVXPY solver fails.

**Active constraints reported using actual sensor readings** (`T_current`), not MPC predicted values.

---

### 4.3 XAI Layer (`xai_layer.py`)

SHAP KernelExplainer applied to the MPC black-box function.

**7 explanation features:**

| Feature | Description |
|---------|-------------|
| `temp_zone1/2/3` | Zone server temperatures (°C) |
| `ambient_temp` | External ambient temperature (°C) |
| `water_level_min` | Minimum reservoir level across all sources (%) |
| `regulatory_utilization` | Current water usage as % of policy limit |
| `cost_weight_ratio` | α/β energy-to-water cost weight ratio |

**Design decisions:**
- Background dataset: k-means summary of 30 historical feature vectors (10 clusters)
- Explainer rebuilt every 25 steps to reflect changing system state
- `nsamples=12` for real-time performance within 5-second control budget
- Output: ranked features by absolute SHAP value, primary driver identified

---

### 4.4 LLM Explainer (`llm_explainer.py`)

Calls **Claude Haiku** (`claude-haiku-4-5-20251001`) with sensor data, MPC result, and SHAP attribution to generate a conversational plain-English briefing.

**Prompt design:**
- Instructs Claude to speak like a knowledgeable engineer, not produce a structured report
- Explicitly passes actual sensor temperatures (not MPC predicted values) labelled as "Sensor temps (ACTUAL current)"
- Covers: triggered constraint → SHAP driver in plain English → valve actions → 30-second forecast
- Graceful fallback to rule-based explanation when API key is unavailable

---

### 4.5 HMI Dashboard (`app.py`)

Streamlit dashboard auto-refreshing every 5 seconds.

**Layout:**

| Section | Content |
|---------|---------|
| Header | Phase label, step counter, solver, cycle time |
| Alert banner | Critical/warning/normal status |
| Zone Temperatures | Line chart with T_max reference lines, highlighted current reading |
| Reservoir Levels | 4 gauge dials with warning (20%) and critical (10%) thresholds |
| MPC Valve Positions | Horizontal bar chart |
| Active Constraints | Colour-coded constraint panel |
| Airflow | Bar chart per zone |
| SHAP Attribution | Horizontal bar chart ranked by absolute SHAP value |
| AI Operator Assistant | Conversational plain-English briefing from Claude |
| Audit Log | Last 10 decisions table |
| Manual Override | Human-in-the-loop valve sliders |

**Key fix:** `load_audit_log()` detects simulation restarts by finding where step counter goes backwards, and filters entries with `step > current_step` to prevent stale data from previous runs contaminating the chart.

---

### 4.6 Policy Constraints (`config/policy_constraints.json`)

Defines all safety and regulatory limits:
- Zone temperature limits (Zone 1: 35°C, Zone 2: 33°C, Zone 3: 34°C)
- Water withdrawal limits per source (Source A: 3000 L/hr, Source B: 2000 L/hr, Source C: 1500 L/hr)
- Combined daily limit: 90,000 L
- Valve min/max positions per zone/source mapping
- Cost parameters: energy 0.12 $/kWh, water 0.003 $/L

---

## 5. Data Flow

```
Every 5 seconds:
1. SensorDataSimulator.simulate_step()     → sensor dict
2. MPCEngine.solve(T, ambient, reservoirs) → u_optimal, constraints, predicted_temps
3. XAILayer.explain(sensor, mpc_result)    → shap_values, ranked_features, primary_driver
4. LLMExplainer.generate_explanation()     → plain-English briefing, alert string
5. Write current_state.json (atomic)       → Streamlit reads this
6. Append to audit_log.json               → Streamlit plots history
```

---

## 6. Deployment Architecture (AWS EC2)

```
EC2 Instance (t3.medium, Ubuntu 22.04)
├── screen session "control"   → python3 src/main.py
└── screen session "dashboard" → streamlit run src/app.py --server.port 8501
```

- Both processes share the `data/` folder on the same filesystem
- Dashboard accessible at `http://<EC2_PUBLIC_IP>:8501`
- Security group: port 22 (SSH), port 8501 (Streamlit)

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CLARABEL solver | Open-source, faster than ECOS for this QP size, no license cost |
| KernelSHAP with nsamples=12 | Balances explanation quality against 5-second control budget |
| Rebuild explainer every 25 steps | Adapts background distribution as system state evolves |
| Atomic JSON writes (write to .tmp then rename) | Prevents Streamlit from reading partial files |
| Sensor temps in constraints (not predicted) | Ensures HMI shows truthful current state, not model artefacts |
| Rule-based fallback in both MPC and LLM | System remains operational without internet or solver availability |
