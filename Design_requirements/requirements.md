# Requirements Document
## AI-MPC Datacenter Cooling — SAS (Smart Autonomous Systems)

---

## 1. Project Overview

**Project Name:** AI-MPC Datacenter Cooling with XAI and LLM Explanation  
**Hackathon:** AWS AI for Bharat Hackathon  
**Problem Domain:** Sustainable datacenter operations under water scarcity  

---

## 2. Functional Requirements

### 2.1 Control System

| ID | Requirement |
|----|-------------|
| F1 | The system shall run a full control cycle every 5 seconds autonomously |
| F2 | The MPC shall solve a constrained quadratic program to determine optimal valve positions for 4 cooling valves |
| F3 | The MPC shall enforce zone temperature limits (Zone 1: ≤35°C, Zone 2: ≤33°C, Zone 3: ≤34°C) |
| F4 | The MPC shall enforce water withdrawal limits per Policy Sections 4.0–4.3 |
| F5 | The MPC shall dynamically tighten valve bounds when reservoir levels fall below 20% (warning) or 10% (critical) |
| F6 | The system shall fall back to rule-based valve control if the CVXPY solver fails |
| F7 | The system shall detect Zone 2 thermal shutdown conditions and recommend workload migration |

### 2.2 Explainability

| ID | Requirement |
|----|-------------|
| F8 | The XAI layer shall compute SHAP feature attributions for every MPC decision |
| F9 | SHAP values shall be computed within the 5-second control budget |
| F10 | The system shall identify and rank the top contributing features to each control decision |
| F11 | The SHAP explainer shall rebuild its background dataset every 25 steps to reflect current system state |

### 2.3 LLM Explanation

| ID | Requirement |
|----|-------------|
| F12 | The LLM shall generate a conversational plain-English briefing for every control cycle |
| F13 | The explanation shall reference actual sensor readings, not MPC predicted temperatures |
| F14 | The explanation shall cover: active constraint, SHAP primary driver, valve actions, and 30-second forecast |
| F15 | The system shall fall back to a rule-based explanation generator when the Anthropic API is unavailable |
| F16 | If Zone 2 exceeds its limit, the explanation shall clearly state the breach and urge action |

### 2.4 HMI Dashboard

| ID | Requirement |
|----|-------------|
| F17 | The dashboard shall auto-refresh every 5 seconds |
| F18 | Zone temperatures shall be displayed as a line chart with T_max reference lines and a highlighted current reading |
| F19 | Reservoir levels shall be displayed as 4 individual gauge dials with warning and critical threshold markers |
| F20 | Active constraints shall be displayed with severity colouring (red = critical, amber = warning, green = normal) |
| F21 | SHAP attributions shall be displayed as a ranked horizontal bar chart |
| F22 | The AI explanation shall be displayed in a conversational format, not a structured report |
| F23 | The dashboard shall display the last 10 control decisions in an audit log table |
| F24 | The dashboard shall provide manual valve override sliders for human-in-the-loop control |
| F25 | The temperature history chart shall only display data from the current simulation run, not previous runs |

---

## 3. Non-Functional Requirements

### 3.1 Performance

| ID | Requirement |
|----|-------------|
| NF1 | The MPC solver (CLARABEL) shall complete within 50ms per cycle |
| NF2 | The full control cycle (sensor → MPC → SHAP → LLM → persist) shall complete within 10 seconds |
| NF3 | The SHAP KernelExplainer shall use nsamples=12 to maintain real-time performance |
| NF4 | JSON state files shall be written atomically to prevent dashboard from reading partial data |

### 3.2 Reliability

| ID | Requirement |
|----|-------------|
| NF5 | The system shall remain operational if the Anthropic API is unavailable (rule-based fallback) |
| NF6 | The system shall remain operational if the CVXPY solver fails (rule-based valve fallback) |
| NF7 | The control loop shall catch and log all exceptions without stopping |

### 3.3 Accuracy

| ID | Requirement |
|----|-------------|
| NF8 | Active constraint labels shall display actual sensor temperatures, not MPC predicted values |
| NF9 | The MPC 1-step temperature prediction error shall be ≤ ±0.5°C under normal conditions |
| NF10 | The SHAP primary driver shall be correctly identified across all simulation phases |

### 3.4 Deployability

| ID | Requirement |
|----|-------------|
| NF11 | The system shall deploy on a single AWS EC2 t3.medium instance |
| NF12 | The dashboard shall be accessible via browser on port 8501 |
| NF13 | All configuration shall be externalised to `config/policy_constraints.json` |
| NF14 | The Anthropic API key shall be set via environment variable, never hardcoded |

---

## 4. System Constraints

| Constraint | Value |
|------------|-------|
| Control cycle interval | 5 seconds |
| Number of cooling zones | 3 |
| Number of water sources | 4 (A: municipal, B: reclaimed, C: groundwater, Main: combined) |
| Number of valves | 4 |
| MPC prediction horizon | 6 steps (30 seconds) |
| Regulatory water limit | 312.5 L/step combined |
| Zone 2 temperature limit | 33°C (tightest constraint) |
| Reservoir critical threshold | 10% (per source) |
| Reservoir warning threshold | 20% (per source) |

---

## 5. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Control optimisation | CVXPY + CLARABEL | Latest |
| Explainability | SHAP KernelExplainer | Latest |
| LLM | Anthropic Claude Haiku | claude-haiku-4-5-20251001 |
| Dashboard | Streamlit | Latest |
| Visualisation | Plotly | Latest |
| Data processing | NumPy, Pandas | Latest |
| Runtime | Python 3.10+ | 3.10+ |
| Deployment | AWS EC2 Ubuntu 22.04 | t3.medium |

---

## 6. Policy Compliance Requirements

| Policy Section | Requirement |
|----------------|-------------|
| 4.0 | Combined daily water withdrawal ≤ 90,000 L; per-step limit ≤ 312.5 L |
| 4.1 | Source A (municipal) hourly limit ≤ 3,000 L |
| 4.2 | Source B (reclaimed) hourly limit ≤ 2,000 L |
| 4.3 | Source C (groundwater) hourly limit ≤ 1,500 L; emergency use only |

---

## 7. Simulation Scenario Requirements

| Phase | Trigger | Expected MPC Behaviour |
|-------|---------|------------------------|
| Normal (Steps 0–11) | All sources healthy | Minimum-cost baseline valve positions |
| Water Shortage (Steps 12–23) | Source A and B drain simultaneously at high rate | Throttle V1/V3 (Source A), manage V2 (Source B), shift to Source C |
| Critical (Steps 24–35) | Both sources critically low, Zone 2 rising | Emergency groundwater activation, maximum V3 opening |
| Zone Overload (Steps 36+) | Zone 2 exceeds 33°C limit | Shutdown flag raised, workload migration recommended |
