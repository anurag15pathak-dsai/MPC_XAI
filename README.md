# AI-Explained MPC for Sustainable Datacenter Cooling

A full-stack Python application that combines **Model Predictive Control (MPC)**, **SHAP explainability**, and **LLM natural-language generation** to manage datacenter cooling under constrained water availability.

---

## Architecture

```
Layer 1  data_simulator.py   IoT sensor simulation (sine waves + noise)
Layer 2  mpc_engine.py       CVXPY constrained QP optimisation (10-step horizon)
Layer 3  xai_layer.py        SHAP KernelExplainer feature attribution
Layer 4  llm_explainer.py    Anthropic Claude plain-English decision narration
Layer 5  app.py              Streamlit dark industrial HMI dashboard
Layer 6  main.py             Closed-loop orchestrator (5-second control cycle)
```

**Hard constraints enforced by the MPC:**
1. **Thermal Safety** — zone temperatures must never exceed T_max (33–35°C)
2. **Physical Availability** — water usage bounded by live reservoir levels
3. **Policy Compliance** — total withdrawal within regulatory daily limits

---

## Setup

### 1. Clone / navigate to the project
```bash
cd sas-datacenter-cooling
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the API key
```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```
The system runs in **rule-based fallback mode** if the key is absent — all other layers remain fully functional.

---

## Running

Always run commands **from the project root directory**.

### Start the control loop (Terminal 1)
```bash
python src/main.py
```
Runs one full pipeline cycle every 5 seconds. Writes:
- `data/current_state.json` — latest control state (read by dashboard)
- `data/audit_log.json` — rolling audit log (last 200 decisions)

### Start the dashboard (Terminal 2)
```bash
streamlit run src/app.py
```
Opens the Streamlit HMI at `http://localhost:8501`. Auto-refreshes every 5 seconds.

### Run the data simulator standalone
```bash
python src/data_simulator.py
```
Generates `config/policy_constraints.json` and prints sample sensor readings.

---

## Dashboard Features

| Panel | Description |
|-------|-------------|
| Zone Temperature Gauges | Green/yellow/red gauges with T_max thresholds |
| Reservoir Level Gauges | Four water source fill levels |
| Valve Position Chart | MPC-recommended openings with override overlay |
| Active Constraints | Colour-coded CRITICAL / WARNING / OK cards |
| SHAP Feature Attribution | Horizontal bar chart ranked by attribution magnitude |
| AI Explanation | Claude-generated plain-English decision narrative |
| Audit Log | Last 10 timestamped control decisions |
| Manual Override | Per-valve sliders for human-in-the-loop intervention |

---

## Project Structure

```
sas-datacenter-cooling/
├── src/
│   ├── data_simulator.py   # Layer 1 — sensor simulation
│   ├── mpc_engine.py       # Layer 2 — CVXPY MPC
│   ├── xai_layer.py        # Layer 3 — SHAP explainability
│   ├── llm_explainer.py    # Layer 4 — LLM narration
│   ├── main.py             # Layer 6 — orchestrator
│   └── app.py              # Layer 5 — Streamlit dashboard
├── config/
│   └── policy_constraints.json
├── data/                   # Created at runtime
│   ├── current_state.json
│   ├── audit_log.json
│   └── overrides.json      # Written by dashboard override controls
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration

Edit `config/policy_constraints.json` to adjust:
- Zone temperature thresholds (`temperature_thresholds`)
- Daily water withdrawal limits per source (`water_withdrawal_limits`)
- Valve-to-zone-source mapping (`valve_constraints`)
- Energy / water cost weights (`cost_parameters`)

The MPC horizon (default 10 steps) and control interval (default 5 seconds) are set in `src/main.py` and `src/mpc_engine.py` respectively.
