# Vicsek Explorer

Optional Streamlit UI for interactively exploring the Vicsek model from the take-home assignment. This is a **bonus** tool layered on top of the core engine in `vicsek.py`; the written report (`report.pdf`) covers the base assignment only.

## Setup

From the **repo root**:

```bash
pip install -r requirements.txt
```

Requires `streamlit` (listed in root `requirements.txt`).

## Run

From the **repo root**:

```bash
streamlit run explorer/app.py
```

Opens in your browser at http://localhost:8501.

## What it does

The explorer reuses the same simulation code as the batch scripts (`vicsek.py`) but lets you change parameters and view results without rerunning `problem1_dynamics.py` or `problem2_phase.py`.

**Sidebar controls**

| Control | Description |
|---------|-------------|
| Mode | **Fast** (short runs, ~2–5 s) or **Accurate** (longer, smoother estimates) |
| eta | Noise strength (0–1) |
| R | Interaction radius |
| v | Particle speed |
| N | Particle count (50–250) |
| seed | Random seed for reproducibility |

`L = 5.0` and `dt = 0.25` are fixed to match the assignment.

Simulations do **not** run automatically when you move a slider — click a run button in the active tab.

## Tabs

### Dynamics

- **Run dynamics** — runs a trajectory at the sidebar `eta` and shows:
  - instantaneous polarization `P(t)` over time
  - particle snapshot (positions + heading arrows) at the final time
  - metrics: final `P(t)` and a trailing mean over the last 20% of steps
- **Compare eta=0.1 vs eta=0.9** — overlays `P(t)` for low and high noise from the **same** random initial condition (same setup as Problem 1)

### Phase transition

- **Estimate psi(eta) and eta_c** — sweeps `eta` and plots the stationary order parameter `psi(eta)`, then estimates `eta_c` where `psi = 0.5`
- Reference values from batch scripts: `(R=1, v=1) → eta_c ≈ 0.615`; `(R=0.70, v=0.80) → eta_c ≈ 0.52`
- Stochastic estimates vary slightly between Fast and Accurate modes

### About

Short model summary and pointers to `report.pdf` and the batch scripts.

## Mode presets

| | Fast | Accurate |
|---|------|----------|
| Dynamics steps | 200 | 800 |
| psi(eta) grid points | 11 | 21 |
| Seeds per eta | 1 | 3 |
| Burn-in / sample steps | 200 / 400 | 400 / 800 |

## Files

| File | Role |
|------|------|
| `app.py` | Streamlit entry point (layout, session state, tabs) |
| `viz.py` | Matplotlib figure builders |
| `sim.py` | Cached wrappers around `vicsek.py` simulation APIs |

Hysteresis plots are not in the explorer; they are produced by `problem2_phase.py` and saved to `outputs/`.
