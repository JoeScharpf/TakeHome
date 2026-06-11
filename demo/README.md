# Vicsek Browser Demo (Problem 1)

Live **canvas simulation** of Vicsek flocking dynamics — a bonus visual companion to [`problem1_dynamics.py`](../problem1_dynamics.py). Mirrors [`vicsek.py`](../vicsek.py) update rules **qualitatively** (JavaScript RNG differs from Python; not bit-identical).

Problems 2 and 3 (phase transition, optimization) remain in the Python scripts only.

## Open the demo

**Option A — double-click** `demo/index.html` in a browser (no server needed).

**Option B — local server** from the repo root:

```bash
python -m http.server 8000
```

Then visit http://localhost:8000/demo/

## Controls

| Control | Range | Default |
|---------|-------|---------|
| η (noise) | 0.0–1.0 | 0.5 |
| R (interaction radius) | 0.3–1.5 | 1.0 |
| v (speed) | 0.5–1.8 | 1.0 |
| N (particles) | 25–250 | 125 |

**Buttons**

- **Start / Pause** — run or freeze the simulation
- **Reset** — new random initial condition (seed 42)
- **Low noise (η=0.1)** / **High noise (η=0.9)** — set η and reset (Problem 1 comparison)

Fixed assignment constants: `L = 5`, `dt = 0.25`.

## What to expect

- **Low η:** after a short transient, `P(t)` rises and headings align (flocking).
- **High η:** `P(t)` stays low; headings look random (disorder).
- Particles wrap periodically at box edges (Pac-Man style).

Compare qualitatively to [`outputs/problem1_polarization_history.png`](../outputs/problem1_polarization_history.png).

## Files

| File | Role |
|------|------|
| `vicsek.js` | Simulation engine (`VicsekSim`, periodic neighbors, polarization) |
| `main.js` | Canvas rendering, controls, animation loop |
| `index.html` | Page layout |
| `style.css` | UI styling |
