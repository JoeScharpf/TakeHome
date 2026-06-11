# Vicsek Particle Interactions

Take-home assignment simulation for collective motion in a 2D periodic box.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
MPLBACKEND=Agg python problem1_dynamics.py
MPLBACKEND=Agg python problem2_phase.py
MPLBACKEND=Agg python -u problem3_optimize.py
pytest -q
```

Generated figures are saved to `outputs/`. Key plots include `problem1_polarization_history.png`, `problem2_psi_vs_eta.png`, and `problem3_optimized_psi_vs_eta.png`.

Unit tests cover core invariants (polarization, periodic boundaries, step update order). Stochastic transition estimates are validated via simulation outputs, not exact unit tests.

Problem 3 uses a coarse-to-fine grid search over `(R, v)`; target runtime is about 15-25 minutes with the vectorized step implementation.

## Bonus: Interactive explorer

Optional Streamlit UI to explore dynamics and phase transitions interactively. See [`explorer/README.md`](explorer/README.md) for usage details.

```bash
streamlit run explorer/app.py
```

## Bonus: Browser demo (Problem 1)

Live canvas simulation of Vicsek dynamics. See [`demo/README.md`](demo/README.md).

**Live demo:** https://vicsekdemo.vercel.app/

```bash
open demo/index.html
```

Mirrors the Python update rules qualitatively, with sliders for η, R, v, and N plus live polarization P(t).

## Bonus: ML surrogate for eta_c

Lightweight regression surrogate predicting critical noise eta_c from (R, v). See [`surrogate/README.md`](surrogate/README.md).

```bash
MPLBACKEND=Agg python surrogate/train.py
```

## Report

The written report is in `report.pdf` (source: `report.md`). Regenerate the PDF after editing the Markdown:

```bash
pandoc report.md -o report.pdf --resource-path=.
```

## Files

- `vicsek.py` — core simulation (periodic geometry, synchronous updates, order parameter)
- `problem1_dynamics.py` — small vs large noise dynamics
- `problem2_phase.py` — stationary order parameter `psi(eta)`
- `problem3_optimize.py` — grid search for `R`, `v` with `eta_c = 0.5`
- `tests/test_vicsek.py` — unit tests for core simulation invariants
- `report.md` / `report.pdf` — written report
- `explorer/` — optional Streamlit interactive explorer (bonus)
- `demo/` — optional browser canvas demo for Problem 1 dynamics (bonus)
- `surrogate/` — optional ML surrogate f(R, v) -> eta_c (bonus)
