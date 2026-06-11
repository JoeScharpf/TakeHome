# ML Surrogate for eta_c

Bonus extension for Problem 3: train a lightweight surrogate **f(R, v) -> eta_c** using labels from the physics simulation ([`vicsek.py`](../vicsek.py) `estimate_eta_c`).

The simulation remains the source of truth; the model approximates the phase boundary to guide parameter exploration faster.

## Setup

From the repo root:

```bash
pip install -r requirements.txt
```

Requires `scikit-learn` (listed in root `requirements.txt`).

## Run

From the repo root:

```bash
MPLBACKEND=Agg python surrogate/train.py
```

### CLI modes

| Command | Purpose |
|---------|---------|
| `python surrogate/train.py` | Load `surrogate/outputs/surrogate_dataset_default.csv` if present, else generate default grid |
| `python surrogate/train.py --quick` | 5x5 smoke test (~few minutes) |
| `python surrogate/train.py --accurate` | Heavier 7x7 grid, 2 seeds (~20-40 min) |
| `python surrogate/train.py --regenerate` | Force new simulation labels (default grid) |

`--quick` and `--accurate` are mutually exclusive. Each mode writes and loads its own dataset file so smoke tests do not overwrite default results.

## Outputs

Written to `surrogate/outputs/` (per mode, e.g. `_default`, `_quick`, `_accurate`):

| File pattern | Description |
|--------------|-------------|
| `surrogate_dataset_{mode}.csv` | Simulation labels `(R, v, eta_c)` |
| `surrogate_model_results_{mode}.txt` | Train/test MAE, best model, surrogate vs Problem 3 point |
| `surrogate_eta_c_heatmap_{mode}.png` | Predicted eta_c field + eta_c=0.5 contour |

## Models

- **Ridge** (scaled linear baseline)
- **RandomForestRegressor** (primary; `n_estimators=300`, `min_samples_leaf=2`)
- **GradientBoostingRegressor** (comparison)

Held-out MAE uses a single 75/25 split; labels are stochastic, so treat metrics as approximate.

## Framing

> Problem 3 required many stochastic simulations to estimate eta_c. I used those evaluations as labels and trained a random forest surrogate to approximate the phase boundary. The physics simulation remains authoritative; the surrogate speeds exploration.
