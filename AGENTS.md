# Repository Collaboration Notes

**Scope:** Applies to the entire repository.

## Instructions
- This file captures the strategic plan and is for informational purposes only. It does **not** impose additional constraints on code style, tooling, or workflow beyond existing repository and project standards.

## Strategic Plan Overview

### Detailed, copy-ready task plan (with commands, file stubs, tests, CI)

Below extends the P0–P6 plan with **exact file trees, code stubs, commands, tests, and acceptance gates**. Keep PRs **small & vertical**; each PR must **run end-to-end** for its scope (wheel install + Kaggle smoke where relevant).

---

## P0 — Hydra / Packaging BLOCKERS

### P0-T1: Package the full Hydra config tree

**Create/ensure this tree (under source package, not repo root):**

```
src/leadlag/configs/
  config.yaml
  agent/
    ppo.yaml
    dqn.yaml
    a2c.yaml
    sac.yaml
    td3.yaml
  data/
    sp500_sector.yaml
    crypto_top.yaml
  features/
    base.yaml
    signature.yaml
    leadlag.yaml
    signature_leadlag.yaml
  split/
    walk_forward_purged.yaml
  training/
    base.yaml
    smoke.yaml
    paper.yaml
  hardware/
    gpu.yaml
    auto.yaml
  reporting/
    base.yaml           # if referenced
  rewards/
    default.yaml        # if referenced
  scenario/
    fixed_30.yaml
    fixed_90.yaml
    dynamic_adaptive.yaml
    rl_ppo.yaml
```

**Minimal `config.yaml` example:**

```yaml
# src/leadlag/configs/config.yaml
defaults:
  - agent: ppo
  - data: sp500_sector
  - features: base
  - split: walk_forward_purged
  - training: smoke
  - hardware: gpu
  - _self_
env:
  action_space: discrete3
policy:
  name: mlp
window:
  lookback: 128
target:
  horizon: 1
costs:
  fee_bps: 1
slippage:
  bps: 2
logging:
  run_id: ${now:%Y%m%d}-${agent.name}-${hydra.job.num}
```

**Acceptance**

* `defaults` compose without errors.
* All group files exist and load with overrides (see P0-T5 tests).

---

### P0-T2: Ensure YAMLs ship in the wheel

**`pyproject.toml` (setuptools example):**

```toml
[build-system]
requires = ["setuptools>=68", "wheel", "build"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
leadlag = ["configs/**/*.yaml"]

[tool.setuptools.packages.find]
where = ["src"]
```

**Optional `MANIFEST.in` (affects sdist):**

```
recursive-include src/leadlag/configs *.yaml
```

**Local check**

```bash
python -m build
pip install -U dist/*.whl
python - <<'PY'
import importlib.resources as ir
p = ir.files('leadlag').joinpath('configs')
ys = list(p.rglob('*.yaml'))
print('yaml_count', len(ys))
assert len(ys) >= 10, "YAMLs not packaged"
PY
```

---

### P0-T3: Single Hydra entry point

**File**

```python
# src/leadlag/pipelines/run_full_suite.py
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("Config sources:", [s.provider for s in HydraConfig.get().runtime.config_sources])
    # TODO: orchestrate data → features → train → evaluate → write metrics/artifacts
    # IMPORTANT: write outputs to ${hydra:run.dir}
    # return or raise on failure

if __name__ == "__main__":
    main()
```

**Acceptance**

```bash
python -m leadlag.pipelines.run_full_suite --cfg job
```

* Starts without ConfigNotFound/Composition errors.
* Prints composed defaults.

---

### P0-T4: Remove stale `configs/` references

**Command**

```bash
rg -n "configs/" -S | rg -v "src/leadlag/configs"
```

* Replace repo-root references with packaged path or relative to module.
* Update any `initialize_config_dir` calls to use `src/leadlag/configs`.

**Acceptance**: ripgrep returns **0** stale hits; tests pass.

---

### P0-T5: Compose & wheel smoke tests

**Test file**

```python
# tests/test_hydra_compose.py
import pytest
pytest.importorskip("hydra")
from hydra import initialize, compose

def test_defaults_compose():
    with initialize(version_base=None, config_path="../src/leadlag/configs"):
        cfg = compose(config_name="config")
    assert cfg.agent is not None

def test_common_overrides():
    with initialize(version_base=None, config_path="../src/leadlag/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "agent=ppo","training=smoke","hardware=gpu",
                "data=sp500_sector","split=walk_forward_purged",
                "features=signature_leadlag"
            ]
        )
    assert cfg.features is not None
```

**CI job (snippet)**
`.github/workflows/ci.yml`:

```yaml
- name: Build wheel & smoke
  run: |
    python -m build
    pip install -U dist/*.whl
    python -m pytest -q tests/test_hydra_compose.py
    python -m leadlag.pipelines.run_full_suite --cfg job
```

**Acceptance**: CI green; entry point runs from wheel.

---

## P1 — Leakage-free Splits (Nested-ready)

### P1-T1: Purged/Embargoed module

```python
# src/leadlag/cv/purged.py
from dataclasses import dataclass
import numpy as np
from typing import Iterator, Tuple

@dataclass
class WalkForwardPurged:
    n_splits: int = 6
    embargo_frac: float = 0.01

    def split(self, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        fold = n // self.n_splits
        emb = max(0, int(self.embargo_frac * n))
        for k in range(self.n_splits):
            t0 = k * fold
            t1 = n if k == self.n_splits - 1 else (k + 1) * fold
            mask = np.ones(n, dtype=bool)
            mask[max(0, t0 - emb):min(n, t1 + emb)] = False
            train_idx = np.where(mask)[0]
            test_idx = np.arange(t0, t1)
            yield train_idx, test_idx

@dataclass
class PurgedKFold:
    n_splits: int = 3
    embargo_frac: float = 0.01
    # similar logic for inner folds
```

### P1-T2: Hydra profile & export

```yaml
# src/leadlag/configs/split/walk_forward_purged.yaml
scheme: walk_forward_purged
n_splits: 6
embargo_frac: 0.01
nested_tuning:
  enabled: true
  inner_n_splits: 3
  inner_embargo_frac: 0.01
```

**Pipeline export**: write `results/<run_id>/splits.csv` with columns:
`window, train_start, train_end, test_start, test_end, embargo_frac`.

### P1-T3: Tests

```python
# tests/test_purged_cv.py
import numpy as np
from leadlag.cv.purged import WalkForwardPurged

def test_no_overlap_with_embargo():
    N=1000; cv=WalkForwardPurged(5, 0.02)
    for tr, te in cv.split(N):
        assert not np.intersect1d(tr, te).size
```

**Acceptance**: tests pass; `splits.csv` produced per run.

---

## P2 — Trading Realism (t→t+1, costs, constraints)

### P2-T1: Enforce t→t+1 execution

* In `env/trading_env.py` use **next bar open** for fills.
* Log both **signal_time** and **exec_time**.

### P2-T2: Costs & slippage

```python
commission = abs(delta_pos) * exec_price * cfg.costs.fee_bps * 1e-4
slip       = abs(delta_pos) * exec_price * cfg.slippage.bps * 1e-4
pnl        = prev_pos * (next_close - curr_close) - commission - slip
```

* Presets: 0/1/5 bps via Hydra.
* Aggregate per window.

### P2-T3: Constraints & metrics

* Config: `env.leverage_cap`, `env.allow_short`.
* Track `Turnover = sum(|Δposition|)`, `Exposure = mean(|position|)`.
* Persist in metrics rows.

**Tests**

* Toy uptrend: higher bps → lower net PnL.
* Verify fill uses `next_open` (not same-bar).

**Acceptance**: metrics include `Turnover/Exposure`; synthetic tests pass.

---

## P3 — Reporting Unification

### P3-T1: Canonical `metrics.csv` writer

```python
# src/leadlag/reporting/metrics_writer.py
from pathlib import Path
import pandas as pd

SCHEMA = [
 "experiment_id","agent","action_space","policy",
 "features_signature","signature_depth","features_leadlag","time_channel",
 "lookback","horizon","universe","timeframe","split_scheme",
 "cost_fee_bps","slippage_bps","reward","seed","window_index",
 "Sharpe","Sortino","MaxDD","Turnover","PnL","Exposure"
]

def write_metrics(out_dir, rows):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in SCHEMA:
        if c not in df: df[c] = None
    df = df[SCHEMA]
    df.to_csv(out / "metrics.csv", index=False)
    return df
```

### P3-T2: Schema validation (guard)

```python
# tests/test_metrics_schema.py
import pandas as pd
from leadlag.reporting.metrics_writer import SCHEMA
def test_schema_columns(tmp_path):
    p = tmp_path/"metrics.csv"
    pd.DataFrame([{c: None for c in SCHEMA}]).to_csv(p, index=False)
    df = pd.read_csv(p)
    assert list(df.columns) == SCHEMA
```

### P3-T3: Aggregator

```python
# scripts/aggregate_metrics.py (or notebook cell)
import pandas as pd, glob, os
rows=[]
for m in glob.glob("/kaggle/working/results/*/metrics.csv"):
    df = pd.read_csv(m); df["run_dir"]=os.path.dirname(m); rows.append(df)
pd.concat(rows, ignore_index=True).to_csv("/kaggle/working/paper_outputs/all_metrics_raw.csv", index=False)
```

**Acceptance**: every run emits valid `metrics.csv`; aggregation produces `all_metrics_raw.csv`.

---

## P4 — Fairness & Reproducibility

### P4-T1: Seed utility

```python
# src/leadlag/utils/repro.py
import os, random, numpy as np
def set_all_seeds(seed=0, cudnn_deterministic=True):
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = not cudnn_deterministic
    except Exception:
        pass
```

### P4-T2: Equal env-steps & manifest

* Enforce `training.total_env_steps` for all agents.
* Write `run_manifest.json`:

```python
# src/leadlag/utils/manifest.py
import json, sys, subprocess
def write_manifest(path, seed, env_steps_reported, env_steps_actual, extra=None):
    commit = subprocess.getoutput("git rev-parse --short HEAD") or "unknown"
    info = dict(
      commit=commit, seed=seed,
      env_steps_reported=env_steps_reported, env_steps_actual=env_steps_actual,
      python=sys.version
    )
    if extra: info.update(extra)
    with open(path, "w") as f: json.dump(info, f, indent=2)
```

### P4-T3: Hydra profiles

```yaml
# src/leadlag/configs/hardware/gpu.yaml
device: cuda
mixed_precision: amp
n_envs: 8
num_workers: 4
pin_memory: true
```

```yaml
# src/leadlag/configs/training/smoke.yaml
total_env_steps: 100000
seeds: [0,1]
windows: 2
# src/leadlag/configs/training/paper.yaml
total_env_steps: 500000
seeds: [0,1,2,3,4,5,6]
windows: 6
```

**Acceptance**: logs show identical budgets; manifest saved per run.

---

## P5 — Stats + Calibration + Multi-seed

### P5-T1: Stats library & CLI

**Library outline**

```python
# src/leadlag/eval/stats.py
import numpy as np, pandas as pd
def sharpe(returns, rf=0.0): ...
def sortino(returns, rf=0.0): ...
def max_drawdown(equity): ...
def hac_confint_sharpe(returns, lags=None): ...
def psr_dsr(returns): ...
def spa_reality_check(table_of_utilities, benchmark_col): ...
def mcs(models_metrics_df, alpha=0.1): ...
```

**CLI**

```python
# src/leadlag/eval/stats_cli.py
# Reads /results/*/metrics.csv → writes to /paper_outputs/
# Outputs: hac_sharpe_ci.csv, psr_dsr_pvalues.csv, spa_table.csv, mcs_table.csv,
# figures: forest.png, heatmap.png, pnl.png; paper_results.md (summary)
```

**Run**

```bash
python -m leadlag.eval.stats_cli \
  --results /kaggle/working/results \
  --out /kaggle/working/paper_outputs \
  --benchmark "Sharpe" --alpha 0.05
```

### P5-T2: Calibration CLI

```python
# src/leadlag/eval/calibration_cli.py
# Computes CRPS, pinball losses, PIT histogram, PI coverage → paper_outputs/
```

Outputs:

* `calibration.csv` (CRPS, pinball@{0.1,0.5,0.9}, coverage@{80,90,95})
* `pit_hist.png`, `coverage_plot.png`

### P5-T3: Multi-seed aggregation

* Orchestrate N seeds; write `/results/*/metrics.csv` per seed.
* Build `/paper_outputs/aggregate.csv` with mean/std/95% CI per metric/agent.

**Acceptance**: CSVs & figures appear; `paper_results.md` summarizes tables.

---

## P6 — One-click reproduce & anonymous packaging

### P6-T1: Reproduce-all

```bash
# scripts/reproduce_all.sh
set -euo pipefail
ENTRY="python -m leadlag.pipelines.run_full_suite"
OUT="/kaggle/working/paper_outputs"
RES="/kaggle/working/results"
$ENTRY training=paper hardware=gpu split=walk_forward_purged
python -m leadlag.eval.stats_cli --results "$RES" --out "$OUT" --benchmark Sharpe
python -m leadlag.eval.calibration_cli --results "$RES" --out "$OUT"
python scripts/aggregate_metrics.py
```

### P6-T2: Pack OpenReview (anonymous)

```bash
# scripts/pack_openreview.sh
set -euo pipefail
ZIP="artifact_anonymous.zip"
ROOT="$(git rev-parse --show-toplevel)"
find "$ROOT" -name ".git" -prune -o -print | grep -vE "(names|emails|affiliations)" >/dev/null
# Copy minimal code + configs + scripts + README_ANON.md
# Scrub file metadata if needed (touch/zip -X)
zip -Xr "$ZIP" src/leadlag configs scripts README_ANON.md LICENSE
# Ensure size <= 100MB
```

**`README_ANON.md`**: concise run steps (no names/URLs).

### P6-T3: Kaggle anonymous path

* Notebook cells:

  1. Upload `artifact_anonymous.zip` to Dataset → mount in notebook.
  2. `!pip -q install /kaggle/input/<anon-dataset>/artifact_anonymous.zip` or unzip & `pip install -e .`
  3. `!bash scripts/reproduce_all.sh`
* Outputs in `/kaggle/working/paper_outputs/`.

**Acceptance**: single click regenerates all artifacts in a clean Kaggle session.

---

## Optional+: Baselines & HTML Report

### Baselines

* `src/leadlag/models/classical/arima.py`, `ets.py`; minimal light-deep baseline.
* Hydra presets under `configs/agent/`.
* Produce `paper_outputs/main_results.csv`.

### HTML report

* `src/leadlag/reporting/html_report.py` → `paper_outputs/report.html` combining:

  * Main results, ablations, HAC CIs, PSR/DSR, SPA/MCS, calibration, equity curves.

---

## Global PR checklist (paste into each PR description)

* [ ] Runs from **wheel**: `python -m build && pip install dist/*.whl`
* [ ] **Entry point** runs: `python -m leadlag.pipelines.run_full_suite --cfg job`
* [ ] **Tests** added/updated (unit + any synthetic guards)
* [ ] **Artifacts** written to `${hydra:run.dir}` and/or `/kaggle/working/...`
* [ ] **Docs** updated (README/CONFIG ref)
* [ ] **No stale refs** to repo-root `configs/` (`rg "configs/"` clean)

---

## Risk guards to add to CI (lightweight)

* **Leakage guard**: assert zero overlap (P1 tests).
* **Budget guard**: assert equal `total_env_steps` per agent.
* **Latency/cost guard**: synthetic case affected by bps & next-open fills.
* **Schema guard**: `metrics.csv` columns match exactly.
* **Wheel guard**: build→install→run entry point.

---

## Minimal Make targets (optional)

```makefile
build:
python -m build

wheel-smoke: build
pip install -U dist/*.whl
python -m leadlag.pipelines.run_full_suite --cfg job

test:
pytest -q

kaggle-smoke:
python -m leadlag.eval.stats_cli --results /kaggle/working/results --out /kaggle/working/paper_outputs
```

---

### Definition of “Refactor Success” (final gate)

* Entry point OK from **wheel** and **Kaggle** (no Hydra errors).
* `splits.csv` valid; **t→t+1** and **costs/slippage** applied; `Turnover/Exposure` logged.
* Canonical `metrics.csv` for every run; aggregation OK.
* Stats & Calibration CLIs produce required CSVs/figures/summary MD.
* Multi-seed `aggregate.csv`; complete `run_manifest.json`.
* `reproduce_all.sh` + `artifact_anonymous.zip` regenerate full paper artifacts deterministically.

