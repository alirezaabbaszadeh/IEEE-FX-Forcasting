# PCC Claim Freeze Summary

## Validated run configuration
The claim freeze locks the temporal transformer reference stack to the default hydra
configuration committed with the benchmark. That manifest establishes a reproducible seed,
chronological walk-forward windows, and data preparation settings that eliminate leakage for the
EURUSD and GBPUSD pairs across three forecast horizons.【F:configs/default.yaml†L1-L38】 The same
profile constrains the model to two residual blocks with four-head attention, a 128-dimensional
convolutional stem, and an LSTM bottleneck widened to 256 hidden units with 0.1 dropout to stabilise
probabilistic training.【F:configs/default.yaml†L40-L55】 Multirun seeds and governance caps (12 epochs
for the temporal transformer and a 32-trial HPO ceiling) remain part of the locked state so audit
replays can match validation runs byte-for-byte.【F:configs/default.yaml†L57-L77】 The reproducibility
manifest at `configs/governance/claim_freeze.yaml` records the freeze timestamp and references the
calibration profile to rehydrate the PCC configuration during evaluation.【F:configs/governance/claim_freeze.yaml†L1-L7】

## Calibration hyperparameters
Purged conformal calibration is frozen to the `configs/inference/pcc.yaml` profile. The alpha level of
0.1 targets 90% prediction intervals, while embargo=2 preserves the temporal gap between calibration
and evaluation windows to avoid residual leakage.【F:configs/inference/pcc.yaml†L1-L7】 Calibration only
consumes validation splits, requires at least 30 chronologically ordered samples, and reuses past
windows when available so the same residual pool is presented whenever the claim is revalidated.

## Improvement thresholds and audits
The research success criteria mandate at least a 2% average CRPS improvement or a 2–3% reduction in
coverage error across half of the evaluated pair–horizon cells before a claim can be published.【F:docs/research_plan.md†L37-L45】
PCC ablations satisfy that requirement by logging embargo and weighting variants into manifests that
highlight any cell falling short of the ≥2% improvement bar, ensuring the freeze retains only those
settings that clear the gate.【F:paper_outputs/ablations/pcc_ablation.md†L1-L18】 Aggregated outputs include
CRPS and coverage deltas for every weighting option, so reviewers can confirm the exponential decay
configuration retained for the freeze outperforms the uniform baseline by the mandated margin without
rerunning the sweep.【F:paper_outputs/ablations/pcc_ablation.md†L12-L18】
