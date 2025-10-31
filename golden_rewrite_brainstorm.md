# Golden Rewrite Brainstorm & Breakdown

## Purpose
Clarify the investigative steps required before defining the definitive golden-standard roadmap.

## Current Understanding
- Experiments `v_01`–`v_10` implement differing combinations of data pipelines, feature sets, and model components.
- Reported “Model V8” performance is based on incomplete and inconsistent experimentation; best configuration remains ambiguous.
- Configuration drift, naming inconsistencies, and limited documentation prevent reuse and comparison.

## Brainstormed Goals
1. **Legacy Mapping** – Extract architecture, training, and evaluation flows for every version.
2. **Comparative Matrix** – Tabulate component combinations, dataset usage, and hyperparameters to explain performance variance.
3. **Reproducibility Baseline** – Record seeds, dependencies, data provenance, and evaluation splits currently used.
4. **Risk Register** – Capture sources of leakage, missing data handling, and statistical blind spots.
5. **Stakeholder Questions** – List open questions that must be answered before drafting implementation tasks.

## Breakdown of Next Steps
1. **Collect Artefacts**
   - Pull scripts, configs, and notebooks from each `v_*` directory.
   - Note undocumented utilities or external dependencies.
2. **Trace Data Flow**
   - Document where raw inputs originate, preprocessing applied, and how train/validation/test splits are produced.
   - Flag instances of potential look-ahead or inconsistent normalization.
3. **Catalog Model Components**
   - Identify shared layers, attention mechanisms, MoE setups, and alternative baselines.
   - Mark which versions support extension to transformers or classical statistical methods.
4. **Compare Training Protocols**
   - Align optimizer settings, schedules, loss functions, early-stopping logic, and evaluation metrics.
5. **Synthesize Findings**
   - Build a unified comparison table and narrative summary.
   - Derive mandatory requirements for the future golden repository from observed gaps.

## Deliverables from Investigation Phase
- Component comparison table (`docs/legacy_comparison.md`).
- Data and evaluation flow diagram.
- Reproducibility audit checklist highlighting missing items.
- List of unanswered questions requiring expert input.

## Exit Criteria for Planning
- Consensus on which legacy components to preserve, modify, or retire.
- Verified understanding of Model V8’s actual configuration and performance limitations.
- Prioritized backlog of refactoring tasks derived from documented evidence.
