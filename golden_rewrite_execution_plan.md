# Golden Rewrite Execution Plan

## Overview
This document translates the brainstormed objectives into an actionable, trackable execution roadmap. It is intended to be maintained alongside `agent.md` and `golden_rewrite_brainstorm.md` so that every contributor can see the current stage, responsibilities, and acceptance criteria.

- **Governance Rhythm:** Weekly steering review with the Technical Lead (TL), Research Lead (RL), and Quality & Compliance Lead (QL). Daily async status updates in the project tracker referencing milestone IDs below.
- **Milestone States:** `Planned → In Progress → Under Review → Accepted`. Progression requires meeting every acceptance criterion and capturing evidence in the repository (`docs/` or `reports/`).
- **Evidence Indexing:** Each deliverable must be stored under `docs/golden/<milestone_id>/` with a `metadata.json` capturing author, date, inputs, and associated git commit.

## Roles & Responsibilities
| Role | Primary Owner | Core Responsibilities |
| --- | --- | --- |
| Technical Lead (TL) | TBD | Architecture decisions, ensures compatibility with future transformer/legacy models, signs off on automation readiness. |
| Research Lead (RL) | TBD | Validates experimental design, statistical testing, and ensures ablation coverage. |
| Data Governance Lead (DL) | TBD | Oversees data provenance, licensing, preprocessing reproducibility, and embargo enforcement. |
| Quality & Compliance Lead (QL) | TBD | Maintains reproducibility, CI, documentation quality, and audit trails. |
| Infrastructure Engineer (IE) | TBD | Manages compute benchmarking scripts, environment locking, and workflow automation. |
| Interpretability Specialist (IS) | TBD | Designs attention/MoE analysis, sanity checks, and reporting templates. |

Each milestone below names the accountable role (A) and supporting roles (S). Owners must be assigned before the milestone can move to `In Progress`.

## Phase 0 — Legacy Comprehension (Week 0–1)
### Milestone P0.M1 — Legacy Inventory Freeze
- **A:** TL & RL
- **S:** DL
- **Inputs:** `v_01`–`v_10` directories, existing notes.
- **Tasks:**
  1. Snapshot the current legacy directories and record checksums.
  2. Build a table of scripts/notebooks per version and their entry points.
  3. Document external dependencies (packages, data sources, credentials).
- **Deliverables:** `docs/golden/P0.M1/legacy_inventory.md`, `docs/golden/P0.M1/checksums.csv`.
- **Acceptance Criteria:**
  - All version directories have documented entry points and dependency notes.
  - Checksums verified via reproducible command listed in `metadata.json`.
- **Target Duration:** 4 working days.

### Milestone P0.M2 — Experiment Flow Trace
- **A:** DL
- **S:** TL, RL
- **Tasks:**
  1. Diagram data flow from raw source to evaluation metrics for each version.
  2. Identify normalization, leakage risks, and split strategies.
  3. Log unanswered questions requiring SME review.
- **Deliverables:** `docs/golden/P0.M2/data_flow_diagrams/`, `open_questions.md`.
- **Acceptance Criteria:**
  - Every version has a data flow diagram with train/val/test boundaries.
  - Leakage risks annotated with severity and mitigation ideas.
  - Questions triaged with owners and due dates.
- **Target Duration:** 5 working days (overlaps last 2 days of P0.M1).

### Exit Gate P0
- Milestones P0.M1 and P0.M2 accepted.
- Kick-off review approving scope for Phase 1.

## Phase 1 — Planning & Alignment (Week 2)
### Milestone P1.M1 — Requirements Consolidation
- **A:** RL
- **S:** TL, QL
- **Tasks:**
  1. Map “Golden Standard” requirements to specific repository artifacts.
  2. Define quantitative acceptance thresholds (e.g., CI width, reproducibility tolerance).
  3. Produce a RACI chart for later phases.
- **Deliverables:** `docs/golden/P1.M1/requirements_matrix.xlsx`, `docs/golden/P1.M1/raci.md`.
- **Acceptance Criteria:**
  - Every requirement has a mapped artifact and measurable acceptance criterion.
  - RACI chart covers all roles and upcoming milestones.
- **Target Duration:** 3 working days.

### Milestone P1.M2 — Migration Strategy
- **A:** TL
- **S:** IE, RL
- **Tasks:**
  1. Define how legacy components migrate to the new modular architecture.
  2. Sequence component extraction (data loaders, models, evaluation) into work packages.
  3. Specify deprecation plan for non-viable variants.
- **Deliverables:** `docs/golden/P1.M2/migration_plan.md`, `work_packages.csv`.
- **Acceptance Criteria:**
  - Work packages include scope, prerequisites, and estimated effort.
  - Migration plan aligns with requirements matrix and highlights high-risk transitions.
- **Target Duration:** 4 working days.

### Exit Gate P1
- Requirements matrix and migration plan approved by steering review.
- Owners assigned for all Phase 2 milestones.

## Phase 2 — Foundational Infrastructure (Weeks 3–5)
### Milestone P2.M1 — Configuration & Reproducibility Backbone
- **A:** IE
- **S:** TL, QL
- **Tasks:**
  1. Design single-source configuration schema (YAML + validation).
  2. Implement seed control utilities covering Python, NumPy, framework, CUDA.
  3. Draft reproducibility checklist and automated verification script.
- **Deliverables:** Schema draft (`docs/golden/P2.M1/config_schema.yaml`), validation plan, reproducibility script prototype.
- **Acceptance Criteria:**
  - Schema reviewed by TL and RL; validation rules documented.
  - Prototype script reproduces RNG setup on a sample notebook.
  - Checklist signed off by QL.
- **Target Duration:** 1.5 weeks.

### Milestone P2.M2 — Data Loader Framework
- **A:** DL
- **S:** IE
- **Tasks:**
  1. Specify API for multi-pair, multi-horizon, walk-forward loader.
  2. Document timezone normalization, embargo logic, and missing data handling.
  3. Produce test plan covering leakage and DST edge cases.
- **Deliverables:** `docs/golden/P2.M2/data_loader_spec.md`, `test_plan.md`.
- **Acceptance Criteria:**
  - Spec reviewed by TL and RL.
  - Test plan includes automated and manual checks with acceptance thresholds.
- **Target Duration:** 1.5 weeks (parallel with P2.M1).

### Exit Gate P2
- Configuration schema, reproducibility script prototype, and data loader spec approved.
- Steering review authorizes development of execution engines.

## Phase 3 — Experiment Engine & Evaluation (Weeks 6–8)
### Milestone P3.M1 — Multi-Run Orchestrator Design
- **A:** TL
- **S:** IE, RL
- **Tasks:**
  1. Define orchestrator architecture (scheduler, run tracking, metadata output).
  2. Outline aggregation logic for mean/SD/CI and bootstrap workflows.
  3. Prepare interface specification for plugging in models and datasets.
- **Deliverables:** `docs/golden/P3.M1/orchestrator_design.md`, interface diagrams.
- **Acceptance Criteria:**
  - Supports ≥5-run repetitions with deterministic seed assignment.
  - Metadata spec includes git commit, hardware, driver versions.
- **Target Duration:** 2 weeks.

### Milestone P3.M2 — Evaluation & Statistical Testing Protocols
- **A:** RL
- **S:** QL
- **Tasks:**
  1. Draft statistical testing blueprint (ANOVA/Tukey, Diebold-Mariano, etc.).
  2. Define calibration, uncertainty, and effect size reporting templates.
  3. Document decision rules for significance, multiple comparisons, and power analysis.
- **Deliverables:** `docs/golden/P3.M2/statistical_protocol.md`, reporting templates.
- **Acceptance Criteria:**
  - Protocol reviewed by external statistical advisor (recorded in metadata).
  - Templates include guidance for macro/micro averages and stratified summaries.
- **Target Duration:** 2 weeks (aligned with P3.M1).

### Exit Gate P3
- Orchestrator and statistical protocols baselined.
- Implementation tickets spawned with references to design docs.

## Phase 4 — Interpretability & Benchmarking (Weeks 9–11)
### Milestone P4.M1 — Interpretability Framework Specification
- **A:** IS
- **S:** TL, RL
- **Tasks:**
  1. Describe pipelines for attention heatmaps, MoE gating analysis, and IG comparisons.
  2. Define sanity checks and perturbation tests.
  3. Outline storage format for interpretability artifacts.
- **Deliverables:** `docs/golden/P4.M1/interpretability_spec.md`, sanity check matrix.
- **Acceptance Criteria:**
  - Supports reproducible extraction with fixed seeds.
  - Artifact metadata schema defined (e.g., gating entropy summaries).
- **Target Duration:** 1.5 weeks.

### Milestone P4.M2 — Compute Benchmark Blueprint
- **A:** IE
- **S:** TL, QL
- **Tasks:**
  1. Define benchmarking methodology (warm-up policy, measurement counts).
  2. Specify hardware baseline, precision modes, and reporting fields.
  3. Plan automation for collecting latency/throughput statistics.
- **Deliverables:** `docs/golden/P4.M2/benchmark_plan.md`, measurement templates.
- **Acceptance Criteria:**
  - Benchmark plan reviewed by TL and RL for fairness across models.
  - Reporting template includes p50/p90/p99, mean, stdev, and energy estimates.
- **Target Duration:** 1.5 weeks.

### Exit Gate P4
- Interpretability and benchmarking specs accepted.
- Tooling implementation scheduled for Phase 5.

## Phase 5 — Implementation & Integration (Weeks 12–18)
This phase converts specifications into code; actual implementation tasks will be tracked in separate tickets but must adhere to the design docs above.

### Milestone P5.M1 — Core Infrastructure Implementation
- **A:** TL
- **S:** IE, DL
- **Tasks:**
  1. Implement configuration system, seed utilities, and reproducibility scripts.
  2. Build data loader modules per spec, including tests.
  3. Establish CI pipelines for linting, unit tests, and smoke training.
- **Deliverables:** Code modules, CI configs, updated docs.
- **Acceptance Criteria:**
  - CI passes on minimal fixture dataset.
  - Reproducibility script validated against baseline experiment.
  - Documentation updated with usage instructions.
- **Target Duration:** 3 weeks.

### Milestone P5.M2 — Experiment Engine & Evaluation Implementation
- **A:** RL
- **S:** TL, IE, QL
- **Tasks:**
  1. Implement multi-run orchestrator, logging, and aggregation.
  2. Integrate statistical testing workflows and reporting templates.
  3. Produce sample runs demonstrating macro/micro averages and stratified analyses.
- **Deliverables:** Orchestrator code, statistical notebooks, example reports.
- **Acceptance Criteria:**
  - Example experiment reproduces legacy V8 components with ≥5 runs.
  - Confidence intervals and effect sizes computed automatically.
  - Reports stored per metadata standard.
- **Target Duration:** 3 weeks (overlaps last week of P5.M1).

### Milestone P5.M3 — Interpretability & Benchmark Tooling Implementation
- **A:** IS
- **S:** IE, TL
- **Tasks:**
  1. Implement attention/MoE visualization modules and sanity checks.
  2. Deliver compute benchmarking scripts and automation.
  3. Validate outputs on at least one legacy configuration.
- **Deliverables:** Tooling modules, sample artifacts.
- **Acceptance Criteria:**
  - Interpretability outputs include metadata and pass perturbation checks.
  - Benchmark results report mean, stdev, p50/p90/p99 latencies.
- **Target Duration:** 2 weeks.

### Exit Gate P5
- All core tooling integrated.
- Sign-off meeting confirming readiness for full experimental campaign.

## Phase 6 — Experimental Campaign & Publication (Weeks 19–24)
### Milestone P6.M1 — Legacy Model Migration & Validation
- **A:** TL
- **S:** RL, DL
- **Tasks:**
  1. Port V1–V10 configurations into new system.
  2. Run reproducibility checks and align results versus historical records.
  3. Document deviations and remediation plans.
- **Deliverables:** Config files, comparison tables, validation report.
- **Acceptance Criteria:**
  - Each legacy model has a documented config and validation outcome.
  - Deviations categorized with root-cause analysis.
- **Target Duration:** 3 weeks.

### Milestone P6.M2 — Golden Experiment Suite Execution
- **A:** RL
- **S:** TL, IE, QL, IS
- **Tasks:**
  1. Execute full multi-run, multi-pair, multi-horizon campaign per requirements.
  2. Collect statistical tests, calibration metrics, interpretability artifacts, and benchmarks.
  3. Compile economic realism analyses and uncertainty reports.
- **Deliverables:** `reports/golden_suite/`, summary manuscript appendices.
- **Acceptance Criteria:**
  - All required metrics logged with metadata and CI coverage.
  - Statistical significance documented with effect sizes and multiple-comparison controls.
  - Interpretability and benchmarking outputs archived.
- **Target Duration:** 4 weeks.

### Milestone P6.M3 — Publication Package & Release
- **A:** QL
- **S:** TL, RL, DL
- **Tasks:**
  1. Finalize documentation (README, Model Card, CITATION.cff, Reproducibility Checklist).
  2. Prepare release notes, changelog, and tag (`v1.0-golden`).
  3. Assemble artifact bundle with persistent identifier plan.
- **Deliverables:** Docs, release tag, archival bundle manifest.
- **Acceptance Criteria:**
  - External reviewer checklist signed off.
  - One-command reproduction script validated end-to-end.
  - Release candidate approved by steering review.
- **Target Duration:** 2 weeks.

### Exit Gate P6
- Golden-standard repository ready for external dissemination.
- Post-mortem logged capturing lessons learned and future extensions.

## Tracking & Updates
- Update this document at the end of each steering review with status notes.
- Use milestone IDs (e.g., `P3.M2`) in commit messages, pull requests, and issue trackers.
- Archive superseded versions under `docs/golden/_archive/` with timestamps.

