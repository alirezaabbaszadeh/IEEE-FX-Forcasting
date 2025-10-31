# Agent Overview

This repository currently hosts ten legacy experiment variants (`v_01` … `v_10`). Before writing any new code, the agent must:

1. Map the structure, dependencies, and divergence between versions.
2. Document risks, missing provenance, and inconsistencies that block a golden-standard rewrite.
3. Produce stepwise plans only after the architecture and experiment flow are fully understood.

## Operating Principles
- Do **not** introduce new packages or pipelines until the existing design is catalogued.
- Capture findings directly inside the repository so future contributors can follow the investigative trail.
- Prefer documentation, diagrams, and analytical notes over implementation at this stage.
- Escalate unclear legacy behaviours for human review instead of guessing.

## Immediate Priorities
- Inventory data preprocessing, feature engineering, and training loops across all versions.
- Identify which combinations of components led to the strong-but-underdocumented “Model V8”.
- Surface blockers for reproducibility, evaluation integrity, and configuration alignment.
