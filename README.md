# IEEE FX Forecasting — Legacy Consolidation Workspace

This repository is a staging ground for **merging ten legacy Forex forecasting experiments (`v_01`–`v_10`) into a single, modular research platform**. The goal is to eliminate duplicated code, capture undocumented behaviours, and provide an auditable foundation for future peer-review cycles.

## Current Objective

- **Unify historical projects:** Consolidate disparate pipelines, scripts, and utilities so that shared capabilities live behind a common interface instead of being re-implemented for every experiment.
- **Establish a modular architecture:** Define clear module boundaries (data ingestion, feature engineering, modelling, evaluation) that can be recomposed for new studies without destabilising existing results.
- **Support reviewer expectations:** Document provenance, configuration, and evaluation flows so that research outputs can be defended and extended when addressing reviewer feedback.

These objectives are derived from the legacy analysis captured in `agent.md`, `golden_rewrite_brainstorm.md`, and `golden_rewrite_execution_plan.md`.

## Repository Structure

```
v_01/ … v_10/      Legacy experiment variants awaiting consolidation
agent.md           Operating principles for catalogue-first refactoring
golden_*/          Brainstorm and execution plan for the golden rewrite
```

Each `v_*` directory contains an isolated snapshot of the historical pipeline. Part of the consolidation effort is to catalogue the differences across these directories before extracting reusable components.

## Workstream Highlights

The high-level execution roadmap lives in [`golden_rewrite_execution_plan.md`](golden_rewrite_execution_plan.md). Current focus areas include:

1. **Legacy Inventory:** Freeze the state of all versions, enumerate entry points, and capture dependency footprints.
2. **Experiment Flow Trace:** Diagram data preparation, training, and evaluation to surface inconsistencies and leakage risks.
3. **Requirements & Migration Strategy:** Translate reviewer-facing requirements into modular design tasks and sequence the extraction of shared infrastructure.

Progress artefacts will be published under `docs/golden/<milestone_id>/` as milestones move from planning to acceptance.

## Contributing

Contributions should prioritise documentation, analysis, and refactoring plans until the legacy landscape is fully mapped. When proposing changes:

- Reference the relevant milestone in `golden_rewrite_execution_plan.md`.
- Include evidence (notes, diagrams, tables) that explains how the change advances the consolidation goal.
- Avoid introducing new external dependencies until the unified architecture is baselined.

## License

This project is released under the [MIT License](LICENSE).
