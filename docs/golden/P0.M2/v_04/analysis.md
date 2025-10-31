# v_04 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Residual Conv+MHA blocks]
    lstm[BiLSTM stack]
    aggregator[AdditiveAttention aggregator]
    head[Dense regression head]
    trainer[Trainer callbacks]
    evaluator[Evaluator & HistoryManager]
    data --> loader
    loader --> sequencer
    sequencer --> feature
    feature --> lstm
    lstm --> aggregator
    aggregator --> head
    head --> trainer
    trainer --> evaluator
```

## Leakage Assessment

### Strengths
- AdditiveAttention aggregator keeps temporal weighting explicit and auditable.
- Dependency bundle isolates version-specific hooks, reducing accidental cross-version reuse.

### Risks
- AdditiveAttention parameters not exposed in config, obstructing reproducibility if defaults change.
- Train/val split identical to v_03 retains narrow validation coverage.

### Recommended Mitigations
- Surface AdditiveAttention key dimensions in configuration and widen validation split during reruns.

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
