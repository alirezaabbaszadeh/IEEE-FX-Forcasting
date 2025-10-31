# v_01 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Conv blocks]
    lstm[BiLSTM stack]
    aggregator[Single-head Attention]
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
- Chronological splits via ChronologicalDataLoader prevent future leakage.
- StandardScaler fits exclusively on training data before reuse for validation/test.

### Risks
- Validation/test ratios (3%) may be too small to detect drift.
- Block configuration defaults omit regularisation, increasing overfit risk.

### Recommended Mitigations
- Consider lowering train_ratio or enabling dropout/batchnorm toggles for robust evaluation.

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
