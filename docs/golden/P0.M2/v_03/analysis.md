# v_03 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Residual Conv+MHA blocks]
    lstm[BiLSTM stack]
    aggregator[Post-LSTM Attention]
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
- Residual topology maintains chronological sequencing without cross-window mixing.
- Trainer callbacks mirror v_01 safeguards (early stopping, LR scheduling).

### Risks
- Skip connections with BatchNorm could reintroduce scale drift if statistics differ across splits.
- No guardrails for block depth—deep configs might exhaust validation coverage.

### Recommended Mitigations
- Record block_configs alongside run artefacts and cap residual depth per data volume.

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
