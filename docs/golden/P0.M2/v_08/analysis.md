# v_08 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Residual Conv+MHA blocks]
    lstm[BiLSTM stack]
    aggregator[MultiHeadAttention + Mix-of-Experts]
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
- Validation patience tightened (20 epochs) reducing prolonged training on stale improvements.
- Explicit configuration for attention heads and MoE units improves reproducibility.

### Risks
- Train/val split remains 96/2/2 causing highly variable validation metrics.
- No explicit seed recorded for NumPy/pandas ingestion in Run.py.

### Recommended Mitigations
- Record dataset index of split boundaries per run; extend val ratio to >=0.05 for evaluation studies.

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
