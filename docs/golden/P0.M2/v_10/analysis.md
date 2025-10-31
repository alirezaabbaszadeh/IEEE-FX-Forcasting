# v_10 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Residual Conv blocks]
    lstm[BiLSTM stack]
    aggregator[Hybrid MHA + AdditiveAttention + Mix-of-Experts]
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
- Combines strengths of MultiHeadAttention and AdditiveAttention with MoE to benchmark hybrid aggregator.
- Configuration exposes both attention mechanisms for tuning.

### Risks
- Complex aggregator chain increases risk of silent dtype mismatches under mixed precision.
- Validation split unchanged (96/2/2) leaving leakage detection difficult.

### Recommended Mitigations
- Add integration tests that run under mixed precision and float32 to compare outputs.

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
