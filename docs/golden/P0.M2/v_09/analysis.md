# v_09 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Residual Conv blocks]
    lstm[BiLSTM stack]
    aggregator[AdditiveAttention + Mix-of-Experts]
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
- AdditiveAttention emphasises interpretability by exposing attention weights.
- MoE leaky ReLU slope configurable per default_config.

### Risks
- Default_config keeps dropout at zero despite AdditiveAttention potentially overfitting.
- No documentation on how AdditiveAttention key/value sizes align with LSTM outputs.

### Recommended Mitigations
- Capture attention weight summaries in HistoryManager outputs.

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
