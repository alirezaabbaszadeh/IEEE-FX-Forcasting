# v_05 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Residual Conv+MHA blocks]
    lstm[BiLSTM stack]
    aggregator[Post-LSTM MultiHeadAttention]
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
- Optional dense bridge before output encourages separation of sequence summarisation and regression head.
- Mix-of-Experts scaffolding present for experimentation but off by default, limiting surprise coupling.

### Risks
- MultiHeadAttention head after BiLSTM may attend across validation timesteps if stateful training is enabled (currently false but undocumented).
- Dropout remains disabled despite increased model capacity.

### Recommended Mitigations
- Explicitly document assumption about stateful=False training and evaluate dropout>0 runs.

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
