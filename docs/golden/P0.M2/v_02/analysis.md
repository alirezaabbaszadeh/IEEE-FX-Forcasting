# v_02 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Configurable Conv+MHA blocks]
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
- Mixed precision toggle captured in config to ensure deterministic arithmetic.
- Attention layers remain downstream of BiLSTM preserving temporal causality.

### Risks
- block_configs sourced externally—improper ordering could leak future context if pooling misconfigured.
- 0.02 validation slice is narrow for early stopping decisions.

### Recommended Mitigations
- Document canonical block_configs and enforce minimum validation horizon (>=5%).

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
