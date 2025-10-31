# v_06 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Residual Conv stack]
    lstm[BiLSTM stack]
    aggregator[Unused Mix-of-Experts hook]
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
- Maintains proven residual stack while preparing Mix-of-Experts interface without enabling it.
- Dependency bundle consistent with orchestrator expectations.

### Risks
- Unused Mix-of-Experts layer could diverge from production path if future hooks enable it without tests.
- Configuration still omits dropout/l2 for larger receptive fields.

### Recommended Mitigations
- Remove dead-code Mix-of-Experts or add smoke tests before activation.

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
