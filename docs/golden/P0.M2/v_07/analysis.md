# v_07 – Data Flow & Leakage Review

## Data Flow Diagram

```mermaid
flowchart TD
    data[Input CSV]
    loader[DataLoader\nChronological split]
    sequencer[Sequenced partitions\n(train/val/test)]
    feature[Residual Conv+MHA blocks]
    lstm[BiLSTM stack]
    aggregator[Active Mix-of-Experts]
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
- Mix-of-Experts gating documented with explicit hyperparameters.
- Mermaid diagram emphasises gating path for review.

### Risks
- Mix-of-Experts gate learns from flattened sequence outputs—verify scaling to prevent exploding logits.
- No evaluation of expert-specialisation captured in history artefacts.

### Recommended Mitigations
- Log gate weights per epoch and monitor norm clipping requirements.

## Validation Notes
- Confirm split ratios and block configurations via the companion inventory checklist before reruns.
