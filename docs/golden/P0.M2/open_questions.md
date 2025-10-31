# Open Questions

- What canonical `block_configs` were used in historical reruns of v_02–v_10, and where are they stored for reproducibility?
- Did any production training enable stateful RNN execution, which would alter the attention context described in v_05–v_10 diagrams?
- How were validation windows selected when only 2–3% of data was reserved—were rolling evaluations performed outside these scripts?
- Are Mix-of-Experts gate weights or attention heatmaps archived anywhere for auditability (Trainer/Evaluator do not log them by default)?
- Which exact seeds (NumPy, pandas, TensorFlow) were injected during the acclaimed v_08 experiments beyond the TensorFlow seed in each module?
