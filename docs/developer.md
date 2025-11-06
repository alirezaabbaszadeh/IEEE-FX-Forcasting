# Developer guide

## Leak guard remediation

The walk-forward dataset builder now validates that each feature window stays at least one
forecasting horizon ahead of its target timestamp. When this guard raises a `ValueError` during
`prepare_datasets`, use the following checklist to remediate the issue:

- Inspect the offending split named in the error to confirm whether raw timestamps bunch up within
  the forecast horizon (for example, duplicate ticks or mismatched resampling).
- Realign feature columns with the correct timestamps before exporting CSVs. Avoid shifting features
  forward in time or mixing measurements from multiple clocks.
- Remove or aggregate irregular observations that compress the gap below the configured horizon.
- If the data is legitimately denser than the requested horizon, update the forecasting horizon or
  the model lookback so that each target remains strictly after the final feature timestamp.

Run `pytest tests/test_leak.py` locally to reproduce the failure after applying fixes. The leak guard
is enforced automatically by the CI workflows, so merges will be blocked until the test passes.
