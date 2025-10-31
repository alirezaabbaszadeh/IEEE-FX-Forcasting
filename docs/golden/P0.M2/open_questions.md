# P0.M2 Open Questions and Missing Artefacts

## Outstanding Data & Asset Gaps
- **Where should the canonical CSV datasets live?** Every `Run.py` defaults to `csv/EURUSD_Candlestick_1_Hour_BID_01.01.2010-18.02.2025.csv`, yet none of the version folders contain a `csv/` directory. The current repository state cannot execute the pipelines without external guidance on sourcing or regenerating these files.【F:v_08/Run.py†L40-L92】【d873a1†L1-L4】

## Architecture Clarifications Needed
- **Are the placeholder residual `block_configs` exhaustive?** Versions 3–7 ship a single default residual block in their `Run.py` files and comments invite adding more entries manually. We need confirmation of the original block depth for each experiment to avoid regressing the intended architectures.【F:v_07/Run.py†L150-L196】
- **Should Version 6 activate the Mix-of-Experts layer?** `ModelBuilder` V6 declares the `MixOfExperts` class for “completeness” but never wires it into `build_model()`. Clarify whether the omission is intentional or if the forward graph lost a component during refactors.【F:v_06/ModelBuilder.py†L1-L80】【24a596†L1-L6】

## Operational Behaviours to Document
- **Mixed-precision expectations remain implicit.** Each `Run.py` enables TensorFlow mixed precision by default but does not document GPU/driver prerequisites or the fallback plan when only CPU is available. Explicit guidance is needed to ensure reproducible training across environments.【F:v_08/Run.py†L12-L47】
