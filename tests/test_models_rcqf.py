import pytest

torch = pytest.importorskip("torch")

from src.models.deep.rcqf import RCQFConfig, RCQFModel


def test_rcqf_forward_returns_quantiles_and_median():
    config = RCQFConfig(
        input_features=4,
        time_steps=12,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        bidirectional=True,
        num_regimes=3,
        gating_hidden_size=16,
        quantile_levels=(0.05, 0.5, 0.95),
    )
    model = RCQFModel(config)
    inputs = torch.randn(6, config.time_steps, config.input_features)

    outputs = model(inputs, return_dict=True)
    assert outputs["median"].shape == (6, 1)
    assert outputs["quantiles"].shape == (6, len(config.quantile_levels))
    assert outputs["per_regime_quantiles"].shape == (
        6, config.num_regimes, len(config.quantile_levels)
    )
    probs = outputs["regime_probabilities"]
    assert torch.allclose(probs.sum(dim=1), torch.ones(6), atol=1e-6)


def test_rcqf_forward_tensor_interface():
    config = RCQFConfig(input_features=2, time_steps=5)
    model = RCQFModel(config)
    inputs = torch.randn(2, config.time_steps, config.input_features)
    median = model(inputs)
    assert median.shape == (2, 1)
