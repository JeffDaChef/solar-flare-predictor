import numpy as np
import pytest

from nn.validate_torch import HAS_TORCH

if HAS_TORCH:
    from nn.validate_torch import compare_lstm, compare_mlp


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_matches_pytorch():
    errors = compare_mlp(np.random.default_rng(0))
    assert errors["forward"] < 1e-8
    assert errors["grad"] < 1e-7


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_lstm_matches_pytorch():
    errors = compare_lstm(np.random.default_rng(1))
    assert errors["forward"] < 1e-8
    assert errors["grad"] < 1e-7
