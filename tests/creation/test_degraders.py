import pytest
from rail.creation.degradation import *
import numpy as np
import pandas as pd


@pytest.fixture
def data():
    x = np.random.normal(loc=26, scale=1, size=(100, 7))
    x[:, 0] = np.linspace(0, 2, x.shape[0])
    x = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return x


@pytest.mark.parametrize(
    "true_wavelen,wrong_wavelen,frac_wrong",
    [
        ("fake", 200, 0.5),
        (100, "fake", 0.5),
        (100, 200, "fake"),
        (-1, 200, 0.5),
        (100, -1, 0.5),
        (100, 200, -1),
        (100, 200, 2),
    ],
)
def test_LineConfusion_bad_inputs(true_wavelen, wrong_wavelen, frac_wrong):
    with pytest.raises(ValueError):
        LineConfusion(true_wavelen, wrong_wavelen, frac_wrong)


def test_LineConfusion_returns_correct_shape(data):
    degrader = LineConfusion(100, 200, 0.5)
    degraded_data = degrader(data)
    assert degraded_data.shape == data.shape


def test_LineConfusion_no_negative_redshifts(data):
    degrader = LineConfusion(200, 100, 0.5)
    degraded_data = degrader(data)
    assert all(degraded_data["redshift"].values >= 0)


@pytest.mark.parametrize("pivot_redshift", ["fake", -1])
def test_InvRedshiftIncompleteness_bad_inputs(pivot_redshift):
    with pytest.raises(ValueError):
        InvRedshiftIncompleteness(pivot_redshift)


def test_InvRedshiftIncompleteness_returns_correct_shape(data):
    degrader = InvRedshiftIncompleteness(1)
    degraded_data = degrader(data)
    assert degraded_data.shape[0] < data.shape[0]
    assert degraded_data.shape[1] == data.shape[1]


@pytest.mark.parametrize(
    "degrader", [LineConfusion(100, 200, 0.5), InvRedshiftIncompleteness(1)]
)
def test_random_seed(degrader, data):
    degraded_data1 = degrader(data, seed=0)
    degraded_data2 = degrader(data, seed=0)
    assert np.allclose(degraded_data1.values, degraded_data2.values)
