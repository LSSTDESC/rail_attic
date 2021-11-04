from typing import Type
import numpy as np
import pandas as pd
import pytest
from rail.creation.degradation import *


@pytest.fixture
def data():
    """Some dummy data to use below."""

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(100, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in a pandas DataFrame
    return pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])


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
def test_LineConfusion_bad_params(true_wavelen, wrong_wavelen, frac_wrong):
    """Test bad parameters that should raise ValueError"""
    with pytest.raises(ValueError):
        LineConfusion(true_wavelen, wrong_wavelen, frac_wrong)


def test_LineConfusion_returns_correct_shape(data):
    """Make sure LineConfusion doesn't change shape of data"""

    degrader = LineConfusion(100, 200, 0.5)
    degraded_data = degrader(data)

    assert degraded_data.shape == data.shape


def test_LineConfusion_no_negative_redshifts(data):
    """Make sure that LineConfusion never returns a negative redshift"""

    degrader = LineConfusion(200, 100, 0.5)
    degraded_data = degrader(data)

    assert all(degraded_data["redshift"].to_numpy() >= 0)


@pytest.mark.parametrize("pivot_redshift", ["fake", -1])
def test_InvRedshiftIncompleteness_bad_params(pivot_redshift):
    """Test bad parameters that should raise ValueError"""
    with pytest.raises(ValueError):
        InvRedshiftIncompleteness(pivot_redshift)


def test_InvRedshiftIncompleteness_returns_correct_shape(data):
    """Make sure returns same number of columns, fewer rows"""
    degrader = InvRedshiftIncompleteness(1)
    degraded_data = degrader(data)
    assert degraded_data.shape[0] < data.shape[0]
    assert degraded_data.shape[1] == data.shape[1]


@pytest.mark.parametrize(
    "cuts,error",
    [
        (1, TypeError),
        ({"u": "cut"}, TypeError),
        ({"u": dict()}, TypeError),
        ({"u": [1, 2, 3]}, ValueError),
        ({"u": [1, "max"]}, TypeError),
        ({"u": [2, 1]}, ValueError),
        ({"u": TypeError}, TypeError),
    ],
)
def test_QuantityCut_bad_params(cuts, error):
    """Test bad parameters that should return Type and Value errors"""
    with pytest.raises(error):
        QuantityCut(cuts)


def test_QuantityCut_returns_correct_shape(data):
    """Make sure QuantityCut is returning the correct shape"""

    cuts = {
        "u": 0,
        "y": (1, 2),
    }
    degrader = QuantityCut(cuts)

    degraded_data = degrader(data)

    assert degraded_data.shape == data.query("u < 0 & y > 1 & y < 2").shape


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"m5": 1}, TypeError),
        ({"tvis": False}, TypeError),
        ({"nYrObs": False}, TypeError),
        ({"airmass": False}, TypeError),
        ({"extendedSource": False}, TypeError),
        ({"sigmaSys": False}, TypeError),
        ({"magLim": False}, TypeError),
        ({"ndFlag": False}, TypeError),
        ({"highSNR": 0}, TypeError),
        ({"tvis": -1}, ValueError),
        ({"nYrObs": -1}, ValueError),
        ({"airmass": -1}, ValueError),
        ({"extendedSource": -1}, ValueError),
        ({"sigmaSys": -1}, ValueError),
        ({"bandNames": False}, TypeError),
        ({"nVisYr": False}, TypeError),
        ({"gamma": False}, TypeError),
        ({"Cm": False}, TypeError),
        ({"msky": False}, TypeError),
        ({"theta": False}, TypeError),
        ({"km": False}, TypeError),
        ({"nVisYr": {}}, ValueError),
        ({"gamma": {}}, ValueError),
        ({"Cm": {}}, ValueError),
        ({"msky": {}}, ValueError),
        ({"theta": {}}, ValueError),
        ({"km": {}}, ValueError),
        ({"nVisYr": {f"{b}": False for b in "ugrizy"}}, TypeError),
        ({"gamma": {f"{b}": False for b in "ugrizy"}}, TypeError),
        ({"Cm": {f"{b}": False for b in "ugrizy"}}, TypeError),
        ({"msky": {f"{b}": False for b in "ugrizy"}}, TypeError),
        ({"theta": {f"{b}": False for b in "ugrizy"}}, TypeError),
        ({"km": {f"{b}": False for b in "ugrizy"}}, TypeError),
        ({"nVisYr": {f"{b}": -1 for b in "ugrizy"}}, ValueError),
        ({"gamma": {f"{b}": -1 for b in "ugrizy"}}, ValueError),
        ({"theta": {f"{b}": -1 for b in "ugrizy"}}, ValueError),
        ({"km": {f"{b}": -1 for b in "ugrizy"}}, ValueError),
    ],
)
def test_LSSTErrorModel_bad_params(settings, error):
    """Test bad parameters that should raise Value and Type errors."""
    with pytest.raises(error):
        LSSTErrorModel(**settings)


@pytest.mark.parametrize("m5,highSNR", [({}, False), ({"u": 23}, True)])
def test_LSSTErrorModel_returns_correct_shape(m5, highSNR, data):
    """Test that the LSSTErrorModel returns the correct shape"""

    degrader = LSSTErrorModel(m5=m5, highSNR=highSNR)

    degraded_data = degrader(data)

    assert degraded_data.shape == (data.shape[0], 2 * data.shape[1] - 1)


@pytest.mark.parametrize("m5,highSNR", [({}, False), ({"u": 23}, True)])
def test_LSSTErrorModel_magLim(m5, highSNR, data):
    """Test that no mags are above magLim"""

    magLim = 27
    ndFlag = np.nan
    degrader = LSSTErrorModel(magLim=magLim, ndFlag=ndFlag, m5=m5, highSNR=highSNR)

    degraded_data = degrader(data)
    degraded_mags = degraded_data.iloc[:, 1:].to_numpy()

    assert degraded_mags[~np.isnan(degraded_mags)].max() < magLim


@pytest.mark.parametrize("highSNR", [False, True])
def test_LSSTErrorModel_get_limiting_mags(highSNR):

    degrader = LSSTErrorModel(highSNR=highSNR)

    # make sure that the 1-sigma single-visit limiting mags match
    assert np.allclose(
        list(degrader.get_limiting_mags().values()),
        list(degrader._all_m5.values()),
        rtol=1e-4,
    )

    # make sure the coadded mags are deeper
    assert np.all(
        np.array(list(degrader.get_limiting_mags(coadded=True).values()))
        > np.array(list(degrader.get_limiting_mags(coadded=False).values()))
    )


@pytest.mark.parametrize(
    "degrader",
    [
        LineConfusion(100, 200, 0.01),
        InvRedshiftIncompleteness(1),
        LSSTErrorModel(),
        LSSTErrorModel(highSNR=True),
    ],
)
def test_random_seed(degrader, data):
    """Test control with random seeds."""

    # make sure setting the same seeds yields the same output
    degraded_data1 = degrader(data, seed=0)
    degraded_data2 = degrader(data, seed=0)
    assert degraded_data1.equals(degraded_data2)

    # make sure setting different seeds yields different output
    degraded_data3 = degrader(data, seed=1).to_numpy()
    assert not degraded_data1.equals(degraded_data3)