from typing import Type
import numpy as np
import pandas as pd
import pytest
from rail.creation.degradation import *


@pytest.fixture
def data():
    np.random.seed(0)
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
def test_BandCut_bad_inputs(cuts, error):
    with pytest.raises(error):
        BandCut(cuts)


def test_BandCut_returns_correct_shape(data):
    cuts = {
        "u": 0,
        "y": (1, 2),
    }
    degrader = BandCut(cuts)
    degraded_data = degrader(data)
    assert degraded_data.shape == data.query("u < 0 & y > 1 & y < 2").shape


def test_BandCut_repr_is_string():
    assert isinstance(BandCut({}).__repr__(), str)


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
        ({"nVisYr": {f"lsst_{b}": False for b in "ugrizy"}}, TypeError),
        ({"gamma": {f"lsst_{b}": False for b in "ugrizy"}}, TypeError),
        ({"Cm": {f"lsst_{b}": False for b in "ugrizy"}}, TypeError),
        ({"msky": {f"lsst_{b}": False for b in "ugrizy"}}, TypeError),
        ({"theta": {f"lsst_{b}": False for b in "ugrizy"}}, TypeError),
        ({"km": {f"lsst_{b}": False for b in "ugrizy"}}, TypeError),
        ({"nVisYr": {f"lsst_{b}": -1 for b in "ugrizy"}}, ValueError),
        ({"gamma": {f"lsst_{b}": -1 for b in "ugrizy"}}, ValueError),
        ({"theta": {f"lsst_{b}": -1 for b in "ugrizy"}}, ValueError),
        ({"km": {f"lsst_{b}": -1 for b in "ugrizy"}}, ValueError),
    ],
)
def test_LSSTErrorModel_bad_inputs(settings, error):
    with pytest.raises(error):
        LSSTErrorModel(**settings)


# I will test with and without an explicit list of m5's
def test_LSSTErrorModel_returns_correct_shape(data):

    bandNames = {f"lsst_{b}": b for b in "ugrizy"}

    degrader = LSSTErrorModel(bandNames=bandNames)
    degraded_data = degrader(data)
    assert degraded_data.shape == (data.shape[0], 2 * data.shape[1] - 1)

    m5 = {"lsst_u": 23}
    degrader = LSSTErrorModel(bandNames=bandNames, m5=m5)
    degraded_data = degrader(data)
    assert degraded_data.shape == (data.shape[0], 2 * data.shape[1] - 1)


# I will test with and without an explicit list of m5's
def test_LSSTErrorModel_magLim(data):
    bandNames = {f"lsst_{b}": b for b in "ugrizy"}
    magLim = 27

    degrader = LSSTErrorModel(bandNames=bandNames, magLim=magLim)
    degraded_data = degrader(data)
    degraded_mags = degraded_data[bandNames.values()].to_numpy()
    assert degraded_mags[degraded_mags < 99].max() < magLim

    m5 = {"lsst_u": 23}
    degrader = LSSTErrorModel(bandNames=bandNames, magLim=magLim, m5=m5)
    degraded_data = degrader(data)
    degraded_mags = degraded_data[bandNames.values()].to_numpy()
    assert degraded_mags[degraded_mags < 99].max() < magLim


def test_LSSTErrorModel_repr_is_string():
    # I will pass an explicit m5 to make sure the if statements checking
    # for explicit m5's are triggered during the test
    m5 = {"lsst_u": 23}
    assert isinstance(LSSTErrorModel(m5=m5).__repr__(), str)