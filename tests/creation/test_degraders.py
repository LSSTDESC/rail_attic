import os
from typing import Type

import numpy as np
import pandas as pd
import pytest
from rail.core.data import DATA_STORE, TableHandle
from rail.core.utilStages import ColumnMapper
from rail.creation.degradation import *


@pytest.fixture
def data():
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(100, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return DS.add_data("data", df, TableHandle, path="dummy.pd")


@pytest.fixture
def data_forspec():
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(200000, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return DS.add_data("data_forspec", df, TableHandle, path="dummy_forspec.pd")


@pytest.mark.parametrize(
    "true_wavelen,wrong_wavelen,frac_wrong,errortype",
    [
        ("fake", 200, 0.5, TypeError),
        (100, "fake", 0.5, TypeError),
        (100, 200, "fake", TypeError),
        (-1, 200, 0.5, ValueError),
        (100, -1, 0.5, ValueError),
        (100, 200, -1, ValueError),
        (100, 200, 2, ValueError),
    ],
)
def test_LineConfusion_bad_params(true_wavelen, wrong_wavelen, frac_wrong, errortype):
    """Test bad parameters that should raise ValueError"""
    with pytest.raises(errortype):
        LineConfusion.make_stage(
            true_wavelen=true_wavelen,
            wrong_wavelen=wrong_wavelen,
            frac_wrong=frac_wrong,
        )


def test_LineConfusion_returns_correct_shape(data):
    """Make sure LineConfusion doesn't change shape of data"""

    degrader = LineConfusion.make_stage(
        name="xx", true_wavelen=100, wrong_wavelen=200, frac_wrong=0.5
    )
    degraded_data = degrader(data).data

    assert degraded_data.shape == data.data.shape
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


def test_LineConfusion_no_negative_redshifts(data):
    """Make sure that LineConfusion never returns a negative redshift"""

    degrader = LineConfusion.make_stage(
        true_wavelen=100, wrong_wavelen=200, frac_wrong=0.5
    )
    degraded_data = degrader(data).data

    assert all(degraded_data["redshift"].to_numpy() >= 0)
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


@pytest.mark.parametrize(
    "pivot_redshift,errortype", [("fake", TypeError), (-1, ValueError)]
)
def test_InvRedshiftIncompleteness_bad_params(pivot_redshift, errortype):
    """Test bad parameters that should raise ValueError"""
    with pytest.raises(errortype):
        InvRedshiftIncompleteness.make_stage(pivot_redshift=pivot_redshift)


def test_InvRedshiftIncompleteness_returns_correct_shape(data):
    """Make sure returns same number of columns, fewer rows"""
    degrader = InvRedshiftIncompleteness.make_stage(pivot_redshift=1.0)
    degraded_data = degrader(data).data
    assert degraded_data.shape[0] < data.data.shape[0]
    assert degraded_data.shape[1] == data.data.shape[1]
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


@pytest.mark.parametrize(
    "percentile_cut,redshift_cut,errortype",
    [(-1, 1, ValueError), (101, 1, ValueError), (99, -1, ValueError)],
)
def test_GridSelection_bad_params(percentile_cut, redshift_cut, errortype):
    """Test bad parameters that should raise ValueError"""
    with pytest.raises(errortype):
        GridSelection.make_stage(
            percentile_cut=percentile_cut, redshift_cut=redshift_cut
        )


def test_GridSelection_returns_correct_shape(data):
    import pdb

    """Make sure returns 2 more columns, fewer rows"""
    degrader = GridSelection.make_stage(pessimistic_redshift_cut=1.0)
    degraded_data = degrader(data).data
    # pdb.set_trace()
    assert degraded_data.shape[0] < data.data.shape[0]
    assert degraded_data.shape[1] == data.data.shape[1] - 1
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


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
        QuantityCut.make_stage(cuts=cuts)


def test_QuantityCut_returns_correct_shape(data):
    """Make sure QuantityCut is returning the correct shape"""

    cuts = {
        "u": 0,
        "y": (1, 2),
    }
    degrader = QuantityCut.make_stage(cuts=cuts)

    degraded_data = degrader(data).data

    assert degraded_data.shape == data.data.query("u < 0 & y > 1 & y < 2").shape
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"m5": 1}, TypeError),
        ({"tvis": "xx"}, TypeError),
        ({"nYrObs": "xx"}, TypeError),
        ({"airmass": "xx"}, TypeError),
        ({"extendedSource": "xx"}, TypeError),
        ({"sigmaSys": "xx"}, TypeError),
        ({"magLim": "xx"}, TypeError),
        ({"ndFlag": "xx"}, TypeError),
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
        LSSTErrorModel.make_stage(**settings)


@pytest.mark.parametrize("m5,highSNR", [({}, False), ({"u": 23}, True)])
def test_LSSTErrorModel_returns_correct_shape(m5, highSNR, data):
    """Test that the LSSTErrorModel returns the correct shape"""

    degrader = LSSTErrorModel.make_stage(m5=m5, highSNR=highSNR)

    degraded_data = degrader(data).data

    assert degraded_data.shape == (data.data.shape[0], 2 * data.data.shape[1] - 1)
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


@pytest.mark.parametrize("m5,highSNR", [({}, False), ({"u": 23}, True)])
def test_LSSTErrorModel_magLim(m5, highSNR, data):
    """Test that no mags are above magLim"""

    magLim = 27
    ndFlag = np.nan
    degrader = LSSTErrorModel.make_stage(
        magLim=magLim, ndFlag=ndFlag, m5=m5, highSNR=highSNR
    )

    degraded_data = degrader(data).data
    degraded_mags = degraded_data.iloc[:, 1:].to_numpy()

    assert degraded_mags[~np.isnan(degraded_mags)].max() < magLim
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


@pytest.mark.parametrize("highSNR", [False, True])
def test_LSSTErrorModel_get_limiting_mags(highSNR):

    degrader = LSSTErrorModel.make_stage(highSNR=highSNR)

    # make sure that the 5-sigma single-visit limiting mags match
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

    # test that _get_NSR is the inverse of get_limiting_mags(coadded=True)
    for SNR in [1, 5, 10, 100]:
        limiting_mags = degrader.get_limiting_mags(Nsigma=SNR, coadded=True)
        limiting_mags = np.array(list(limiting_mags.values()))
        NSR = degrader._get_NSR(limiting_mags, degrader.config["bandNames"].keys())
        assert np.allclose(1 / NSR, SNR)


def test_LSSTErrorModel_extended(data):
    n_samples = len(data.data)
    rng = np.random.default_rng(0)
    data.data["major"] = np.abs(
        rng.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * rng.random(n_samples)
    data.data["minor"] = data.data["major"] * b_to_a

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", errorType="auto"
    )
    errorModel_auto(data)
    errorModel_auto.__repr__()

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", errorType="gaap"
    )
    errorModel_gaap(data)
    errorModel_gaap.__repr__()

    os.remove(
        errorModel_auto.get_output(
            errorModel_auto.get_aliased_tag("output"), final_name=True
        )
    )
    os.remove(
        errorModel_gaap.get_output(
            errorModel_gaap.get_aliased_tag("output"), final_name=True
        )
    )


@pytest.mark.parametrize("errType, errortype", [("sakana", ValueError)])
def test_LSSTErrorModel_bad_type(errType, errortype):
    """Test bad parameters that should raise ValueError"""
    with pytest.raises(errortype):
        LSSTErrorModel.make_stage(errorType=errType)


@pytest.mark.parametrize(
    "degrader",
    [
        LineConfusion.make_stage(true_wavelen=100, wrong_wavelen=200, frac_wrong=0.01),
        InvRedshiftIncompleteness.make_stage(pivot_redshift=1.0),
        LSSTErrorModel.make_stage(),
        LSSTErrorModel.make_stage(highSNR=True),
    ],
)
def test_random_seed(degrader, data):
    """Test control with random seeds."""

    # make sure setting the same seeds yields the same output
    degraded_data1 = degrader(data, seed=0).data
    degraded_data2 = degrader(data, seed=0).data
    assert degraded_data1.equals(degraded_data2)

    # make sure setting different seeds yields different output
    degraded_data3 = degrader(data, seed=1).data.to_numpy()
    assert not degraded_data1.equals(degraded_data3)
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


def test_SpecSelection(data):

    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"{band}_err": f"mag_err_{band}_lsst" for band in bands}
    rename_dict.update({f"{band}": f"mag_{band}_lsst" for band in bands})
    standard_colnames = [f"mag_{band}_lsst" for band in "ugrizy"]

    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )
    data = col_remapper_test(data)

    degrader_GAMA = SpecSelection_GAMA.make_stage()
    degrader_GAMA(data)
    degrader_GAMA.__repr__()

    os.remove(
        degrader_GAMA.get_output(
            degrader_GAMA.get_aliased_tag("output"), final_name=True
        )
    )

    degrader_BOSS = SpecSelection_BOSS.make_stage()
    degrader_BOSS(data)
    degrader_BOSS.__repr__()

    os.remove(
        degrader_BOSS.get_output(
            degrader_BOSS.get_aliased_tag("output"), final_name=True
        )
    )

    degrader_DEEP2 = SpecSelection_DEEP2.make_stage()
    degrader_DEEP2(data)
    degrader_DEEP2.__repr__()

    os.remove(
        degrader_DEEP2.get_output(
            degrader_DEEP2.get_aliased_tag("output"), final_name=True
        )
    )

    degrader_VVDSf02 = SpecSelection_VVDSf02.make_stage()
    degrader_VVDSf02(data)
    degrader_VVDSf02.__repr__()

    degrader_zCOSMOS = SpecSelection_zCOSMOS.make_stage(
        colnames={"i": "mag_i_lsst", "redshift": "redshift"}
    )
    degrader_zCOSMOS(data)
    degrader_zCOSMOS.__repr__()

    os.remove(
        degrader_zCOSMOS.get_output(
            degrader_zCOSMOS.get_aliased_tag("output"), final_name=True
        )
    )

    degrader_HSC = SpecSelection_HSC.make_stage()
    degrader_HSC(data)
    degrader_HSC.__repr__()

    os.remove(
        degrader_HSC.get_output(degrader_HSC.get_aliased_tag("output"), final_name=True)
    )

    degrader_HSC = SpecSelection_HSC.make_stage(percentile_cut=70)
    degrader_HSC(data)
    degrader_HSC.__repr__()

    os.remove(
        degrader_HSC.get_output(degrader_HSC.get_aliased_tag("output"), final_name=True)
    )


def test_SpecSelection_low_N_tot(data_forspec):

    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"{band}_err": f"mag_err_{band}_lsst" for band in bands}
    rename_dict.update({f"{band}": f"mag_{band}_lsst" for band in bands})
    standard_colnames = [f"mag_{band}_lsst" for band in "ugrizy"]

    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )
    data_forspec = col_remapper_test(data_forspec)

    degrader_zCOSMOS = SpecSelection_zCOSMOS.make_stage(N_tot=1)
    degrader_zCOSMOS(data_forspec)

    os.remove(
        degrader_zCOSMOS.get_output(
            degrader_zCOSMOS.get_aliased_tag("output"), final_name=True
        )
    )


@pytest.mark.parametrize("N_tot, errortype", [(-1, ValueError)])
def test_SpecSelection_bad_params(N_tot, errortype):
    """Test bad parameters that should raise TypeError"""
    with pytest.raises(errortype):
        SpecSelection.make_stage(N_tot=N_tot)


@pytest.mark.parametrize("errortype", [(ValueError)])
def test_SpecSelection_bad_colname(data, errortype):
    """Test bad parameters that should raise TypeError"""
    with pytest.raises(errortype):
        degrader_GAMA = SpecSelection_GAMA.make_stage()
        degrader_GAMA(data)


@pytest.mark.parametrize(
    "success_rate_dir, errortype", [("/this/path/should/not/exist", ValueError)]
)
def test_SpecSelection_bad_path(success_rate_dir, errortype):
    """Test bad parameters that should raise TypeError"""
    with pytest.raises(errortype):
        SpecSelection.make_stage(success_rate_dir=success_rate_dir)
