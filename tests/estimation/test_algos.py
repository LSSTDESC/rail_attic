import numpy as np
import pytest
import scipy.special

from rail.core.algo_utils import one_algo
from rail.core.stage import RailStage
from rail.estimation.algos import randomPZ, trainZ

sci_ver_str = scipy.__version__.split(".")


DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def test_random_pz():
    train_config_dict = {}
    estim_config_dict = {
        "rand_width": 0.025,
        "rand_zmin": 0.0,
        "rand_zmax": 3.0,
        "nzbins": 301,
        "hdf5_groupname": "photometry",
        "model": "None",
    }
    # zb_expected = np.array([1.359, 0.013, 0.944, 1.831, 2.982, 1.565, 0.308, 0.157, 0.986, 1.679])
    train_algo = None
    pz_algo = randomPZ.RandomPZ
    results, rerun_results, rerun3_results = one_algo(
        "RandomPZ", train_algo, pz_algo, train_config_dict, estim_config_dict
    )
    # assert np.isclose(results.ancil['zmode'], zb_expected).all()
    # assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()
    # we skip this assert since the random number generator will return
    # different results the second run!


def test_train_pz():
    train_config_dict = dict(
        zmin=0.0, zmax=3.0, nzbins=301, hdf5_groupname="photometry", model="model_train_z.tmp"
    )
    estim_config_dict = dict(hdf5_groupname="photometry", model="model_train_z.tmp")

    zb_expected = np.repeat(0.1445183, 10)
    pdf_expected = np.zeros(shape=(301,))
    pdf_expected[10:16] = [7, 23, 8, 23, 26, 13]
    train_algo = trainZ.Inform_trainZ
    pz_algo = trainZ.TrainZ
    results, rerun_results, rerun3_results = one_algo(
        "TrainZ", train_algo, pz_algo, train_config_dict, estim_config_dict
    )
    assert np.isclose(results.ancil["zmode"], zb_expected).all()
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()


