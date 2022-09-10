import numpy as np
import pytest
from rail.core.stage import RailStage
from rail.core.algo_utils import one_algo
from rail.estimation.algos import randomPZ, sklearn_nn, trainZ
from rail.estimation.algos import pzflow, knnpz
import scipy.special
sci_ver_str = scipy.__version__.split('.')


DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def test_random_pz():
    train_config_dict = {}
    estim_config_dict = {'rand_width': 0.025, 'rand_zmin': 0.0,
                         'rand_zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'model': 'None'}
    # zb_expected = np.array([1.359, 0.013, 0.944, 1.831, 2.982, 1.565, 0.308, 0.157, 0.986, 1.679])
    train_algo = None
    pz_algo = randomPZ.RandomPZ
    results, rerun_results, rerun3_results = one_algo("RandomPZ", train_algo, pz_algo, train_config_dict, estim_config_dict)
    # assert np.isclose(results.ancil['zmode'], zb_expected).all()
    # assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()
    # we skip this assert since the random number generator will return
    # different results the second run!


def test_simple_nn():
    train_config_dict = {'width': 0.025, 'zmin': 0.0, 'zmax': 3.0,
                         'nzbins': 301, 'max_iter': 250,
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    estim_config_dict = {'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    # zb_expected = np.array([0.152, 0.135, 0.109, 0.158, 0.113, 0.176, 0.13 , 0.15 , 0.119, 0.133])
    train_algo = sklearn_nn.Inform_SimpleNN
    pz_algo = sklearn_nn.SimpleNN
    results, rerun_results, rerun3_results = one_algo("SimpleNN", train_algo, pz_algo, train_config_dict, estim_config_dict)
    # assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()


@pytest.mark.parametrize(
    "inputs, zb_expected",
    [(False, [0.15, 0.14, 0.11, 0.14, 0.12, 0.14, 0.15, 0.16, 0.11, 0.12]),
     (True, [0.15, 0.14, 0.15, 0.14, 0.12, 0.14, 0.15, 0.12, 0.13, 0.11]),
     ],
)
def test_pzflow(inputs, zb_expected):
    def_bands = ['u', 'g', 'r', 'i', 'z', 'y']
    refcols = [f"mag_{band}_lsst" for band in def_bands]
    def_maglims = dict(mag_u_lsst=27.79,
                       mag_g_lsst=29.04,
                       mag_r_lsst=29.06,
                       mag_i_lsst=28.62,
                       mag_z_lsst=27.98,
                       mag_y_lsst=27.05)
    def_errnames = dict(mag_err_u_lsst="mag_u_lsst_err",
                        mag_err_g_lsst="mag_g_lsst_err",
                        mag_err_r_lsst="mag_r_lsst_err",
                        mag_err_i_lsst="mag_i_lsst_err",
                        mag_err_z_lsst="mag_z_lsst_err",
                        mag_err_y_lsst="mag_y_lsst_err")
    train_config_dict = dict(zmin=0.0,
                             zmax=3.0,
                             nzbins=301,
                             flow_seed=0,
                             ref_column_name='mag_i_lsst',
                             column_names=refcols,
                             mag_limits=def_maglims,
                             include_mag_errors=inputs,
                             error_names_dict=def_errnames,
                             n_error_samples=3,
                             soft_sharpness=10,
                             soft_idx_col=0,
                             redshift_column_name='redshift',
                             num_training_epochs=50,
                             hdf5_groupname='photometry',
                             model="PZflowPDF.pkl")
    estim_config_dict = dict(hdf5_groupname='photometry',
                             model="PZflowPDF.pkl")

    # zb_expected = np.array([0.15, 0.14, 0.11, 0.14, 0.12, 0.14, 0.15, 0.16, 0.11, 0.12])
    train_algo = pzflow.Inform_PZFlowPDF
    pz_algo = pzflow.PZFlowPDF
    results, rerun_results, rerun3_results = one_algo("PZFlow", train_algo, pz_algo, train_config_dict, estim_config_dict)
    # temporarily remove comparison to "expected" values, as we are getting
    # slightly different answers for python3.7 vs python3.8 for some reason
#    assert np.isclose(results.ancil['zmode'], zb_expected, atol=0.05).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode'], atol=0.05).all()


def test_train_pz():
    train_config_dict = dict(zmin=0.0,
                             zmax=3.0,
                             nzbins=301,
                             hdf5_groupname='photometry',
                             model='model_train_z.tmp')
    estim_config_dict = dict(hdf5_groupname='photometry',
                             model='model_train_z.tmp')

    zb_expected = np.repeat(0.1445183, 10)
    pdf_expected = np.zeros(shape=(301, ))
    pdf_expected[10:16] = [7, 23, 8, 23, 26, 13]
    train_algo = trainZ.Inform_trainZ
    pz_algo = trainZ.TrainZ
    results, rerun_results, rerun3_results = one_algo("TrainZ", train_algo, pz_algo, train_config_dict, estim_config_dict)
    assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()


@pytest.mark.skipif(int(sci_ver_str[0]) < 2 and int(sci_ver_str[1]) < 8,
                    reason="mixmod parameterization known to break for scipy<1.8 due to array broadcast change")
def test_KNearNeigh():
    def_bands = ['u', 'g', 'r', 'i', 'z', 'y']
    refcols = [f"mag_{band}_lsst" for band in def_bands]
    def_maglims = dict(mag_u_lsst=27.79,
                       mag_g_lsst=29.04,
                       mag_r_lsst=29.06,
                       mag_i_lsst=28.62,
                       mag_z_lsst=27.98,
                       mag_y_lsst=27.05)
    train_config_dict = dict(zmin=0.0,
                             zmax=3.0,
                             nzbins=301,
                             trainfrac=0.75,
                             random_seed=87,
                             ref_column_name='mag_i_lsst',
                             column_names=refcols,
                             mag_limits=def_maglims,
                             sigma_grid_min=0.02,
                             sigma_grid_max=0.03,
                             ngrid_sigma=2,
                             leaf_size=2,
                             nneigh_min=2,
                             nneigh_max=3,
                             redshift_column_name='redshift',
                             hdf5_groupname='photometry',
                             model="KNearNeighPDF.pkl")
    estim_config_dict = dict(hdf5_groupname='photometry',
                             model="KNearNeighPDF.pkl")

    # zb_expected = np.array([0.13, 0.14, 0.13, 0.13, 0.11, 0.15, 0.13, 0.14,
    #                         0.11, 0.12])
    train_algo = knnpz.Inform_KNearNeighPDF
    pz_algo = knnpz.KNearNeighPDF
    results, rerun_results, rerun3_results = one_algo("KNN", train_algo, pz_algo, train_config_dict, estim_config_dict)
    # assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()


def test_catch_bad_bands():
    params = dict(bands='u,g,r,i,z,y')
    with pytest.raises(ValueError):
        sklearn_nn.Inform_SimpleNN.make_stage(hdf5_groupname='', **params)
    with pytest.raises(ValueError):
        sklearn_nn.SimpleNN.make_stage(hdf5_groupname='', **params)
