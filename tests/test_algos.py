import numpy as np
import os
import sys
import glob
import pickle
import pytest
import yaml
from rail.core.stage import RailStage
from rail.core.data import DataStore, TableHandle
from rail.estimation.algos import randomPZ, sklearn_nn, flexzboost, trainZ
try:
    from rail.estimation.algos import delightPZ
except ImportError:
    pass
from rail.estimation.algos import bpz_lite, pzflow, knnpz
import scipy
sci_ver_str = scipy.__version__.split('.')


traindata = 'tests/data/training_100gal.hdf5'
validdata = 'tests/data/validation_10gal.hdf5'
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def one_algo(key, single_trainer, single_estimator, train_kwargs, estim_kwargs):
    """
    A basic test of running an estimator subclass
    Run inform, write temporary trained model to
    'tempmodelfile.tmp', run photo-z algorithm.
    Then, load temp modelfile and re-run, return
    both datasets.
    """
    DS.clear()
    training_data = DS.read_file('training_data', TableHandle, traindata)
    validation_data = DS.read_file('validation_data', TableHandle, validdata)

    if single_trainer is not None:
        train_pz = single_trainer.make_stage(**train_kwargs)
        train_pz.inform(training_data)

    pz = single_estimator.make_stage(name=key, **estim_kwargs)
    estim = pz.estimate(validation_data)

    copy_estim_kwargs = estim_kwargs.copy()
    model_file = copy_estim_kwargs.pop('model', 'None')

    if model_file != 'None':
        copy_estim_kwargs['model'] = model_file
        pz_2 = single_estimator.make_stage(name=f"{pz.name}_copy", **copy_estim_kwargs)
        estim_2 = pz_2.estimate(validation_data)
    else:
        pz_2 = None
        estim_2 = estim

    if single_trainer is not None and 'model' in single_trainer.output_tags():
        copy3_estim_kwargs = estim_kwargs.copy()
        copy3_estim_kwargs['model'] = train_pz.get_handle('model')
        pz_3 = single_estimator.make_stage(name=f"{pz.name}_copy3", **copy3_estim_kwargs)
        estim_3 = pz_3.estimate(validation_data)
    else:
        pz_3 = None
        estim_3 = estim

    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))
    if pz_2 is not None:
        os.remove(pz_2.get_output(pz_2.get_aliased_tag('output'), final_name=True))

    if pz_3 is not None:
        os.remove(pz_3.get_output(pz_3.get_aliased_tag('output'), final_name=True))
    model_file = estim_kwargs.get('model', 'None')
    if model_file != 'None':
        try:
            os.remove(model_file)
        except FileNotFoundError:
            pass
    return estim.data, estim_2.data, estim_3.data


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


def test_flexzboost():
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'trainfrac': 0.75, 'bumpmin': 0.02,
                         'bumpmax': 0.35, 'nbump': 3,
                         'sharpmin': 0.7, 'sharpmax': 2.1,
                         'nsharp': 3, 'max_basis': 35,
                         'basis_system': 'cosine',
                         'regression_params': {'max_depth': 8,
                                               'objective':
                                               'reg:squarederror'},
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    estim_config_dict = {'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    # zb_expected = np.array([0.13, 0.13, 0.13, 0.12, 0.12, 0.13, 0.12, 0.13,
    #                         0.12, 0.12])
    train_algo = flexzboost.Inform_FZBoost
    pz_algo = flexzboost.FZBoost
    results, rerun_results, rerun3_results = one_algo("FZBoost", train_algo, pz_algo, train_config_dict, estim_config_dict)
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


@pytest.mark.skipif('rail.estimation.algos.delightPZ' not in sys.modules,
                    reason="delightPZ not installed!")
def test_delight():
    with open("./tests/delightPZ.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config_dict['model_file'] = "None"
    config_dict['hdf5_groupname'] = 'photometry'
    train_algo = delightPZ.Inform_DelightPZ
    pz_algo = delightPZ.delightPZ
    results, rerun_results, rerun3_results = one_algo("Delight", train_algo, pz_algo, config_dict, config_dict)
    zb_expected = np.array([0.18, 0.01, -1., -1., 0.01, -1., -1., -1., 0.01, 0.01])
    assert np.isclose(results.ancil['zmode'], zb_expected, atol=0.03).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()
    # get delight to clean up after itself
    for pattern in ['rail/estimation/data/SED/ssp_*Myr_z008_fluxredshiftmod.txt',
                    'rail/estimation/data/SED/*_B2004a_fluxredshiftmod.txt',
                    'rail/estimation/data/FILTER/DC2LSST_*_gaussian_coefficients.txt',
                    'examples/estimation/tmp/delight_data/galaxies*.txt',
                    'parametersTest*.cfg']:
        files = glob.glob(pattern)
        for file_ in files:
            os.remove(file_)
    os.removedirs('examples/estimation/tmp/delight_data')


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
    os.remove('TEMPZFILE.out')


def test_catch_bad_bands():
    params = dict(bands='u,g,r,i,z,y')
    with pytest.raises(ValueError):
        flexzboost.Inform_FZBoost.make_stage(hdf5_groupname='', **params)
    with pytest.raises(ValueError):
        flexzboost.FZBoost.make_stage(hdf5_groupname='', **params)
    with pytest.raises(ValueError):
        sklearn_nn.Inform_SimpleNN.make_stage(hdf5_groupname='', **params)
    with pytest.raises(ValueError):
        sklearn_nn.SimpleNN.make_stage(hdf5_groupname='', **params)


def test_bpz_train():
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'dz': 0.01, 'hdf5_groupname': 'photometry',
                         'nt_array': [8], 'model': 'testmodel_bpz.pkl'}
    train_algo = bpz_lite.Inform_BPZ_lite
    DS.clear()
    training_data = DS.read_file('training_data', TableHandle, traindata)
    train_stage = train_algo.make_stage(**train_config_dict)
    train_stage.inform(training_data)
    expected_keys = ['fo_arr', 'kt_arr', 'zo_arr', 'km_arr', 'a_arr', 'mo', 'nt_array']
    with open("testmodel_bpz.pkl", "rb") as f:
        tmpmodel = pickle.load(f)
    for key in expected_keys:
        assert key in tmpmodel.keys()


def test_bpz_lite():
    train_config_dict = {}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'columns_file': "./examples/estimation/configs/test_bpz.columns",
                         'spectra_file': "SED/CWWSB4.list",
                         'madau_flag': 'no',
                         'no_prior': False,
                         'prior_band': 'mag_i_lsst',
                         'prior_file': 'hdfn_gen',
                         'p_min': 0.005,
                         'gauss_kernel': 0.0,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'hdf5_groupname': 'photometry',
                         'nt_array': [8],
                         'model': 'testmodel_bpz.pkl'}
    zb_expected = np.array([0.18, 2.89, 0.12, 0.19, 2.97, 2.78, 0.1, 0.23,
                            2.98, 2.92])
    train_algo = None
    pz_algo = bpz_lite.BPZ_lite
    results, rerun_results, rerun3_results = one_algo("BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict)
    assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()


def test_bpz_wHDFN_prior():
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'columns_file': "./examples/estimation/configs/test_bpz.columns",
                         'spectra_file': "SED/CWWSB4.list",
                         'madau_flag': 'no',
                         'bands': 'ugrizy',
                         'prior_band': 'mag_i_lsst',
                         'prior_file': 'flat',
                         'p_min': 0.005,
                         'gauss_kernel': 0.1,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'hdf5_groupname': 'photometry',
                         'nt_array': [1, 2, 5],
                         'model': './examples/estimation/CWW_HDFN_prior.pkl'}
    zb_expected = np.array([0.18, 2.88, 0.12, 0.15, 2.97, 2.78, 0.11, 0.19,
                            2.98, 2.92])

    validation_data = DS.read_file('validation_data', TableHandle, validdata)
    pz = bpz_lite.BPZ_lite.make_stage(name='bpz_hdfn', **estim_config_dict)
    results = pz.estimate(validation_data)
    assert np.isclose(results.data.ancil['zmode'], zb_expected).all()
    DS.clear()
    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))


def test_bpz_lite_wkernel_flatprior():
    train_config_dict = {}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'columns_file': "./examples/estimation/configs/test_bpz.columns",
                         'spectra_file': "SED/CWWSB4.list",
                         'madau_flag': 'no',
                         'bands': 'ugrizy',
                         'prior_band': 'mag_i_lsst',
                         'prior_file': 'flat',
                         'p_min': 0.005,
                         'gauss_kernel': 0.1,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'hdf5_groupname': 'photometry'}
    # zb_expected = np.array([0.18, 2.88, 0.12, 0.15, 2.97, 2.78, 0.11, 0.19,
    #                         2.98, 2.92])
    train_algo = None
    pz_algo = bpz_lite.BPZ_lite
    results, rerun_results, rerun3_results = one_algo("BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict)
    # assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()


def test_missing_groupname_keyword():
    config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                   'trainfrac': 0.75, 'bumpmin': 0.02,
                   'bumpmax': 0.35, 'nbump': 3,
                   'sharpmin': 0.7, 'sharpmax': 2.1,
                   'nsharp': 3, 'max_basis': 35,
                   'basis_system': 'cosine',
                   'regression_params': {'max_depth': 8,
                                             'objective':
                                             'reg:squarederror'}}
    with pytest.raises(ValueError):
        _ = flexzboost.FZBoost.make_stage(**config_dict)


def test_wrong_modelfile_keyword():
    DS.clear()
    config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                   'trainfrac': 0.75, 'bumpmin': 0.02,
                   'bumpmax': 0.35, 'nbump': 3,
                   'sharpmin': 0.7, 'sharpmax': 2.1,
                   'nsharp': 3, 'max_basis': 35,
                   'basis_system': 'cosine',
                   'hdf5_groupname': 'photometry',
                   'regression_params': {'max_depth': 8,
                                             'objective':
                                             'reg:squarederror'},
                   'model': 'nonexist.pkl'}
    with pytest.raises(FileNotFoundError):
        pz_algo = flexzboost.FZBoost.make_stage(**config_dict)
        assert pz_algo.model is None
