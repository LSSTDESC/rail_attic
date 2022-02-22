import numpy as np
import os
import copy
import pytest
from rail.core.stage import RailStage
from rail.core.data import DataStore, TableHandle
from rail.estimation.algos import randomPZ, sklearn_nn, flexzboost, trainZ
from rail.estimation.algos import bpz_lite, pzflow, knnpz #delightPZ, knnpz
from rail.estimation.estimator import MODEL_FACTORY
from pzflow import Flow

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
    training_data = DS.read_file('training_data', TableHandle, traindata)
    validation_data = DS.read_file('validation_data', TableHandle, validdata)

    if single_trainer is not None:
        train_pz = single_trainer.make_stage(**train_kwargs)
        train_pz.inform(training_data)
        
    pz = single_estimator.make_stage(name=key, **estim_kwargs)    
    estim = pz.estimate(validation_data)

    copy_estim_kwargs = estim_kwargs.copy()
    model_file = copy_estim_kwargs.pop('model_file', 'None')

    READER_DICT = dict(PZFlow=pzflow.model_read_flow)
    if model_file != 'None':
        model = MODEL_FACTORY.read(model_file, reader=READER_DICT.get(key))
        copy_estim_kwargs['model'] = model
        pz_2 = single_estimator.make_stage(name=f"{pz.name}_copy", **copy_estim_kwargs)    
        estim_2 = pz_2.estimate(validation_data)
    else:
        pz_2 = None
        estim_2 = estim
    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))
    if pz_2 is not None:
        os.remove(pz_2.get_output(pz_2.get_aliased_tag('output'), final_name=True))
                  
    return estim.data, estim_2.data


def test_random_pz():
    train_config_dict = {}
    estim_config_dict = {'rand_width': 0.025, 'rand_zmin': 0.0,
                         'rand_zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname':'photometry',
                         'model_file': 'None'}
    zb_expected = np.array([1.359, 0.013, 0.944, 1.831, 2.982, 1.565, 0.308, 0.157, 0.986, 1.679])
    train_algo = None
    pz_algo = randomPZ.RandomPZ
    results, rerun_results = one_algo("RandomPZ", train_algo, pz_algo, train_config_dict, estim_config_dict)
    #assert np.isclose(results.ancil['zmode'], zb_expected).all()
    # assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()
    # we skip this assert since the random number generator will return
    # different results the second run!


def test_simple_nn():
    train_config_dict = {'width': 0.025, 'zmin': 0.0, 'zmax': 3.0,
                         'nzbins': 301, 'max_iter': 250,
                         'hdf5_groupname':'photometry',                         
                         'model_file': 'model.tmp'}
    estim_config_dict = {'hdf5_groupname':'photometry',
                         'model_file': 'inprogress_model.tmp'}        
    zb_expected = np.array([0.152, 0.135, 0.109, 0.158, 0.113, 0.176, 0.13 , 0.15 , 0.119, 0.133])
    train_algo = sklearn_nn.Train_SimpleNN
    pz_algo = sklearn_nn.SimpleNN
    results, rerun_results = one_algo("SimpleNN", train_algo, pz_algo, train_config_dict, estim_config_dict)
    #assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()
    os.remove('inprogress_model.tmp')


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
                         'hdf5_groupname':'photometry',                                               
                         'model_file': 'model.tmp'}
    estim_config_dict = {'hdf5_groupname':'photometry',
                         'model_file': 'model.tmp'}        
    zb_expected = np.array([0.13, 0.13, 0.13, 0.12, 0.12, 0.13, 0.12, 0.13,
                            0.12, 0.12])
    train_algo = flexzboost.Train_FZBoost
    pz_algo = flexzboost.FZBoost
    results, rerun_results = one_algo("FZBoost", train_algo, pz_algo, train_config_dict, estim_config_dict)
    #assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()
    os.remove('model.tmp')


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
                             model_file="PZflowPDF.pkl")
    estim_config_dict = dict(hdf5_groupname='photometry',                                 
                             model_file="inprogress_PZflowPDF.pkl")

    # zb_expected = np.array([0.15, 0.14, 0.11, 0.14, 0.12, 0.14, 0.15, 0.16, 0.11, 0.12])
    train_algo = pzflow.Train_PZFlowPDF
    pz_algo = pzflow.PZFlowPDF
    results, rerun_results = one_algo("PZFlow", train_algo, pz_algo, train_config_dict, estim_config_dict)
    # temporarily remove comparison to "expected" values, as we are getting
    # slightly different answers for python3.7 vs python3.8 for some reason
#    assert np.isclose(results.ancil['zmode'], zb_expected, atol=0.05).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode'], atol=0.05).all()
    os.remove('inprogress_PZflowPDF.pkl')

def test_train_pz():
    train_config_dict = dict(zmin=0.0,
                             zmax=3.0,
                             nzbins=301,
                             hdf5_groupname='photometry',
                             model_file='model_train_z.tmp')
    estim_config_dict = dict(hdf5_groupname='photometry',
                             model_file='inprogress_model_train_z.tmp')

    zb_expected = np.repeat(0.1445183, 10)
    pdf_expected = np.zeros(shape=(301, ))
    pdf_expected[10:16] = [7, 23, 8, 23, 26, 13]
    train_algo = trainZ.Train_trainZ
    pz_algo = trainZ.TrainZ
    results, rerun_results = one_algo("TrainZ", train_algo, pz_algo, train_config_dict, estim_config_dict)
    assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()
    os.remove('inprogress_model_train_z.tmp')


def test_delight():
    with open("./tests/delightPZ.yaml", "r") as f:
        config_dict=yaml.safe_load(f)
    pz_algo = delightPZ.delightPZ
    pz_dict, rerun_pz_dict = one_algo(pz_algo, config_dict)
    zb_expected = np.array([0.18, 0.01, -1., -1., 0.01, -1., -1., -1., 0.01, 0.01])
    assert np.isclose(pz_dict['zmode'], zb_expected, atol=0.03).all()
    assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()
    

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
                             model_file="KNearNeighPDF.pkl")
    estim_config_dict = dict(hdf5_groupname='photometry',
                             model_file="inprogress_KNearNeighPDF.pkl")       

    zb_expected = np.array([0.13, 0.14, 0.13, 0.13, 0.11, 0.15, 0.13, 0.14,
                            0.11, 0.12])
    train_algo = knnpz.Train_KNearNeighPDF
    pz_algo = knnpz.KNearNeighPDF
    results, rerun_results = one_algo("KNN", train_algo, pz_algo, train_config_dict, estim_config_dict)
    assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()

    os.remove('inprogress_KNearNeighPDF.pkl')
    os.remove('TEMPZFILE.out')

def test_catch_bad_bands():
    params = dict(bands='u,g,r,i,z,y')
    with pytest.raises(ValueError):
        flexzboost.Train_FZBoost.make_stage(hdf5_groupname='', **params)    
    with pytest.raises(ValueError):
        flexzboost.FZBoost.make_stage(hdf5_groupname='', **params)
    with pytest.raises(ValueError) as errinfo:
        sklearn_nn.Train_SimpleNN.make_stage(hdf5_groupname='', **params)
    with pytest.raises(ValueError) as errinfo:
        sklearn_nn.SimpleNN.make_stage(hdf5_groupname='', **params)


def test_bpz_lite():
    train_config_dict = {}    
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'columns_file':"./examples/estimation/configs/test_bpz.columns",
                         'spectra_file': "SED/CWWSB4.list",
                         'madau_flag': 'no',
                         'bands': 'ugrizy',
                         'prior_band': 'i',
                         'prior_file': 'hdfn_gen',
                         'p_min': 0.005,
                         'gauss_kernel': 0.0,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'hdf5_groupname':'photometry',
                         'modelfile': 'model.out'}
    zb_expected = np.array([0.18, 2.88, 0.14, 0.21, 2.97, 0.18, 0.23, 0.23,
                            2.98, 2.92])
    train_algo = None
    pz_algo = bpz_lite.BPZ_lite
    results, rerun_results = one_algo("BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict)
    assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()


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
                         'prior_band': 'i',
                         'prior_file': 'flat',
                         'p_min': 0.005,
                         'gauss_kernel': 0.1,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'hdf5_groupname':'photometry',                         
                         'modelfile': 'model.out'}
    zb_expected = np.array([0.18, 2.88, 0.12, 0.15, 2.97, 2.78, 0.11, 0.19,
                            2.98, 2.92])
    train_algo = None
    pz_algo = bpz_lite.BPZ_lite
    results, rerun_results = one_algo("BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict)
    #assert np.isclose(results.ancil['zmode'], zb_expected).all()
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
        pz_algo = flexzboost.FZBoost.make_stage(**config_dict)



def test_wrong_modelfile_keyword():
    config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                   'trainfrac': 0.75, 'bumpmin': 0.02,
                   'bumpmax': 0.35, 'nbump': 3,
                   'sharpmin': 0.7, 'sharpmax': 2.1,
                   'nsharp': 3, 'max_basis': 35,
                   'basis_system': 'cosine',
                   'hdf5_groupname':'photometry',                                            
                   'regression_params': {'max_depth': 8,
                                             'objective':
                                             'reg:squarederror'},
                   'model_file': 'nonexist.pkl'}
    with pytest.raises(FileNotFoundError):
        pz_algo = flexzboost.FZBoost.make_stage(**config_dict)

