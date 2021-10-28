import numpy as np
import os
import copy
import yaml
from tables_io.ioUtils import readHdf5ToDict, iterHdf5ToDict
import pytest
from rail.estimation.algos import randomPZ, sklearn_nn, flexzboost, trainZ
from rail.estimation.algos import delightPZ, bpz_lite


test_base_yaml = './tests/test.yaml'


def one_algo(single_estimator, single_input):
    """
    A basic test of running an estimator subclass
    Run inform, write temporary trained model to
    'tempmodelfile.tmp', run photo-z algorithm.
    Then, load temp modelfile and re-run, return
    both datasets.
    """

    pz = single_estimator(test_base_yaml, single_input)
    trainfile = pz.trainfile
    training_data = readHdf5ToDict(trainfile, pz.groupname)
    pz.inform_dict = single_input['run_params']['inform_options']
    pz.inform(training_data)
    # set chunk size to pz.num_rows to ensure all run in one chunk
    oversize_rows = pz.num_rows + 4  # test proper chunking truncation
    for _, end, data in iterHdf5ToDict(pz.testfile,
                                       oversize_rows,
                                       pz.hdf5_groupname):
        pz_dict = pz.estimate(data)
    assert end == pz.num_rows
    xinputs = single_input['run_params']
    assert len(pz.zgrid) == np.int32(xinputs['nzbins'])

    pz.load_pretrained_model()
    for _, end, data in iterHdf5ToDict(pz.testfile, pz.num_rows,
                                       pz.hdf5_groupname):
        rerun_pz_dict = pz.estimate(data)
    pz.output_format = 'qp'
    for _, end, data in iterHdf5ToDict(pz.testfile, pz.num_rows,
                                       pz.hdf5_groupname):
        _ = pz.estimate(data)
    # add a test load for no config dict
    # check that all keys are present
    # if single_estimator is not delightPZ.delightPZ:
    noconfig_pz = single_estimator(test_base_yaml)
    for key in single_input['run_params'].keys():
        assert key in noconfig_pz.config_dict['run_params']
    return pz_dict, rerun_pz_dict


def test_random_pz():
    config_dict = {'run_params': {'rand_width': 0.025, 'rand_zmin': 0.0,
                                  'rand_zmax': 3.0, 'nzbins': 301,
                                  'inform_options': {'save_train': True,
                                                     'modelfile': 'model.tmp'}
                                  }}
    zb_expected = np.array([1.969, 2.865, 2.913, 0.293, 0.722, 2.606, 1.642,
                            2.157, 2.777, 1.851])
    pz_algo = randomPZ.randomPZ
    pz_dict, rerun_pz_dict = one_algo(pz_algo, config_dict)
    assert np.isclose(pz_dict['zmode'], zb_expected).all()
    # assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()
    # we skip this assert since the random number generator will return
    # different results the second run!


def test_simple_nn():
    config_dict = {'run_params': {'width': 0.025, 'zmin': 0.0, 'zmax': 3.0,
                                  'nzbins': 301, 'max_iter': 250,
                                  'inform_options': {'save_train': True,
                                                     'modelfile': 'model.tmp'}
                                  }}
    zb_expected = np.array([0.133, 0.123, 0.085, 0.145, 0.123, 0.155, 0.136,
                            0.157, 0.128, 0.13])
    pz_algo = sklearn_nn.simpleNN
    pz_dict, rerun_pz_dict = one_algo(pz_algo, config_dict)
    assert np.isclose(pz_dict['zmode'], zb_expected).all()
    assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()
    os.remove('model.tmp')


def test_flexzboost():
    config_dict = {'run_params': {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                                  'trainfrac': 0.75, 'bumpmin': 0.02,
                                  'bumpmax': 0.35, 'nbump': 3,
                                  'sharpmin': 0.7, 'sharpmax': 2.1,
                                  'nsharp': 3, 'max_basis': 35,
                                  'basis_system': 'cosine',
                                  'regression_params': {'max_depth': 8,
                                                        'objective':
                                                            'reg:squarederror'},
                                  'inform_options': {'save_train': True,
                                                     'modelfile': 'model.tmp'}
                                  }}
    zb_expected = np.array([0.13, 0.13, 0.13, 0.12, 0.12, 0.13, 0.12, 0.13,
                            0.12, 0.12])
    pz_algo = flexzboost.FZBoost
    pz_dict, rerun_pz_dict = one_algo(pz_algo, config_dict)
    assert np.isclose(pz_dict['zmode'], zb_expected).all()
    assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()
    os.remove('model.tmp')


def test_train_pz():
    config_dict = {'run_params': {'zmin': 0.0,
                                  'zmax': 3.0, 'nzbins': 301,
                                  'inform_options': {'save_train': True,
                                                     'modelfile': 'model.tmp'}
                                  }}
    zb_expected = np.repeat(0.1445183, 10)
    pdf_expected = np.zeros(shape=(301, ))
    pdf_expected[10:16] = [7, 23, 8, 23, 26, 13]
    pz_algo = trainZ.trainZ
    pz_dict, rerun_pz_dict = one_algo(pz_algo, config_dict)
    print(pz_dict['zmode'])
    assert np.isclose(pz_dict['zmode'], zb_expected).all()
    assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()

def test_delight():
    with open("./tests/delightPZ.yaml", "r") as f:
        config_dict=yaml.safe_load(f)
    pz_algo = delightPZ.delightPZ
    pz_dict, rerun_pz_dict = one_algo(pz_algo, config_dict)
    zb_expected = np.array([0.17, 0.01, -1., -1., 0.01, -1., -1., -1., 0.01, 0.01])
    assert np.isclose(pz_dict['zmode'], zb_expected).all()
    assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()
    
def test_catch_bad_bands():
    params = copy.deepcopy(flexzboost.def_param)
    params['run_params']['bands'] = 'u,g,r,i,z,y'
    with pytest.raises(ValueError):
        flexzboost.FZBoost(test_base_yaml, params)

    params = copy.deepcopy(sklearn_nn.def_param)
    params['run_params']['bands'] = 'u,g,r,i,z,y'
    with pytest.raises(ValueError) as errinfo:
        sklearn_nn.simpleNN(test_base_yaml, params)


def test_bpz_lite():
    cdict = {'run_params': {'zmin': 0.0, 'zmax': 3.0,
                            'dz': 0.01,
                            'nzbins': 301,
                            'data_path': None,
                            'columns_file':
                                "./examples/estimation/configs/test_bpz.columns",
                            'spectra_file': "SED/CWWSB4.list",
                            'madau_flag': 'no',
                            'bands': 'ugrizy',
                            'prior_band': 'i',
                            'prior_file': 'hdfn_gen',
                            'p_min': 0.005,
                            'gauss_kernel': 0.0,
                            'zp_errors':
                                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                            'mag_err_min': 0.005,
                            'inform_options': {'save_train': True,
                                               'modelfile': 'model.out'
                                               }}}
    zb_expected = np.array([0.18, 2.88, 0.14, 0.21, 2.97, 0.18, 0.23, 0.23,
                            2.98, 2.92])
    pz_algo = bpz_lite.BPZ_lite
    pz_dict, rerun_pz_dict = one_algo(pz_algo, cdict)
    print(pz_dict['zmode'])
    assert np.isclose(pz_dict['zmode'], zb_expected).all()
    assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()


def test_bpz_lite_wkernel_flatprior():
    cdict = {'run_params': {'zmin': 0.0, 'zmax': 3.0,
                            'dz': 0.01,
                            'nzbins': 301,
                            'data_path': None,
                            'columns_file':
                                "./examples/estimation/configs/test_bpz.columns",
                            'spectra_file': "SED/CWWSB4.list",
                            'madau_flag': 'no',
                            'bands': 'ugrizy',
                            'prior_band': 'i',
                            'prior_file': 'flat',
                            'p_min': 0.005,
                            'gauss_kernel': 0.1,
                            'zp_errors':
                                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                            'mag_err_min': 0.005,
                            'inform_options': {'save_train': True,
                                               'modelfile': 'model.out'
                                               }}}
    zb_expected = np.array([0.18, 2.88, 0.12, 0.15, 2.97, 2.78, 0.11, 0.19,
                            2.98, 2.92])
    pz_algo = bpz_lite.BPZ_lite
    pz_dict, rerun_pz_dict = one_algo(pz_algo, cdict)
    print(pz_dict['zmode'])
    assert np.isclose(pz_dict['zmode'], zb_expected).all()
    assert np.isclose(pz_dict['zmode'], rerun_pz_dict['zmode']).all()


def test_missing_modelfile_keyword():
    config_dict = {'run_params': {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                                  'trainfrac': 0.75, 'bumpmin': 0.02,
                                  'bumpmax': 0.35, 'nbump': 3,
                                  'sharpmin': 0.7, 'sharpmax': 2.1,
                                  'nsharp': 3, 'max_basis': 35,
                                  'basis_system': 'cosine',
                                  'regression_params': {'max_depth': 8,
                                                        'objective':
                                                        'reg:squarederror'},
                                  'inform_options': {'load_train': True}
                                  }}
    pz_algo = flexzboost.FZBoost
    pz = pz_algo(test_base_yaml, config_dict)
    with pytest.raises(KeyError):
        pz.load_pretrained_model()


def test_wrong_modelfile_keyword():
    config_dict = {'run_params': {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                                  'trainfrac': 0.75, 'bumpmin': 0.02,
                                  'bumpmax': 0.35, 'nbump': 3,
                                  'sharpmin': 0.7, 'sharpmax': 2.1,
                                  'nsharp': 3, 'max_basis': 35,
                                  'basis_system': 'cosine',
                                  'regression_params': {'max_depth': 8,
                                                        'objective':
                                                        'reg:squarederror'},
                                  'inform_options': {'save_train': True,
                                                     'modelfile': 'nonexist.pkl'}
                                  }}
    pz_algo = flexzboost.FZBoost
    pz = pz_algo(test_base_yaml, config_dict)
    with pytest.raises(FileNotFoundError):
        pz.load_pretrained_model()
