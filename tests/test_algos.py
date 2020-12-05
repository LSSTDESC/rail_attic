import numpy as np
from rail.estimation.utils import iter_chunk_hdf5_data
from rail.estimation.algos import randomPZ, sklearn_nn, flexzboost, trainZ


test_base_yaml = './tests/test.yaml'


def one_algo(single_estimator, single_input):
    """
    A basic test of running an estimator subclass
    """

    pz = single_estimator(test_base_yaml, single_input)
    pz.inform()
    # set chunk size to pz.num_rows to ensure all run in one chunk
    for _, end, data in iter_chunk_hdf5_data(pz.testfile, pz.num_rows,
                                             pz.hdf5_groupname):
        pz_dict = pz.estimate(data)
    assert end == pz.num_rows
    xinputs = single_input['run_params']
    assert len(pz.zgrid) == np.int32(xinputs['nzbins'])
    return pz_dict


def test_random_pz():
    config_dict = {'run_params': {'rand_width': 0.025, 'rand_zmin': 0.0,
                                  'rand_zmax': 3.0, 'nzbins': 301}}
    zb_expected = np.array([1.969, 2.865, 2.913, 0.293, 0.722, 2.606, 1.642,
                            2.157, 2.777, 1.851])
    pz_algo = randomPZ.randomPZ
    pz_dict = one_algo(pz_algo, config_dict)
    assert np.isclose(pz_dict['zmode'], zb_expected).all()


def test_simple_nn():
    config_dict = {'run_params': {'width': 0.025, 'zmin': 0.0, 'zmax': 3.0,
                                  'nzbins': 301}}
    zb_expected = np.array([0.133, 0.123, 0.085, 0.145, 0.123, 0.155, 0.136,
                            0.157, 0.128, 0.13])
    pz_algo = sklearn_nn.simpleNN
    pz_dict = one_algo(pz_algo, config_dict)
    assert np.isclose(pz_dict['zmode'], zb_expected).all()


def test_flexzboost():
    config_dict = {'run_params': {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                                  'trainfrac': 0.75, 'bumpmin': 0.02,
                                  'bumpmax': 0.35, 'nbump': 3,
                                  'sharpmin': 0.7, 'sharpmax': 2.1,
                                  'nsharp': 3, 'max_basis': 35,
                                  'basis_system': 'cosine',
                                  'regression_params': {'max_depth': 8,
                                                        'objective':
                                                        'reg:squarederror'}
                                  }}
    zb_expected = np.array([0.13, 0.13, 0.13, 0.12, 0.12, 0.13, 0.12, 0.13,
                            0.12, 0.12])
    pz_algo = flexzboost.FZBoost
    pz_dict = one_algo(pz_algo, config_dict)
    assert np.isclose(pz_dict['zmode'], zb_expected).all()


def test_train_pz():
    config_dict = {'run_params': {'zmin': 0.0,
                                  'zmax': 3.0, 'nzbins': 301}}
    zb_expected = np.repeat(0.1445183, 10)
    pdf_expected = np.zeros(shape=(301, ))
    pdf_expected[10:16] = [7, 23, 8, 23, 26, 13]
    pz_algo = trainZ.trainZ
    pz_dict = one_algo(pz_algo, config_dict)
    print(pz_dict['zmode'])
    assert np.isclose(pz_dict['zmode'], zb_expected).all()
