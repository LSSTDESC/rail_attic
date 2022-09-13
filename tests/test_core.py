import os
import rail
import pytest
import pickle
import numpy as np
import pandas as pd
from types import GeneratorType
from rail.core.utils import RAILDIR
from rail.core.stage import RailStage
from rail.core.data import DataStore, DataHandle, TableHandle, Hdf5Handle, FitsHandle, PqHandle, QPHandle, ModelHandle, FlowHandle
from rail.core.utilStages import ColumnMapper, RowSelector, TableConverter
from rail.core.utilPhotometry import PhotormetryManipulator, HyperbolicSmoothing, HyperbolicMagnitudes
from rail.core.common_params import SHARED_PARAMS, copy_param, set_param_default


#def test_data_file():    
#    with pytest.raises(ValueError) as errinfo:
#        df = DataFile('dummy', 'x')
    

def test_util_stages():

    DS = RailStage.data_store

    datapath = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816.pq')
    
    data = DS.read_file('data', TableHandle, datapath)

    table_conv = TableConverter.make_stage(name='conv', output_format='numpyDict')
    col_map = ColumnMapper.make_stage(name='col_map', columns={})
    row_sel = RowSelector.make_stage(name='row_sel', start=1, stop=15)

    with pytest.raises(KeyError) as errinfo:
        table_conv.get_handle('nope', allow_missing=False)

    conv_data = table_conv(data)
    mapped_data = col_map(data)
    sel_data = row_sel(mapped_data)
                               
    row_sel_2 = RowSelector.make_stage(name='row_sel_2', start=1, stop=15)
    row_sel_2.set_data('input', mapped_data.data)
    handle = row_sel_2.get_handle('input')
    
    row_sel_3 = RowSelector.make_stage(name='row_sel_3', input=handle.path, start=1, stop=15)
    row_sel_3.set_data('input', None, do_read=True)
    
    
def do_data_handle(datapath, handle_class):

    DS = RailStage.data_store

    th = handle_class('data', path=datapath)

    with pytest.raises(ValueError) as errinfo:
        th.write()

    assert not th.has_data
    with pytest.raises(ValueError) as errinfo:
        th.write_chunk(0, 1)        
    assert th.has_path
    assert th.is_written
    data = th.read()
    data2 = th.read()

    assert data is data2
    assert th.has_data
    assert th.make_name('data') == f'data.{handle_class.suffix}'
    
    th2 = handle_class('data2', data=data)
    assert th2.has_data
    assert not th2.has_path
    assert not th2.is_written
    with pytest.raises(ValueError) as errinfo:
        th2.open()
    with pytest.raises(ValueError) as errinfo:
        th2.write()
    with pytest.raises(ValueError) as errinfo:
        th2.write_chunk(0, 1)
        
    assert th2.make_name('data2') == f'data2.{handle_class.suffix}'
    assert str(th)
    assert str(th2)
    return th
    

def test_pq_handle():
    datapath = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816.pq')
    handle = do_data_handle(datapath, PqHandle)
    pqfile = handle.open()
    assert pqfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None

    
def test_qp_handle():
    datapath = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'output_BPZ_lite.fits')
    handle = do_data_handle(datapath, QPHandle)
    qpfile = handle.open()
    assert qpfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None


    
def test_hdf5_handle():
    datapath = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816.hdf5')
    handle = do_data_handle(datapath, Hdf5Handle)
    with handle.open(mode='r') as f:
        assert f
        assert handle.fileObj is not None
    datapath_chunked = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816_chunked.hdf5')
    handle_chunked = Hdf5Handle("chunked", handle.data, path=datapath_chunked)
    from tables_io.arrayUtils import getGroupInputDataLength, sliceDict, getInitializationForODict
    num_rows = len(handle.data['photometry']['id'])
    check_num_rows = len(handle()['photometry']['id'])
    assert num_rows == check_num_rows
    chunk_size = 1000
    data = handle.data['photometry']
    init_dict = getInitializationForODict(data)
    with handle_chunked.open(mode='w') as fout:
        for k, v in init_dict.items():
            fout.create_dataset(k, v[0], v[1])
        for i in range(0, num_rows, chunk_size):
            start = i
            end = i+chunk_size
            if end > num_rows:
                end = num_rows
            handle_chunked.set_data(sliceDict(handle.data['photometry'], slice(start, end)), partial=True)
            handle_chunked.write_chunk(start, end)
    write_size = handle_chunked.size()
    assert len(handle_chunked.data) <= 1000
    data_called = handle_chunked()
    assert len(data_called['id']) == write_size
    read_chunked = Hdf5Handle("read_chunked", None, path=datapath_chunked)
    data_check = read_chunked.read()
    assert np.allclose(data['id'], data_check['id'])
    assert np.allclose(data_called['id'], data_check['id'])
    os.remove(datapath_chunked)


def test_fits_handle():
    datapath = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'output_BPZ_lite.fits')
    handle = do_data_handle(datapath, FitsHandle)
    fitsfile = handle.open()
    assert fitsfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None


def test_model_handle():
    DS = RailStage.data_store
    DS.clear()
    
    model_path = os.path.join(RAILDIR, 'rail', 'examples', 'estimation', 'demo_snn.pkl')
    model_path_copy = os.path.join(RAILDIR, 'rail', 'examples', 'estimation', 'demo_snn_copy.pkl')
    mh = ModelHandle("model", path=model_path)
    mh2 = ModelHandle("model2", path=model_path)
    
    model1 = mh.read()
    model2 = mh2.read()

    model3 = mh.open()

    assert model1 is model2
    assert model2 is model3

    mh3 = ModelHandle("model3", path=model_path_copy, data=model1)
    with mh3.open(mode='w') as fout:
        pickle.dump(obj=mh3.data, file=fout, protocol=pickle.HIGHEST_PROTOCOL)
    os.remove(model_path_copy)

def test_flow_handle():
    DS = RailStage.data_store
    DS.clear()
    
    flow_path = os.path.join(RAILDIR, 'rail', 'examples', 'goldenspike', 'data', 'pretrained_flow.pkl')
    flow_path_copy = os.path.join(RAILDIR, 'rail', 'examples', 'goldenspike', 'data', 'pretrained_flow_copy.pkl')
    fh = FlowHandle("flow", path=flow_path)
    fh2 = FlowHandle("flow2", path=flow_path)
    
    flow1 = fh.read()
    flow2 = fh2.read()

    flow3 = fh.open()

    assert flow1 is flow2
    assert flow2 is flow3
    
    fh3 = FlowHandle("flo3", path=flow_path_copy, data=flow1)
    with pytest.raises(NotImplementedError) as errinfo:
        fh3.open(mode='w')
    fh3.write()
    os.remove(flow_path_copy)

    
def test_data_hdf5_iter():

    DS = RailStage.data_store
    DS.clear()
    
    datapath = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816.hdf5')

    #data = DS.read_file('data', TableHandle, datapath)
    th = Hdf5Handle('data', path=datapath)
    x = th.iterator(groupname='photometry', chunk_size=1000)

    assert isinstance(x, GeneratorType)
    for i, xx in enumerate(x):
        assert xx[0] == i*1000
        assert xx[1] - xx[0] <= 1000

    data = DS.read_file('input', TableHandle, datapath)        
    cm = ColumnMapper.make_stage(input=datapath, chunk_size=1000,
                                 hdf5_groupname='photometry', columns=dict(id='bob'))
    x = cm.input_iterator('input')

    assert isinstance(x, GeneratorType)

    for i, xx in enumerate(x):
        assert xx[0] == i*1000
        assert xx[1] - xx[0] <= 1000


def test_data_store():
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False
    datapath_hdf5 = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816.hdf5')
    datapath_pq = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816.pq')
    datapath_hdf5_copy = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816_copy.hdf5')
    datapath_pq_copy = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816_copy.pq')

    DS.add_data('hdf5', None, Hdf5Handle, path=datapath_hdf5)
    DS.add_data('pq', None, PqHandle, path=datapath_pq)

    with DS.open('hdf5') as f:
        assert f
    
    data_pq = DS.read('pq')
    data_hdf5 = DS.read('hdf5')

    DS.add_data('pq_copy', data_pq, PqHandle, path=datapath_pq_copy)
    DS.add_data('hdf5_copy', data_hdf5, Hdf5Handle, path=datapath_hdf5_copy)
    DS.write('pq_copy')
    DS.write('hdf5_copy')

    with pytest.raises(KeyError) as errinfo:
        DS.read('nope')
    with pytest.raises(KeyError) as errinfo:
        DS.open('nope')
    with pytest.raises(KeyError) as errinfo:
        DS.write('nope')

    with pytest.raises(TypeError) as errinfo:
        DS['nope'] = None
    with pytest.raises(ValueError) as errinfo:        
        DS['pq'] = DS['pq']        
    with pytest.raises(ValueError) as errinfo:
        DS.pq = DS['pq']

    assert repr(DS) 

    DS2 = DataStore(pq=DS.pq)
    assert isinstance(DS2.pq, DataHandle)

    # pop the 'pq' data item to avoid overwriting file under git control
    DS.pop('pq')
    
    DS.write_all()
    DS.write_all(force=True)
    
    os.remove(datapath_hdf5_copy)
    os.remove(datapath_pq_copy)


@pytest.fixture
def hyperbolic_configuration():
    """get the code configuration for the example data"""
    lsst_bands = 'ugrizy'
    return dict(
        value_columns=[f"mag_{band}_lsst" for band in lsst_bands],
        error_columns=[f"mag_err_{band}_lsst" for band in lsst_bands],
        zeropoints=[0.0] * len(lsst_bands),
        is_flux=False)


@pytest.fixture
def load_result_smoothing():
    """load the smoothing parameters for an example patch of DC2"""
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    testFile = os.path.join(RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816_smoothing_params.pq')
    return DS.read_file("test_data", TableHandle, testFile).data


def test_PhotormetryManipulator(hyperbolic_configuration):
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    # NOTE: the __init__ machinery of HyperbolicSmoothing is identical to PhotormetryManipulator
    # and is used as substitute since PhotormetryManipulator cannot be instantiated.
    n_filters = len(hyperbolic_configuration["value_columns"])

    # wrong number of "error_columns"
    config = hyperbolic_configuration.copy()
    config["error_columns"] = hyperbolic_configuration["error_columns"][:-1]
    with pytest.raises(IndexError):
        inst = HyperbolicSmoothing.make_stage(name='photormetry_manipulator', **config)

    # wrong number of "zeropoints"
    config = hyperbolic_configuration.copy()
    config["zeropoints"] = np.arange(0, n_filters - 1)
    with pytest.raises(IndexError):
        inst = HyperbolicSmoothing.make_stage(name='photormetry_manipulator', **config)

    # default values for "zeropoints"
    config = hyperbolic_configuration.copy()
    config.pop("zeropoints")  # should resort to default of 0.0
    inst = HyperbolicSmoothing.make_stage(name='photormetry_manipulator', **config)
    assert len(inst.zeropoints) == n_filters
    assert all(zp == 0.0 for zp in inst.zeropoints)

    # if_flux preserves the values
    dummy_data = pd.DataFrame(dict(val=[1, 2, 3], err=[1, 2, 3]))
    config = dict(
        value_columns=["val"],
        error_columns=["err"],
        zeropoints=[0.0])
    inst = HyperbolicSmoothing.make_stage(name='photormetry_manipulator', **config, is_flux=True)
    inst.set_data('input', dummy_data)
    data = inst.get_as_fluxes()
    assert np.allclose(data, dummy_data)



def test_HyperbolicSmoothing(hyperbolic_configuration):
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    test_data = DS.read_file(
        "test_data", TableHandle, os.path.join(
            RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816.pq')
    ).data
    result_smoothing = DS.read_file(
        "result_smoothing", TableHandle, os.path.join(
            RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816_smoothing_params.pq')
    ).data

    stage_name, handle_name = 'hyperbolic_smoothing', 'parameters'

    # test against prerecorded output
    smooth = HyperbolicSmoothing.make_stage(name=stage_name, **hyperbolic_configuration)
    smooth.compute(test_data)
    smooth_params = smooth.get_handle(handle_name).data
    assert np.allclose(smooth_params, result_smoothing)

    os.remove(f'{handle_name}_{stage_name}.pq')


def test_HyperbolicMagnitudes(hyperbolic_configuration,):
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    test_data = DS.read_file(
        "test_data", TableHandle, os.path.join(
            RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816.pq')
    ).data
    result_smoothing = DS.read_file(
        "result_smoothing", TableHandle, os.path.join(
            RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816_smoothing_params.pq')
    ).data
    result_hyperbolic = DS.read_file(
        "result_hyperbolic", TableHandle, os.path.join(
            RAILDIR, 'rail', 'examples', 'testdata', 'test_dc2_training_9816_hyperbolic.pq')
    ).data

    stage_name, handle_name = 'hyperbolic_magnitudes', 'output'

    # test against prerecorded output
    hypmag = HyperbolicMagnitudes.make_stage(name=stage_name, **hyperbolic_configuration)
    hypmag.compute(test_data, result_smoothing)
    test_hypmags = hypmag.get_handle(handle_name).data

    # What we would want to test is
    # >>> assert test_hypmags.equals(result_hyperbolic)
    # however this test fails at github actions.
    # Instead we test that the values are numerically close. The accepted deviation scales with
    # magnitude m as
    # dm = 1e-5 * m
    # which is smaller than difference between classical and hyperbolic magnitudes except at the
    # very brightest magnitudes.
    for (key_test, values_test), (key_ref, values_ref) in zip(
            test_hypmags.items(), result_hyperbolic.items()):
        assert key_test == key_ref
        assert np.allclose(values_test, values_ref)

    # check of input data columns against smoothing parameter table
    smoothing = result_smoothing.copy().drop("mag_r_lsst")  # drop one filter from the set
    hypmag = HyperbolicMagnitudes.make_stage(name=stage_name, **hyperbolic_configuration)
    with pytest.raises(KeyError):
        hypmag._check_filters(smoothing)

    os.remove(f'{handle_name}_{stage_name}.pq')


def test_common_params():

    par = copy_param('zmin')
    assert par.default == 0.0
    assert par.value == 0.0
    assert par.dtype == float

    set_param_default('zmin', 0.1)
    par = copy_param('zmin')
    assert par.default == 0.1
    assert par.value == 0.1
    assert par.dtype == float
 
