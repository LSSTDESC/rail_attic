import os
import rail
import pytest
import pickle
import numpy as np
from types import GeneratorType
from rail.core.stage import RailStage
from rail.core.data import DataStore, DataHandle, TableHandle, Hdf5Handle, PqHandle, QPHandle, ModelHandle, FlowHandle
from rail.core.utilStages import ColumnMapper, RowSelector, TableConverter


#def test_data_file():    
#    with pytest.raises(ValueError) as errinfo:
#        df = DataFile('dummy', 'x')
    

def test_util_stages():

    DS = RailStage.data_store

    raildir = os.path.dirname(rail.__file__)
    datapath = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816.pq')
    
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

    raildir = os.path.dirname(rail.__file__)
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
    raildir = os.path.dirname(rail.__file__)
    datapath = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816.pq')
    handle = do_data_handle(datapath, PqHandle)
    pqfile = handle.open()
    assert pqfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None

    
def test_qp_handle():
    raildir = os.path.dirname(rail.__file__)
    datapath = os.path.join(raildir, '..', 'tests', 'data', 'output_BPZ_lite.fits')
    handle = do_data_handle(datapath, QPHandle)
    qpfile = handle.open()
    assert qpfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None


    
def test_hdf5_handle():
    raildir = os.path.dirname(rail.__file__)
    datapath = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816.hdf5')
    handle = do_data_handle(datapath, Hdf5Handle)
    with handle.open(mode='r') as f:
        assert f
        assert handle.fileObj is not None
    datapath_chunked = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816_chunked.hdf5')
    handle_chunked = Hdf5Handle("chunked", handle.data, path=datapath_chunked)
    from tables_io.arrayUtils import getGroupInputDataLength, sliceDict, getInitializationForODict
    num_rows = len(handle.data['photometry']['id'])
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
            handle_chunked.data = sliceDict(handle.data['photometry'], slice(start, end))
            handle_chunked.write_chunk(start, end)
    read_chunked = Hdf5Handle("read_chunked", None, path=datapath_chunked)
    data_check = read_chunked.read()
    assert np.allclose(data['id'], data_check['id'])
    os.remove(datapath_chunked)


def test_model_handle():
    DS = RailStage.data_store
    DS.clear()
    
    raildir = os.path.dirname(rail.__file__)
    model_path = os.path.join(raildir, '..', 'examples', 'estimation', 'demo_snn.pkl')
    model_path_copy = os.path.join(raildir, '..', 'examples', 'estimation', 'demo_snn_copy.pkl')
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
    
    raildir = os.path.dirname(rail.__file__)
    flow_path = os.path.join(raildir, '..', 'examples', 'goldenspike', 'data', 'pretrained_flow.pkl')
    flow_path_copy = os.path.join(raildir, '..', 'examples', 'goldenspike', 'data', 'pretrained_flow_copy.pkl')
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
    
    raildir = os.path.dirname(rail.__file__)
    datapath = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816.hdf5')

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
    raildir = os.path.dirname(rail.__file__)
    datapath_hdf5 = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816.hdf5')
    datapath_pq = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816.pq')
    datapath_hdf5_copy = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816_copy.hdf5')
    datapath_pq_copy = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816_copy.pq')

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
