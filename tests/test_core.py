import os
import rail
from rail.core.stage import RailStage
from rail.core.data import TableHandle
from rail.core.utilStages import ColumnMapper, RowSelector, TableConverter


def test_util_stages():

    DS = RailStage.data_store

    raildir = os.path.dirname(rail.__file__)
    datapath = os.path.join(raildir, '..', 'tests', 'data', 'test_dc2_training_9816.hdf5')
    
    data = DS.read_file('data', TableHandle, datapath)
    
    table_conv = TableConverter.make_stage(name='conv', hdf5_groupname='photometry', output_format='pandasDataFrame')
    col_map = ColumnMapper.make_stage(name='col_map', columns={})
    row_sel = RowSelector.make_stage(name='row_sel', start=1, stop=15)

    conv_data = table_conv(data)
    mapped_data = col_map(conv_data)
    sel_data = row_sel(mapped_data)
                               
