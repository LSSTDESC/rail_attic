import os

import ceci
import numpy as np

import rail
from rail.core.data import TableHandle
from rail.core.stage import RailPipeline, RailStage
from rail.core.utils import RAILDIR
from rail.core.utilStages import ColumnMapper, TableConverter
from rail.creation.degradation.lsst_error_model import LSSTErrorModel
from rail.creation.degradation.quantityCut import QuantityCut

def test_pipeline():
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    input_file = os.path.join(RAILDIR, "rail/examples_data/goldenspike_data/data//test_flow_data.pq")
    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"mag_{band}_lsst_err": f"mag_err_{band}_lsst" for band in bands}
    post_grid = [float(x) for x in np.linspace(0.0, 5, 21)]

    lsst_error_model_test = LSSTErrorModel.make_stage(name="lsst_error_model_test", bandNames=band_dict)

    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )

    table_conv_test = TableConverter.make_stage(name="table_conv_test", output_format="numpyDict", seed=12345)

    pipe = ceci.Pipeline.interactive()
    stages = [
        lsst_error_model_test,
        col_remapper_test,
        table_conv_test,
    ]
    for stage in stages:
        pipe.add_stage(stage)

    col_remapper_test.connect_input(lsst_error_model_test)
    table_conv_test.connect_input(col_remapper_test)

    pipe.initialize(dict(input=input_file), dict(output_dir=".", log_dir=".", resume=False), None)

    pipe.save("stage.yaml")

    pr = ceci.Pipeline.read("stage.yaml")
    pr.run()

    os.remove("stage.yaml")
    os.remove("stage_config.yml")

    outputs = pr.find_all_outputs()
    for output_ in outputs.values():
        try:
            os.remove(output_)
        except FileNotFoundError:
            pass
    logfiles = [f"{stage.instance_name}.out" for stage in pr.stages]
    for logfile_ in logfiles:
        try:
            os.remove(logfile_)
        except FileNotFoundError:
            pass


def test_golden_v2():
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()
    pipe = RailPipeline()

    input_file = os.path.join(RAILDIR, "rail/examples_data/goldenspike_data/data//test_flow_data.pq")
    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"mag_{band}_lsst_err": f"mag_err_{band}_lsst" for band in bands}
    post_grid = [float(x) for x in np.linspace(0.0, 5, 21)]

    pipe.lsst_error_model_test = LSSTErrorModel.build(
        bandNames=band_dict,
    )

    pipe.col_remapper_test = ColumnMapper.build(
        connections=dict(input=pipe.lsst_error_model_test.io.output),
        hdf5_groupname="",
        columns=rename_dict,
    )

    pipe.table_conv_test = TableConverter.build(
        connections=dict(input=pipe.col_remapper_test.io.output),
        output_format="numpyDict",
        seed=12345,
    )

    pipe.initialize(dict(input=input_file), dict(output_dir=".", log_dir=".", resume=False), None)
    pipe.save("stage.yaml")

    pr = ceci.Pipeline.read("stage.yaml")
    pr.run()
