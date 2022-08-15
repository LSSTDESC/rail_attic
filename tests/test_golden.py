import os

import ceci
import rail
from rail.core.stage import RailStage
from rail.core.utilStages import ColumnMapper, TableConverter
from rail.creation.degradation import LSSTErrorModel
from rail.creation.engines.flowEngine import FlowCreator


def test_goldenspike():
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

    RAIL_DIR = os.path.join(os.path.dirname(rail.__file__), "..")
    flow_file = os.path.join(RAIL_DIR, "examples/goldenspike/data/pretrained_flow.pkl")
    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"mag_{band}_lsst_err": f"mag_err_{band}_lsst" for band in bands}

    flow_creator_test = FlowCreator.make_stage(
        name="flow_creator_test", flow=flow_file, n_samples=50
    )

    lsst_error_model_test = LSSTErrorModel.make_stage(
        name="lsst_error_model_test", bandNames=band_dict
    )

    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )

    table_conv_test = TableConverter.make_stage(
        name="table_conv_test", output_format="numpyDict", seed=12345
    )

    pipe = ceci.Pipeline.interactive()
    stages = [
        flow_creator_test,
        lsst_error_model_test,
        col_remapper_test,
        table_conv_test,
    ]
    for stage in stages:
        pipe.add_stage(stage)

    lsst_error_model_test.connect_input(flow_creator_test)
    col_remapper_test.connect_input(lsst_error_model_test)
    table_conv_test.connect_input(col_remapper_test)

    pipe.initialize(
        dict(flow=flow_file), dict(output_dir=".", log_dir=".", resume=False), None
    )

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
