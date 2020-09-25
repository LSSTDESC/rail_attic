import sys
import os
import yaml
import rail
from rail.estimation.utils import *
from rail.estimation.estimator import Estimator


# Note: This is where 'base.yaml' actually belongs, but how to make it so
def main(argv):
    if len(argv) == 2:
        # this is in case hiding the base yaml is wanted
        base_config = 'base.yaml'
        input_yaml = argv[1]
    elif len(argv) == 3:
        base_config = argv[1]
        input_yaml = argv[2]
    else:
        print(len(argv))
        print("Usage: main <yaml file>")
        sys.exit()

    with open(input_yaml, 'r') as f:
        run_dict = yaml.safe_load(f)

    name = run_dict['run_params']['class_name']

    try:
        Estimator._find_subclass(name)
    except KeyError:
        raise ValueError(f"Class name {name} for PZ code is not defined")

    code = Estimator._find_subclass(name)
    print(f"code name: {name}")

    pz = code(base_config, run_dict)
    pz.inform()
    outfile = f"{name}.hdf5"

    if 'output_dir' in run_dict['run_params']:
        saveloc = os.path.join(pz.outpath,
        run_dict['run_params']['output_dir'], outfile)
    else:
        saveloc = os.path.join(pz.outpath, outfile)
    outf = initialize_writeout(saveloc, pz.num_rows, pz.nzbins)

    for start, end, data in iter_chunk_hdf5_data(pz.testfile,
                                                 pz._chunk_size,
                                                 'photometry'):
        pz_dict = pz.estimate(data)
        write_out_chunk(outf, pz_dict, start, end)
        print("writing " + name + f"[{start}:{end}]")

    finalize_writeout(outf, pz.zgrid)

    print("finished")


if __name__ == "__main__":
    main(sys.argv)
