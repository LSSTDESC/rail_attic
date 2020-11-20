import sys
import os
import yaml
from rail.estimation.utils import initialize_writeout, iter_chunk_hdf5_data
from rail.estimation.utils import write_out_chunk, finalize_writeout
from rail.estimation.estimator import Estimator


# Note: This is where 'base.yaml' actually belongs, but how to make it so
def main(argv):
    if len(argv) == 2:
        # this is in case hiding the base yaml is wanted
        base_config = 'base.yaml'
        input_yaml = argv[1]
    elif len(argv) == 3:
        input_yaml = argv[1]
        base_config = argv[2]
    else:
        print(len(argv))
        print("Usage: main <config yaml file> [base config yaml]")
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
    if 'run_name' in run_dict['run_params']:
        outfile = run_dict['run_params']['run_name'] + '.hdf5'
    else:
        outfile = 'output.hdf5'

    saveloc = os.path.join(pz.outpath, name, outfile)

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
