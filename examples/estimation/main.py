import sys
import os
import yaml
from rail.fileIO import load_training_data
from rail.fileIO import initialize_writeout, iter_chunk_hdf5_data
from rail.fileIO import write_out_chunk, finalize_writeout
from rail.fileIO import write_qp_output_chunk, initialize_qp_output
from rail.fileIO import qp_reformat_output
from rail.estimation.estimator import Estimator


# Note: This is where 'base.yaml' actually belongs, but how to make it so
def main(argv):
    if len(argv) == 2:
        # this is in case hiding the base yaml is wanted
        input_yaml = argv[1]
        base_config = 'base.yaml'
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
    
    pz.inform_dict = run_dict['run_params']['inform_options']
    if pz.inform_dict['load_model']:
        # note: specific options set in subclasss func def
        pz.load_pretrained_model()
    else:
        trainfile = pz.trainfile
        train_fmt = trainfile.split(".")[-1]
        training_data = load_training_data(trainfile, 
                                           train_fmt,
                                           pz.groupname)
        pz.inform(training_data)

    if 'run_name' in run_dict['run_params']:
        outfile = run_dict['run_params']['run_name'] + '.hdf5'
        tmpfile = "temp_" + outfile
    else:
        outfile = 'output.hdf5'

    if pz.output_format == 'qp':
        tmploc = os.path.join(pz.outpath, name, tmpfile)
        outfile = run_dict['run_params']['run_name'] + "_qp.hdf5"
    saveloc = os.path.join(pz.outpath, name, outfile)

    if pz.output_format == 'qp':
        initialize_qp_output(saveloc)
    else:
        outf = initialize_writeout(saveloc, pz.num_rows, pz.nzbins)

    for chunk, (start, end, data) in enumerate(iter_chunk_hdf5_data(pz.testfile,
                                                                    pz._chunk_size,
                                                                    'photometry')):
        pz_data_chunk = pz.estimate(data)
        if pz.output_format == 'qp':
            write_qp_output_chunk(tmploc, saveloc, pz_data_chunk, chunk)
        else:
            write_out_chunk(outf, pz_data_chunk, start, end)
        print("writing " + name + f"[{start}:{end}]")

    num_chunks = end // pz._chunk_size
    if end % pz._chunk_size > 0:
        num_chunks += 1

    if pz.output_format == 'qp':
        qp_reformat_output(tmploc, saveloc, num_chunks)
    else:
        finalize_writeout(outf, pz.zgrid)

    print("finished")


if __name__ == "__main__":
    main(sys.argv)
