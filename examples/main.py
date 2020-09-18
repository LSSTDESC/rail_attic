import os, sys, inspect
import yaml
import rail
from rail.estimation.utils import *
from rail.estimation.estimator import Estimator


#Note: This is where 'base.yaml' actually belongs, but how to make it so 
def main(argv):
    if len(argv) == 2:
        #this is in case hiding the base yaml is wanted
        base_config =  os.path.join(os.path.dirname(inspect.getfile(rail)), '../examples/base.yaml')
        input_yaml = argv[1]
    elif len(argv) == 3:
        base_config = argv[1]
        input_yaml = argv[2]
    else:
        print(len(argv))
        print("Usage: main <yaml file>")
        sys.exit()
    name = input_yaml.split("/")[-1].split(".")[0]

    with open(input_yaml, 'r') as f:
        config_dict = yaml.safe_load(f)
    # with open(input_yaml, 'r') as f:
    #     base_config = yaml.safe_load(f)

    print(config_dict)
    run_dict = config_dict

    try:
        run_dict['class_name'] = Estimator._find_subclass(name)
    except KeyError:
        raise ValueError(f"Class name {name} for PZ code is not defined")

    code = Estimator._find_subclass(name)
    print(f"code name: {code}")

    pz = code(base_config, run_dict)
    
    pz.train()

    outf = initialize_writeout(pz.saveloc, pz.num_rows, pz.nzbins)
    
    for start, end, data in iter_chunk_hdf5_data(pz.testfile, pz._chunk_size,
                                                 'photometry'):
        pz_dict = pz.estimate(data)
        write_out_chunk(outf, pz_dict, start, end)
        print("finished " + name)

    finalize_writeout(outf, pz.zgrid)
        
    print("finished")

if __name__=="__main__":
    main(sys.argv)
