import sys
import yaml
from rail.estimation.estimator import Estimator as BaseEstimation
from utils import base_yaml
from algos import *

#Note: This is where 'base.yaml' actually belongs, but how to make it so 
def main(argv):
    if len(argv) == 2:
        #this is in case hiding the base yaml is wanted
        base_yaml =  os.path.join(os.path.dirname(inspect.getfile(rail)),'estimation/base.yaml')
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

    print(config_dict)
    run_dict = config_dict

    try:
        run_dict['class_name'] = BaseEstimation._find_subclass(name)
    except KeyError:
        raise ValueError(f"Class name {name} for PZ code is not defined")

    code = BaseEstimation._find_subclass(name)
    print(f"code name: {code}")

    pz = code(base_config,run_dict)
    
    pz.train()

    pz.run_photoz()

    pz.write_out()
    
    print("finished")

if __name__=="__main__":
    main(sys.argv)
