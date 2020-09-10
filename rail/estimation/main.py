import sys
import yaml
from estimator import Estimator as BaseEstimation
from utils import base_yaml
import algos

#Note: This is where 'base.yaml' actually belongs, but how to make it so 
def main(argv):
    if len(argv) != 2:
        print(len(argv))
        print("Usage: main <yaml file>")
        sys.exit()
    input_yaml = argv[1]
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

    pz = code(run_dict)
    
    pz.train()

    pz.run_photoz()

    pz.write_out()
    
    print("finished")

if __name__=="__main__":
    main(sys.argv)
