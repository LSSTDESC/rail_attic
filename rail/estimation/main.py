import sys
import yaml
# import genericpipev2 as npipe
from estimator import Estimator as BaseEstimation
from utils import base_yaml
import algos

def main(argv):
    if len(argv) != 2:
        print(len(argv))
        print("Usage: main <yaml file>")
        exit()
    input_yaml = argv[1]
    name = input_yaml.split("/")[-1].split(".")[0]
#     name = input_yaml[:-5]

    with open(input_yaml, 'r') as f:
        config_dict = yaml.safe_load(f)

    print(config_dict)
    run_dict = config_dict#['run_params']
#     name =  run_dict['class_name']

    try:
#         run_dict['class_name'] = npipe.Tomographer._find_subclass(name)
        run_dict['class_name'] = BaseEstimation._find_subclass(name)
    except KeyError:
        raise ValueError(f"Class name {name} for PZ code is not defined")

#     code = npipe.Tomographer._find_subclass(name)
    code = BaseEstimation._find_subclass(name)
    print(f"code name: {code}")

    pz = code(run_dict)
    
    pz.train()

    pz.run_photoz()

    pz.write_out()
    

    print("finished")

if __name__=="__main__":
    main(sys.argv)
