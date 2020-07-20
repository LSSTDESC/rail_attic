import sys
import yaml
import genericpipev2 as npipe

def main(argv):
    if len(argv) != 2:
        print(len(argv))
        print("Usage: main <yaml file>")
        exit()
    input_yaml = argv[1]

    with open(input_yaml, 'r') as f:
        base_dict = yaml.safe_load(f)

    print(base_dict)
    run_dict = base_dict['run_params']
    name =  run_dict['class_name']

    try:
#         run_dict['class_name'] = npipe.Tomographer._find_subclass(name)
        run_dict['class_name'] = npipe.BaseEstimation._find_subclass(name)
    except KeyError:
        raise ValueError(f"Class name {name} for PZ code is not defined")

#     code = npipe.Tomographer._find_subclass(name)
    code = npipe.BaseEstimation._find_subclass(name)
    print(f"code name: {code}")

    pz = code(base_dict)
    
    pz.train()

    pz.run_photoz()

    pz.write_out()
    

    print("finished")

if __name__=="__main__":
    main(sys.argv)
