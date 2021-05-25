from rail.estimation.utils import check_and_print_params

incomplete_dict = {'run_params': {'zmin': 0.0}}
complete_dict = {'run_params': {'zmin': 0.0, 'zmax': 3.0}}
desc_dict = {'zmin': "zmin: minimum z", 'zmax': "zmax: max z"}


def test_incomplete_dict():
    # check that feeding incomplete dict returns all values
    # in complete_dict, i.e. check adding a key
    new_dict = check_and_print_params(incomplete_dict,
                                      complete_dict,
                                      desc_dict)
    for key in complete_dict['run_params'].keys():
        assert key in new_dict['run_params'].keys()
