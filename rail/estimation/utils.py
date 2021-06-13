def check_and_print_params(config_dict, default_param_dict, desc_dict):
    """
    Function that tests that all required parameters are in file, if
    not, then add default value.  It also prints a short description
    of each parameter along with the value being used.
    Inputs:
    -------
    config_dict: dict
      dictionary of input params
    Output:
    -------
    updated_dict: dict
      updated parameter dictionary
    """
    print("summary of input parameters:")
    params_dict = default_param_dict['run_params']
    for key in params_dict.keys():
        if key not in config_dict['run_params'].keys():
            print(f"param {key} not included in config, using default value")
            config_dict['run_params'][key] = params_dict[key]
        print(f"{key}: {desc_dict[key]}")
        print(f"{key} value: {config_dict['run_params'][key]}")
    return config_dict
