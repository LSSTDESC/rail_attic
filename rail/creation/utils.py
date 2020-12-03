param_defaults = {}
param_defaults['bands'] = ['u', 'g', 'r', 'i', 'z', 'y']

# err params from Table 2 from https://arxiv.org/pdf/0805.2366.pdf
# 10 yr limiting mags from https://www.lsst.org/scientists/keynumbers
# currently these limitting mags are set 2 mags dimmer than actual LSST vals
param_defaults['err_params'] = {'gamma_u': 0.038,
                                'gamma_g': 0.039,
                                'gamma_r': 0.039,
                                'gamma_i': 0.039,
                                'gamma_z': 0.039,
                                'gamma_y': 0.039,
                                'm5_u': 28.1,
                                'm5_g': 29.4,
                                'm5_r': 29.5,
                                'm5_i': 28.8,
                                'm5_z': 28.1,
                                'm5_y': 26.9}

zinfo = {'zmin': 0.,
         'zmax': 2.,
         'dz': 0.02}
