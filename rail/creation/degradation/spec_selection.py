""" LSST Model for photometric errors """

from numbers import Number
from typing import Iterable, List, Optional, Tuple

import numpy as np
from rail.creation.degradation import Degrader


class SpecSelection(Degrader):
    """
    A toy degrader which only print out a given column

    Parameters
    ----------
    colname: string, the name of the spec survey
    """

    name = 'specselection'
    config_options = Degrader.config_options.copy()
    config_options.update(**{"survey_name": "GAMA"})

    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)
        self.selections = {'all': self.selection_all,
                            'GAMA': self.selection_GAMA,
                          'BOSS': self.selection_BOSS,
                          'WiggleZ': self.selection_WiggleZ}
        # validate the settings
        self._validate_settings()


    def _validate_settings(self):
        """
        Validate all the settings.
        """

        # check that highSNR is boolean
        
        if isinstance(self.config["survey_name"], str) is not True:
            raise TypeError("survey name must be a string.")
        elif self.config["survey_name"] not in self.selections.keys():
            raise ValueError("survey name must be in "+str([*self.selections]))

            
    
    
    def selection_all(self, data):
        return
        
    def selection_GAMA(self, data):
        self.mask *= (data["r"] < 17.7)
        return
    
    def selection_BOSS(self, data):
        mask = (np.abs(data["g"]) < 99.0) & (np.abs(data["r"]) < 99.0)
        mask &= (np.abs(data["r"]) < 99.0) & (np.abs(data["i"]) < 99.0)
        # cut quantities (unchanged)
        c_p = 0.7 * (data["g"]-data["r"]) + 1.2 * (data["r"]-data["i"] - 0.18)
        c_r = (data["r"]-data["i"]) - (data["g"]-data["r"]) / 4.0 - 0.18
        d_r = (data["r"]-data["i"]) - (data["g"]-data["r"]) / 8.0
        # defining the LOWZ sample
        # we cannot apply the r_psf - r_cmod cut
        low_z = (
            (data["r"] > 16.0) &
            (data["r"] < 20.0) &  # 19.6
            (np.abs(c_r) < 0.2) &
            (data["r"] < 13.35 + c_p / 0.3))  # 13.5, 0.3
        # defining the CMASS sample
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = (
            (data["i"] > 17.5) &
            (data["i"] < 20.1) &  # 19.9
            (d_r > 0.55) &
            (data["i"] < 19.98 + 1.6 * (d_r - 0.7)) &  # 19.86, 1.6, 0.8
            ((data["r"]-data["i"]) < 2.0))
        # NOTE: we ignore the CMASS sparse sample
        self.mask *= (low_z | cmass)

        return
    
    def selection_WiggleZ(self, data):
        ###############################################################
        #   based on Drinkwater+10                                    #
        ###############################################################
        # colour masking
        colour_mask = (np.abs(data["g"]) < 99.0) & (np.abs(data["r"]) < 99.0)
        colour_mask &= (np.abs(data["r"]) < 99.0) & (np.abs(data["i"]) < 99.0)
        colour_mask &= (np.abs(data["u"]) < 99.0) & (np.abs(data["i"]) < 99.0)
        colour_mask &= (np.abs(data["g"]) < 99.0) & (np.abs(data["i"]) < 99.0)
        colour_mask &= (np.abs(data["r"]) < 99.0) & (np.abs(data["z"]) < 99.0)
        # photometric cuts
        # we cannot reproduce the FUV, NUV, S/N and position matching cuts
        include = (
            (data["r"] > 20.0) &
            (data["r"] < 22.5))
        exclude = (
            (data["g"] < 22.5) &
            (data["i"] < 21.5) &
            (data["r"]-data["i"] < (data["g"]-data["r"] - 0.1)) &
            (data["r"]-data["i"] < 0.4) &
            (data["g"]-data["r"] > 0.6) &
            (data["r"]-data["z"] < 0.7 * (data["g"]-data["r"])))
        mask = (include & ~exclude)
        # update the internal state
        self.mask *= mask
        return
        
    def run(self):
        """
        Run the toy
        """       
        
        # get the bands and bandNames present in the data
        data = self.get_data('input', allow_missing=True)
        
        self.mask = np.product(~np.isnan(data.to_numpy()), axis=1)        
        
        selection = self.config["survey_name"]
        selection_func = self.selections[selection]
        print("Applying the selection from "+selection+" survey...")
        selection_func(data)
        data_selected = data.iloc[np.where(self.mask==1)[0]]
        
        self.add_data('output', data_selected)


    def __repr__(self):  
        """
        Define how the model is represented and printed.
        """

        selection = self.config["survey_name"]

        # start message
        printMsg = "Applying the selection from "+selection+" survey."

        return printMsg
