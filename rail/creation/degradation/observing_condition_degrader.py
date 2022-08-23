"""Degrader applied to the magnitude error based on a set of input observing condition maps"""

import numpy as np
import pandas as pd
import healpy as hp
import os
import pickle

from rail.creation.degradation import Degrader
from ceci.config import StageParameter as Param

#### To do list:
### - Tables io
### - Other formats than fits: .hs (needs healsparse), and .npz (needs rubin)


class ObsCondition(Degrader):
    """
    Degrader that generates magnitude errors using LSSTErrorModel
    
    Example: 
    error_model_param = ObsCondition(obs_config_file)
    errModel = LSSTErrorModel(obs_cond = error_model_param)
    data_with_errs = errModel(data)
    
    This function takes a set of observation condition maps and 
    convert them into a data file which is then passed on to 
    LSSTErrorModel for generating magnitude errors based on 
    observing conditions.
    
    Parameters
    ----------
    obs_config_file:
        A configuration file which specifies for each observational
        conditions listed in LSSTErrorModel, the directory of the 
        systematic maps to read in.
        
    Example config_file: see example_obs_config
    """
    
    name = 'ObsCondition'
    config_options = Degrader.config_options.copy()
    config_options.update(
        obs_config_file=Param(
            str, os.path.join(os.path.dirname(__file__),
            "../../../examples/creation/data/example_obs_config.ini"),
            msg="The path to the directory containing the config file."
        )
    )
    
    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)
        
        # read the configuration file
        self.obs_cond_path = self._read_obs_config()
        
        # validate parameters provided in configuration file
        self._validate_obs_config()
            
    
    def _read_obs_config(self) -> dict:
        """
        Read in the configuration file into a dictionary with a
        similar structure as the LSSTErrorModel
        """
        
        #dictionary to store all the information
        obs_cond_path = collections.OrderedDict()

        #read all the lines
        lines=open(self.obs_config_file,'r').readlines() 
        nline=len(lines)
        
        #now go through all the lines and read the parameter information
        for tmp in lines:
            tsplit=tmp.split()
            if(tmp=='\n' or tmp[0]=='#'):
                continue
            elif(tsplit[1]=='='):
                if(len(tsplit)==3):
                    if(tsplit[0] == "nside"):
                         obs_cond_path[tsplit[0]]=np.int(tsplit[2])
                    elif(tsplit[0] == "nYrObs"):
                         obs_cond_path[tsplit[0]]=np.float(tsplit[2])
                    else:
                         obs_cond_path[tsplit[0]]=tsplit[2]
            else:
                raise ValueError("Incorrect format for the configuration file. "
                                + "Error in line: "
                                + temp)
        
        print("Configuration file initialised.")
        return obs_cond_path
        
    
    def _validate_obs_config(self):
        """
        Validate the input
        """
        obs_cond_path = self.obs_cond_path
        
        #check keys:
        general_info = [
            "nside", "mask",
            "nYrObs", "weight",
            "savefile",
        ]
        
        obs_maps = [
            "m5","nVisYr","airmass",
            "gamma","msky","theta","km",
        ]
        
        obs_maps_bands = []
        for obs in obs_maps:
            for band in ["u","g","r","i","z","y"]:
                obs_maps_bands.append(obs+"_"+band)
        
        #check if all of the keys are included:
        if(len(set(obs_cond_path.keys())-set(general_info).union(set(obs_maps_bands)))!=0):
            raise ValueError("Incomplete or extra keywords are passed to the configuration. "
                            + "The following keys shoud be passed: "
                            + str(genera_info) + "\n"
                            + str(obs_maps_bands))
        
        #Check keys:
        for keys in obs_cond_path:
            if key == "nside":
                #check positive and correct input for healpix
                if obs_cond_path[key]<0:
                    raise ValueError("nside must be positive.")
                elif np.log2(obs_cond_path[key]).is_integer() is not True:
                    raise ValueError("nside must be powers of 2")
            elif key == "nYrObs":
                #check positive
                if obs_cond_path[key]<0:
                    raise ValueError("Number of years of observation must be positive.")
            elif key == "mask":
                #cannot be empty
                if obs_cond_path[key] == "empty":
                    raise ValueError("Mask must be provided.")
            #check path exists
            ###need to check if outroot exists
            elif key == "savefile":
                #check the directory exists
                s = obs_cond_path[key].split("/")
                path = obs_cond_path[key][:-len(s[-1])]
                if os.path.exists(path) is not True:
                    raise ValueError("Saving directory does not exist.")
            else:
                if os.path.exists(obs_cond_path[key]) is not True:
                    raise ValueError("The following file is not found: "
                                    + obs_cond_path[key])
    

    def _get_maps(self) -> dict:
        """
        Load in the maps from the directory
        A note on nVisYr: input map usually in terms of total number of exposures,
                          so manually divide the map by nYrObs
        """
        
        obs_maps = [
            "m5","nVisYr","airmass",
            "gamma","msky","theta","km",
        ]
        
        obs_cond_path = self.obs_cond_path
        
        maps = collections.OrderedDict()
        
        # load mask
        mask = hp.read_map(obs_cond_path["mask"])
        if (mask<0).any():
            # set negative values (if any) to zero
            mask[mask<0]=0
        pixels = np.arange(int(obs_cond_path["nside"]**2*12))[mask.astype(bool)]
        maps["pixels"] = pixels
        
        # load nYrObs
        maps["nYrObs"] = obs_cond_path["nYrObs"]
        
        # load weight
        if obs_cond_path["weight"] != "empty":
            maps["weight"] = obs_cond_path["weight"][mask.astype(bool)]
        
        # load everything else
        for obs in obs_maps:
            maps[obs] = {}
            for band in ["u","g","r","i","z","y"]:
                if obs_cond_path[obs+"_"+band] != "empty"
                    maps[obs][band] = obs_cond_path[obs+"_"+band][mask.astype(bool)]
                    #in case of nVisYr, divide by nYrObs 
                    if obs == "nVisYr":
                        maps[obs][band] /= float(maps["nYrObs"])
            if not maps[obs]:
                # delete the key since it is not used
                del maps[obs]
        
        return maps
    
    def _write_output(self, output):
        """
        Save the data into a pikle file
        ###or tables io? 
        """
        with open(self.obs_cond_path["savefile"],'wb') as fout:
            pickle.dump(output,fout,pickle.HIGHEST_PROTOCOL)
        
        
    def run(self):
        obs_cond = self._get_maps()
        self._write_output(obs_cond)
        
        
    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        
        # start message
        printMsg = "Loaded observing conditions from configuration file: "
        
        printMsg += self.obs_config_file
        
        return printMsg
    