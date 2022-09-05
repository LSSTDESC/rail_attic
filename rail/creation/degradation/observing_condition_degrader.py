"""Degrader applied to the magnitude error based on a set of input observing condition maps"""

import numpy as np
import pandas as pd
import healpy as hp
import os
import yaml

from photerr import LsstErrorModel

from rail.creation.degradation import Degrader
from ceci.config import StageParameter as Param

#### To do list:
### - Other formats than fits: .hs (needs healsparse), and .npz (needs rubin)
### - Should allow input argument for all lsst error model


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
            "../../../examples/creation/data/example_obs_config.yml"),
            msg="The path to the directory containing the config file in yaml format."
        )
    )
    
    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)
        
        # read the configuration file
        self.obs_cond_path = self._read_obs_config()
        
        # validate parameters provided in configuration file
        self._validate_obs_config()
        
        # initiate self.maps
        self.maps = {}
        
        # load the survey condition maps if the validation is passed
        self._get_maps()
            
    
    def _read_obs_config(self) -> dict:
        """
        Read in the configuration file into a dictionary with a
        similar structure as the LSSTErrorModel
        """
        
        with open(self.config["obs_config_file"], "r") as stream:
            try:
                obs_cond_path = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        print("Configuration file initialised.")
        return obs_cond_path
        
    
    def _validate_obs_config(self):
        """
        Validate the input
        """
        obs_cond_path = self.obs_cond_path
        
        # A directory to the survey condition
        # map should be provided if
        # these keys are included in the
        # config file
        obs_cond_keys = [
            "m5",
            "nVisYr",
            "airmass",
            "gamma",
            "msky",
            "theta",
            "km",
            "tvis",
        ]
        
        # Other lsst_error_model keys that
        # can be passed in the config file
        lsst_error_model_keys = [
            "nYrObs",
            "Cm",
            "sigmaSys",
            "sigLim",
            "ndMode",
            "ndFlag",
            "absFlux",
            "extendedType",
            "aMin",
            "aMax",
            "majorCol",
            "minorCol",
            "decorrelate",
            "highSNR",
            "errLoc",
            "renameDict",
        ]
        
        # Keys that are not included in
        # lsst_error_model, contains the
        # map information and weight for
        # assigning galaxies to pixels.
        additional_keys = [
            "nside", 
            "mask",
            "weight",
            "nVisYr_tot",
        ]
        
        all_keys = obs_cond_keys + lsst_error_model_keys + additional_keys
        
        # Check if extra keys not in 
        # this list have been passed:
        if(len(set(obs_cond_path.keys())-set(all_keys))!=0):
            extra_keys = set(obs_cond_path.keys())-set(all_keys)
            raise ValueError("Extra keywords are passed to the configuration: \n"
                            + str(extra_keys))
           
        # Check necessary parameters are included:
        if "nside" not in list(obs_cond_path.keys()):
            raise ValueError("nside needs to be provided for the input maps.")
        if "mask" not in list(obs_cond_path.keys()):
            raise ValueError("mask needs to be provided for the input maps.")
        
        # Check data type for the keys:
        # Note that LSSTErrorModel checks
        # the data type for its parameters,
        # so here we only check the additional 
        # parameters and the file paths
        for key in obs_cond_path.keys():
            
            # Check nside should be positive and powers of two
            if key == "nside":
                if obs_cond_path[key]<0:
                    raise ValueError("nside must be positive.")
                elif np.log2(obs_cond_path[key]).is_integer() is not True:
                    raise ValueError("nside must be powers of two.")
            
            # Check if nVisYr_tot is boolean
            if key == "nVisYr_tot":
                if type(obs_cond_path[key])!=bool:
                    raise ValueError("nVisYr_tot must be boolean.")
                    
            # Check input paths exist
            elif key in (obs_cond_keys + ["mask", "weight"]):
                # band-independent keys:
                if key in ["airmass", "tvis", "mask", "weight"]:
                    if os.path.exists(obs_cond_path[key]) is not True:
                        raise ValueError("The following file is not found: "
                                        + obs_cond_path[key])
                # band-dependent keys
                else:
                    for band in obs_cond_path[key].keys():
                        if os.path.exists(obs_cond_path[key][band]) is not True:
                            raise ValueError("The following file is not found: "
                                        + obs_cond_path[key][band])
    
    
    def _get_maps(self):
        """
        Load in the maps from the directory
        A note on nVisYr: input map usually in terms of total number of exposures,
                          so manually divide the map by nYrObs
        """
        
        obs_cond_keys = [
            "m5",
            "nVisYr",
            "airmass",
            "gamma",
            "msky",
            "theta",
            "km",
            "tvis",
        ]
        
        obs_cond_path = self.obs_cond_path
        
        #maps = collections.OrderedDict()
        maps = {}
        
        # Load mask
        mask = hp.read_map(obs_cond_path["mask"])
        if (mask<0).any():
            # set negative values (if any) to zero
            mask[mask<0]=0
        pixels = np.arange(int(obs_cond_path["nside"]**2*12))[mask.astype(bool)]
        maps["pixels"] = pixels
        
        # Load all other maps in the obs_cond_keys and weight
        for key in obs_cond_path.keys():
            if key in (obs_cond_keys + ["weight"]):
                # band-independent keys:
                if key in ["airmass", "tvis", "weight"]:
                     maps[key] = hp.read_map(obs_cond_path[key])[pixels]
                # band-dependent keys
                else:
                    maps[key] = {}
                    for band in obs_cond_path[key].keys():
                        maps[key][band] = hp.read_map(obs_cond_path[key][band])[pixels]
            elif key not in ["nside", "nVisYr_tot", "mask"]:   
                # copy all other lsst_error_model parameters supplied
                maps[key] = obs_cond_path[key]
        
        if "nVisYr" in list(obs_cond_path.keys()):
            if "nYrObs" not in list(obs_cond_path.keys()):
                # Set to default:
                maps["nYrObs"]=10.
            if "nVisYr_tot" not in list(obs_cond_path.keys()):
                # Set to default:
                obs_cond_path["nVisYr_tot"] = True
            if  obs_cond_path["nVisYr_tot"] == True:
                # For each band, compute the average number of visits per year
                for band in maps["nVisYr"].keys():
                    maps["nVisYr"][band] /= float(maps["nYrObs"])
                    
        self.maps = maps
        
    
    def get_pixel_conditions(self, 
                             pixel: int
                            ) -> dict:
        obs_cond_keys = [
            "m5",
            "nVisYr",
            "airmass",
            "gamma",
            "msky",
            "theta",
            "km",
            "tvis",
        ]
        
        allpix = self.maps["pixels"]
        ind = allpix==pixel
        
        obs_conditions = {}
        for key in (self.maps).keys():
            # For keys that may contain the survey condition maps
            if key in obs_cond_keys:
                # band-independent keys:
                if key in ["airmass", "tvis"]:
                     obs_conditions[key] = float(self.maps[key][ind])
                # band-dependent keys:
                else:
                    obs_conditions[key] = {}
                    for band in (self.maps[key]).keys():
                        obs_conditions[key][band] = float(self.maps[key][band][ind])
            # For other keys in LSSTErrorModel:
            elif key not in ["pixels","weights"]:
                obs_conditions[key] = self.maps[key]
        # obs_conditions should now only contain the LSSTErrorModel keys
        return obs_conditions
    
    
    def assign_pixels(self, 
                      catalog: pd.DataFrame
                     ) -> pd.DataFrame:
        """
        assign the pixels to the input catalog
        """
        pixels = self.maps["pixels"]
        if "weights" in list((self.maps).keys()):
            weights = self.maps["weights"]
        else:
            weights = None
        assigned_pix = np.random.choice(pixels, size=len(catalog), replace=True, p=weights)
        #make it a DataFrame object
        assigned_pix = pd.DataFrame(assigned_pix, columns=["pixel"])
        catalog = pd.concat([catalog, assigned_pix], axis=1)
        
        return catalog
        
    def run(self):
        catalog = self.get_data("input", allow_missing=True)
        
        # assign each galaxy to a pixel
        catalog = self.assign_pixels(catalog)

        # loop over each pixel
        pixel_cat_list = []
        for pixel, pixel_cat in catalog.groupby("pixel"):
            # get the observing conditions for this pixel
            obs_conditions = self.get_pixel_conditions(pixel)

            # calculate photometric errors for this pixel
            errorModel = LsstErrorModel(**obs_conditions)
            
            # reset the index
            index = pixel_cat.index
            pixel_cat = pixel_cat.set_index(np.arange(len(pixel_cat)))
            obs_cat = errorModel(pixel_cat, random_state=np.random.default_rng())
            obs_cat = obs_cat.set_index(index)
            
            # add this pixel catalog to the list
            pixel_cat_list.append(obs_cat)

        # recombine all the pixels into a single catalog
        catalog = pd.concat(pixel_cat_list)
        
        # sort index
        catalog = catalog.sort_index()

        self.add_data("output", catalog)
        
        
    def __repr__(self):
        """
        Define how the model is represented and printed.
        """
        
        # start message
        printMsg = "Loaded observing conditions from configuration file: "
        
        printMsg += self.config["obs_config_file"]
        
        return printMsg
    