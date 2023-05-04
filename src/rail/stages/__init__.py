import pkgutil
import rail
import importlib

from rail.estimation.estimator import *
from rail.estimation.summarizer import *
from rail.estimation.algos.NZDir import *
from rail.estimation.algos.naiveStack import *
from rail.estimation.algos.pzflow import *
from rail.estimation.algos.randomPZ import *
from rail.estimation.algos.pointEstimateHist import *
from rail.estimation.algos.knnpz import *
from rail.estimation.algos.simpleSOM import *
from rail.estimation.algos.somocluSOM import *
from rail.estimation.algos.trainZ import *
from rail.estimation.algos.varInference import *

from rail.creation.degrader import *
from rail.creation.degradation.grid_selection import *
from rail.creation.degradation.observing_condition_degrader import *
from rail.creation.degradation.spectroscopic_degraders import *
from rail.creation.degradation.spectroscopic_selections import *
from rail.creation.degradation.quantityCut import *
from rail.creation.degradation.lsst_error_model import *

from rail.creation.engine import *
from rail.creation.engines.flowEngine import *
#from rail.creation.engines.galaxy_population_components import *
from rail.creation.engines.dsps_photometry_creator import *
from rail.creation.engines.dsps_sed_modeler import *

from rail.evaluation.evaluator import Evaluator


STAGE_DICT = {}
BASE_STAGES = []

def print_rail_packages():
    """Print all the packages that are available in the RAIL ecosystem"""
    for pkg in pkgutil.iter_modules(rail.__path__, rail.__name__ + '.'):
        print(f"{pkg[1]} @ {pkg[0].path}")


def list_rail_packages():
    """List all the packages that are available in the RAIL ecosystem"""
    return [pkg for pkg in pkgutil.iter_modules(rail.__path__, rail.__name__ + '.')]
      

def import_all_packages():
    """Import all the packages that are available in the RAIL ecosystem"""
    pkgs = list_rail_packages()
    for pkg in pkgs:
        print(f"Importing {pkg[1]}")
        imported_module = importlib.import_module(pkg[1])
        

def attach_stages():
    """Attach all the available stqges to this module

    This allow you to do 'from rail.stages import *'
    """
    from rail.core.stage import RailStage
    rail.stages.STAGE_DICT.clear()
    rail.stages.STAGE_DICT['none'] = []
    rail.stages.BASE_STAGES.clear()

    n_base_classes = 0
    n_stages = 0
    
    for stage_name, stage_info in RailStage.incomplete_pipeline_stages.items():
        if stage_info[0] in [RailStage]:
            continue
        rail.stages.BASE_STAGES.append(stage_info[0])
        rail.stages.STAGE_DICT[stage_info[0].__name__] = []            
        n_base_classes += 1
        
    for stage_name, stage_info in RailStage.pipeline_stages.items():
        setattr(rail.stages, stage_name, stage_info[0])   
        n_stages += 1
        
    for stage_name, stage_info in RailStage.pipeline_stages.items():
        for possible_base in rail.stages.BASE_STAGES:
            baseclass = "none"
            if issubclass(stage_info[0], possible_base):
                baseclass = possible_base.__name__
                break            
        rail.stages.STAGE_DICT[baseclass].append(stage_name)

    print(f"Attached {n_base_classes} base classes and {n_stages} fully formed stages to rail.stages")
        

def import_and_attach_all():    
    import_all_packages()
    attach_stages()


def print_stage_dict():
    """Print an dict of all the RailSages organized by their base class"""
    for key, val in STAGE_DICT.items():
        print(f"BaseClass {key}")
        for vv in val:
            print(f"  {vv}")
    
        
