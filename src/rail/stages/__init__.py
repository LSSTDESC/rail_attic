import pkgutil
import rail
import importlib
import setuptools

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
PACKAGES = {}
NAMESPACE_PATH_DICT = {}
NAMESPACE_MODULE_DICT = {}
MODULE_DICT = {}
MODULE_PATH_DICT = {}
TREE = {}


def list_rail_packages():
    """List all the packages that are available in the RAIL ecosystem"""
    rail.stages.PACKAGES = {pkg.name:pkg for pkg in pkgutil.iter_modules(rail.__path__, rail.__name__ + '.')}
    return rail.stages.PACKAGES


def print_rail_packages():
    """Print all the packages that are available in the RAIL ecosystem"""
    for pkg_name, pkg in rail.stages.PACKAGES.items():
        print(f"{pkg_name} @ {pkg.path}")

        
def list_rail_namespaces(verbose=False):
    """List all the namespaces within rail"""
    rail.stages.NAMESPACE_PATH_DICT.clear()

    for path_ in rail.__path__:
        namespaces = setuptools.find_namespace_packages(path_)
        for namespace_ in namespaces:
            # exclude stuff that starts with 'example'
            if namespace_.find('example') == 0:
                continue
            if namespace_ in rail.stages.NAMESPACE_PATH_DICT:
                rail.stages.NAMESPACE_PATH_DICT[namespace_].append(path_)
            else:
                rail.stages.NAMESPACE_PATH_DICT[namespace_] = [path_]

    if not verbose:
        return
    
    for key, val in rail.stages.NAMESPACE_PATH_DICT.items():
        print(f"Namespace {key}")
        for vv in val:
            print(f"     {vv}")


def list_rail_modules():
    """List all modules within rail"""
    rail.stages.MODULE_DICT.clear()
    rail.stages.MODULE_PATH_DICT.clear()
    rail.stages.NAMESPACE_MODULE_DICT.clear()
    if not rail.stages.NAMESPACE_PATH_DICT:
        list_rail_namespaces()
    for key, val in rail.stages.NAMESPACE_PATH_DICT.items():
        rail.stages.NAMESPACE_MODULE_DICT[key] = []
        for vv in val:
            fullpath = os.path.join(vv, key.replace('.', '/'))
            modules = [pkg for pkg in pkgutil.iter_modules([fullpath], rail.__name__ + '.' + key + '.')]
            for module_ in modules:
                if module_ in rail.stages.MODULE_DICT:
                    rail.stages.MODULE_DICT[module_.name].append(key)
                else:
                    rail.stages.MODULE_DICT[module_.name] = [key]
                rail.stages.NAMESPACE_MODULE_DICT[key].append(module_)
                rail.stages.MODULE_PATH_DICT[module_.name] = module_[0].path
                
    for key, val in rail.stages.MODULE_DICT.items():
        print(f"Module {key}")
        for vv in val:
            print(f"     {vv}")

    for key, val in rail.stages.NAMESPACE_MODULE_DICT.items():
        print(f"Namespace {key}")
        for vv in val:
            print(f"     {vv}")
    return rail.stages.MODULE_PATH_DICT
            
def build_namespace_tree():
    rail.stages.TREE.clear()    
    if not rail.stages.NAMESPACE_MODULE_DICT:
        list_rail_modules()

    if not rail.stages.PACKAGES:
        list_rail_packages()

    level_dict = {}
    for key in rail.stages.NAMESPACE_MODULE_DICT.keys():
        count = key.count('.')
        if count in level_dict:
            level_dict[count].append(key)
        else:
            level_dict[count] = [key]

    depth = max(level_dict.keys())
    for current_depth in range(depth+1):
        for key in level_dict[current_depth]:
            nsname = f"rail.{key}"
            if current_depth == 0:
                nsname = f"rail.{key}"
                
                rail.stages.TREE[key] = rail.stages.NAMESPACE_MODULE_DICT[key]
            else:
                parent_key = '.'.join(key.split('.')[0:current_depth])
                if parent_key in rail.stages.TREE:
                    rail.stages.TREE[parent_key].append({key:rail.stages.NAMESPACE_MODULE_DICT[key]})


def pretty_print_tree(the_dict=None, indent=""):
    if the_dict is None:
        the_dict = rail.stages.TREE
    for key, val in the_dict.items():
        nsname = f"rail.{key}"
        if nsname in rail.stages.PACKAGES:
            pkg_type = "Package"
        else:
            pkg_type = "Namespace"

        print(f"{indent}{pkg_type} {nsname}")
        for vv in val:
            if isinstance(vv, dict):
                pretty_print_tree(vv, indent=indent+"    ")
            else:
                print(f"    {indent}{vv.name}")
    

def import_all_packages():
    """Import all the packages that are available in the RAIL ecosystem"""
    pkgs = list_rail_packages()
    for pkg in pkgs.keys():
        print(f"Importing {pkg}")
        imported_module = importlib.import_module(pkg)


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


def do_pkg_api_rst(key, val):

    api_pkg_toc = f"rail.{key} package\n"
    api_pkg_toc += "="*len(api_pkg_toc)

    api_pkg_toc += \
f"""
.. automodule:: rail.{key}
    :members:
    :undoc-members:
    :show-inheritance:

Submodules
----------

.. toctree::
    :maxdepth: 4

"""
    
    for vv in val:
        if isinstance(vv, dict):
            for k3, v3 in vv.items():
                for v4 in v3:
                    api_pkg_toc += f"    {v4.name}.rst\n"
        else:
            api_pkg_toc += f"    {vv.name}.rst\n"                        

    with open(os.path.join('api', f"rail.{key}.rst"), 'w') as apitocfile:
        apitocfile.write(api_pkg_toc)


def do_namespace_api_rst(key, val):

    api_pkg_toc = f"{key} namespace\n"
    api_pkg_toc += "="*len(api_pkg_toc)

    api_pkg_toc += \
"""

.. py:module:: rail.estimation

Subpackages
-----------

.. toctree::
    :maxdepth: 4

{sub_packages}

Submodules
----------

.. toctree::
    :maxdepth: 4

{sub_modules}
"""

    sub_packages = ""
    sub_modules = ""
    for vv in val:
        if isinstance(vv, dict):
            for k3, v3 in vv.items():
                do_namespace_api_rst(k3, v3)
                sub_packages += f"    rail.{k3}\n"
        else:
            sub_modules += f"    {vv.name}\n"
    api_pkg_toc = api_pkg_toc.format(sub_packages=sub_packages, sub_modules=sub_modules)
            
    with open(os.path.join('api', f"rail.{key}.rst"), 'w') as apitocfile:
        apitocfile.write(api_pkg_toc)
     

def do_api_rst():
    if not rail.stages.TREE:
        build_namespace_tree()

        apitoc = \
"""API Documentation
=================

Information on specific functions, classes, and methods.

.. toctree::

"""
    
    for key, val in rail.stages.TREE.items():        
        nsname = f"rail.{key}"
        nsfile = os.path.join('api', f"{nsname}.rst")
        apitoc += f"    {nsfile}\n"

        if nsname in rail.stages.PACKAGES:
            do_pkg_api_rst(key, val)
        else:
            do_namespace_api_rst(key, val) 
        

    with open('api.rst', 'w') as apitocfile:
        apitocfile.write(apitoc)
