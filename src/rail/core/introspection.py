import pkgutil
import setuptools
import os
import importlib

import rail


class RailEnv:

    PACKAGES = {}
    NAMESPACE_PATH_DICT = {}
    NAMESPACE_MODULE_DICT = {}
    MODULE_DICT = {}
    MODULE_PATH_DICT = {}
    TREE = {}
    STAGE_DICT = {}
    BASE_STAGES = []

    @classmethod
    def list_rail_packages(cls):
        """List all the packages that are available in the RAIL ecosystem"""
        cls.PACKAGES = {pkg.name:pkg for pkg in pkgutil.iter_modules(rail.__path__, rail.__name__ + '.')}
        return cls.PACKAGES

    
    @classmethod
    def print_rail_packages(cls):
        """Print all the packages that are available in the RAIL ecosystem"""
        if not cls.PACKAGES:
            cls.list_rail_packages()
        for pkg_name, pkg in cls.PACKAGES.items():
            print(f"{pkg_name} @ {pkg[0].path}")
        return

    @classmethod
    def list_rail_namespaces(cls):
        """List all the namespaces within rail"""
        cls.NAMESPACE_PATH_DICT.clear()
        
        for path_ in rail.__path__:
            namespaces = setuptools.find_namespace_packages(path_)
            for namespace_ in namespaces:
                # exclude stuff that starts with 'example'
                if namespace_.find('example') == 0:
                    continue
                if namespace_ in cls.NAMESPACE_PATH_DICT:  # pragma: no cover
                    cls.NAMESPACE_PATH_DICT[namespace_].append(path_)
                else:
                    cls.NAMESPACE_PATH_DICT[namespace_] = [path_]

        return cls.NAMESPACE_PATH_DICT

                    
    @classmethod
    def print_rail_namespaces(cls):
        """Print all the namespaces that are available in the RAIL ecosystem"""
        if not cls.NAMESPACE_PATH_DICT:
            cls.list_rail_namespaces()
        for key, val in cls.NAMESPACE_PATH_DICT.items():
            print(f"Namespace {key}")
            for vv in val:
                print(f"     {vv}")
        return


    @classmethod
    def list_rail_modules(cls):
        """List all modules within rail"""
        cls.MODULE_DICT.clear()
        cls.MODULE_PATH_DICT.clear()
        cls.NAMESPACE_MODULE_DICT.clear()
        if not cls.NAMESPACE_PATH_DICT:  # pragma: no cover
            cls.list_rail_namespaces()
        for key, val in cls.NAMESPACE_PATH_DICT.items():
            cls.NAMESPACE_MODULE_DICT[key] = []
            for vv in val:
                fullpath = os.path.join(vv, key.replace('.', '/'))
                modules = [pkg for pkg in pkgutil.iter_modules([fullpath], rail.__name__ + '.' + key + '.')]
                for module_ in modules:
                    if module_ in cls.MODULE_DICT:  # pragma: no cover
                        cls.MODULE_DICT[module_.name].append(key)
                    else:
                        cls.MODULE_DICT[module_.name] = [key]
                    cls.NAMESPACE_MODULE_DICT[key].append(module_)
                    cls.MODULE_PATH_DICT[module_.name] = module_[0].path

        return cls.MODULE_PATH_DICT


    @classmethod
    def print_rail_modules(cls):
        """Print all the moduels that are available in the RAIL ecosystem"""
        if not cls.MODULE_DICT:
            cls.list_rail_modules()
        
        for key, val in cls.MODULE_DICT.items():
            print(f"Module {key}")
            for vv in val:
                print(f"     {vv}")

        for key, val in cls.NAMESPACE_MODULE_DICT.items():
            print(f"Namespace {key}")
            for vv in val:
                print(f"     {vv}")
        return


    @classmethod
    def build_rail_namespace_tree(cls):
        """Build a tree of the namespaces and packages in rail"""
        cls.TREE.clear()
        if not cls.NAMESPACE_MODULE_DICT:  # pragma: no cover
            cls.list_rail_modules()

        if not cls.PACKAGES:  # pragma: no cover
            cls.list_rail_packages()

        level_dict = {}
        for key in cls.NAMESPACE_MODULE_DICT.keys():
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
                    cls.TREE[key] = cls.NAMESPACE_MODULE_DICT[key]
                else:
                    parent_key = '.'.join(key.split('.')[0:current_depth])
                    if parent_key in cls.TREE:
                        cls.TREE[parent_key].append({key:cls.NAMESPACE_MODULE_DICT[key]})

        return cls.TREE
        
    @classmethod
    def pretty_print_tree(cls, the_dict=None, indent=""):
        """Utility function to help print the namespace tree

        This can be called recurisvely to walk the tree structure, which has nested dicts

        Parameters
        ----------
        the_dict:  dict | None
            Current dictionary to print, if None it will print cls.TREE
    
        indent:  str
            Indentation string prepended to each line
        """
        if the_dict is None:  # pragma: no cover
            the_dict = cls.TREE
        for key, val in the_dict.items():
            nsname = f"rail.{key}"
            if nsname in cls.PACKAGES:
                pkg_type = "Package"
            else:
                pkg_type = "Namespace"

            print(f"{indent}{pkg_type} {nsname}")
            for vv in val:
                if isinstance(vv, dict):
                    cls.pretty_print_tree(vv, indent=indent+"    ")
                else:
                    print(f"    {indent}{vv.name}")
        return

                    
    @classmethod
    def print_rail_namespace_tree(cls):
        """Print the namespace tree in a nice way"""
        if not cls.TREE:
            cls.build_rail_namespace_tree()
        cls.pretty_print_tree(cls.TREE)
        return
        
    @classmethod
    def do_pkg_api_rst(cls, basedir, key, val):
        """Build the api rst file for a rail package"""
        
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
            if isinstance(vv, dict):  # pragma: no cover
                for k3, v3 in vv.items():
                    for v4 in v3:
                        api_pkg_toc += f"    {v4.name}.rst\n"
            else:
                api_pkg_toc += f"    {vv.name}.rst\n"                        

        with open(os.path.join(basedir, 'api', f"rail.{key}.rst"), 'w') as apitocfile:
            apitocfile.write(api_pkg_toc)
        return

            
    @classmethod
    def do_namespace_api_rst(cls, basedir, key, val):
        """Build the api rst file for a rail namespace"""

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
                    cls.do_namespace_api_rst(basedir, k3, v3)
                    sub_packages += f"    rail.{k3}\n"
            else:
                sub_modules += f"    {vv.name}\n"
        api_pkg_toc = api_pkg_toc.format(sub_packages=sub_packages, sub_modules=sub_modules)
            
        with open(os.path.join(basedir, 'api', f"rail.{key}.rst"), 'w') as apitocfile:
            apitocfile.write(api_pkg_toc)
        return

            
    @classmethod
    def do_api_rst(cls, basedir='.'):
        if not cls.TREE:  # pragma: no cover
            cls.build_rail_namespace_tree()

        apitoc = \
"""API Documentation
=================

Information on specific functions, classes, and methods.

.. toctree::

"""
        try:
            os.makedirs(basedir)
        except:
            pass

        try:
            os.makedirs(os.path.join(basedir, 'api'))
        except:  # pragma: no cover
            pass

        for key, val in cls.TREE.items():        
            nsname = f"rail.{key}"
            nsfile = os.path.join('api', f"{nsname}.rst")
            apitoc += f"    {nsfile}\n"

            if nsname in cls.PACKAGES:
                cls.do_pkg_api_rst(basedir, key, val)
            else:
                cls.do_namespace_api_rst(basedir, key, val) 

        with open(os.path.join(basedir, 'api.rst'), 'w') as apitocfile:
            apitocfile.write(apitoc)

        return


    @classmethod
    def import_all_packages(cls):
        """Import all the packages that are available in the RAIL ecosystem"""
        pkgs = cls.list_rail_packages()
        for pkg in pkgs.keys():
            try:
                imported_module = importlib.import_module(pkg)
                print(f"Imported {pkg}")
            except Exception as msg:
                print(f"Failed to import {pkg} because: {str(msg)}")


    @classmethod
    def attach_stages(cls, to_module):
        """Attach all the available stages to this module
        
        This allow you to do 'from rail.stages import *'
        """
        from rail.core.stage import RailStage
        cls.STAGE_DICT.clear()
        cls.STAGE_DICT['none'] = []
        cls.BASE_STAGES.clear()
        
        n_base_classes = 0
        n_stages = 0

        for stage_name, stage_info in RailStage.incomplete_pipeline_stages.items():
            if stage_info[0] in [RailStage]:
                continue
            cls.BASE_STAGES.append(stage_info[0])
            cls.STAGE_DICT[stage_info[0].__name__] = []
            n_base_classes += 1

        for stage_name, stage_info in RailStage.pipeline_stages.items():
            setattr(to_module, stage_name, stage_info[0])
            n_stages += 1

        for stage_name, stage_info in RailStage.pipeline_stages.items():
            baseclass = "none"
            for possible_base in cls.BASE_STAGES:
                if issubclass(stage_info[0], possible_base):
                    baseclass = possible_base.__name__
                    break
            cls.STAGE_DICT[baseclass].append(stage_name)

        print(f"Attached {n_base_classes} base classes and {n_stages} fully formed stages to rail.stages")
        return


    @classmethod
    def print_rail_stage_dict(cls):
        """Print an dict of all the RailSages organized by their base class"""
        for key, val in cls.STAGE_DICT.items():
            print(f"BaseClass {key}")
            for vv in val:
                print(f"  {vv}")
