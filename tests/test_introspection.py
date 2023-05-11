import tempfile
import pkgutil
import setuptools

import rail
from rail.core import RailEnv
import rail.stages


def test_print_rail_packages():
    RailEnv.print_rail_packages()

    
def test_print_rail_namespaces():
    RailEnv.print_rail_namespaces()


def test_print_rail_modules():
    RailEnv.print_rail_modules()


def test_print_rail_namespace_tree():
    RailEnv.print_rail_namespace_tree()


def test_import_and_attach_all():
    rail.stages.import_and_attach_all()
    RailEnv.print_rail_stage_dict()


def test_api_rst():
    with tempfile.TemporaryDirectory() as tmpdirname:
        RailEnv.do_api_rst(tmpdirname)
