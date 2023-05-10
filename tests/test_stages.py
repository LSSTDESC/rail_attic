import tempfile
import rail.stages


def test_print_rail_packages():
    rail.stages.print_rail_packages()

    
def test_print_rail_namespaces():
    rail.stages.print_rail_namespaces()


def test_print_rail_modules():
    rail.stages.print_rail_modules()


def test_print_rail_namespace_tree():
    rail.stages.print_rail_namespace_tree()


def test_import_and_attach_all():
    rail.stages.import_and_attach_all()
    rail.stages.print_rail_stage_dict()


def test_api_rst():
    with tempfile.TemporaryDirectory() as tmpdirname:
        rail.stages.do_api_rst(tmpdirname)
