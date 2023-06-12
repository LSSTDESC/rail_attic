""" Utility functions """

import os
import rail
import rail.core

RAILDIR = os.path.abspath(os.path.join(os.path.dirname(rail.core.__file__), '..', '..'))


def find_rail_file(relpath):
    """Find a file somewhere in rail by searching the namespace path

    This lets us avoid issues that the paths can be different depending
    on if we have installed things from source or not
    """
    for path_ in rail.__path__:
        fullpath = os.path.abspath(os.path.join(path_, relpath))
        if os.path.exists(fullpath):
            return fullpath
    raise ValueError(f"Could not file {relpath} in {rail.__path__}")
