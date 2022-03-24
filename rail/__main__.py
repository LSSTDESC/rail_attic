"""Module level main function needed for python -m rail"""

# This file must exist with these contents
import sys
import rail  #pylint: disable=unused-import
from rail.core.stage import RailStage

if __name__ == "__main__":
    sys.exit(RailStage.main())
