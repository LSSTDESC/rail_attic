"""Call through to ceci.main"""
import sys
from ceci.main import main as ceci_main


def main():  #pragma: no cover
    """Call-through to main from ceci"""
    ceci_main()

if __name__ == "__main__":  #pragma: no cover
    sys.exit(main())
