************
Installation
************

First, it is recommended that you create a new virtual environment for RAIL.
For example, to create a conda environment named "rail" that has the latest version of python and pip, run the command `conda create -n rail pip`.
You can then run the command `conda activate rail` to activate this environment.

Now to install RAIL, you need to:
1. [Clone this repo](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository) to your local workspace.
2. Change directories so that you are in the RAIL root directory.
3. Run one of the following commands, depending on your use case:

  - If you are not developing RAIL and just want to install the package for use in some other project, you can run the command `pip install .[all]`. This will download the entire RAIL package. 
  If you only want to install the dependencies for a piece of RAIL, you can change the install option. E.g. to install only the dependencies for the Creation Module or the Estimation Module, run `pip install .[creation]` or `pip install .[estimation]` respectively. For other install options, look at the keys for the `extras_require` dictionary at the top of `setup.py`.
  - If you are developing RAIL, you should install with the `-e` flag, e.g. `pip install -e .[all]`. This means that any changes you make to the RAIL codebase will propagate to imports of RAIL in your scripts and notebooks.

Note the Creation Module depends on pzflow, which has an optional GPU-compatible installation.
For instructions, see the [pzflow Github repo](https://github.com/jfcrenshaw/pzflow/).

Once you have installed RAIL, you can import the package (via `import rail`) in any of your scripts and notebooks.
For examples demonstrating how to use the different pieces, see the notebooks in the `examples/` directory.
  
Requirements
============

RAIL requires Python version 3.6 or later.  To run the code, there are the following dependencies:

- numpy
- pandas
- pyyaml
- pzflow
- qp@git+https://github.com/LSSTDESC/qp
- scikit-learn
- scipy
- tables-io
- yml