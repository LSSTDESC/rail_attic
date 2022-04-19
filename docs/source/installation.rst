************
Installation
************

RAIL has multiple dependencies that are sensitive to out-of-date code versions, therefore it is strongly recommended that you create a new dedicated virtual environment for RAIL to avoid problems with pip/conda failing to update some packages that you have previously installed during installation of RAIL.  Due to changes in `scipy` array broadcasting with v1.8.0, we also strongly recommend using scipy >= 1.8.0, and note that scipy 1.8.0 requires python >= 3.8, and therefore recommend against using python versions of 3.7 or below.

If you want to add the conda environment that you are about to create as a kernel that you can use in a Jupyter notebook, see below in the :ref: `Adding your kernel to jupyter` section.

To create a conda environment named "[name-for-your-env]" that has a specific version of python (in this case 3.9) and pip, run the command:
`conda create -n [name-for-your-env] pip python=3.9`.
Where you have replaced [name-for-your-env] with whatever name you wish to use, e.g. `rail`.
You can then run the command `conda activate [name-for-your-env]` to activate this environment.  We are now ready to install RAIL.  Before we do that, though, we note that the particular estimator `Delight` is built with `Cython` and uses `openmp`.  Mac has dropped native support for `openmp`, which will likely cause problems when trying to run the `delightPZ` estimation code in RAIL.  See the notes below for instructions on installing Delight if you wish to use this particular estimator.

Now to install RAIL, you need to:
1. `Clone this repo <https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository>`_ to your local workspace.
2. Change directories so that you are in the RAIL root directory.
3. Run one of the following commands, depending on your use case:

  - If you are not developing RAIL and just want to install the package for use in some other project, you can run the command `pip install .[all]` (or `pip install '.[all]'` if you are using zsh, note the single quotes). This will download the entire RAIL package.
    - If you are installing RAIL on a Mac, as noted above the `delightPZ` estimator requires that your machine's `gcc` be set up to work with `openmp`. If you are installing on a Mac and do not plan on using `delightPZ`, then you can simply install RAIL with `pip install .[base]` rather than `pip install .[all]`, which will skip the Delight package.  If you are on a Mac and *do* expect to run `delightPZ`, then follow the instructions `here <https://github.com/LSSTDESC/Delight/blob/master/Mac_installation.md>`_ to install Delight before running `pip install .[all]`.
  If you only want to install the dependencies for a specific piece of RAIL, you can change the install option. E.g. to install only the dependencies for the Creation Module or the Estimation Module, run `pip install .[creation]` or `pip install .[estimation]` respectively. For other install options, look at the keys for the `extras_require` dictionary at the top of `setup.py`.
  - If you are developing RAIL, you should install with the `-e` flag, e.g. `pip install -e .[all]`. This means that any changes you make to the RAIL codebase will propagate to imports of RAIL in your scripts and notebooks.

Note the Creation Module depends on pzflow, which has an optional GPU-compatible installation.
For instructions, see the `pzflow Github repo <https://github.com/jfcrenshaw/pzflow/>`_.

On some systems that are slightly out of date, e.g. an older version of python's `setuptools`, there can be some problems installing packages hosted on GitHub rather than PyPi.  We recommend that you update your system; however, some users have still reported problems with installation of subpackages necessary for `FZBoost` and `bpz_lite`.  If this occurs, try the following procedure:

If all of the estimation algorithms are listed as `not avaialble` there may have been a problem installing `qp`.  Try:
- cd to a directory where you wish to clone qp and run `git clone https://github.com/LSSTDESC/qp.git`
- cd to the qp directory and run `python setup.py install`
- cd to the directory where you cloned RAIL, and reinstall with `pip install .[all]`, or `pip install '.[all]'` if using zsh

For FZBoost:
- install `xgboost` with the command `pip install xgboost==0.90.0`
- install FlexCode with `pip install FlexCode[all]
- ensure that you are in the directory where you cloned RAIL, and reinstall with `pip install .[all]`, or `pip install '.[all]'` if using zsh

For bpz_lite:
- cd to a directory where you wish to clone the DESC_BPZ package and run `git clone https://github.com/LSSTDESC/DESC_BPZ.git`
- cd to the DESC_BPZ directory and run `python setup.py install` (add `--user` if you are on a shared system such as NERSC)
- cd to the directory where you cloned RAIL, and reinstall with `pip install .[all]`, or `pip install '.[all]'` if using zsh


Once you have installed RAIL, you can import the package (via `import rail`) in any of your scripts and notebooks.
For examples demonstrating how to use the different pieces, see the notebooks in the `examples/` directory.


Adding your kernel to jupyter
========================
If you want to use the kernel that you have just created to run RAIL example demos, then you may need to explicitly add an ipython kernel.  You may need to first install ipykernel with `conda install ipykernel`.  You can do then add your kernel with the following command, making sure that you have the conda environment that you wish to add activated.  From your environment, execute the command:
`python -m ipykernel install --user --name [nametocallnewkernel]`
(you may or may not need to prepend `sudo` depending on your permissions).  When you next start up Jupyter you should see a kernel with your new name as an option, including using the Jupyter interface at NERSC.


Requirements
============

RAIL requires Python version 3.6 or later.  To run the code, there are the following dependencies:

- ceci
- numpy
- pandas
- pyyaml
- pzflow
- qp@git+https://github.com/LSSTDESC/qp
- scikit-learn
- scipy
- seaborn
- tables-io
- yml
