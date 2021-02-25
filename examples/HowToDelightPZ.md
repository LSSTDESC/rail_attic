# DelightPZ README.md

- sylvielsstfr, 
  - creation date : Feb 24th 2021
  - update : Feb 25th 2021

## Delight reference package

To work temporily with the DelightPZ package,
you have to have installed the Delight
from github here:

**https://github.com/sylvielsstfr/Delight**


and you must select the branch **desc_rail**

Then install Delight according the instructions here

- **https://delight.readthedocs.io/en/latest/install.html**

- pip install -r requirements.txt

(Notice the coloredlogs has been added)

then

- python setup.py build_ext --inplace
- python setup.py install --user



It is important to note that the Delight **setup.py** make that its own data files 
(FILTERS and SED) are installed somewhere by setuptools.

The user use the setuptools to guess where these data are installed.

The *scripts* in Delight are translated in python modules that can be used by RAIL.

Thus a new module has been created in :

- **Delight/interfaces/rail**

(To avoid too much code inside RAIL)

## LSSTDESC/RAIL

Installation from here 
- **https://github.com/LSSTDESC/RAIL**


For the moment, the following files are under developpment in the branch **issue/49/delight**

- **RAIL/examples/configs/delightPZ.yaml**
- **RAIL/rail/estimation/algos/delightPZ.py**



delightPZ as Delight to generate the Delight 
configuration file **parametersTest.cfg**
in a temporary directory (see elightPZ.yaml) .






