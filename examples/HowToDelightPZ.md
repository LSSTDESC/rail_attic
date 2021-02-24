# DelightPZ README.md

- sylvielsstfr, Freeb 24th 2021

## Delight reference package

To work temporaryly with the DelightPZ package,
you have to have installed the Delight
from github here:

**https://github.com/sylvielsstfr/Delight**


and you must select the branch **desc_rail**

Then install delight according the instructions here

- **https://delight.readthedocs.io/en/latest/install.html**

- pip install -r requirements.txt

(Notice the coloredlogs has been added)

then

- python setup.py build_ext --inplace
- python setup.py install


To works, delight needs a bunch of datasets in :
  - Delight/data>

To ease the interface with RAIL, an interface package is being
added in 

- Delight/interfaces/rail



## LSSTDESC/RAIL

Installation from here 
- **https://github.com/LSSTDESC/RAIL**


For the moment, the following files are under developpment in the branch **issue/49/delight**

- RAIL/examples/configs/delightPZ.yaml
- RAIL/rail/estimation/algos/delightPZ.py


For the moment Delight needs temporarily the configuration file
put here

- **RAIL/tests/data/parametersTest.cfg**

Note **parametersTest.cfg** contains some absolute path
to Delight data. It must be adapted for each installation.

It will change later.




