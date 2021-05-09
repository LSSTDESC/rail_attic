# DelightPZ README.md

- sylvielsstfr, 
  - creation date : Feb 24th 2021
  - update : May 9th 2021

## Delight reference package

- You must have a working Delight package installed.

- Get it from https://github.com/LSSTDESC/Delight

- Then install Delight according the instructions here

- **https://delight.readthedocs.io/en/latest/install.html**


Get the Delight data https://github.com/LSSTDESC/Delight/tree/master/data

in some path defined in RAIL config file (see file **DelightPZ.yaml**).



## LSSTDESC/RAIL

### Installation of RAIL

Installation from here 
- **https://github.com/LSSTDESC/RAIL**

      git clone git@github.com:LSSTDESC/RAIL.git
      cd RAIL
  

For the moment, the following files are under developpment in the branch **issue/49/delight**

      git checkout issue/49/delight


- **RAIL/examples/configs/delightPZ.yaml**
- **RAIL/rail/estimation/algos/delightPZ.py**
- **RAIL/rail/estimation/algos/include_delightPZ/**

Note **include_delightPZ** includes scripts interfaces from Delight.



      rail
      ├── README.md
      ├── __init__.py
      ├── creation
      │   ├── README.md
      │   └── __init__.py
      ├── estimation
      │   ├── README.md
      │   ├── __init__.py
      │   ├── algos
      │   │   ├── __init__.py
      │   │   ├── delightPZ.py
      │   │   ├── flexzboost.py
      │   │   ├── include_delightPZ
      │   │   │   ├── __init__.py
      │   │   │   ├── calibrateTemplateMixturePriors.py
      │   │   │   ├── convertDESCcat.py
      │   │   │   ├── delightApply.py
      │   │   │   ├── delightLearn.py
      │   │   │   ├── getDelightRedshiftEstimation.py
      │   │   │   ├── makeConfigParam.py
      │   │   │   ├── processFilters.py
      │   │   │   ├── processSEDs.py
      │   │   │   ├── simulateWithSEDs.py
      │   │   │   └── templateFitting.py
      │   │   ├── randomPZ.py
      │   │   ├── sklearn_nn.py
      │   │   └── trainZ.py
      │   ├── estimator.py
      │   └── utils.py
      └── evaluation
      ├── README.md
      └── __init__.py


### Build RAIL


    python setup.py install


### Run RAIL


- delightPZ generates the configuration file **parametersTest.cfg** required by Delight

The path of the configuration file is defined in **DelightPZ.yaml** (RAIL/example/configs) .

#### RAIL temporay files

The recommended temporary structure in **examples/** directory from where one issue the command to run RAILS with Delight:

    python main.py config/delightPZ.yaml

- Some temporary directories must be created in **examples/** for Delight input and output data.

- The recommended structure is the following

#### For DC2 data
tree tmp

    tmp
    ├── delight_data
    ├── delight_indata
    │   ├── BROWN_SEDs
    │   ├── CWW_SEDs
    │   └── FILTERS



#### For internal mock data (simulation)
tree tmpsim

    tmpsim
    ├── delight_data
    ├── delight_indata
    │   ├── BROWN_SEDs
    │   ├── CWW_SEDs
    │   └── FILTERS


Note the directories BROWN_SEDs, CWW_SEDs, FILTERS must be copied from Delight installation
https://github.com/LSSTDESC/Delight/tree/master/data .

