#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:11:59
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --error="test.err"
#SBATCH --output="test.out"

                                                                                              
module load python/3.7-anaconda-2019.07
module swap PrgEnv-intel PrgEnv-gnu
module load PrgEnv-gnu
module unload darshan
module load h5py-parallel
module load cfitsio/3.47
module load gsl/2.5

export CECI_SETUP="/global/projecta/projectdirs/lsst/groups/PZ/FlexZBoost/FlexZPipe/setup-flexz-cori-update"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PYTHONPATH:/global/projecta/projectdirs/lsst/groups/PZ/Packages/descformats/lib/python3.7/site-packages:/global/projecta/projectdirs/lsst/groups/PZ/Packages/ceci/lib/python3.7/site-packages:/global/projecta/projectdirs/lsst/groups/PZ/Packages/parsl0.5.2/lib/python3.7/site-packages:/global/homes/s/schmidt9/DESC/software/RAIL/RAIL/rail/estimation


srun -n 1 python3 -m genericpipe GenZPipe --photometry_catalog=./test1000_h5pyfmt_new.h5 --config=./config.yml --photoz_pdfs=./testout.hdf5 --mpi
