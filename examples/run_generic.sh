#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:29:59
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --error="generic_err.err"
#SBATCH --output="generic_out.out"

module load python/3.7-anaconda-2019.07
module swap PrgEnv-intel PrgEnv-gnu
module load PrgEnv-gnu
module unload darshan
module load h5py-parallel
module load cfitsio/3.47
module load gsl/2.5


export CECI_SETUP="/global/projecta/projectdirs/lsst/groups/PZ/RAIL/examples/setup_ceci"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PYTHONPATH:/global/projecta/projectdirs/lsst/groups/PZ/RAIL/rail/estimation:/global/projecta/projectdirs/lsst/groups/PZ/Packages/descformats/lib/python3.7/site-packages:/global/projecta/projectdirs/lsst/groups/PZ/Packages/ceci/lib/python3.7/site-packages:/global/projecta/projectdirs/lsst/groups/PZ/Packages/parsl0.5.2/lib/python3.7/site-packages


srun -n 8 python3 -m genericpipe GenZPipe --photometry_catalog=./testdata.hdf5 --config=./config.yml --photoz_pdfs=./testpdfs.hdf5 --mpi
