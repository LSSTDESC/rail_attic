from ceci import PipelineStage
from descformats import TextFile, HDFFile, YamlFile
# from txpipe.data_types import PhotozPDFFile                                               
import os
import pandas as pd
import sys
import numpy as np
import time
import scipy.stats
                         
class GenZPipe(PipelineStage):
    """A generic example Pipeline stage to create a random bunch of 
       fake photo-z PDFs
       The code will be set up to read and write hdf5 files in the
       same formats as BPZPipe as a TXPipe pipeline stage
    """
    name = "GenZPipe"
    #                                                                                      
    inputs = [
        ('photometry_catalog', HDFFile),]
    outputs = [
        ('photoz_pdfs', HDFFile),]
    config_options = {
        "chunk_rows": 1000,
        "bands": ["u","g","r","i","z","y"],
        "has_redshift": True, # does the test file have redshift?
        # if so, read in and append to output file.
        "nz": 300, # Number of redshift grid points
        "zmax": 3.0, # maximum redshift for grid
    }

    def run(self):
        """The main run function to launch the pipeline stage
        This reads in the config parameters from either the default config
        parameters listed in config_options, or from the config.yaml file
        """
        starttime = time.time()
        os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"
        os.environ["CECI_SETUP"]="/global/projecta/projectdirs/lsst/groups/PZ/FlexZBoost/FlexZPipe/setup-flexz-cori-update"

        # Columns we will need from the data                                               
        bands = self.config['bands']
        cols =  [f'mag_{band}_lsst' for band in bands]
        cols += [f'mag_err_{band}_lsst' for band in bands]
        cols += ["id"]
        has_sz = self.config['has_redshift']
        if has_sz:
            cols += ["redshift"]

        # set up redshift grid
        nz = self.config['nz']
        zmax = self.config['zmax']
        z_grid = np.linspace(0.0,zmax,nz)
        
        # Prepare the output HDF5 file                                                     
        output_file = self.prepare_output(z_grid)

        # Amount of data to load at once                                                   
        chunk_rows = self.config['chunk_rows']
        # Loop through chunks of the data.                                                 
        # Parallelism is handled in the iterate_input function -                           
        # each processor will only be given the sub-set of data it is                      
        # responsible for.  The HDF5 parallel output mode means they can                   
        # all write to the file at once too.                                               
        for start, end, data in self.iterate_hdf('photometry_catalog', "photometry", cols,\
                                                 chunk_rows):
            print(f"Process {self.rank} running photo-z for rows {start}-{end}")

            # Calculate modifications to raw input data, in this case colors                                     
            new_data = self.preprocess_data(data)
            # run the core photo-z pdf estimation                                                              
            point_estimates, pdfs = self.estimate_pdfs(z_grid, new_data,nz)

            # Save this chunk of data                                                      
            self.write_output(output_file, start, end, pdfs, point_estimates)

        # Synchronize processors                                                           
        if self.is_mpi():
            self.comm.Barrier()
        output_file.close()
        endtime = time.time()
        print(f"finished, took {endtime - starttime} seconds")

    def prepare_output(self, z_grid):
        """                                                                                
        Prepare the output HDF5 file for writing.                                          
        Note that this is done by all the processes if running in parallel;                
        that is part of the design of HDF5.                                                
                                                                                           
        Parameters                                                                         
        ----------                                                                         
        nobj: int                                                                          
            Number of objects in the catalog                                               
                                                                                           
        z_grid: np 1d array                                                                          
            Redshift grid points that p(z) will be calculated on.
        Returns                                                                            
        -------                                                                            
        f: h5py.File object                                                                
            The output file, opened for writing.                                           
                                                                                           
        """
        has_sz = self.config['has_redshift']
        print(f'has_sz: {has_sz}')
        # Work out how much space we will need.                                            
        cat = self.open_input("photometry_catalog")
        ids = np.array(cat['photometry/id'])
        nobj = ids.size
        nz = len(z_grid)
        if has_sz == True:
            szs = np.array(cat['photometry/redshift'])
        cat.close()

        
        # Open the output file.                                                            
        # This will automatically open using the HDF5 mpi-io driver                        
        # if we are running under MPI and the output type is parallel                      
        f = self.open_output('photoz_pdfs', parallel=True)
        # Create the space for output data                                                 
        groupid = f.create_group('id')
        groupid.create_dataset('galaxy_id', (nobj,), dtype = 'i8')
        if has_sz == True:
            groupsz =f.create_group('true_redshift')
            groupsz.create_dataset('specz', (nobj,), dtype='f4')
        grouppt = f.create_group('point_estimates')
        grouppt.create_dataset('z_mode', (nobj,), dtype='f4')
        group = f.create_group('pdf')
        group.create_dataset("zgrid", (nz,), dtype='f4')
        group.create_dataset("pdf", (nobj,nz), dtype='f4')

        # One processor writes the redshift axis, ids, and true redshifts
        # to output.                                
        if self.rank==0:
            groupid['galaxy_id'][:] = ids
            group['zgrid'][:] = z_grid
            if has_sz == True:
                groupsz['specz'][:] = szs
        return f


    def preprocess_data(self, data):
        """
        This function makes a new set of data with the i-magnitude and
        the colors and color errors (just mag errors in quadrature)
        input:
          data: iterate_hdf data 
        returns:
          df: pandas dataframe of data
          
        """
        bands = self.config['bands']
        numfilts = len(bands)
        #read in the i-band magnitude, calculate colors and color
        #errors for the other bands, stick in a dataframe for simplicity
        

        i_mag = data[f'mag_i_lsst']
        tmpdict = {f'i_mag':i_mag}
        df = pd.DataFrame(tmpdict)
        for xx in range(numfilts-1):
            df[f'color_{bands[xx]}{bands[xx+1]}']= \
	    np.array(data[f'mag_{bands[xx]}_lsst']) -\
            np.array(data[f'mag_{bands[xx+1]}_lsst'])

            df[f'color_err_{bands[xx]}{bands[xx+1]}'] = np.sqrt(\
            np.array(data[f'mag_err_{bands[xx]}_lsst'])**2.0 +\
            np.array(data[f'mag_err_{bands[xx+1]}_lsst'])**2.0)

        #new_data = df.to_numpy()
        #return new_data
        return df
        
    def estimate_pdfs(self, zgrid,new_data, nz):
        """
        function to actually compute the PDFs and point estimates
        inputs:
        zgrid: grid of points on which to calculate ata
          np ndarray
        new_data: some data
          pandas df of data columns
        nz: integer
          number of redshift grid points to evaluate the model on
        Returns:
        point_estimates: numpy nd-array
          point estimates 
        pdfs:
          p(z) evaluated on nz grid points
       """
        ngal = len(new_data['i_mag'])
        point_estimates = np.zeros(ngal)

        zmax =self.config['zmax']
        medians = np.random.uniform(0.0,zmax,size=ngal)
        sigmas = 0.05*(1.+medians)
        pdfs = np.empty((ngal,nz),dtype='f4')
        for i, (mu,sigma) in enumerate(zip(medians,sigmas)):
            pdf = scipy.stats.lognorm.pdf(zgrid, s = sigma, scale = mu)
            pdfs[i] = pdf
            point_estimates[i] = mu

        return point_estimates, pdfs


    def write_output(self, output_file, start, end, pdfs, point_estimates):
        """                                                                                
        Write out a chunk of the computed PZ data.                                       
        Parameters                                                                         
        ----------                                                                         
        output_file: h5py.File                                                             
            The object we are writing out to                                               
        start: int                                                                         
            The index into the full range of data that this chunk starts at                
        end: int                                                                           
            The index into the full range of data that this chunk ends at                  
        pdfs: array of shape (n_chunk, n_z)                                                
            The output PDF values                                                          
        point_estimates: array of shape (3, n_chunk)                                       
            Point-estimated photo-zs for each of the 5 metacalibrated variants             
        """
        group = output_file['pdf']
        group['pdf'][start:end] = pdfs
        grouppt = output_file['point_estimates']
        grouppt[f'z_mode'][start:end] = point_estimates

