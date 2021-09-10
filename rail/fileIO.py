# Note: 'utils.py' is terrible!  These i/o functions don't really belong
# here, so the bad name is a reminder to deal with this.
import os
import h5py
import pandas as pd
import numpy as np
from astropy.table import Table, vstack


def initialize_qp_output(outfile):
    """
    Since we are appending to astropy tables hdf5 files, this just
    checks for the output file names and deletes them if present.
    """
    basedir = os.path.dirname(outfile)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    basename, ext = os.path.splitext(outfile)
    metaname = basename + "_meta" + ext
    if os.path.exists(outfile):
        os.remove(outfile)
    if os.path.exists(metaname):
        os.remove(metaname)


def write_qp_output_chunk(tmpfile, outfile, qp_chunk, chunk_num):
    """
    Writes out a chunk of qp data to a single hdf5 file (outfile)
    Note: We are just going to use "append", so this assumes
    that the data is coming in the same order as it was fed in.
    We will have to think of a new output writing scheme when we
    switch to parallel processing where chunks may be out o sequence
    Inputs:
    -------
    tmpfile: str
      name of temp file
    outfile: str
      name of final output file
    qp_chunk: qp Ensemble object
      chunk of qp data in default format
    chunk_num: int
      number of chunk, will make a path based on it
    """
    basename, ext = os.path.splitext(outfile)
    metaname = basename + "_meta" + ext
    pathname = f'chunk_{chunk_num}'
    tables = qp_chunk.build_tables()
    if chunk_num == 0:
        tables['meta'].write(metaname, overwrite=True)
    tables['data'].write(tmpfile, path=pathname, append=True)


def qp_reformat_output(tmpfile, outfile, num_chunks):
    """
    The current quick and dirty implementation spits out iterator
    chunks in individual astropy table "paths", this function will
    simply 'vstack' those back into a single file
    Inputs:
    -------
    tmpfile: str
      name of temporary hdf5 file with chunked data
    outfile: str
      name of final outputfile
    num_chunks: int
      the number of astropy tables in outfile to combine
    """
    for i in range(num_chunks):
        chunk = f"chunk_{i}"
        tmptable = Table.read(tmpfile, path=chunk)
        if i == 0:
            bigtable = tmptable
        else:
            bigtable = vstack([bigtable, tmptable])
    bigtable.write(outfile, overwrite=True)
    os.remove(tmpfile)
