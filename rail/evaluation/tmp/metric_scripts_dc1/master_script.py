
from __future__ import print_function, absolute_import

import argparse
import numpy as np
import qp

""" Script for injesting and processing photometric redshift data for the DESC
PhotoZ Working Group Data Callenge 1 (DC1).

The goal is compute summary plots and
statistics from the output PDFs of various photometric redshift estimation
codes. This script should be generalized enough to handle the outputs from
all the different codes used to process the DC1 data. To this end we use the
qp software package to injest and process the PDF data to avoid multiple
transformations and approximations that may degrade the information in the
PDFs and complicate the interpretation of our computed summaries.

The summaries we will use include qualitative and quanitative measures. They
are:

Qualitative
-----------
QQ Plot - Plots of CDF quantiles of probability and N(z) against the true N(z)
    of the simulated data. This gives a comparison of the overal PDF for a
    given sample of galaxies.

Quantitative
------------
RMSE - Root mean squared error on the a single point redshift estimate. Does
    not give full information on the PDf. Useful to have but may be shelved.
Kolmogorov-Smirnoff
Anderson Darling/Cramer von Mises
Kullback-Leibler - Full sample PDF compared to true N(z) distribution.
"""


def load_gridded(catalog_file_name, pz_file_name, z_spec_col,
                 z_min, z_max, z_step):
    """ Load a files that are sampled on a reqular grid.

    Load data files that come from codes such as LePHARE and BPZ which
    sample their PDFs at regular grid points.

    Parameters
    ----------
    catalog_file_name : str
        Name of the catalog file to load containing z_estimated and z_spec
    pz_file_name : str
        Name of file containing gridded PDF information
    z_min : float
        Minimum redshift of PDFs
    z_max : float
        Maximum redshift of PDFs. z_max is defined as inclusive in this calse
    z_step : float
        Step size in redshift for PDFs
    z_spec_col : int
       Column number of spectroscopic redshift.

    Returns
    -------
    A tubple gontaining a list of qp.PDF objects for each estimated pdf
    in the file and a qp.PDF of the true N(z) created from samples of the
    distribution.
    """

    # Load our data and create a the array of redshifts used in the grid.
    z_array = np.arange(z_min, z_max + z_step / 2., z_step)
    z_trues = np.loadtxt(catalog_file_name, usecols=z_spec_col)
    gridded_pdfs = np.loadtxt(pz_file_name)

    # Create our "true" PDF using the samples from the inputed data file.
    true_pdf = qp.PDF(samples=z_trues)

    # Create a qp.Ensamble objecct for each of the estimated pdfs.
    estimated_pdfs = qp.Ensemble(gridded_pdfs.shape[0],
                                 gridded=(z_array, gridded_pdfs))

    return (estimated_pdfs, true_pdf)


if __name__ == "__main__":
    pass
