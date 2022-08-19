"""
first pass implementation of pzflow estimator
First pass will ignore photometric errors and just do
things in terms of magnitudes, we will expand in a
future update
"""

import numpy as np

from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.data import FlowHandle, TableHandle
import pandas as pd
import qp


def computemeanstd(df):
    """Compute colors from the magnitudes and compute their
    means and stddevs for data whitening

    Parameters
    ----------
    df: pandas dataframe
        ordered dict of raw input data

    Returns
    -------
    means, stds: numpy arrays
         means and stddevs for the mags and colors
    """
    tmpdict = {}
    tmpdict['redshift'] = df['redshift']
    tmpdict['i'] = df['mag_i_lsst']
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    usecols = ['redshift', 'i']
    for k in range(5):
        newcol = f'{bands[k]}{bands[k+1]}'
        usecols.append(newcol)
        tmpdict[f'{bands[k]}{bands[k+1]}'] = \
            df[f'mag_{bands[k]}_lsst'] - \
            df[f'mag_{bands[k+1]}_lsst']
    tmpdf = pd.DataFrame(tmpdict)
    tmpdata = np.array(tmpdf[usecols])
    means = tmpdata.mean(axis=0)
    stds = tmpdata.std(axis=0)
    return means, stds


def_bands = ['u', 'g', 'r', 'i', 'z', 'y']
refcols = [f"mag_{band}_lsst" for band in def_bands]
all_columns = refcols.copy()
for band in def_bands:
    all_columns.append(f"mag_{band}_lsst_err")
def_maglims = dict(mag_u_lsst=27.79,
                   mag_g_lsst=29.04,
                   mag_r_lsst=29.06,
                   mag_i_lsst=28.62,
                   mag_z_lsst=27.98,
                   mag_y_lsst=27.05)
def_errornames=dict(mag_err_u_lsst="mag_u_lsst_err",
                    mag_err_g_lsst="mag_g_lsst_err",
                    mag_err_r_lsst="mag_r_lsst_err",
                    mag_err_i_lsst="mag_i_lsst_err",
                    mag_err_z_lsst="mag_z_lsst_err",
                    mag_err_y_lsst="mag_y_lsst_err")


class Inform_PZFlowPDF(CatInformer):
    """ Subclass to train a pzflow-based estimator
    """
    name = 'Inform_PZFlowPdf'
    outputs = [('model', FlowHandle)]
    config_options = CatInformer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="min z"),
                          zmax=Param(float, 3.0, msg="max_z"),
                          nzbins=Param(int, 301, msg="num z bins"),
                          flow_seed=Param(int, 0, msg="seed for flow"),
                          ref_column_name=Param(str, 'mag_i_lsst',
                                                msg="name for reference column"),
                          column_names=Param(list, refcols,
                                             msg="column names to be used in flow"),
                          mag_limits=Param(dict, def_maglims,
                                           msg="1 sigma mag limits"),
                          include_mag_errors=Param(bool, False,
                                                   msg="Boolean flag on whether to marginalize"
                                                       "over mag errors (NOTE: much slower on CPU!)"),
                          error_names_dict=Param(dict, def_errornames,
                                                 msg="dictionary to rename error columns"),
                          n_error_samples=Param(int, 1000,
                                                msg="umber of error samples in marginalization"),
                          soft_sharpness=Param(int, 10,
                                               msg="sharpening paremeter for SoftPlus"),
                          soft_idx_col=Param(int, 0,
                                             msg="index column for SoftPlus"),
                          redshift_column_name=Param(str, 'redshift',
                                                     msg="name of redshift column"),
                          num_training_epochs=Param(int, 50,
                                                    msg="number flow training epochs"))


    def __init__(self, args, comm=None):
        """Constructor, build the CatInformer, then do PZFlow specific setup
        """
        CatInformer.__init__(self, args, comm=comm)
        usecols = self.config.column_names.copy()
        allcols = usecols.copy()
        if self.config.include_mag_errors:  # only include errors if option set
            for item in self.config.error_names_dict:
                allcols.append(self.config.error_names_dict[item])
        usecols.append(self.config.redshift_column_name)
        allcols.append(self.config.redshift_column_name)
        self.usecols = usecols
        self.allcols = allcols

    def run(self):
        """
        train a flow based on the training data
        This is mostly based off of the pzflow example notebook
        """
        from pzflow import Flow
        from pzflow.bijectors import Chain, ColorTransform, InvSoftplus
        from pzflow.bijectors import StandardScaler, RollingSplineCoupling
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  #pragma: no cover
            training_data = self.get_data('input')

        input_df = pd.DataFrame(training_data)
        flowdf = input_df[self.usecols]
        # replace nondetects
        # will fancy this up later with a flow to sample from truth
        for col in self.config.column_names:
            flowdf.loc[np.isclose(flowdf[col], 99.), col] = self.config.mag_limits[col]

        # compute means and stddevs for StandardScalar transform
        col_means, col_stds = computemeanstd(flowdf)
        ref_idx = flowdf.columns.get_loc(self.config.ref_column_name)
        mag_idx = [flowdf.columns.get_loc(col) for col in self.config.column_names]
        nlayers = flowdf.shape[1]
        bijector = Chain(
            ColorTransform(ref_idx, mag_idx),
            InvSoftplus(self.config.soft_idx_col, self.config.soft_sharpness),
            StandardScaler(col_means, col_stds),
            RollingSplineCoupling(nlayers),
        )
        self.model = Flow(flowdf.columns, bijector, seed=self.config.flow_seed)
        _ = self.model.train(flowdf[self.usecols], epochs=self.config.num_training_epochs,
                             verbose=True)
        self.model.save(self.get_output('model'))



class PZFlowPDF(CatEstimator):
    """CatEstimator which uses PZFlow
    """
    name = 'PZFlowPDF'
    inputs = [('model', FlowHandle),
              ('input', TableHandle)]
    config_options = CatEstimator.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          flow_seed=Param(int, 0, msg="seed for flow"),
                          ref_column_name=Param(str, 'mag_i_lsst',
                                                msg="name for reference column"),
                          column_names=Param(list, refcols,
                                             msg="column names to be used in flow"),
                          mag_limits=Param(dict, def_maglims,
                                           msg="1 sigma mag limits"),
                          include_mag_errors=Param(bool, False,
                                                   msg="Boolean flag on whether to marginalize"
                                                       "over mag errors (NOTE: much slower on CPU!)"),
                          error_names_dict=Param(dict, def_errornames,
                                                 msg="dictionary to rename error columns"),
                          n_error_samples=Param(int, 1000,
                                                 msg="umber of error samples in marginalization"),
                          redshift_column_name=Param(str, 'redshift',
                                                     msg="name of redshift column"))


    def __init__(self, args, comm=None):
        CatEstimator.__init__(self, args, comm=comm)
        usecols = self.config.column_names.copy()
        allcols = usecols.copy()
        if self.config.include_mag_errors:  #pragma: no cover
            for item in self.config.error_names_dict:
                allcols.append(self.config.error_names_dict[item])
        usecols.append(self.config.redshift_column_name)
        allcols.append(self.config.redshift_column_name)
        self.usecols = usecols
        self.allcols = allcols
        self.zgrid = None

    def _process_chunk(self, start, end, data, first):
        """
        calculate and return PDFs for each galaxy using the trained flow
        """
        # flow expects dataframe
        test_df = pd.DataFrame(data)
        if self.config.include_mag_errors:  #pragma: no cover
            # rename the error columns to end in _err!
            test_df.rename(columns=self.config.error_names_dict, inplace=True)
        flow_df = test_df[self.allcols]
        # replace nondetects
        if self.config.include_mag_errors:  #pragma: no cover
            err_names = list(self.config.error_names_dict.values())
            for col, err_col in zip(self.config.column_names, err_names):
                # set error to 0.7525 for 1 sigma detection 2.5log10(2) = .75257
                flow_df.loc[np.isclose(flow_df[col], 99.), err_col] = 0.75257
                flow_df.loc[np.isclose(flow_df[col], 99.), col] = self.config.mag_limits[col]
        else:
            for col in self.config.column_names:
                flow_df.loc[np.isclose(flow_df[col], 99.), col] = self.config.mag_limits[col]

        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        if self.config.include_mag_errors:  #pragma: no cover
            pdfs = self.model.posterior(flow_df,
                                        column=self.config.redshift_column_name,
                                        seed=self.config.flow_seed,
                                        grid=self.zgrid,
                                        err_samples=self.config.n_error_samples)
        else:
            pdfs = self.model.posterior(flow_df,
                                        column=self.config.redshift_column_name,
                                        grid=self.zgrid)
        zmode = np.array([self.zgrid[np.argmax(pdf)] for pdf in pdfs]).flatten()
        qp_distn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=pdfs))
        qp_distn.set_ancil(dict(zmode=zmode))
        self._do_chunk_output(qp_distn, start, end, first)
