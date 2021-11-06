"""
first pass implementation of pzflow estimator
First pass will ignore photometric errors and just do
things in terms of magnitudes, we will expand in a
future update
"""

import numpy as np
from rail.estimation.estimator import Estimator as BaseEstimation
from rail.estimation.utils import check_and_print_params
import pandas as pd
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus
from pzflow.bijectors import StandardScaler, RollingSplineCoupling
import qp


def computemeanstd(df):
    """
      compute colors from the magnitudes and compute their
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
def_maglims = dict(mag_u_lsst=27.79,
                   mag_g_lsst=29.04,
                   mag_r_lsst=29.06,
                   mag_i_lsst=28.62,
                   mag_z_lsst=27.98,
                   mag_y_lsst=27.05)
def_param = dict(run_params=dict(zmin=0.0,
                                 zmax=3.0,
                                 nzbins=301,
                                 ref_column_name='mag_i_lsst',
                                 column_names=refcols,
                                 mag_limits=def_maglims,
                                 soft_sharpness=10,
                                 soft_idx_col=0,
                                 redshift_column_name='redshift',
                                 num_training_epocs=200,
                                 inform_options=dict(save_train=True,
                                                     load_model=False,
                                                     modelfile="PZflowPDF.pkl")
                                 )
                 )

desc_dict = dict(zmin="min z",
                 zmax="max_z",
                 nzbins="num z bins",
                 ref_column_name="name for reference column",
                 column_names="column names to be used in flow",
                 mag_limits="1 sigma mag limits",
                 soft_sharpness="sharpening paremeter for SoftPlus",
                 soft_idx_col="index column for SoftPlus",
                 redshift_column_name="name of redshift column",
                 num_training_epocs="number flow training epocs",
                 inform_options="inform options"
                 )


class PZFlowPDF(BaseEstimation):
    """
    Subclass to implement a pzflow-based estimator
    """
    def __init__(self, base_dict, config_dict='None'):
        """
        Parameters
        ----------
        base_dict: dict
          dictionary of variables from base.yaml-type file
        config_dict: dict
          dictionary of all variables read in from the run_params
          values in the yaml file
        """
        if config_dict == "None":
            print("No config file supplied, using default parameters")
            config_dict = def_param
        config_dict = check_and_print_params(config_dict, def_param,
                                             desc_dict)
        super().__init__(base_dict, config_dict)

        inputs = config_dict['run_params']
        self.zmin = inputs['zmin']
        self.zmax = inputs['zmax']
        self.nzbins = inputs['nzbins']
        self.refcol = inputs['ref_column_name']
        self.col_names = inputs['column_names']
        self.maglims = inputs['mag_limits']
        self.sharpness = inputs['soft_sharpness']
        self.idx_col = inputs['soft_idx_col']
        self.redshiftname = inputs['redshift_column_name']
        self.trainepochs = inputs['num_training_epochs']
        self.inform_options = inputs['inform_options']
        usecols = self.col_names.copy()
        usecols.append(self.redshiftname)
        self.usecols = usecols

    def inform(self, training_data):
        """
        train a flow based on the training data
        This is mostly based off of the pzflow example notebook
        """
        input_df = pd.DataFrame(training_data)

        flowdf = input_df[self.usecols]
        # replace nondetects
        # will fancy this up later with a flow to sample from truth
        for col in self.col_names:
            flowdf.loc[np.isclose(flowdf[col], 99.), col] = self.maglims[col]

        # compute means and stddevs for StandardScalar transform
        col_means, col_stds = computemeanstd(flowdf)

        ref_idx = flowdf.columns.get_loc(self.refcol)
        mag_idx = [flowdf.columns.get_loc(col) for col in self.col_names]
        nlayers = flowdf.shape[1]
        bijector = Chain(
            ColorTransform(ref_idx, mag_idx),
            InvSoftplus(self.idx_col, self.sharpness),
            StandardScaler(col_means, col_stds),
            RollingSplineCoupling(nlayers),
        )
        model = Flow(flowdf.columns, bijector)
        _ = model.train(flowdf[self.usecols], epochs=self.trainepochs,
                        verbose=True)
        self.model = model
        if self.inform_options['save_train']:
            model.save(self.inform_options['modelfile'])

    def load_pretrained_model(self):
        try:
            modelfile = self.inform_options['modelfile']
        except KeyError:
            print("inform_options['modelfile'] not specified, exiting!")
            raise KeyError("inform_options['modelfile'] not found!")
        try:
            self.model = Flow(file=modelfile)
            print(f"success in loading {modelfile}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {modelfile} not found!")

    def estimate(self, test_data):
        """
        calculate and return PDFs for each galaxy using the trained flow
        """
        # flow expects dataframe
        test_df = pd.DataFrame(test_data)
        flow_df = test_df[self.usecols]
        # replace nondetects
        for col in self.col_names:
            flow_df.loc[np.isclose(flow_df[col], 99.), col] = self.maglims[col]

        self.zgrid = np.linspace(self.zmin, self.zmax, self.nzbins)
        pdfs = self.model.posterior(flow_df, column=self.redshiftname,
                                    grid=self.zgrid)
        if self.output_format == 'qp':
            qp_distn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid,
                                                        yvals=pdfs))
            return qp_distn
        else:
            zmode = np.array([self.zgrid[np.argmax(pdf)]
                              for pdf in pdfs]).flatten()
            pz_dict = {'zmode': zmode, 'pz_pdf': pdfs}
            return pz_dict
