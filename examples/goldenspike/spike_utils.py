import numpy as np
import pandas as pd
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling


def transformdata(df):
    tmpdict = {}
    tmpdict['redshift'] = df['redshift']
    tmpdict['i'] = df['mag_i_lsst']
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    usecols = ['redshift', 'i']
    for k in range(5):
        newcol = f'{bands[k]}{bands[k+1]}'
        usecols.append(newcol)
        tmpdict[f'{bands[k]}{bands[k+1]}'] = df[f'mag_{bands[k]}_lsst'] - df[f'mag_{bands[k+1]}_lsst']
    tmpdf = pd.DataFrame(tmpdict)
    return np.array(tmpdf[usecols])


def generateflow(data=None, hpix=None, frac=20):
    """
    generate a pzflow object from the input data
    Parameters:
    -----------
    data: dataframe
      pandas dataframe input GCR-like data
    hpix: int
      healpix number if reading from GCR
    frac: int
      fraction of GCR data from healpix to keep, i.e. df[::frac]

    Returns:
    flow: pzflow
      flow to generate data modeled on the input data
    """
    ref_idx = data.columns.get_loc("mag_i_lsst")
    mag_idx = [data.columns.get_loc(col) for col in ["mag_u_lsst", "mag_g_lsst", "mag_r_lsst",
                                                     "mag_i_lsst", "mag_z_lsst", "mag_y_lsst"]]
    column_idx = 0
    sharpness = 10
    data_temp = transformdata(data)
    means = data_temp.mean(axis=0)
    stds = data_temp.std(axis=0)
    del data_temp
    nlayers = data.shape[1]
    bijector = Chain(
        ColorTransform(ref_idx, mag_idx),
        InvSoftplus(column_idx, sharpness),
        StandardScaler(means, stds),
        RollingSplineCoupling(nlayers),
        )
    flow = Flow(data.columns, bijector)
    usecols = ['redshift', 'mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    losses = flow.train(data[usecols], epochs=200, verbose=True)
    return flow # , losses
