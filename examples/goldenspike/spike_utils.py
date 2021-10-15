import numpy as np
import photErrorModel
import pandas as pd
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling


filterList = ["LSSTu", "LSSTg", "LSSTr", "LSSTi", "LSSTz", "LSSTy"]
nFilter = len(filterList)
pars = {}
pars["tvis"] = 30.
pars["sigmaSys"] = 0.0025   # expected irreducible error
pars["nYrObs"] = 10          # number of years of observations
# number of visits per year
pars["nVisYr"] = {'LSSTu': 5.6, 'LSSTg': 8, 'LSSTr': 18.4, 'LSSTi': 18.4, 'LSSTz': 16, 'LSSTy': 16}
# band dependent parameter
pars["gamma"] = {'LSSTu': 0.038, 'LSSTg': 0.039, 'LSSTr': 0.039, 'LSSTi': 0.039, 'LSSTz': 0.039, 'LSSTy': 0.039}
# band dependent parameter
pars["Cm"] = {'LSSTu': 23.09, 'LSSTg': 24.42, 'LSSTr': 24.44, 'LSSTi': 24.32, 'LSSTz': 24.16, 'LSSTy': 23.73}
# sky brightness
pars["msky"] = {'LSSTu': 22.99, 'LSSTg': 22.26, 'LSSTr': 21.2, 'LSSTi': 20.48, 'LSSTz': 19.6, 'LSSTy': 18.61}
# seeing
pars["theta"] = {'LSSTu': 0.81, 'LSSTg': 0.77, 'LSSTr': 0.73, 'LSSTi': 0.71, 'LSSTz': 0.69, 'LSSTy': 0.68}
# extinction coefficient
pars["km"] = {'LSSTu': 0.491, 'LSSTg': 0.213, 'LSSTr': 0.126, 'LSSTi': 0.096, 'LSSTz': 0.069, 'LSSTy': 0.170}
pars["airMass"] = 1.2       # air mass
pars["extendedSource"] = 0. # extended source model: simple constant added to m5 (makes m5 fainter)
pars["minFlux"] = 2.5e-40   # set minimum allowed flux (equivalent to mag=99)


def make_errors(mags, band):
    errmod = photErrorModel.LSSTErrorModel(pars)
    ngal = len(mags)
    magerr = np.zeros(ngal)
    for i in range(ngal):
        magerr[i] = errmod.getMagError(mags[i], band)
    return magerr


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
