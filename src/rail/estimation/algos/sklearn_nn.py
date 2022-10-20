"""
Example code that implements a simple Neural Net predictor
for z_mode, and Gaussian centered at z_mode with base_width
read in fromfile and pdf width set to base_width*(1+zmode).
"""

import numpy as np
# from numpy import inf
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer
import qp


def_filt = ['u', 'g', 'r', 'i', 'z', 'y']
def_bands = [f"mag_{band}_lsst" for band in def_filt]

def make_color_data(data_dict, bands, ref_band, nondet_val):
    """
    make a dataset consisting of the i-band mag and the five colors

    Returns
    --------
    input_data: `ndarray` array of imag and 5 colors
    """
    input_data = data_dict[ref_band]
    # make colors and append to input data
    for i in range(len(bands)-1):
        # replace the non-detect 99s with 28.0 just arbitrarily for now
        band1 = data_dict[bands[i]]
        # band1err = data_dict[f'mag_err_{bands[i]}_lsst']
        band2 = data_dict[bands[i+1]]
        # band2err = data_dict[f'mag_err_{bands[i+1]}_lsst']
        # for j,xx in enumerate(band1):
        #    if np.isclose(xx,99.,atol=.01):
        #        band1[j] = band1err[j]
        #        band1err[j] = 1.0
        # for j,xx in enumerate(band2):
        #    if np.isclose(xx,99.,atol=0.01):
        #        band2[j] = band2err[j]
        #        band2err[j] = 1.0
        for band in [band1, band2]:
            if np.isnan(nondet_val): # pragma: no cover
                nondetmask = np.isnan(band)
            else: # pragma: no cover
                nondetmask = np.isclose(band, nondet_val)
            band[nondetmask] = 28.0
        input_data = np.vstack((input_data, band1-band2))
    return input_data.T


def regularize_data(data):
    """Utility function to prepare data for sklearn"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data)
    regularized_data = scaler.transform(data)
    return regularized_data


class Inform_SimpleNN(CatInformer):
    """
    Subclass to train a simple point estimate Neural Net photoz
    rather than actually predict PDF, for now just predict point zb
    and then put an error of width*(1+zb).  We'll do a "real" NN
    photo-z later.
    """

    name = 'Inform_SimpleNN'
    config_options = CatInformer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
                          zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
                          nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
                          width=Param(float, 0.05, msg="The ad hoc base width of the PDFs"),
                          bands=Param(list, def_bands, msg="bands to use in estimation"),
                          ref_band=Param(str, "mag_i_lsst", msg="reference magnitude"),
                          nondetect_val=Param(float, 99.0, msg="value to be replaced with magnitude limit for non detects"),
                          max_iter=Param(int, 500,
                                         msg="max number of iterations while "
                                         "training the neural net.  Too low a value will cause an "
                                         "error to be printed (though the code will still work, just"
                                          "not optimally)"))


    def __init__(self, args, comm=None):
        """ Constructor:
        Do CatInformer specific initialization """
        CatInformer.__init__(self, args, comm=comm)
        if self.config.ref_band not in self.config.bands:
            raise ValueError("ref_band not present in bands list! ")

    def run(self):
        """Train the NN model
        """
        import sklearn.neural_network as sknn
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  #pragma: no cover
            training_data = self.get_data('input')
        speczs = training_data['redshift']
        print("stacking some data...")
        color_data = make_color_data(training_data, self.config.bands,
                                     self.config.ref_band, self.config.nondetect_val)
        input_data = regularize_data(color_data)
        simplenn = sknn.MLPRegressor(hidden_layer_sizes=(12, 12),
                                     activation='tanh', solver='lbfgs',
                                     max_iter=self.config.max_iter)
        simplenn.fit(input_data, speczs)
        self.model = simplenn
        self.add_data('model', self.model)


class SimpleNN(CatEstimator):
    """
    Subclass to implement a simple point estimate Neural Net photoz
    rather than actually predict PDF, for now just predict point zb
    and then put an error of width*(1+zb).  We'll do a "real" NN
    photo-z later.
    """
    name = 'SimpleNN'
    config_options = CatEstimator.config_options.copy()
    config_options.update(width=Param(float, 0.05, msg="The ad hoc base width of the PDFs"),
                          ref_band=Param(str, "mag_i_lsst", msg="reference magnitude"),
                          nondetect_val=Param(float, 99.0, msg="value to be replaced with magnitude limit for non detects"),
                          bands=Param(list, def_bands, msg="bands to use in estimation"))

    def __init__(self, args, comm=None):
        """ Constructor:
        Do CatEstimator specific initialization """
        CatEstimator.__init__(self, args, comm=comm)
        if self.config.ref_band not in self.config.bands:
            raise ValueError("ref_band is not in list of bands!")

    def _process_chunk(self, start, end, data, first):
        color_data = make_color_data(data, self.config.bands,
                                     self.config.ref_band, self.config.nondetect_val)
        input_data = regularize_data(color_data)
        zmode = np.round(self.model.predict(input_data), 3)
        widths = self.config.width * (1.0+zmode)
        qp_dstn = qp.Ensemble(qp.stats.norm, data=dict(loc=np.expand_dims(zmode, -1), #pylint: disable=no-member
                                                       scale=np.expand_dims(widths, -1)))
        qp_dstn.set_ancil(dict(zmode=zmode))
        self._do_chunk_output(qp_dstn, start, end, first)
