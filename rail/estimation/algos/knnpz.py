"""
quick implementation of k nearest neighbor estimator
First pass will ignore photometric errors and just do
things in terms of magnitudes, we will expand in a
future update
"""

import numpy as np
import copy
import pickle
from rail.estimation.estimator import Estimator as BaseEstimation
from rail.estimation.utils import check_and_print_params
from rail.evaluation.metrics.cdeloss import CDELoss
from sklearn.neighbors import KDTree
import pandas as pd
import qp


def_bands = ['u', 'g', 'r', 'i', 'z', 'y']
refcols = [f"mag_{band}_lsst" for band in def_bands]
allcols = refcols.copy()
for band in def_bands:
    allcols.append(f"mag_{band}_lsst_err")
def_maglims = dict(mag_u_lsst=27.79,
                   mag_g_lsst=29.04,
                   mag_r_lsst=29.06,
                   mag_i_lsst=28.62,
                   mag_z_lsst=27.98,
                   mag_y_lsst=27.05)
def_param = dict(run_params=dict(zmin=0.0,
                                 zmax=3.0,
                                 nzbins=301,
                                 trainfrac=0.75,
                                 random_seed=87,
                                 ref_column_name='mag_i_lsst',
                                 column_names=refcols,
                                 mag_limits=def_maglims,
                                 sigma_grid_min=0.01,
                                 sigma_grid_max=0.075,
                                 ngrid_sigma=10,
                                 leaf_size=15,
                                 nneigh_min=3,
                                 nneigh_max=7,
                                 redshift_column_name='redshift',
                                 inform_options=dict(save_train=True,
                                                     load_model=False,
                                                     modelfile="PZflowPDF.pkl")
                                 )
                 )

desc_dict = dict(zmin="min z",
                 zmax="max_z",
                 nzbins="num z bins",
                 trainfrac="fraction of training data used to make tree, rest used to set best sigma",
                 random_seed="int, random seed for reproducibility",
                 ref_column_name="name for reference column",
                 column_names="column names to be used in NN, *ASSUMED TO BE IN INCREASING WL ORDER!*",
                 mag_limits="1 sigma mag limits",
                 sigma_grid_min="minimum value of sigma for grid check",
                 sigma_grid_max="maximum value of sigma for grid check",
                 ngrid_sigma="number of grid points in sigma check",
                 leaf_size="int, min leaf size for KDTree",
                 nneigh_min="int, min number of near neighbors to use for PDF fit",
                 nneigh_max="int, max number of near neighbors to use ofr PDF fit",
                 redshift_column_name="name of redshift column",
                 inform_options="inform options"
                 )


class KNearNeighPDF(BaseEstimation):
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
        self.zgrid = np.linspace(self.zmin, self.zmax, self.nzbins)
        self.trainfrac = inputs['trainfrac']
        self.seed = inputs['random_seed']
        self.refcol = inputs['ref_column_name']
        self.col_names = inputs['column_names']
        self.maglims = inputs['mag_limits']
        self.sig_min = inputs['sigma_grid_min']
        self.sig_max = inputs['sigma_grid_max']
        self.nsig = inputs['ngrid_sigma']
        self.leaf_size = inputs['leaf_size']
        self.nn_min = inputs['nneigh_min']
        self.nn_max = inputs['nneigh_max']
        self.redshiftname = inputs['redshift_column_name']
        self.inform_options = inputs['inform_options']
        usecols = self.col_names.copy()
        usecols.append(self.redshiftname)
        self.usecols = usecols

    def computecolordata(self, df):
        newdict = {}
        newdict['x'] = df[self.refcol]
        nbands = len(self.col_names)-1
        for k in range(nbands):
            newdict[f'x{k}'] = df[self.col_names[k]] - df[self.col_names[k+1]]
        newdf = pd.DataFrame(newdict)
        coldata = newdf.to_numpy()

        return coldata

    def makepdf(self, dists, ids, szs, sigma):
        sigmas = np.full_like(dists, sigma)
        weights = 1./dists
        weights /= weights.sum(axis=1, keepdims=True)
        norms = np.sum(weights, axis=1)
        means = szs[ids]

        pdfs = qp.Ensemble(qp.mixmod, data=dict(means=means, stds=sigmas, weights=weights))
        return pdfs

    def inform(self, training_data):
        """
        train a KDTree on a fraction of the training data
        """
        input_df = pd.DataFrame(training_data)
        knndf = input_df[self.usecols]
        # replace nondetects
        # will fancy this up later with a flow to sample from truth
        for col in self.col_names:
            knndf.loc[np.isclose(knndf[col], 99.), col] = self.maglims[col]

        self.trainszs = np.array(knndf[self.redshiftname])
        colordata = self.computecolordata(knndf)
        nobs = colordata.shape[0]
        np.random.seed(self.seed)  # set seed for reproducibility
        perm = np.random.permutation(nobs)
        ntrain = round(nobs * self.trainfrac)
        xtrain_data = colordata[perm[:ntrain]]
        train_data = copy.deepcopy(xtrain_data)
        val_data = colordata[perm[ntrain:]]
        xtrain_sz = self.trainszs[perm[:ntrain]].copy()
        train_sz = np.array(copy.deepcopy(xtrain_sz))
        np.savetxt("TEMPZFILE.out", train_sz)
        val_sz = np.array(self.trainszs[perm[ntrain:]])
        print(f"split into {len(train_sz)} training and {len(val_sz)} validation samples")
        tmpmodel = KDTree(train_data, leaf_size=self.leaf_size)
        # Find best sigma and n_neigh by minimizing CDE Loss
        bestloss = 1e20
        bestsig = self.sig_min
        bestnn = self.nn_min
        siggrid = np.linspace(self.sig_min, self.sig_max, self.nsig)
        for sig in siggrid:
            for nn in range(self.nn_min, self.nn_max+1):
                print(f"sigma: {sig} num neigh: {nn}...")
                dists, idxs = tmpmodel.query(val_data, k=nn)
                ens = self.makepdf(dists, idxs, train_sz, sig)
                cdelossobj = CDELoss(ens, self.zgrid, val_sz)
                cdeloss = cdelossobj.evaluate().statistic
                print(f"sigma: {sig} num neigh: {nn} loss: {cdeloss}")
                if cdeloss < bestloss:
                    bestsig = sig
                    bestnn = nn
                    bestloss = cdeloss
        self.numneigh = bestnn
        self.sigma = bestsig
        print(f"\n\n\nbest fit values are sigma={self.sigma} and numneigh={self.numneigh}\n\n\n")
        # remake tree with full dataset!
        model = KDTree(colordata, leaf_size=self.leaf_size)
        self.model = model
        if self.inform_options['save_train']:
            pickledict = dict(model=model, bestsig=self.sigma, nneigh=self.numneigh, truezs=self.trainszs)
            with open(self.inform_options['modelfile'], 'wb') as f:
                pickle.dump(file=f, obj=pickledict,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def load_pretrained_model(self):
        try:
            modelfile = self.inform_options['modelfile']
        except KeyError: #pragma: no cover
            print("inform_options['modelfile'] not specified, exiting!")
            raise KeyError("inform_options['modelfile'] not found!")
        try:
            with open(modelfile, 'rb') as infp:
                newdict = pickle.load(infp)
            self.sigma = newdict['bestsig']
            self.numneigh = newdict['nneigh']
            self.model = newdict['model']
            self.trainszs = newdict['truezs']
        except KeyError: #pragma: no cover
            raise KeyError("missing pretrained model component")
        print(f"loaded KDTree from file, using sigma of {self.sigma} and {self.numneigh} neighbors")

    def estimate(self, test_data):
        """
        calculate and return PDFs for each galaxy using the trained flow
        """
        # flow expects dataframe
        test_df = pd.DataFrame(test_data)
        knn_df = test_df[self.usecols]

        # replace nondetects
        for col in self.col_names:
            knn_df.loc[np.isclose(knn_df[col], 99.), col] = self.maglims[col]

        testcolordata = self.computecolordata(knn_df)
        dists, idxs = self.model.query(testcolordata, k=self.numneigh)
        test_ens = self.makepdf(dists, idxs, self.trainszs, self.sigma)

        if self.output_format == 'qp':
            return test_ens
        else:
            zmode = test_ens.mode(grid=self.zgrid)
            pdfs = test_ens.pdf(self.zgrid)
            pz_dict = {'zmode': zmode, 'pz_pdf': pdfs}
            return pz_dict
