"""
Port of *some* parts of BPZ, not the entire codebase.
Much of the code is directly ported from BPZ, written
by Txitxo Benitez and Dan Coe (Benitez 2000), which
was modified by Will Hartley and Sam Schmidt to make
it python3 compatible.  It was then modified to work
with TXPipe and ceci by Joe Zuntz and Sam Schmidt
for BPZPipe.  This version for RAIL removes a few
features and concentrates on just predicting the PDF.

Missing from full BPZ:
-no tracking of 'best' type/TB
-no "interp" between templates
-no ODDS, chi^2, ML quantities
-plotting utilities
-no output of 2D probs (maybe later add back in)
-no 'cluster' prior mods
-no 'ONLY_TYPE' mode

"""

import os
import numpy as np
import glob
import qp
import rail
from ceci.config import StageParameter as Param
from rail.estimation.estimator import Estimator
from desc_bpz.useful_py3 import get_str, get_data, match_resol




class BPZ_lite(Estimator):
    """Estimator subclass to implement basic marginalized PDF for BPZ
    """
    config_options = Estimator.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="min z for grid"),
                          zmax=Param(float, 3.0, msg="max z for grid"),
                          dz=Param(float, 0.01, msg="delta z in grid"),
                          nzbins=Param(int, 301, msg="# of bins in zgrid"),
                          data_path=Param(str, "None",
                                          msg="data_path (str): file path to the "
                                          "SED, FILTER, and AB directories.  If left to "
                                          "default `None` it will use the install "
                                          "directory for rail + estimation/data"),
                          columns_file=Param(str, './examples/estimation/configs/test_bpz.columns',
                                             msg="name of the file specifying the columns"),
                          spectra_file=Param(str, 'SED/CWWSB4.list',
                                             msg="name of the file specifying the list of SEDs to use"),
                          madau_flag=Param(str, 'no',
                                           msg="set to 'yes' or 'no' to set whether to include intergalactic "
                                               "Madau reddening when constructing model fluxes"),
                          bands=Param(str, 'ugrizy',
                                      msg="the list of filter bands used by BPZ, e.g for LSST we would "
                                          "use 'ugrizy'"),
                          prior_band=Param(str, 'i',
                                           msg="specifies which band the magnitude/type prior is trained in, e.g. 'i'"),
                          prior_file=Param(str, 'hdfn_gen',
                                           msg="prior_file (str): the file "
                                           "name of the prior, which should be located "
                                           "in the root BPZ directory.  If the "
                                           "full prior file is named e.g. "
                                           "'prior_dc2_lsst_trained_model.py' then we should "
                                           "set the value to 'dc2_lsst_trained_model', as "
                                           "'prior_' is added to the name by BPZ"),
                          p_min=Param(float, 0.005,
                                      msg="BPZ sets all values of "
                                      "the PDF that are below p_min*peak_value to 0.0, "
                                      "p_min controls that fractional cutoff"),
                          gauss_kernel=Param(float, 0.0,
                                              msg="gauss_kernel (float): BPZ "
                                              "convolves the PDF with a kernel if this is set "
                                              "to a non-zero number"),
                          zp_errors=Param(np.ndarray, np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                                          msg="BPZ adds these values in quadrature to the photometric errors"),
                          mag_err_min=Param(float, 0.005,
                                            msg="a minimum floor for the magnitude errors to prevent a "
                                            "large chi^2 for very very bright objects"))

    def __init__(self, args, comm=None):
        """Constructor, build the Estimator, then do BPZ specific setup
        """
        Estimator.__init__(self, args, comm=comm)

        datapath = self.config['data_path']
        if datapath is None or datapath == "None":
            railpath = os.path.dirname(rail.__file__)
            tmpdatapath = os.path.join(railpath, "estimation/data")
            os.environ["BPZDATAPATH"] = tmpdatapath
            self.data_path = tmpdatapath
        else:  #pragma: no cover
            self.data_path = datapath
            os.environ['BPZDATAPATH'] = self.data_path
        if not os.path.exists(self.data_path): #pragma: no cover
            raise FileNotFoundError("BPZDATAPATH " + self.data_path
                                    + " does not exist! Check value of "
                                    + "data_path in config file!")

        # load the template fluxes from the AB files
        self.flux_templates = self._load_templates()

        # Load the AB files, or if they don't exist, create from SEDs*filters

    def _load_templates(self):

        # The redshift range we will evaluate on
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        z = self.zgrid

        data_path = self.data_path
        columns_file = self.config.columns_file
        ignore_rows = ['M_0', 'OTHER', 'ID', 'Z_S']
        filters = [f for f in get_str(columns_file, 0) if f not in ignore_rows]

        spectra_file = os.path.join(data_path, self.config.spectra_file)
        spectra = [s[:-4] for s in get_str(spectra_file)]

        nt = len(spectra)
        nf = len(filters)
        nz = len(z)
        flux_templates = np.zeros((nz, nt, nf))

        # make a list of all available AB files in the AB directory
        ab_file_list = glob.glob(os.path.join(data_path, "AB")+"/*.AB")
        ab_file_db = [os.path.split(x)[-1] for x in ab_file_list]

        for i, s in enumerate(spectra):
            for j, f in enumerate(filters):
                model = f"{s}.{f}.AB"
                if model not in ab_file_db:  #pragma: no cover
                    self._make_new_ab_file(s, f)
                model_path = os.path.join(data_path, "AB", model)
                zo, f_mod_0 = get_data(model_path, (0, 1))
                flux_templates[:, i, j] = match_resol(zo, f_mod_0, z)

        return flux_templates

    def _make_new_ab_file(self, spectrum, filter_):  #pragma: no cover
        from desc_bpz.bpz_tools_py3 import ABflux

        new_file = f"{spectrum}.{filter_}.AB"
        print(f"  Generating new AB file {new_file}....")
        ABflux(spectrum, filter, self.config.madau)

    def _preprocess_magnitudes(self, data):
        from desc_bpz.bpz_tools_py3 import e_mag2frac

        bands = self.config.bands

        # Load the magnitudes
        zp_frac = e_mag2frac(self.config.zp_errors)

        # Only one set of mag errors
        mag_errs = np.array([data[f'mag_err_{b}_lsst'] for b in bands]).T

        # But many sets of mags, for now
        # Group the magnitudes and errors into one big array
        mags = np.array([data[f'mag_{b}_lsst'] for b in bands]).T

        # Clip to min mag errors.
        # JZ: Changed the max value here to 20 as values in the lensfit
        # catalog of ~ 200 were causing underflows below that turned into
        # zero errors on the fluxes and then nans in the output
        np.clip(mag_errs, self.config.mag_err_min, 20, mag_errs)

        # Convert to pseudo-fluxes
        flux = 10.0**(-0.4*mags)
        flux_err = flux * (10.0**(0.4*mag_errs) - 1.0)

        # Check if an object is seen in each band at all.
        # Fluxes not seen at all are listed as infinity in the input,
        # so will come out as zero flux and zero flux_err.
        # Check which is which here, to use with the ZP errors below
        seen1 = (flux > 0) & (flux_err > 0)
        seen = np.where(seen1)
        # unseen = np.where(~seen1)
        # replace Joe's definition with more standard BPZ style
        nondetect = 99.
        nondetflux = 10.**(-0.4*nondetect)
        unseen = np.isclose(flux, nondetflux, atol=nondetflux*0.5)

        # replace mag = 99 values with 0 flux and 1 sigma limiting magnitude
        # value, which is stored in the mag_errs column for non-detects
        # NOTE: We should check that this same convention will be used in
        # LSST, or change how we handle non-detects here!
        flux[unseen] = 0.
        flux_err[unseen] = 10.**(-0.4*np.abs(mag_errs[unseen]))

        # Add zero point magnitude errors.
        # In the case that the object is detected, this
        # correction depends onthe flux.  If it is not detected
        # then BPZ uses half the errors instead
        add_err = np.zeros_like(flux_err)
        add_err[seen] = ((zp_frac*flux)**2)[seen]
        add_err[unseen] = ((zp_frac*0.5*flux_err)**2)[unseen]
        flux_err = np.sqrt(flux_err**2 + add_err)

        # Convert non-observed objects to have zero flux
        # and enormous error, so that their likelihood will be
        # flat. This follows what's done in the bpz script.
        nonobserved = -99.
        unobserved = np.isclose(mags, nonobserved)
        flux[unobserved] = 0.0
        flux_err[unobserved] = 1e108

        # Upate the input dictionary with new things we have calculated
        data['flux'] = flux
        data['flux_err'] = flux_err
        data['mags'] = mags
        return data

    def _estimate_pdf(self, flux_templates, kernel, flux, flux_err, mag_0, z):

        from desc_bpz.bpz_tools_py3 import p_c_z_t, prior

        prior_file = self.config.prior_file
        p_min = self.config.p_min

        nt = flux_templates.shape[1]

        # The likelihood and prior...
        pczt = p_c_z_t(flux, flux_err, flux_templates)
        L = pczt.likelihood

        # old prior code returns NoneType for prior if "flat" or "none"
        # just hard code the no prior case for now for backward compatibility
        if prior_file in ['flat', 'none']:  #pragma: no cover
            P = np.ones(L.shape)
        else:
            P = prior(z, mag_0, prior_file, nt, ninterp=0)  # hardcode interp 0

        post = L * P
        # Right now we jave the joint PDF of p(z,template). Marginalize
        # over the templates to just get p(z)
        post_z = post.sum(axis=1)

        # Convolve with Gaussian kernel, if present
        if kernel is not None:  #pragma: no cover
            post_z = np.convolve(post_z, kernel, 1)

        # Find the mode
        zmode = self.zgrid[np.argmax(post_z)]

        # Trim probabilities
        # below a certain threshold pct of p_max
        p_max = post_z.max()
        post_z[post_z < (p_max * p_min)] = 0

        # Normalize in the same way that BPZ does
        # But, only normalize if the elements don't sum to zero
        # if they are all zero, just leave p(z) as all zeros, as no templates
        # are a good fit.
        if not np.isclose(post_z.sum(), 0.0):
            post_z /= post_z.sum()

        return post_z, zmode

    def run(self):
        """
        This will likely mostly be copied from BPZPipe code
        """
        test_data = self.get_data('input', allow_missing=True)['photometry']
        test_data = self._preprocess_magnitudes(test_data)

        m_0_col = self.config.bands.index(self.config.prior_band)

        nz = len(self.zgrid)
        ng = test_data['mags'].shape[0]

        # Set up Gauss kernel for extra smoothing, if needed
        if self.config.gauss_kernel > 0:  #pragma: no cover
            dz = self.config.dz
            x = np.arange(-3.*self.config.gauss_kernel,
                          3.*self.config.gauss_kernel + dz/10., dz)
            kernel = np.exp(-(x/self.config.gauss_kernel)**2)
        else:
            kernel = None

        pdfs = np.zeros((ng, nz))
        zmode = np.zeros(ng)
        flux_temps = self.flux_templates
        zgrid = self.zgrid
        # Loop over all ng galaxies!
        for i in range(ng):
            mag_0 = test_data['mags'][i, m_0_col]
            flux = test_data['flux'][i]
            flux_err = test_data['flux_err'][i]
            pdfs[i], zmode[i] = self._estimate_pdf(flux_temps,
                                                    kernel, flux,
                                                    flux_err, mag_0,
                                                    zgrid)

        qp_dstn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=pdfs))
        qp_dstn.set_ancil(dict(zmode=zmode))
        self.add_data('output', qp_dstn)
