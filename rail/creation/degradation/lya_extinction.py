"""Degrader that simulates Lyman-alpha extinction for high-redshift sources."""

from rail.creation.degradation import Degrader


class LyaExtinction(Degrader):
    """Degrader that simulates Lyman-alpha extinction for high-z sources.

    This degrader only simulates the mean Lyman-alpha extinction.
    The mean optical depth of the Lyman-alpha Forest is assumed to take
    the following form:
        tau(z) = tau_0 * (1 + z) ^ tau_gamma
    The default values used for tau_0 and tau_gamma come from
    https://arxiv.org/pdf/1904.01110.pdf

    To calculate the mean extinction, we must specify the mean shape of
    the galaxy SEDs over the photometric band in question. We parameterize
    the mean shape using a power law: z ^ s_gamma.

    Parameters
    ----------
    band_name : str
        The name of the band that the Lyman-alpha extinction is being
        applied to.
    band_throughput : str
        Path to a file containing the bandpass's throughput.
        For details about the throughput, see the notes below.
    tau_0 : float, default=5.54e-3
        Normalization of Lyman-alpha optical depth
    tau_gamma : float, default=3.182
        Power law exponent for Lyman-alpha optical depth
    s_gamma : float, default=0
        Exponent of the power law that specifies the mean shape of the
        galaxy SEDs over the support of the photometric band.

    Notes
    -----
    The band_throughput is assumed to be saved in a plain text file,
    consisting of two columns:
        (1) the wavelength in angstroms
        (2) the band's throughput
    This is the format provided by the SVO Filter Profile Service
    (http://svo2.cab.inta-csic.es/theory/fps/).

    It is not assumed that the throughput is normalized, but it is
    assumed that the detector is a photon counter, so that the flux from
    an SED $f(\lambda)$ observed in a bandpass with throughput $T(\lambda)$
    is proportional to $\int d\lambda \lambda * T(\lambda) * f(\lambda)$
    """

    name = "LyaExtinction"
    config_options = Degrader.config_options.copy()
    config_options.update(
        band_name=str,
        band_throughput=str,
        tau_0=5.54e-3,
        tau_gamma=3.182,
        s_gamma=0.0,
    )

    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)

    def run(self):
        """Run method.

        Adds magnitude decrements to account for Lyman-alpha exctinction.

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """
        data = self.get_data("input")

        #

        self.add_data("output", data)
