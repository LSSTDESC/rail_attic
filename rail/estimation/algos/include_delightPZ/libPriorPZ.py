#######################################################################################
#
# script : libpriorPZ
#
# Provide a library of priors on photoZ
#
# author : Sylvie Dagoret-Campagne
# affiliation : IJCLab/IN2P3/CNRS
#
# from https://github.com/ixkael/Photoz-tools
#
######################################################################################
import sys
import numpy as np
from scipy.interpolate import interp1d
from pprint import pprint

import coloredlogs
import logging


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s, %(name)s[%(process)d] %(levelname)s %(message)s')


def mknames(nt):
    return ['Elliptical ' + str(i + 1) for i in range(nt[0])] \
           + ['Spiral ' + str(i + 1) for i in range(nt[1])] \
           + ['Starburst ' + str(i + 1) for i in range(nt[2])]



# This is the prior HDFN prior from Benitez 2000, adapted from the BPZ code.
# This could be replaced with any redshift, magnitude, and type distribution.
def bpz_prior(z, m, nt):
    """
    bpz_prior(z, m, nt):

    - z grid of redshift
    - m maximum magnitude
    - nt : number of types

    """
    nz = len(z)
    momin_hdf = 20.
    if m > 32.: m = 32.
    if m < 20.: m = 20.
    # nt Templates = nell Elliptical + nsp Spiral + nSB starburst
    try:  # nt is a list of 3 values
        nell, nsp, nsb = nt
    except:  # nt is a single value
        nell = 1  # 1 Elliptical in default template set
        nsp = 2  # 2 Spirals in default template set
        nsb = nt - nell - nsp  # rest Irr/SB
    nn = nell, nsp, nsb
    nt = sum(nn)
    # See Table 1 of Benitez00
    a = 2.465, 1.806, 0.906
    zo = 0.431, 0.390, 0.0626
    km = 0.0913, 0.0636, 0.123
    k_t = 0.450, 0.147
    a = np.repeat(a, nn)
    zo = np.repeat(zo, nn)
    km = np.repeat(km, nn)
    k_t = np.repeat(k_t, nn[:2])

    # Fractions expected at m = 20: 35% E/S0, 50% Spiral, 15% Irr
    fo_t = 0.35, 0.5
    fo_t = fo_t / np.array(nn[:2])
    fo_t = np.repeat(fo_t, nn[:2])

    dm = m - momin_hdf
    zmt = np.clip(zo + km * dm, 0.01, 15.)
    zmt_at_a = zmt ** (a)
    zt_at_a = np.power.outer(z, a)

    # Morphological fractions
    nellsp = nell + nsp
    f_t = np.zeros((len(a),), float)
    f_t[:nellsp] = fo_t * np.exp(-k_t * dm)
    f_t[nellsp:] = (1. - np.add.reduce(f_t[:nellsp])) / float(nsb)

    # Formula: zm=zo+km*(m_m_min)  and  p(z|T,m)=(z**a)*exp(-(z/zm)**a)
    p_i = zt_at_a[:nz, :nt] * np.exp(-np.clip(zt_at_a[:nz, :nt] / zmt_at_a[:nt], 0., 700.))

    # This eliminates the very low level tails of the priors
    norm = np.add.reduce(p_i[:nz, :nt], 0)
    p_i[:nz, :nt] = np.where(np.less(p_i[:nz, :nt] / norm[:nt], 1e-2 / float(nz)),
                             0., p_i[:nz, :nt] / norm[:nt])
    norm = np.add.reduce(p_i[:nz, :nt], 0)
    p_i[:nz, :nt] = p_i[:nz, :nt] / norm[:nt] * f_t[:nt]
    return p_i  # return 2D template nz x nt


def libPriorPZ(z_grid,maglim,nt=8):
    """

    :return:
    """

    msg = "--- libPriorPZ"
    #logger.info(msg)

    # Just some boolean indexing of templates used. Needed later for some BPZ fcts.
    selectedtemplates = np.repeat(False, nt)

    # Using all templates
    templatetypesnb = (1, 2, 5)  # nb of ellipticals, spirals, and starburst used in the 8-template library.
    selectedtemplates[:] = True

    # Uncomment that to use three templates using
    # templatetypesnb = (1,1,1) #(1,2,8-3)
    # selectedtemplates[0:1] = True
    nt = sum(templatetypesnb)

    ellipticals = ['El_B2004a.sed'][0:templatetypesnb[0]]
    spirals = ['Sbc_B2004a.sed', 'Scd_B2004a.sed'][0:templatetypesnb[1]]
    irregulars = ['Im_B2004a.sed', 'SB3_B2004a.sed', 'SB2_B2004a.sed',
                  'ssp_25Myr_z008.sed', 'ssp_5Myr_z008.sed'][0:templatetypesnb[2]]
    template_names = [nm.replace('.sed', '') for nm in ellipticals + spirals + irregulars]

    # Use the p(z,t,m) distribution defined above
    m = maglim  # some reference magnitude
    p_z__t_m = bpz_prior(z_grid, m, templatetypesnb) #  2D template nz x nt

    # Convenient function for template names
    def mknames(nt):
        return ['Elliptical ' + str(i + 1) for i in range(nt[0])] \
               + ['Spiral ' + str(i + 1) for i in range(nt[1])] \
               + ['Starburst ' + str(i + 1) for i in range(nt[2])]

    names = mknames(templatetypesnb)

    return p_z__t_m # return 2D template nz x nt





if __name__ == "__main__":  # pragma: no cover
    # execute only if run as a script


    msg="Start libpriorPZ.py"
    logger.info(msg)
    logger.info("--- libPriorPZ ---")

    z_grid_binsize = 0.001
    z_grid_edges = np.arange(0.0, 3.0, z_grid_binsize)
    z_grid = (z_grid_edges[1:] + z_grid_edges[:-1]) / 2.

    m = 26.0  # some reference magnitude
    nt=8

    p_z__t_m = libPriorPZ(z_grid,maglim=m,nt=nt)

    np.set_printoptions(threshold=20, edgeitems=10, linewidth=140,
                        formatter=dict(float=lambda x: "%.3e" % x))  # float arrays %.3g
    print(p_z__t_m )
