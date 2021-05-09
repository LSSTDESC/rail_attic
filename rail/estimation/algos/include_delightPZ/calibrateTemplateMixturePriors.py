
import sys
from mpi4py import MPI
import numpy as np
from scipy.interpolate import interp1d
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
import scipy.stats
import matplotlib.pyplot as plt
import emcee
import corner

import coloredlogs
import logging


# Create a logger object.
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')





def calibrateTemplateMixturePriors(configfilename,make_plot=False):
    """

    :param configfilename:
    :return:
    """
    # Parse parameters file

    logger.info("--- calibrate Template Mixture Priors ---")

    paramFileName = configfilename
    params = parseParamFile(paramFileName, verbose=False)

    comm = MPI.COMM_WORLD
    threadNum = comm.Get_rank()
    numThreads = comm.Get_size()

    DL = approx_DL()
    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
    numZ = redshiftGrid.size

    # Locate which columns of the catalog correspond to which bands.
    bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,refBandColumn = readColumnPositions(params, prefix="training_")

    dir_seds = params['templates_directory']
    dir_filters = params['bands_directory']
    lambdaRef = params['lambdaRef']
    sed_names = params['templates_names']
    numBands = bandIndices.size
    nt = len(sed_names)


    f_mod = np.zeros((numZ, nt, len(params['bandNames'])))

    # model of flux-redshift for each template
    for t, sed_name in enumerate(sed_names):
        f_mod[:, t, :] = np.loadtxt(dir_seds + '/' + sed_name + '_fluxredshiftmod.txt')

    numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))

    msg = 'Number of Training Objects ' + str(numObjectsTraining)
    logger.info(msg)

    numMetrics = 7 + len(params['confidenceLevels'])


    allFluxes = np.zeros((numObjectsTraining, numBands))
    allFluxesVar = np.zeros((numObjectsTraining, numBands))

    redshifts = np.zeros((numObjectsTraining, 1))
    fmod_atZ = np.zeros((numObjectsTraining, nt, numBands))

    # Now loop over training set to compute likelihood function
    loc = - 1
    trainingDataIter = getDataFromFile(params, 0, numObjectsTraining,prefix="training_", getXY=False)

    # loop on traning
    for z, ell, bands, fluxes, fluxesVar, bCV, fCV, fvCV in trainingDataIter:
        loc += 1
        allFluxes[loc, :] = fluxes
        allFluxesVar[loc, :] = fluxesVar
        redshifts[loc, 0] = z

        # loop on SED
        for t, sed_name in enumerate(sed_names):
            for ib, b in enumerate(bands):
                fmod_atZ[loc, t, ib] = ell * np.interp(z, redshiftGrid,f_mod[:, t, b])

    zZmax = redshifts[:, 0] / redshiftGrid[-1]


    pmin = np.repeat(0., 2*nt)
    pmax = np.repeat(200., 2*nt)

    ndim, nwalkers = 2*nt, 100

    def approx_flux_likelihood_multiobj(
            f_obs,  # no, nf
            f_obs_var,  # no, nf
            f_mod,  # no, nt, nf
            ell_hat,  # 1
            ell_var,  # 1
            returnChi2=False,
            normalized=True):

        assert len(f_obs.shape) == 2
        assert len(f_obs_var.shape) == 2
        assert len(f_mod.shape) == 3
        no, nt, nf = f_mod.shape
        f_obs_r = f_obs[:, None, :]
        var = f_obs_var[:, None, :]
        invvar = np.where(f_obs_r / var < 1e-6, 0.0, var ** -1.0)  # nz * nt * nf
        FOT = np.sum(f_mod * f_obs_r * invvar, axis=2) \
              + ell_hat / ell_var  # no * nt
        FTT = np.sum(f_mod ** 2 * invvar, axis=2) \
              + 1. / ell_var  # no * nt
        FOO = np.sum(f_obs_r ** 2 * invvar, axis=2) \
              + ell_hat ** 2 / ell_var  # no * nt
        sigma_det = np.prod(var, axis=2)
        chi2 = FOO - FOT ** 2.0 / FTT  # no * nt
        denom = np.sqrt(FTT)
        if normalized:
            denom *= np.sqrt(sigma_det * (2 * np.pi) ** nf * ell_var)
        like = np.exp(-0.5 * chi2) / denom  # no * nt
        if returnChi2:
            return chi2
        else:
            return like

    def lnprob(params, nt, allFluxes, allFluxesVar, zZmax, fmod_atZ, pmin, pmax):
        if np.any(params > pmax) or np.any(params < pmin):
            return - np.inf

        alphas0 = dirichlet(params[0:nt], rsize=1).ravel()[None, :]  # 1, nt
        alphas1 = dirichlet(params[nt:2 * nt], rsize=1).ravel()[None, :]  # 1, nt
        alphas_atZ = zZmax[:, None] * (alphas1 - alphas0) + alphas0  # no, nt
        # fmod_atZ: no, nt, nf
        fmod_atZ_t = (fmod_atZ * alphas_atZ[:, :, None]).sum(axis=1)[:, None, :]
        # no, 1, nf
        sigma_ell = 1e3
        like_grid = approx_flux_likelihood_multiobj(allFluxes, allFluxesVar, fmod_atZ_t, 1,
                                                    sigma_ell ** 2.).ravel()  # no,
        eps = 1e-305
        ind = like_grid > eps
        theprob = np.log(like_grid[ind]).sum()
        return theprob


    p0 = [pmin + (pmax-pmin)*np.random.uniform(0, 1, size=ndim) for i in range(nwalkers)]

    for i in range(10):
        print(lnprob(p0[i], nt, allFluxes, allFluxesVar, zZmax, fmod_atZ, pmin, pmax))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,threads=4,args=[nt, allFluxes, allFluxesVar, zZmax,fmod_atZ, pmin, pmax])
    pos, prob, state = sampler.run_mcmc(p0, 200)
    sampler.reset()
    sampler.run_mcmc(pos, 2000)
    print("Mean acceptance fraction: {0:.3f}" .format(np.mean(sampler.acceptance_fraction)))
    samples = sampler.chain.reshape((-1, ndim))
    lnprob = sampler.lnprobability.reshape((-1, 1))

    params_mean = samples.mean(axis=0)
    params_std = samples.std(axis=0)

    if make_plot:
        fig, axs = plt.subplots(4, 5, figsize=(16, 8))
        axs = axs.ravel()
        for i in range(ndim):
            axs[i].hist(samples[:, i], 50, color="k", histtype="step")
            axs[i].axvspan(params_mean[i]-params_std[i],params_mean[i]+params_std[i], color='gray', alpha=0.5)
            axs[i].axvline(params_mean[i], c='k', lw=2)
        fig.tight_layout()
        fig.savefig('priormixture_parameters.pdf')

        fig = plot_params(params_mean)
        fig.savefig('priormixture_meanparameters.pdf')

    print("params_mean", params_mean)
    print("params_std", params_std)

    alphas = params_mean[0:nt]
    alpha0 = np.sum(alphas)
    print("alpha0:", ', '.join(['%.2g' % x for x in alphas / alpha0]))
    print("alpha0 err:", ', '.join(['%.2g' % x for x in np.sqrt(alphas*(alpha0-alphas)/alpha0**2/(alpha0+1))]))

    alphas = params_mean[nt:2*nt]
    alpha0 = np.sum(alphas)
    print("alpha1:", ', '.join(['%.2g' % x for x in alphas / alpha0]))
    print("alpha1 err:", ', '.join(['%.2g' % x for x in np.sqrt(alphas*(alpha0-alphas)/alpha0**2/(alpha0+1))]))



    alphas = params_mean[0:nt]
    betas = params_mean[nt:2 * nt]

    alpha0 = np.sum(alphas)
    print("p_t:", ' '.join(['%.2g' % x for x in alphas / alpha0]))
    print("p_z_t:", ' '.join(['%.2g' % x for x in betas]))
    print("p_t err:", ' '.join(['%.2g' % x for x in np.sqrt(alphas * (alpha0 - alphas) / alpha0 ** 2 / (alpha0 + 1))]))




    if make_plot:
        fig = corner.corner(samples)
        fig.savefig("trianglemixture.pdf")


def plot_params(params):
    fig, axs = plt.subplots(4, 4, figsize=(16, 8))
    axs = axs.ravel()
    alphas = params[0:nt]
    alpha0 = np.sum(alphas)
    dirsamples = dirichlet(alphas, 1000)
    for i in range(nt):
        mean = alphas[i]/alpha0
        std = np.sqrt(alphas[i] * (alpha0-alphas[i]) / alpha0**2 / (alpha0+1))
        axs[i].axvspan(mean-std, mean+std, color='gray', alpha=0.5)
        axs[i].axvline(mean, c='k', lw=2)
        axs[i].axvline(1/nt, c='k', lw=1, ls='dashed')
        axs[i].set_title('alpha0 = '+str(alphas[i]))
        axs[i].set_xlim([0, 1])
        axs[i].hist(dirsamples[:, i], 50, color="k", histtype="step")
    alphas = params[nt:2*nt]
    alpha0 = np.sum(alphas)
    dirsamples = dirichlet(alphas, 1000)
    for i in range(nt):
        mean = alphas[i]/alpha0
        std = np.sqrt(alphas[i] * (alpha0-alphas[i]) / alpha0**2 / (alpha0+1))
        axs[nt+i].axvspan(mean-std, mean+std, color='gray', alpha=0.5)
        axs[nt+i].axvline(mean, c='k', lw=2)
        axs[nt+i].axvline(1/nt, c='k', lw=1, ls='dashed')
        axs[nt+i].set_title('alpha1 = '+str(alphas[i]))
        axs[nt+i].set_xlim([0, 1])
        axs[nt+i].hist(dirsamples[:, i], 50, color="k", histtype="step")
    fig.tight_layout()
    return fig

#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script


    msg=" calibrateTemplateMixturePriors "
    logger.info(msg)


    logger.debug("__name__:"+__name__)
    logger.debug("__file__:"+__file__)

    logger.info(" calibrate Template Priors ---")

    if len(sys.argv) < 2:
        raise Exception('Please provide a parameter file')

    calibrateTemplateMixturePriors(sys.argv[1])





