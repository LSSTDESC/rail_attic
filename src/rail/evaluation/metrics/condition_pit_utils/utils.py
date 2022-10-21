from scipy.stats import binom
from matplotlib import pyplot as plt
import numpy as np


def normalize(cde_estimates, x_grid, tol=1e-6, max_iter=200):
    """Normalizes conditional density estimates to be non-negative and
    integrate to one.

    :param cde_estimates: a numpy array or matrix of conditional density estimates.
    :param tol: float, the tolerance to accept for abs(area - 1).
    :param max_iter: int, the maximal number of search iterations.
    :returns: the normalized conditional density estimates.
    :rtype: numpy array or matrix.

    """
    if cde_estimates.ndim == 1:
        normalized_cde = _normalize(cde_estimates, x_grid, tol, max_iter)
    else:
        normalized_cde = np.apply_along_axis(_normalize, 1, cde_estimates, x_grid, tol=tol, max_iter=max_iter)
    return normalized_cde


def _normalize(density, x_grid, tol=1e-6, max_iter=500):
    """Normalizes a density estimate to be non-negative and integrate to
    one.

    :param density: a numpy array of density estimates.
    :param z_grid: an array, the grid points at the density is estimated.
    :param tol: float, the tolerance to accept for abs(area - 1).
    :param max_iter: int, the maximal number of search iterations.
    :returns: the normalized density estimate.
    :rtype: numpy array.

    """
    hi = np.max(density)
    lo = 0.0

    area = np.trapz(np.maximum(density, 0.0), x_grid)
    if area == 0.0:
        # replace with uniform if all negative density
        density[:] = 1 / (x_grid.max() - x_grid.min())
    elif area < 1:
        density /= area
        density[density < 0.0] = 0.0
        return density

    for _ in range(max_iter):
        mid = (hi + lo) / 2
        area = np.trapz(np.maximum(density - mid, 0.0), x_grid)
        if abs(1.0 - area) <= tol:
            break
        if area < 1.0:
            hi = mid
        else:
            lo = mid

    # update in place
    density -= mid
    density[density < 0.0] = 0.0

    return density


def kolmogorov_smirnov_statistic(cdf_test, cdf_ref):
    """
    cdf_test: CDF of the test distribution (array)
    cdf_ref: CDF of the reference distribution on the same grid (array)
    """
    ks = np.max(np.abs(cdf_test - cdf_ref), axis=-1)

    return ks


def cramer_von_mises(cdf_test, cdf_ref):
    """
    cdf_test: CDF of the test distribution (1D array)
    cdf_ref: CDF of the reference distribution on the same grid (1D array)
    """
    diff = (cdf_test - cdf_ref) ** 2

    cvm2 = np.trapz(diff, cdf_ref, axis=-1)
    return np.sqrt(cvm2)


def anderson_darling_statistic(cdf_test, cdf_ref, n_tot=1):
    """
    cdf_test: CDF of the test distribution (1D array)
    cdf_ref: CDF of the reference distribution on the same grid (1D array)
    n_tot:Scaling factor equal to the number of PDFs used to construct ECDF
    """
    num = (cdf_test - cdf_ref) ** 2
    den = cdf_ref * (1 - cdf_ref)

    ad2 = n_tot * np.trapz((num / den), cdf_ref, axis=-1)
    return np.sqrt(ad2)


def get_pit(cdes: np.ndarray, z_grid: np.ndarray, z_test: np.ndarray) -> np.ndarray:
    """
    Calculates PIT based on CDE

    cdes: a numpy array of conditional density estimates;
        each row corresponds to an observation, each column corresponds to a grid
        point
    z_grid: a numpy array of the grid points at which cde_estimates is evaluated
    z_test: a numpy array of the true z values corresponding to the rows of cde_estimates

    returns: A numpy array of values

    """
    # flatten the input arrays to 1D
    z_grid = np.ravel(z_grid)
    z_test = np.ravel(z_test)

    # Sanity checks
    nrow_cde, ncol_cde = cdes.shape
    n_samples = z_test.shape[0]
    n_grid_points = z_grid.shape[0]

    if nrow_cde != n_samples:
        raise ValueError(
            "Number of samples in CDEs should be the same as in z_test."
            "Currently %s and %s." % (nrow_cde, n_samples)
        )
    if ncol_cde != n_grid_points:
        raise ValueError(
            "Number of grid points in CDEs should be the same as in z_grid."
            "Currently %s and %s." % (nrow_cde, n_grid_points)
        )

    z_min = np.min(z_grid)
    z_max = np.max(z_grid)
    z_delta = (z_max - z_min) / (n_grid_points - 1)

    # Vectorized implementation using masked arrays
    pit = np.ma.masked_array(cdes, (z_grid > z_test[:, np.newaxis]))
    pit = np.trapz(pit, z_grid)

    return np.array(pit)


def plot_pit(pit_values, ci_level, n_bins=30, y_true=None, ax=None, **fig_kw):
    """
    Plots the PIT/HPD histogram and calculates the confidence interval for the bin values, were the PIT/HPD values follow an uniform distribution

    @param values: a numpy array with PIT/HPD values
    @param ci_level: a float between 0 and 1 indicating the size of the confidence level
    @param x_label: a string, populates the x_label of the plot
    @param n_bins: an integer, the number of bins in the histogram
    @param figsize: a tuple, the plot size (width, height)
    @param ylim: a list of two elements, including the lower and upper limit for the y axis

    @returns The matplotlib figure object with the histogram of the PIT/HPD values and the CI for the uniform distribution
    """

    # Extract the number of CDEs
    n = pit_values.shape[0]

    # Creating upper and lower limit for selected uniform band
    ci_quantity = (1 - ci_level) / 2
    low_lim = binom.ppf(q=ci_quantity, n=n, p=1 / n_bins)
    upp_lim = binom.ppf(q=ci_level + ci_quantity, n=n, p=1 / n_bins)

    # Creating figure

    if ax is None:
        fig, ax = plt.subplots(1, 2, **fig_kw)

    # plot PIT histogram
    ax[0].hist(pit_values, bins=n_bins)
    ax[0].axhline(y=low_lim, color="grey")
    ax[0].axhline(y=upp_lim, color="grey")
    ax[0].axhline(y=n / n_bins, label="Uniform Average", color="red")
    ax[0].fill_between(
        x=np.linspace(0, 1, 100),
        y1=np.repeat(low_lim, 100),
        y2=np.repeat(upp_lim, 100),
        color="grey",
        alpha=0.2,
    )
    ax[0].set_xlabel("PIT Values")
    ax[0].legend(loc="best")

    # plot P-P plot
    prob_theory = np.linspace(0.01, 0.99, 100)
    prob_data = [np.sum(pit_values < i) / len(pit_values) for i in prob_theory]
    # # plot Q-Q
    # quants = np.linspace(0, 100, 100)
    # quant_theory = quants/100.
    # quant_data = np.percentile(pit_values,quants)

    ax[1].scatter(prob_theory, prob_data, marker=".")
    ax[1].plot(prob_theory, prob_theory, c="k", ls="--")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("Expected Cumulative Probability")
    ax[1].set_ylabel("Empirical Cumulative Probability")
    xlabels = np.linspace(0, 1, 6)[1:]
    ax[1].set_xticks(xlabels)
    ax[1].set_aspect("equal")
    if y_true is not None:
        ks = kolmogorov_smirnov_statistic(prob_data, prob_theory)
        ad = anderson_darling_statistic(prob_data, prob_theory, len(y_true))
        cvm = cramer_von_mises(prob_data, prob_theory)
        ax[1].text(0.05, 0.9, f"KS:  ${ks:.3f} $", fontsize=15)
        ax[1].text(0.05, 0.84, f"CvM:  ${cvm:.3f} $", fontsize=15)
        ax[1].text(0.05, 0.78, f"AD:  ${ad:.2f} $", fontsize=15)

    return fig, ax