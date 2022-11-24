from __future__ import annotations

from abc import ABC, abstractmethod

from astropy.cosmology import FLRW
import numpy as np
from pandas import DataFrame, IntervalIndex, Series
from numpy.typing import NDArray

from rail.estimation.algos.yet_another_wizz.correlation import CorrelationFunction
from rail.estimation.algos.yet_another_wizz.utils import CustomCosmology, get_default_cosmology


class BinFactory:

    def __init__(
        self,
        zmin: float,
        zmax: float,
        nbins: int,
        cosmology: FLRW | CustomCosmology | None = None
    ):
        if zmin >= zmax:
            raise ValueError("'zmin' >= 'zmax'")
        if cosmology is None:
            cosmology = get_default_cosmology()
        self.cosmology = cosmology
        self.zmin = zmin
        self.zmax = zmax
        self.nbins = nbins

    def linear(self, **kwargs) -> NDArray[np.float_]:
        return np.linspace(self.zmin, self.zmax, self.nbins + 1)

    def comoving(self, **kwargs) -> NDArray[np.float_]:
        cbinning = np.linspace(
            self.cosmology.comoving_distance(self.zmin).value,
            self.cosmology.comoving_distance(self.zmax).value,
            self.nbins + 1)
        # construct a spline mapping from comoving distance to redshift
        zarray = np.linspace(0, 10.0, 5000)
        carray = self.cosmology.comoving_distance(zarray).value
        return np.interp(cbinning, xp=carray, fp=zarray)  # redshift @ cbinning

    def adaptive(self, redshifts: NDArray[np.float_]) -> NDArray[np.float_]:
        # find equal number bins within the given redshift limits
        mask = (redshifts >= self.zmin) & (redshifts < self.zmax)
        z_sorted = np.sort(redshifts[mask])
        idx = np.linspace(0, len(z_sorted) - 1, self.nbins + 1, dtype=np.int_)
        binning = z_sorted[idx]
        # fix the limits
        binning[0] = self.zmin
        binning[-1] = self.zmax
        return binning

    def logspace(self, **kwargs) -> NDArray[np.float_]:
        logbinning = np.linspace(
            np.log(1.0 + self.zmin), np.log(1.0 + self.zmax), self.nbins + 1)
        return np.exp(logbinning) - 1.0

    @staticmethod
    def check(zbins: NDArray[np.float_]) -> None:
        if np.any(np.diff(zbins) <= 0):
            raise ValueError("redshift bins are not monotonicaly increasing")

    def get(self, method: str, **kwargs) -> NDArray[np.float_]:
        try:
            return getattr(self, method)(**kwargs)
        except AttributeError:
            raise ValueError(f"invalid binning method '{method}'")


class Nz(ABC):

    def _get_redshift_binwidths(
        self,
        interval_index: IntervalIndex
    ) -> NDArray[np.float_]:
        # compute redshift bin widths
        return np.array([zbin.right - zbin.left for zbin in interval_index])

    @abstractmethod
    def get(self, **kwargs) -> Series:
        raise NotImplementedError

    @abstractmethod
    def get_samples(
        self,
        *,
        n_boot: int = 500,
        seed: int = 12345,
        **kwargs
    ) -> DataFrame:
        raise NotImplementedError


class NzTrue(Nz):

    def __init__(
        self,
        region_counts: DataFrame
    ) -> None:
        self.counts = region_counts
        self.dz = self._get_redshift_binwidths(region_counts.index)

    def get(self, **kwargs) -> Series:
        Nz = np.sum(self.counts, axis=1)
        norm = Nz.sum() * self.dz
        return Nz / norm

    def get_samples(
        self,
        *,
        n_boot: int = 500,
        seed: int = 12345,
        **kwargs
    ) -> DataFrame:
        rng = np.random.default_rng(seed=seed)
        n_regions = len(self.counts.columns)
        patch_idx = rng.integers(0, n_regions, size=[n_boot, n_regions])
        Nz_boot = np.sum(self.counts.to_numpy().T[patch_idx], axis=1)
        nz_boot = Nz_boot / (
            Nz_boot.sum(axis=1)[:, np.newaxis] * self.dz[np.newaxis, :])
        return DataFrame(
            index=self.counts.index,
            columns=np.arange(n_boot),
            data=nz_boot.T)


class NzEstimator(Nz):

    def __init__(
        self,
        cross_corr: CorrelationFunction
    ) -> None:
        self.cross_corr = cross_corr
        self.ref_corr = None
        self.unk_corr = None
        self.dz = self._get_redshift_binwidths(self.cross_corr.dd.binning)

    def add_reference_autocorr(
        self,
        ref_corr: CorrelationFunction
    ) -> None:
        if not self.cross_corr.is_compatible(ref_corr):
            raise ValueError(
                "redshift binning or number of patches do not match")
        self.ref_corr = ref_corr

    def add_unknown_autocorr(
        self,
        unk_corr: CorrelationFunction
    ) -> None:
        if not self.cross_corr.is_compatible(unk_corr):
            raise ValueError(
                "redshift binning or number of patches do not match")
        self.unk_corr = unk_corr

    def get(
        self,
        estimator: str = "DP"
    ) -> Series:
        cross_corr = self.cross_corr.get(estimator)
        if self.ref_corr is None:
            ref_corr = np.float64(1.0)
        else:
            ref_corr = self.ref_corr.get(estimator)
        if self.unk_corr is None:
            unk_corr = np.float64(1.0)
        else:
            unk_corr = self.unk_corr.get(estimator)
        return cross_corr / np.sqrt(self.dz**2 * ref_corr * unk_corr)

    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        return self.cross_corr.generate_bootstrap_patch_indices(n_boot, seed)

    def get_samples(
        self,
        *,
        estimator: str = "DP",
        global_norm: bool = False,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        seed: int = 12345
    ) -> DataFrame:
        if sample_method == "bootstrap":
            patch_idx = self.generate_bootstrap_patch_indices(n_boot, seed)
        else:
            patch_idx = None
        kwargs = dict(
            estimator=estimator,
            global_norm=global_norm,
            sample_method=sample_method,
            patch_idx=patch_idx)
        cross_corr = self.cross_corr.get_samples(**kwargs)
        if self.ref_corr is None:
            ref_corr = np.float64(1.0)
        else:
            ref_corr = self.ref_corr.get_samples(**kwargs)
        if self.unk_corr is None:
            unk_corr = np.float64(1.0)
        else:
            unk_corr = self.unk_corr.get_samples(**kwargs)
        dz_sq = cross_corr.copy()
        for col in dz_sq.columns:
            dz_sq[col] = self.dz**2
        return cross_corr / np.sqrt(dz_sq * ref_corr * unk_corr)
