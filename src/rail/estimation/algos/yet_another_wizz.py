"""
Implementation of yet_another_wizz for easy cross-correlation/clustering
redshifts (arXiv:2007.01846 / A&A 642, A200 (2020)).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import astropy.cosmology
import numpy as np
import pandas as pd
import treecorr
from astropy.cosmology.core import Cosmology
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Interval, IntervalIndex, Series
from treecorr.util import depr_pos_kwargs
from treecorr.binnedcorr2 import estimate_multi_cov
from tqdm import tqdm

from rail.core.data import PqHandle
from rail.core.stage import RailStage


class UniformRandoms:

    def __init__(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float
    ) -> None:
        self.x_min, self.y_min = self.sky2cylinder(ra_min, dec_min)
        self.x_max, self.y_max = self.sky2cylinder(ra_max, dec_max)

    @staticmethod
    def sky2cylinder(
        ra: float | NDArray[np.float_],
        dec: float | NDArray[np.float_]
    ) -> NDArray:
        x = np.deg2rad(ra)
        y = np.sin(np.deg2rad(dec))
        return np.transpose([x, y])
 
    @staticmethod
    def cylinder2sky(
        x: float | NDArray[np.float_],
        y: float | NDArray[np.float_]
    ) -> float | NDArray[np.float_]:
        ra = np.rad2deg(x)
        dec = np.rad2deg(np.arcsin(y))
        return np.transpose([ra, dec])

    def generate(
        self,
        size: int,
        names: list[str, str] | None = None
    ) -> DataFrame:
        x = np.random.uniform(self.x_min, self.x_max, size)
        y = np.random.uniform(self.y_min, self.y_max, size)
        if names is None:
            names = ["ra", "dec"]
        ra, dec = self.cylinder2sky(x, y).T
        return DataFrame({names[0]: ra, names[1]: dec})


class ArrayDict:

    def __init__(
        self,
        array: NDArray,
        key_to_index: dict[Any, np.int_]
    ) -> None:
        if len(array) != len(key_to_index):
            raise ValueError("number of keys and array length do not match")
        self.array = array
        self.map = key_to_index

    def get(self, key: Any) -> NDArray:
        return self.array[self.map[key]]

    def sample(self, keys: Iterable[Any]) -> NDArray:
        idx = [self.map[key] for key in keys]
        return self.array[idx]


@dataclass(frozen=True, repr=False)
class TreeCorrResult:
    interval: Interval
    npatch: tuple(int, int)
    weights: ArrayDict
    total: ArrayDict
    mask: NDArray[np.bool_]

    @classmethod
    def from_countcorr(
        cls,
        interval: Interval,
        correlation: treecorr.NNCorrelation,
        dist_weight_scale: float | None = None
    ) -> TreeCorrResult:
        # compute the pair-separation (a.k.a inverse distance) weights
        npatch = (correlation.npatch1, correlation.npatch2)
        if dist_weight_scale is not None:
            r_weights = correlation.meanr ** dist_weight_scale
            r_weights /= r_weights.mean()
        else:
            r_weights = np.ones_like(correlation.meanr)

        # extract the (cross-patch) pair counts
        key_to_index = {}
        weights = []
        total = []
        for i, (patches, result) in enumerate(correlation.results.items()):
            key_to_index["%i,%i" % patches] = i
            weights.append(np.sum(result.weight * r_weights))
            total.append(result.tot)
        weights = ArrayDict(np.array(weights), key_to_index)
        total = ArrayDict(np.array(total), key_to_index)
        return cls(interval, npatch, weights, total, mask=correlation._ok)


@dataclass(frozen=True, repr=False)
class PairCountData:
    binning: IntervalIndex
    weights: NDArray[np.float_]
    total: NDArray[np.float_]


@dataclass(frozen=True, repr=False)
class TreeCorrResultZbinned:
    binning: IntervalIndex
    npatch: tuple(int, int)
    weights: ArrayDict
    total: ArrayDict
    mask: NDArray[np.bool_]

    @classmethod
    def from_bins(
        cls,
        zbins: Iterable[TreeCorrResult]
    ) -> TreeCorrResultZbinned:
        # check the number of patches
        if len(zbins) == 0:
            raise ValueError("'zbin' contains no entries")
        npatch = zbins[0].npatch
        mask = zbins[0].mask
        for zbin in zbins[1:]:
            if zbin.npatch != npatch:
                raise ValueError("the patch numbers are inconsistent")
            if not np.array_equal(mask, zbin.mask):
                raise ValueError("pair masks are inconsistent")
        keys = set(zbins[0].weights.map)
        for zbin in zbins[1:]:
            if set(zbin.weights.map) != keys:
                raise ValueError("patches are inconsistent")
        keys = zbins[0].weights.map

        # check the ordering of the bins based on the provided intervals
        binning = IntervalIndex([zbin.interval for zbin in zbins])
        if not binning.is_non_overlapping_monotonic:
            raise ValueError(
                "the binning interval is overlapping or not monotonic")
        for this, following in zip(binning[:-1], binning[1:]):
            if this.right != following.left:
                raise ValueError(f"the binning interval is not contiguous")

        # merge the data
        weights = ArrayDict(
            np.column_stack([zbin.weights.array for zbin in zbins]), keys)
        total = ArrayDict(
            np.column_stack([zbin.total.array for zbin in zbins]), keys)
        return cls(binning, npatch, weights, total, mask)

    def __len__(self) -> int:
        return len(self.total)

    def get(self) -> PairCountData:
        return PairCountData(
            self.binning,
            self.weights.array.sum(axis=0),
            self.total.array.sum(axis=0))

    def get_jackknife_samples(
        self,
        global_norm: bool = False,
        **kwargs
    ) -> PairCountData:
        # The treecorr jackknife iterator expects the actual results dictionary
        # with (i,j) keys, whereas the dataframes have "j,i" string columns.
        _dummy_pairs = {
            tuple(str(f) for f in colname.split(",")): None
            for colname in self.paris.columns}
        npatch = max(self.npatch)
        # Furthermore, the iterator expects a single patch index which is
        # dropped in a single realisation.
        weights = []
        total = []
        if global_norm:
            global_total = self.total.array.sum(axis=0)
        for idx in range(npatch):  # simplified leave-one-out iteration
            # we need to use the jackknife iterator twice
            keys_str = list(
                "%i,%i" % patches
                for patches in treecorr.NNCorrelation.JackknifePairIterator(
                    _dummy_pairs, *self.npatch, idx, self.mask))
            weights.append(self.weights.sample(keys_str).sum(axis=0))
            if global_norm:
                total.append(global_total)
            else:
                total.append(self.total.sample(keys_str).sum(axis=0))
        return PairCountData(self.binning, np.array(weights), np.array(total))

    def get_bootstrap_samples(
        self,
        patch_idx: NDArray[np.int_],
        global_norm: bool = False
    ) -> PairCountData:
        # The treecorr bootstrap iterator expects a list of patch indicies which
        # are present in the specific boostrap realisation to generate, i.e.
        # draw N times from (0, ..., N) with repetition. These random patch
        # indices for M realisations should be provided in the [M, N] shaped
        # array 'patch_idx'.
        weights = []
        total = []
        if global_norm:
            global_total = self.total.array.sum(axis=0)
        for idx in patch_idx:  # simplified leave-one-out iteration
            # we need to use the jackknife iterator twice
            keys_str = list(
                "%i,%i" % patches
                for patches in treecorr.NNCorrelation.BootstrapPairIterator(
                    self.weights, *self.npatch, idx, self.mask))
            weights.append(self.weights.sample(keys_str).sum(axis=0))
            if global_norm:
                total.append(global_total)
            else:
                total.append(self.total.sample(keys_str).sum(axis=0))
        return PairCountData(self.binning, np.array(weights), np.array(total))

    def get_samples(
        self,
        method: str,
        **kwargs
    ) -> PairCountData:
        if method == "jackknife":
            return self.get_jackknife_samples(**kwargs)
        elif method == "bootstrap":
            return self.get_bootstrap_samples(**kwargs)
        else:
            raise NotImplementedError(
                f"sampling method '{method}' not implemented")


@dataclass(frozen=True, repr=False)
class CorrelationFunctionZbinned:
    dd: TreeCorrResultZbinned
    dr: TreeCorrResultZbinned | None = field(default=None)
    rd: TreeCorrResultZbinned | None = field(default=None)
    rr: TreeCorrResultZbinned | None = field(default=None)
    npatch: tuple(int, int) = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "npatch", self.dd.npatch)  # since frozen=True
        # check if the minimum required pair counts are provided
        if self.dr is None and self.rr is None:
            raise ValueError("either 'dr' or 'rr' is required")
        if self.dr is None and self.rd is not None:
            raise ValueError("'rd' requires 'dr'")
        # check that the pair counts are compatible
        for kind in ("dr", "rd", "rr"):
            pair_instance = getattr(self, kind)
            if pair_instance is not None:
                if pair_instance.npatch != self.npatch:
                    raise ValueError(f"npatches of '{kind}' do not match 'dd'")
                if np.any(pair_instance.binning != self.dd.binning):
                    raise ValueError(
                        f"redshift binning of '{kind}' do not match 'dd'")

    def is_compatible(
        self,
        other: CorrelationFunctionZbinned
    ) -> bool:
        if self.npatch != other.npatch:
            return False
        if np.any(self.dd.binning != other.dd.binning):
            return False
        return True

    @property
    def estimators(self) -> set[str]:
        estimators = set()
        if self.dr is not None:
            estimators.add("DP")
            if self.rr is not None:
                estimators.add("HM")
                estimators.add("LS")
        if self.rr is not None:
            estimators.add("PH")
        return estimators

    @staticmethod
    def Peebles_Hauser(
        *,
        dd: PairCountData,
        rr: PairCountData,
        **kwargs
    ) -> DataFrame:
        DD = dd.weights
        RR = rr.weights * dd.total / rr.total
        est = DD / RR - 1.0
        return pd.DataFrame(data=est.T, index=dd.binning)

    @staticmethod
    def Davis_Peebles(
        *,
        dd: PairCountData,
        dr: PairCountData,
        **kwargs
    ) -> DataFrame:
        DD = dd.weights
        DR = dr.weights * dd.total / dr.total
        est = DD / DR - 1.0
        return pd.DataFrame(data=est.T, index=dd.binning)

    @staticmethod
    def Hamilton(
        *,
        dd: PairCountData,
        dr: PairCountData,
        rr: PairCountData,
        rd: PairCountData | None = None
    ) -> DataFrame:
        DD = dd.weights
        DR = dr.weights * dd.total / dr.total
        if rd is None:
            RD = DR
        else:
            RD = rd.weights * dd.total / rd.total
        RR = rr.weights * dd.total / rr.total
        est = (DD * RR) / (DR * RD) - 1.0
        return pd.DataFrame(data=est.T, index=dd.binning)

    @staticmethod
    def Landy_Szalay(
        *,
        dd: PairCountData,
        dr: PairCountData,
        rr: PairCountData,
        rd: PairCountData | None = None
    ) -> DataFrame:
        DD = dd.weights
        DR = dr.weights * dd.total / dr.total
        if rd is None:
            RD = DR
        else:
            RD = rd.weights * dd.total / rd.total
        RR = rr.weights * dd.total / rr.total
        est = (DD - (DR + RD) + RR) / RR
        return pd.DataFrame(data=est.T, index=dd.binning)

    def _check_and_select_estimator(
        self,
        estimator: str
    ) -> Callable[[NDArray[np.int_]], DataFrame]:
        if estimator not in self.estimators:
            raise ValueError(f"estimator '{estimator}' not available")
        # select the correct estimator
        method = dict(
            PH=self.Peebles_Hauser,
            DP=self.Davis_Peebles,
            HM=self.Hamilton,
            LS=self.Landy_Szalay
        )[estimator]
        arguments = dict(
            PH=("dd", "rr"),
            DP=("dd", "dr"),
            HM=("dd", "dr", "rd", "rr"),
            LS=("dd", "dr", "rd", "rr")
        )[estimator]
        return method, arguments

    def get(
        self,
        estimator: str
    ) -> Series:
        method, arguments = self._check_and_select_estimator(estimator)
        pairs = {}
        for attr in arguments:
            instance = getattr(self, attr)
            if instance is not None:
                pairs[attr] = instance.get()
        return method(**pairs)[0]

    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        N = max(self.npatch)
        rng = np.random.default_rng(seed=seed)
        return rng.integers(0, N, size=(n_boot, N))

    def get_samples(
        self,
        estimator: str,
        global_norm: bool = False,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        boot_idx: NDArray[np.int_] | None = None
    ) -> DataFrame:
        # check the sampling method
        valid_methods = ("bootstrap", "jackknife")
        if sample_method not in valid_methods:
            opts = ", ".join(f"'{s}'" for s in valid_methods)
            raise ValueError(f"'sample_method' must be either of {opts}")
        # select the sampling method and generate optional bootstrap samples
        method, arguments = self._check_and_select_estimator(estimator)
        if boot_idx is None and sample_method == "bootstrap":
            boot_idx = self.generate_bootstrap_patch_indices(n_boot)
        # resample the patch pair counts
        pairs = {}
        for attr in arguments:
            instance = getattr(self, attr)
            if instance is not None:
                pairs[attr] = instance.get_samples(
                    method=sample_method,
                    global_norm=global_norm,
                    patch_idx=boot_idx)
        # evaluate the correlation estimator
        return method(**pairs)


class Nz(ABC):

    def _get_redshift_binwidths(
        self,
        interval_index: pd.IntervalIndex
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
        boot_idx = rng.integers(0, n_regions, size=[n_boot, n_regions])
        Nz_boot = np.sum(self.counts.to_numpy().T[boot_idx], axis=1)
        nz_boot = Nz_boot / (
            Nz_boot.sum(axis=1)[:, np.newaxis] * self.dz[np.newaxis, :])
        samples = pd.DataFrame(
            index=self.counts.index, columns=np.arange(n_boot), data=nz_boot.T)
        return samples


class NzEstimator(Nz):

    def __init__(
        self,
        cross_corr: CorrelationFunctionZbinned
    ) -> None:
        self.cross_corr = cross_corr
        self.ref_corr = None
        self.unk_corr = None
        self.dz = self._get_redshift_binwidths(self.cross_corr.dd.binning)

    def add_reference_autocorr(
        self,
        ref_corr: CorrelationFunctionZbinned
    ) -> None:
        if not self.cross_corr.is_compatible(ref_corr):
            raise ValueError(
                "redshift binning or number of patches do not match")
        self.ref_corr = ref_corr

    def add_unknown_autocorr(
        self,
        unk_corr: CorrelationFunctionZbinned
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
            patch_idx = self.cross_corr.generate_bootstrap_patch_indices(
                n_boot, seed)  # should be applicable to dr/rd/rr after checks
        else:
            patch_idx = None
        kwargs = dict(
            estimator=estimator,
            global_norm=global_norm,
            sample_method=sample_method,
            boot_idx=patch_idx)
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


def angular_separation(
    r_physical: ArrayLike,
    redshift: float,
    cosmology: Cosmology | None = None
) -> ArrayLike:
    """
    TODO.
    """
    if cosmology is None:  # TODO: is this the most sensible?
        cosmology = astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    f_K = cosmology.comoving_transverse_distance(redshift)  # for 1 radian
    angle_rad = np.asarray(r_physical) * (1.0 + redshift) / f_K.value
    return np.rad2deg(angle_rad)


class CustomNNCorrelation(treecorr.NNCorrelation):

    def copy(self):
        """Make a copy"""
        ret = CustomNNCorrelation.__new__(CustomNNCorrelation)
        for key, item in self.__dict__.items():
            if isinstance(item, np.ndarray):
                # Only items that might change need to by deep copied.
                ret.__dict__[key] = item.copy()
            else:
                # For everything else, shallow copy is fine.
                # In particular don't deep copy config or logger
                # Most of the rest are scalars, which copy fine this way.
                # And the read-only things are all in _ro.
                # The results dict is trickier.  We rely on it being copied in places, but we
                # never add more to it after the copy, so shallow copy is fine.
                ret.__dict__[key] = item
        ret._corr = None # We'll want to make a new one of these if we need it.
        if self._rd is not None:
            ret._rd = self._rd.copy()
        if self._dr is not None:
            ret._dr = self._dr.copy()
        if self._rr is not None:
            ret._rr = self._rr.copy()
        return ret

    def _zero_copy(self, tot):
        # A minimal "copy" with zero for the weight array, and the given value for tot.
        ret = CustomNNCorrelation.__new__(CustomNNCorrelation)
        ret._ro = self._ro
        ret.coords = self.coords
        ret.metric = self.metric
        ret.config = self.config
        ret.meanr = self._zero_array
        ret.meanlogr = self._zero_array
        ret.weight = self._zero_array
        ret.npairs = self._zero_array
        ret.tot = tot
        ret._corr = None
        ret._rr = ret._dr = ret._rd = None
        ret._write_rr = ret._write_dr = ret._write_rd = None
        # This override is really the main advantage of using this:
        setattr(ret, '_nonzero', False)
        return ret

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        This raises a RuntimeError if calculateXi has not been run yet.
        """
        if self._rr is None and self._dr is None:
            raise RuntimeError("You need to call calculateXi before calling estimate_cov.")
        return self.xi.ravel()

    @depr_pos_kwargs
    def calculateXi(self, *, dr=None, rd=None, rr=None):
        r"""Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross correlations of the data and random.

        The rr value is the `NNCorrelation` function for random points.
        For a signal that involves a cross correlations, there should be two random
        cross-correlations: data-random and random-data, given as dr and rd.

        - If dr is None, the simple correlation function :math:`\xi = (DD/RR - 1)` is used.
        - if dr is given and rd is None, then :math:`\xi = (DD - 2DR + RR)/RR` is used.
        - If dr and rd are both given, then :math:`\xi = (DD - DR - RD + RR)/RR` is used.

        where DD is the data NN correlation function, which is the current object.

        .. note::

            The default method for estimating the variance is 'shot', which only includes the
            shot noise propagated into the final correlation.  This does not include sample
            variance, so it is always an underestimate of the actual variance.  To get better
            estimates, you need to set ``var_method`` to something else and use patches in the
            input catalog(s).  cf. `Covariance Estimates`.

        After calling this method, you can use the `BinnedCorr2.estimate_cov` method or use this
        correlation object in the `estimate_multi_cov` function.  Also, the calculated xi and
        varxi returned from this function will be available as attributes.

        Parameters:
            rr (NNCorrelation):     The auto-correlation of the random field (RR)
            dr (NNCorrelation):     The cross-correlation of the data with randoms (DR), if
                                    desired, in which case the Landy-Szalay estimator will be
                                    calculated.  (default: None)
            rd (NNCorrelation):     The cross-correlation of the randoms with data (RD), if
                                    desired. (default: None, which means use rd=dr)

        Returns:
            Tuple containing:

            - xi = array of :math:`\xi(r)`
            - varxi = an estimate of the variance of :math:`\xi(r)`
        """
        if rr is None and dr is None:
            raise ValueError("either of 'dr' or 'rr' must be provided")

        # Each weight value needs to be rescaled by the ratio of total possible pairs.
        if dr is not None:
            if dr.tot == 0:
                raise ValueError("dr has tot=0.")
            drf = self.tot / dr.tot
        if rd is not None:
            if rd.tot == 0:
                raise ValueError("rd has tot=0.")
            rdf = self.tot / rd.tot
        if rr is not None:
            if rr.tot == 0:
                raise ValueError("rr has tot=0.")
            rrf = self.tot / rr.tot

        # Calculate xi based on which randoms are provided.
        if rr is None:
            denom = dr.weight * drf
            self.xi = self.weight - denom
            # Divide by DR.
            if np.any(dr.weight == 0):
                self.logger.warning("Warning: Some bins for the data-randoms had no pairs.")
                denom[dr.weight==0] = 1.  # guard against division by 0.
        else:
            denom = rr.weight * rrf
            if dr is None and rd is None:
                self.xi = self.weight - denom
            elif rd is not None and dr is None:
                self.xi = self.weight - 2.*rd.weight * rdf + denom
            elif dr is not None and rd is None:
                self.xi = self.weight - 2.*dr.weight * drf + denom
            else:
                self.xi = self.weight - rd.weight * rdf - dr.weight * drf + denom
            # Divide by RR in all cases.
            if np.any(rr.weight == 0):
                self.logger.warning("Warning: Some bins for the randoms had no pairs.")
                denom[rr.weight==0] = 1.  # guard against division by 0.
        self.xi /= denom

        # Set up necessary info for estimate_cov

        # First the bits needed for shot noise covariance:
        ddw = self._mean_weight()
        if dr is not None:
            drw = dr._mean_weight()
        if rd is not None:
            rdw = rd._mean_weight()
        if rr is not None:
            rrw = rr._mean_weight()

        # Note: The use of varxi_factor for the shot noise varxi is semi-empirical.
        #       It gives the increase in the variance over the case where RR >> DD.
        #       I don't have a good derivation that this is the right factor to apply
        #       when the random catalog is not >> larger than the data.
        #       When I tried to derive this from first principles, I get the below formula,
        #       but without the **2.  So I'm not sure why this factor needs to be squared.
        #       It seems at least plausible that I missed something in the derivation that
        #       leads to this getting squared, but I can't really justify it.
        #       But it's also possible that this is wrong...
        #       Anyway, it seems to give good results compared to the empirical variance.
        #       cf. test_nn.py:test_varxi
        if rr is None:
            varxi_factor = 1 + drf*drw/ddw
            self._rr_weight = dr.weight * drf  # not good, but there is not better simple solution
        else:
            if dr is None and rd is None:
                varxi_factor = 1 + rrf*rrw/ddw
            elif rd is not None and dr is None:
                varxi_factor = 1 + 2*rdf*rdw/ddw + rrf*rrw/ddw
            elif dr is not None and rd is None:
                varxi_factor = 1 + 2*drf*drw/ddw + rrf*rrw/ddw
            else:
                varxi_factor = 1 + drf*drw/ddw + rdf*rdw/ddw + rrf*rrw/ddw
            self._rr_weight = rr.weight * rrf
        self._var_num = ddw * varxi_factor**2

        # Now set up the bits needed for patch-based covariance
        self._rr = rr
        self._dr = dr
        self._rd = rd

        if len(self.results) > 0:
            # Check that rr,dr,rd use the same patches as dd
            pair_inst = rr if dr is None else dr
            msg_label = "RR" if dr is None else "DR"
            if pair_inst.npatch1 != 1 and pair_inst.npatch2 != 1:
                if pair_inst.npatch1 != self.npatch1 or pair_inst.npatch2 != self.npatch2:
                    raise RuntimeError(
                        f"If using patches, {msg_label} must be run with the same patches as DD")

            if dr is not None and (len(dr.results) == 0 or dr.npatch1 != self.npatch1 or
                                   dr.npatch2 not in (self.npatch2, 1)):
                raise RuntimeError("DR must be run with the same patches as DD")
            if rd is not None and (len(rd.results) == 0 or rd.npatch2 != self.npatch2 or
                                   rd.npatch1 not in (self.npatch1, 1)):
                raise RuntimeError("RD must be run with the same patches as DD")
            if rr is not None and (len(rr.results) == 0 or rr.npatch2 != self.npatch2 or
                                   rr.npatch1 not in (self.npatch1, 1)):
                raise RuntimeError("RR must be run with the same patches as DD")

            # If there are any rr,rd,dr patch pairs that aren't in results (because dr is a cross
            # correlation, and dd,rr may be auto-correlations, or because the d catalogs has some
            # patches with no items), then we need to add some dummy results to make sure all the
            # right pairs are computed when we make the vectors for the covariance matrix.
            add_ij = set()

            if dr is not None and dr.npatch2 != 1:
                for ij in dr.results:
                    if ij not in self.results:
                        add_ij.add(ij)

            if rd is not None and rd.npatch1 != 1:
                for ij in rd.results:
                    if ij not in self.results:
                        add_ij.add(ij)

            if rr is not None and rr.npatch1 != 1:
                for ij in rr.results:
                    if ij not in self.results:
                        add_ij.add(ij)

            if len(add_ij) > 0:
                for ij in add_ij:
                    self.results[ij] = self._zero_copy(0)
                self.__dict__.pop('_ok',None)  # If it was already made, it will need to be redone.

        # Now that it's all set up, calculate the covariance and set varxi to the diagonal.
        self.cov = self.estimate_cov(self.var_method)
        self.varxi = self.cov.diagonal()
        return self.xi, self.varxi

    def _calculate_xi_from_pairs(self, pairs):
        self._sum([self.results[ij] for ij in pairs])
        self._finalize()
        dd = self.weight

        if self._rr is None:
            # Calculate rr and rrf in the normal way based on the same pairs as used for DD.
            pairs1 = [ij for ij in pairs if self._dr._ok[ij[0],ij[1]]]
            self._dr._sum([self._dr.results[ij] for ij in pairs1])
            dd_tot = self.tot

            if self._dr.npatch2 == 1:
                # If r doesn't have patches, then convert all (i,i) pairs to (i,0).
                pairs2 = [(ij[0],0) for ij in pairs if ij[0] == ij[1]]
            else:
                pairs2 = [ij for ij in pairs if self._dr._ok[ij[0],ij[1]]]
            self._dr._sum([self._dr.results[ij] for ij in pairs2])
            dr = self._dr.weight
            drf = dd_tot / self._dr.tot

            denom = dr * drf
            xi = dd - denom

        else:
            if len(self._rr.results) > 0:
                # This is the usual case.  R has patches just like D.
                # Calculate rr and rrf in the normal way based on the same pairs as used for DD.
                pairs1 = [ij for ij in pairs if self._rr._ok[ij[0],ij[1]]]
                self._rr._sum([self._rr.results[ij] for ij in pairs1])
                dd_tot = self.tot
            else:
                # In this case, R was not run with patches.
                # This is not necessarily much worse in practice it turns out.
                # We just need to scale RR down by the relative area.
                # The approximation we'll use is that tot in the auto-correlations is
                # proportional to area**2.
                # So the sum of tot**0.5 when i==j gives an estimate of the fraction of the total area.
                area_frac = np.sum([self.results[ij].tot**0.5 for ij in pairs if ij[0] == ij[1]])
                area_frac /= np.sum([cij.tot**0.5 for ij,cij in self.results.items() if ij[0] == ij[1]])
                # First figure out the original total for all DD that had the same footprint as RR.
                dd_tot = np.sum([self.results[ij].tot for ij in self.results])
                # The rrf we want will be a factor of area_frac smaller than the original
                # dd_tot/rr_tot.  We can effect this by multiplying the full dd_tot by area_frac
                # and use that value normally below.  (Also for drf and rdf.)
                dd_tot *= area_frac

            rr = self._rr.weight
            rrf = dd_tot / self._rr.tot

            if self._dr is not None:
                if self._dr.npatch2 == 1:
                    # If r doesn't have patches, then convert all (i,i) pairs to (i,0).
                    pairs2 = [(ij[0],0) for ij in pairs if ij[0] == ij[1]]
                else:
                    pairs2 = [ij for ij in pairs if self._dr._ok[ij[0],ij[1]]]
                self._dr._sum([self._dr.results[ij] for ij in pairs2])
                dr = self._dr.weight
                drf = dd_tot / self._dr.tot
            if self._rd is not None:
                if self._rd.npatch1 == 1:
                    # If r doesn't have patches, then convert all (i,i) pairs to (0,i).
                    pairs3 = [(0,ij[1]) for ij in pairs if ij[0] == ij[1]]
                else:
                    pairs3 = [ij for ij in pairs if self._rd._ok[ij[0],ij[1]]]
                self._rd._sum([self._rd.results[ij] for ij in pairs3])
                rd = self._rd.weight
                rdf = dd_tot / self._rd.tot
            denom = rr * rrf
            if self._dr is None and self._rd is None:
                xi = dd - denom
            elif self._rd is not None and self._dr is None:
                xi = dd - 2.*rd * rdf + denom
            elif self._dr is not None and self._rd is None:
                xi = dd - 2.*dr * drf + denom
            else:
                xi = dd - rd * rdf - dr * drf + denom

        denom[denom == 0] = 1  # Guard against division by zero.
        self.xi = xi / denom
        self._rr_weight = denom


class NNCorrelationContainer:

    def __init__(self, nncorrs: Iterable[treecorr.NNCorrelation]):
        self.corrs = nncorrs
        self.values = np.array([corr.xi for corr in nncorrs])
        self.errors = np.sqrt([corr.varxi for corr in nncorrs])
        self.var_method = nncorrs[0].var_method
        self._covariance = None

    def __len__(self) -> int:
        return len(self.corrs)

    @property
    def covariance(self):
        if self._covariance is None:
            self._covariance = estimate_multi_cov(
                self.corrs, self.var_method, comm=None)
        return self._covariance


class YetAnotherWizz:

    def __init__(
        self,
        *,
        reference: DataFrame,
        ref_rand: DataFrame,
        unknown: DataFrame,
        unk_rand: DataFrame,
        rmin_Mpc: float = 0.1,
        rmax_Mpc: float = 1.0,
        dist_weight_scale: float | None = None,
        bin_slop: float | None = None,
        num_threads: int | None = None
    ) -> None:
        # set configuration
        self.rmin = rmin_Mpc
        self.rmax = rmax_Mpc
        self.dist_weight_scale = dist_weight_scale
        if num_threads is None:
            num_threads = os.cpu_count()
        self.correlator_config = dict(
            sep_units="degrees",
            metric="Arc",
            nbins=(1 if dist_weight_scale is None else 50),
            var_method="bootstrap",
            bin_slop=bin_slop,
            num_threads=num_threads)
        # set data
        self.reference = reference
        self.ref_rand = ref_rand
        self.unknown = unknown
        self.unk_rand = unk_rand

    def make_catalogue(
        self,
        df: DataFrame
    ) -> treecorr.Catalog:
        return treecorr.Catalog(
            ra=df["ra"], ra_units="degrees",
            dec=df["dec"], dec_units="degrees",
            patch=df["region_idx"])

    def bin_iter(
        self,
        zbins: NDArray[np.float_],
        data: DataFrame,
        rand: DataFrame,
        desc: str | None = None
    ) -> Iterator[tuple[Interval, DataFrame, DataFrame]]:
        data_iter = data.groupby(pd.cut(data["z"], zbins))
        rand_iter = rand.groupby(pd.cut(rand["z"], zbins))
        iterator = zip(data_iter, rand_iter)  # guaranteed to have same length
        if desc is not None:
            iterator = tqdm(iterator, total=len(zbins)-1, desc=desc)
        for (z_interval, data_bin), (_, rand_bin) in iterator:
            yield z_interval, data_bin, rand_bin

    def correlate(
        self,
        z_interval: Interval,
        cat1: treecorr.Catalog,
        cat2: treecorr.Catalog | None = None
    ) -> tuple[Interval, CustomNNCorrelation]:
        ang_min, ang_max = angular_separation(
            [self.rmin, self.rmax], z_interval.mid)
        correlation = CustomNNCorrelation(
            min_sep=ang_min, max_sep=ang_max, **self.correlator_config)
        correlation.process(cat1, cat2)
        return TreeCorrResult.from_countcorr(
            z_interval, correlation, self.dist_weight_scale)

    def crosscorr(
        self,
        zbins: NDArray[np.float_],
        estimator: str = "LS"
    ) -> CorrelationFunctionZbinned:
        cat_unknown = self.make_catalogue(self.unknown)
        cat_unk_rand = self.make_catalogue(self.unk_rand)
        DD = []
        DR = []
        if estimator == "LS":  # otherwise default to Davis-Peebles
            RD = []
            RR = []
        for z_edges, data_bin, rand_bin in self.bin_iter(
                zbins, self.reference, self.ref_rand, desc="w_sp"):
            cat_reference = self.make_catalogue(data_bin)
            cat_ref_rand = self.make_catalogue(rand_bin)
            # correlate
            DD.append(self.correlate(z_edges, cat_reference, cat_unknown))
            DR.append(self.correlate(z_edges, cat_reference, cat_unk_rand))
            if estimator == "LS":
                RD.append(self.correlate(z_edges, cat_ref_rand, cat_unknown))
                RR.append(self.correlate(z_edges, cat_ref_rand, cat_unk_rand))
        DD = TreeCorrResultZbinned.from_bins(DD)
        DR = TreeCorrResultZbinned.from_bins(DR)
        if estimator == "LS":
            RD = TreeCorrResultZbinned.from_bins(RD)
            RR = TreeCorrResultZbinned.from_bins(RR)
        else:
            RD = None
            RR = None
        return CorrelationFunctionZbinned(dd=DD, dr=DR, rd=RD, rr=RR)

    def autocorr(
        self,
        zbins: NDArray[np.float_],
        estimator: str = "LS",
        which: str = "reference"
    ) -> CorrelationFunctionZbinned:
        if which == "reference":
            bin_iter = self.bin_iter(
                zbins, self.reference, self.ref_rand, desc="w_ss")
        elif which == "unknown":
            bin_iter = self.bin_iter(
                zbins, self.unknown, self.unk_rand, desc="w_pp")
        else:
            raise ValueError("'which' must be either of 'reference' or 'unknown'")
        DD = []
        DR = []
        if estimator == "LS":  # otherwise default to Davis-Peebles
            RR = []
        for z_edges, data_bin, rand_bin in bin_iter:
            cat_data_bin = self.make_catalogue(data_bin)
            cat_rand_bin = self.make_catalogue(rand_bin)
            # correlate
            DD.append(self.correlate(z_edges, cat_data_bin))
            DR.append(self.correlate(z_edges, cat_data_bin, cat_rand_bin))
            if estimator == "LS":
                RR.append(self.correlate(z_edges, cat_rand_bin))
        DD = TreeCorrResultZbinned.from_bins(DD)
        DR = TreeCorrResultZbinned.from_bins(DR)
        if estimator == "LS":
            RR = TreeCorrResultZbinned.from_bins(RR)
        else:
            RR = None
        return CorrelationFunctionZbinned(dd=DD, dr=DR, rr=RR)

    def true_redshifts(self, zbins: NDArray[np.float_]) -> NzTrue:
        region_counts = DataFrame(
            index=pd.IntervalIndex.from_breaks(zbins),
            data={
                reg_id: np.histogram(d["z"], zbins)[0]
                for reg_id, d in self.unknown.groupby("region_idx")})
        return NzTrue(region_counts)


class UniformRandomGenerator(RailStage):
    pass


class SpatialRegionCreator(RailStage):
    pass


class SpatialRegionAssign(RailStage):
    pass


class Counts(RailStage):

    def __init__(self, args, comm=None):
        super().__init__(self, args, comm=comm)
        # construct radial bins
        self.r_min = ...
        self.r_max = ...
        r_steps = ...
        self.r_bins = np.logspace(self.r_min, self.r_max, r_steps + 1)


class CrossCorrelationCounts(Counts):
    """
    TODO.
    Don't forget to define a fixed default cosmology!
    """
    name = "CrossCorrelationCounts"
    config_options = RailStage.config_options.copy()
    inputs = [
        ("reference", PqHandle),
        ("unknown", PqHandle),
        ("randoms", PqHandle)]
    outputs = [("counts", PqHandle)]

    def run(self):
        reference = self.get_data("reference", allow_missing=False)
        unknown = self.get_data("unknown", allow_missing=False)
        randoms = self.get_data("randoms", allow_missing=False)

        self.add_data("counts", ...)

    def __call__(
        self,
        reference: DataFrame,
        unknown: DataFrame,
        randoms: DataFrame
    ) -> DataFrame:
        """
        TODO.
        """
        reference = self.set_data("reference", reference)
        unknown = self.set_data("unknown", unknown)
        randoms = self.set_data("randoms", randoms)
        self.run()
        return self.get_handle("counts")


class AutoCorrelationCounts(Counts):
    """
    TODO.
    Don't forget to define a fixed default cosmology!
    """
    name = "AutoCorrelationCounts"
    config_options = RailStage.config_options.copy()
    inputs = [
        ("data", PqHandle),
        ("randoms", PqHandle)]
    outputs = [("counts", PqHandle)]

    def __init__(self, args, comm=None):
        super().__init__(self, args, comm=comm)

    def run(self):
        data = self.get_data("data", allow_missing=False)
        randoms = self.get_data("randoms", allow_missing=False)

        self.add_data("counts", ...)

    def __call__(
        self,
        data: DataFrame,
        randoms: DataFrame
    ) -> DataFrame:
        """
        TODO.
        """
        data = self.set_data("data", data)
        randoms = self.set_data("randoms", randoms)
        self.run()
        return self.get_handle("counts")


class CorrelationCountsMerger(RailStage):
    pass


class CorrelationMesuarements(RailStage):
    pass


class YAWInterface(RailStage):
    pass


# TODO: don't forget entry in README if that is still maintained.
