"""
Implementation of yet_another_wizz for easy cross-correlation/clustering
redshifts (arXiv:2007.01846 / A&A 642, A200 (2020)).
"""
from __future__ import annotations

import dataclasses
import os
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Callable, Collection, Iterable, Iterator, KeysView, Mapping
from typing import Any

import astropy.cosmology
import numpy as np
import pandas as pd
import treecorr
from astropy.cosmology.core import Cosmology
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Interval, IntervalIndex, Series
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


class ArrayDict(Mapping):

    def __init__(
        self,
        keys: Collection[Any],
        array: NDArray
    ) -> None:
        if len(array) != len(keys):
            raise ValueError("number of keys and array length do not match")
        self._array = array
        self._dict = {key: idx for idx, key in enumerate(keys)}

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, key: Any) -> NDArray:
        idx = self._dict[key]
        return self._array[idx]

    def __iter__(self) -> Iterator[NDArray]:
        return self._dict.__iter__()

    def __contains__(self, key: Any) -> bool:
        return key in self._dict

    def items(self) -> list[tuple[Any, NDArray]]:
        # ensure that the items are ordered by the index of each key
        return sorted(self._dict.items(), key=lambda item: item[1])

    def keys(self) -> list[Any]:
        # key are ordered by their corresponding index
        return [key for key, _ in self.items()]

    def values(self) -> list[NDArray]:
        # values are returned in index order
        return [value for value in self._array]

    def get(self, key: Any, default: Any) -> Any:
        try:
            idx = self._dict[key]
        except KeyError:
            return default
        else:
            return self._array[idx]

    def sample(self, keys: Iterable[Any]) -> NDArray:
        idx = [self._dict[key] for key in keys]
        return self._array[idx]

    def as_array(self) -> NDArray:
        return self._array


@dataclasses.dataclass(frozen=True, repr=False)
class PairCountData:
    binning: IntervalIndex
    count: NDArray[np.float_]
    total: NDArray[np.float_]

    def normalise(self) -> NDArray[np.float_]:
        normalised = self.count / self.total
        return DataFrame(data=normalised.T, index=self.binning)


@dataclasses.dataclass(frozen=True, repr=False)
class TreeCorrData:
    npatch: tuple(int, int)
    count: ArrayDict
    total: ArrayDict
    mask: NDArray[np.bool_]
    binning: IntervalIndex

    def __post_init__(self) -> None:
        if self.count.keys() != self.total.keys():
            raise KeyError("keys for 'count' and 'total' are not identical")

    def keys(self) -> list[tuple(int, int)]:
        return self.total.keys()

    @property
    def nbins(self) -> int:
        return len(self.binning)

    @classmethod
    def from_nncorrelation(
        cls,
        interval: Interval,
        correlation: treecorr.NNCorrelation
    ) -> TreeCorrData:
        # extract the (cross-patch) pair counts
        npatch = (correlation.npatch1, correlation.npatch2)
        keys = []
        count = np.empty((len(correlation.results), 1))
        total = np.empty((len(correlation.results), 1))
        for i, (patches, result) in enumerate(correlation.results.items()):
            keys.append(patches)
            count[i] = result.weight
            total[i] = result.tot
        return cls(
            npatch=npatch,
            count=ArrayDict(keys, count),
            total=ArrayDict(keys, total),
            mask=correlation._ok,
            binning=IntervalIndex([interval]))

    @classmethod
    def from_bins(
        cls,
        zbins: Iterable[TreeCorrData]
    ) -> TreeCorrData:
        # check that the data is compatible
        if len(zbins) == 0:
            raise ValueError("'zbins' is empty")
        npatch = zbins[0].npatch
        mask = zbins[0].mask
        keys = tuple(zbins[0].keys())
        nbins = zbins[0].nbins
        for zbin in zbins[1:]:
            if zbin.npatch != npatch:
                raise ValueError("the patch numbers are inconsistent")
            if not np.array_equal(mask, zbin.mask):
                raise ValueError("pair masks are inconsistent")
            if tuple(zbin.keys()) != keys:
                raise ValueError("patches are inconsistent")
            if zbin.nbins != nbins:
                raise IndexError("number of bins is inconsistent")

        # check the ordering of the bins based on the provided intervals
        binning = IntervalIndex.from_tuples([
            zbin.binning.to_tuples()[0]  # contains just one entry
            for zbin in zbins])
        if not binning.is_non_overlapping_monotonic:
            raise ValueError(
                "the binning interval is overlapping or not monotonic")
        for this, following in zip(binning[:-1], binning[1:]):
            if this.right != following.left:
                raise ValueError(f"the binning interval is not contiguous")

        # merge the ArrayDicts
        count = ArrayDict(
            keys, np.column_stack([zbin.count.as_array() for zbin in zbins]))
        total = ArrayDict(
            keys, np.column_stack([zbin.total.as_array() for zbin in zbins]))
        return cls(
            npatch=npatch,
            count=count,
            total=total,
            mask=mask,
            binning=binning)

    def get_patch_count(self) -> DataFrame:
        return DataFrame(
            index=self.binning,
            columns=self.keys(),
            data=self.count.as_array().T)

    def get_patch_total(self) -> DataFrame:
        return DataFrame(
            index=self.binning,
            columns=self.keys(),
            data=self.total.as_array().T)

    def get(self) -> PairCountData:
        return PairCountData(
            binning=self.binning,
            count=self.count.as_array().sum(axis=0),
            total=self.total.as_array().sum(axis=0))

    def get_jackknife_samples(
        self,
        global_norm: bool = False,
        **kwargs
    ) -> PairCountData:
        # The iterator expects a single patch index which is dropped in a single
        # realisation.
        count = []
        total = []
        if global_norm:
            global_total = self.total.as_array().sum(axis=0)
        for idx in range(max(self.npatch)):  # leave-one-out iteration
            # we need to use the jackknife iterator twice
            patches = list(treecorr.NNCorrelation.JackknifePairIterator(
                self.count, *self.npatch, idx, self.mask))
            count.append(self.count.sample(patches).sum(axis=0))
            if global_norm:
                total.append(global_total)
            else:
                total.append(self.total.sample(patches).sum(axis=0))
        return PairCountData(
            binning=self.binning,
            count=np.array(count),
            total=np.array(total))

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
        count = []
        total = []
        if global_norm:
            global_total = self.total.as_array().sum(axis=0)
        for idx in patch_idx:  # simplified leave-one-out iteration
            # we need to use the jackknife iterator twice
            patches = list(treecorr.NNCorrelation.BootstrapPairIterator(
                self.count, *self.npatch, idx, self.mask))
            count.append(self.count.sample(patches).sum(axis=0))
            if global_norm:
                total.append(global_total)
            else:
                total.append(self.total.sample(patches).sum(axis=0))
        return PairCountData(
            binning=self.binning,
            count=np.array(count),
            total=np.array(total))

    def get_samples(
        self,
        method: str,
        **kwargs
    ) -> PairCountData:
        if method == "jackknife":
            samples = self.get_jackknife_samples(**kwargs)
        elif method == "bootstrap":
            samples = self.get_bootstrap_samples(**kwargs)
        else:
            raise NotImplementedError(
                f"sampling method '{method}' not implemented")
        return samples


class CorrelationEstimator(ABC):
    variants: list[CorrelationEstimator] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.variants.append(cls)

    def name(self) -> str:
        return self.__class__.__name__

    @abstractproperty
    def short(self) -> str:
        return "CE"

    @abstractproperty
    def requires(self) -> list[str]:
        return ["dd", "dr", "rr"]

    @abstractproperty
    def optional(self) -> list[str]:
        return ["rd"]

    @abstractmethod
    def __call__(
        self,
        *,
        dd: PairCountData,
        dr: PairCountData,
        rr: PairCountData,
        rd: PairCountData | None = None
    ) -> DataFrame:
        raise NotImplementedError


class PeeblesHauser(CorrelationEstimator):
    short: str = "PH"
    requires = ["dd", "rr"]
    optional = []

    def __call__(
        self,
        *,
        dd: PairCountData,
        rr: PairCountData
    ) -> DataFrame:
        DD = dd.normalise()
        RR = rr.normalise()
        return DD / RR - 1.0


class DavisPeebles(CorrelationEstimator):
    short = "DP"
    requires = ["dd", "dr"]
    optional = []

    def __call__(
        self,
        *,
        dd: PairCountData,
        dr: PairCountData
    ) -> DataFrame:
        DD = dd.normalise()
        DR = dr.normalise()
        return DD / DR - 1.0


class Hamilton(CorrelationEstimator):
    short = "HM"
    requires = ["dd", "dr", "rr"]
    optional = ["rd"]

    def __call__(
        self,
        *,
        dd: PairCountData,
        dr: PairCountData,
        rr: PairCountData,
        rd: PairCountData | None = None
    ) -> DataFrame:
        DD = dd.normalise()
        DR = dr.normalise()
        RD = DR if rd is None else rd.normalise()
        RR = rr.normalise()
        return (DD * RR) / (DR * RD) - 1.0


class LandySzalay(CorrelationEstimator):
    short = "LS"
    requires = ["dd", "dr", "rr"]
    optional = ["rd"]

    def __call__(
        self,
        *,
        dd: PairCountData,
        dr: PairCountData,
        rr: PairCountData,
        rd: PairCountData | None = None
    ) -> DataFrame:
        DD = dd.normalise()
        DR = dr.normalise()
        RD = DR if rd is None else rd.normalise()
        RR = rr.normalise()
        return (DD - (DR + RD) + RR) / RR


@dataclasses.dataclass(frozen=True, repr=False)
class CorrelationFunction:
    dd: TreeCorrData
    dr: TreeCorrData | None = dataclasses.field(default=None)
    rd: TreeCorrData | None = dataclasses.field(default=None)
    rr: TreeCorrData | None = dataclasses.field(default=None)
    npatch: tuple(int, int) = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "npatch", self.dd.npatch)  # since frozen=True
        # check if the minimum required pair counts are provided
        if self.dr is None and self.rr is None:
            raise ValueError("either 'dr' or 'rr' is required")
        if self.dr is None and self.rd is not None:
            raise ValueError("'rd' requires 'dr'")
        # check that the pair counts are compatible
        for kind in ("dr", "rd", "rr"):
            pairs: TreeCorrData | None = getattr(self, kind)
            if pairs is None:
                continue
            if pairs.npatch != self.npatch:
                raise ValueError(f"patches of '{kind}' do not match 'dd'")
            if np.any(pairs.binning != self.dd.binning):
                raise ValueError(f"binning of '{kind}' and 'dd' does not match")

    @property
    def binning(self) -> IntervalIndex:
        return self.dd.binning

    def is_compatible(
        self,
        other: TreeCorrData
    ) -> bool:
        if self.npatch != other.npatch:
            return False
        if np.any(self.binning != other.binning):
            return False
        return True

    @property
    def estimators(self) -> dict[str, CorrelationEstimator]:
        # figure out which of dd, dr, ... are not None
        available = set()
        # iterate all dataclass attributes that are in __init__
        for attr in dataclasses.fields(self):
            if not attr.init:
                continue
            if getattr(self, attr.name) is not None:
                available.add(attr.name)
        # check which estimators are supported
        estimators = {}
        for estimator in CorrelationEstimator.variants:  # registered estimators
            if set(estimator.requires) <= available:
                estimators[estimator.short] = estimator
        return estimators

    def _check_and_select_estimator(
        self,
        estimator: str
    ) -> CorrelationEstimator:
        options = self.estimators
        if estimator not in options:
            opts = ", ".join(sorted(options.keys()))
            raise ValueError(
                f"estimator '{estimator}' not available, options are: {opts}")
        # select the correct estimator
        return options[estimator]()  # return estimator class instance

    def get(
        self,
        estimator: str
    ) -> Series:
        estimator_func = self._check_and_select_estimator(estimator)
        requires = {
            kind: getattr(self, kind).get()
            for kind in estimator_func.requires}
        optional = {
            kind: getattr(self, kind).get()
            for kind in estimator_func.optional
            if getattr(self, kind) is not None}
        return estimator_func(**requires, **optional)[0]

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
        patch_idx: NDArray[np.int_] | None = None
    ) -> DataFrame:
        # set up the sampling method
        valid_methods = ("bootstrap", "jackknife")
        if sample_method not in valid_methods:
            opts = ", ".join(f"'{s}'" for s in valid_methods)
            raise ValueError(f"'sample_method' must be either of {opts}")
        if patch_idx is None and sample_method == "bootstrap":
            patch_idx = self.generate_bootstrap_patch_indices(n_boot)
        sample_kwargs = dict(
            method=sample_method,
            global_norm=global_norm,
            patch_idx=patch_idx)
        # select the sampling method and generate optional bootstrap samples
        estimator_func = self._check_and_select_estimator(estimator)
        requires = {
            kind: getattr(self, kind).get_samples(**sample_kwargs)
            for kind in estimator_func.requires}
        optional = {
            kind: getattr(self, kind).get_samples(**sample_kwargs)
            for kind in estimator_func.optional
            if getattr(self, kind) is not None}
        return estimator_func(**requires, **optional)


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
        patch_idx = rng.integers(0, n_regions, size=[n_boot, n_regions])
        Nz_boot = np.sum(self.counts.to_numpy().T[patch_idx], axis=1)
        nz_boot = Nz_boot / (
            Nz_boot.sum(axis=1)[:, np.newaxis] * self.dz[np.newaxis, :])
        samples = pd.DataFrame(
            index=self.counts.index, columns=np.arange(n_boot), data=nz_boot.T)
        return samples


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
    ) -> tuple[Interval, treecorr.NNCorrelation]:
        ang_min, ang_max = angular_separation(
            [self.rmin, self.rmax], z_interval.mid)
        correlation = treecorr.NNCorrelation(
            min_sep=ang_min, max_sep=ang_max, **self.correlator_config)
        correlation.process(cat1, cat2)
        return TreeCorrData.from_nncorrelation(z_interval, correlation)

    def crosscorr(
        self,
        zbins: NDArray[np.float_],
        estimator: str = "LS"
    ) -> CorrelationFunction:
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
        DD = TreeCorrData.from_bins(DD)
        DR = TreeCorrData.from_bins(DR)
        if estimator == "LS":
            RD = TreeCorrData.from_bins(RD)
            RR = TreeCorrData.from_bins(RR)
        else:
            RD = None
            RR = None
        return CorrelationFunction(dd=DD, dr=DR, rd=RD, rr=RR)

    def autocorr(
        self,
        zbins: NDArray[np.float_],
        estimator: str = "LS",
        which: str = "reference"
    ) -> CorrelationFunction:
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
        DD = TreeCorrData.from_bins(DD)
        DR = TreeCorrData.from_bins(DR)
        if estimator == "LS":
            RR = TreeCorrData.from_bins(RR)
        else:
            RR = None
        return CorrelationFunction(dd=DD, dr=DR, rr=RR)

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
