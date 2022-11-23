from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import numpy as np
import treecorr
from numpy.typing import NDArray
from pandas import DataFrame, Interval, IntervalIndex

from rail.estimation.algos.yet_another_wizz.utils import ArrayDict


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

    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        N = max(self.npatch)
        rng = np.random.default_rng(seed=seed)
        return rng.integers(0, N, size=(n_boot, N))

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
