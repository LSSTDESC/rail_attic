from __future__ import annotations

import os
from collections.abc import Iterator

from astropy.cosmology import FLRW
import numpy as np
import pandas as pd
import treecorr
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Interval
from tqdm import tqdm

from rail.estimation.algos.yet_another_wizz.bootstrap import TreeCorrData
from rail.estimation.algos.yet_another_wizz.correlation import CorrelationFunction
from rail.estimation.algos.yet_another_wizz.redshifts import BinFactory, NzTrue
from rail.estimation.algos.yet_another_wizz.utils import CatalogWrapper, CustomCosmology, get_default_cosmology


class YetAnotherWizz:

    def __init__(
        self,
        *,
        # data samples
        reference: CatalogWrapper,
        ref_rand: CatalogWrapper,
        unknown: CatalogWrapper,
        unk_rand: CatalogWrapper,
        # measurement scales, TODO: implement multiple scales!
        rmin_Mpc: ArrayLike = 0.1,
        rmax_Mpc: ArrayLike = 1.0,
        dist_weight_scale: float | None = None,
        rbin_slop: float | None = None,
        # redshift binning
        zmin: float | None = None,
        zmax: float | None = None,
        n_zbins: int | None = None,
        zbin_method: str = "linear",
        cosmology: FLRW | CustomCosmology | None = None,
        zbins: NDArray | None = None,
        # others
        num_threads: int | None = None
    ) -> None:
        # set data
        self.reference = reference
        self.ref_rand = ref_rand
        self.unknown = unknown
        self.unk_rand = unk_rand
        # configure scales
        self.rmin = rmin_Mpc
        self.rmax = rmax_Mpc
        self.dist_weight_scale = dist_weight_scale
        if cosmology is None:
            cosmology = get_default_cosmology()
        self.cosmology = cosmology
        # configure redshift binning
        if zbins is not None:
            BinFactory.check(zbins)
            self.binning = zbins
        elif zmin is not None and zmax is not None and n_zbins is not None:
            factory = BinFactory(zmin, zmax, n_zbins, cosmology)
            self.binning = factory.get(zbin_method, redshifts=reference["z"])
        else:
            raise ValueError(
                "either 'zbins' or 'zmin', 'zmax', 'n_zbins' must be provided")
        # others
        if num_threads is None:
            num_threads = os.cpu_count()
        # set treecorr configuration
        self.correlator_config = dict(
            sep_units="degrees",
            metric="Arc",
            nbins=(1 if dist_weight_scale is None else 50),
            bin_slop=rbin_slop,
            num_threads=num_threads)

    def get_config(self) -> dict[str, int | float | bool | str | None]:
        raise NotImplementedError  # TODO

    def bin_iter(
        self,
        data: CatalogWrapper,
        rand: CatalogWrapper,
        *,
        desc: str | None = None
    ) -> Iterator[tuple[Interval, CatalogWrapper, CatalogWrapper]]:
        iterator = zip(data.bin_iter(self.binning), rand.bin_iter(self.binning))
        if desc is not None:  # wrap with tqdm
            iterator = tqdm(iterator, total=len(self.binning)-1, desc=desc)
        for (z_interval, data_bin), (_, rand_bin) in iterator:
            yield z_interval, data_bin, rand_bin

    def _rbin_to_angle(self, z: float,) -> tuple[float, float]:
        f_K = self.cosmology.comoving_transverse_distance(z)  # for 1 radian
        angle_rad = np.asarray([self.rmin, self.rmax]) * (1.0 + z) / f_K.value
        ang_min, ang_max = np.rad2deg(angle_rad)
        return ang_min, ang_max

    def correlate(
        self,
        z_interval: Interval,
        cat1: treecorr.Catalog,
        cat2: treecorr.Catalog | None = None
    ) -> tuple[Interval, treecorr.NNCorrelation]:
        ang_min, ang_max = self._rbin_to_angle(z_interval.mid)
        correlation = treecorr.NNCorrelation(
            min_sep=ang_min, max_sep=ang_max, **self.correlator_config)
        correlation.process(cat1, cat2)
        """
        # TODO: implement 1/r weights
        """
        return TreeCorrData.from_nncorrelation(z_interval, correlation)

    def crosscorr(
        self,
        *,
        compute_rr: bool = False,
        progress: bool = False
    ) -> CorrelationFunction:
        dd, dr, rd, rr = [], [], [], []
        for z_edges, cat_reference, cat_ref_rand in self.bin_iter(
                self.reference, self.ref_rand,
                desc=("w_sp" if progress else None)):
            # correlate
            dd.append(self.correlate(z_edges, cat_reference, self.unknown))
            dr.append(self.correlate(z_edges, cat_reference, self.unk_rand))
            if compute_rr:  # otherwise will default to Davis-Peebles
                rd.append(self.correlate(z_edges, cat_ref_rand, self.unknown))
                rr.append(self.correlate(z_edges, cat_ref_rand, self.unk_rand))
        # merge binnned measurements
        DD = TreeCorrData.from_bins(dd)
        DR = TreeCorrData.from_bins(dr)
        RD = TreeCorrData.from_bins(rd) if compute_rr else None
        RR = TreeCorrData.from_bins(rr) if compute_rr else None
        return CorrelationFunction(dd=DD, dr=DR, rd=RD, rr=RR)

    def autocorr(
        self,
        which: str,
        *,
        compute_rr: bool = False,
        progress: bool = False
    ) -> CorrelationFunction:
        if which == "reference":
            bin_iter = self.bin_iter(
                self.reference, self.ref_rand,
                desc=("w_ss" if progress else None))
        elif which == "unknown":
            bin_iter = self.bin_iter(
                self.unknown, self.unk_rand,
                desc=("w_pp" if progress else None))
        else:
            raise ValueError("'which' must be either 'reference' or 'unknown'")
        dd, dr, rr = [], [], []
        for z_edges, cat_data_bin, cat_rand_bin in bin_iter:
            # correlate
            dd.append(self.correlate(z_edges, cat_data_bin))
            dr.append(self.correlate(z_edges, cat_data_bin, cat_rand_bin))
            if compute_rr:  # otherwise will default to Davis-Peebles
                rr.append(self.correlate(z_edges, cat_rand_bin))
        # merge binnned measurements
        DD = TreeCorrData.from_bins(dd)
        DR = TreeCorrData.from_bins(dr)
        RR = TreeCorrData.from_bins(rr) if compute_rr else None
        return CorrelationFunction(dd=DD, dr=DR, rr=RR)

    def true_redshifts(self) -> NzTrue:
        u = self.unknown
        region_counts = DataFrame(
            index=pd.IntervalIndex.from_breaks(self.binning),
            data={patch_id: np.histogram(patch_cat.z, self.binning)[0]
                  for patch_id, patch_cat in self.unknown.patch_iter()})
        return NzTrue(region_counts)
