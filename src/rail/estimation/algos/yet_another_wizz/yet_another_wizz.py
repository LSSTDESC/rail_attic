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
from rail.estimation.algos.yet_another_wizz.utils import CustomCosmology, get_default_cosmology


class YetAnotherWizz:

    def __init__(
        self,
        *,
        # data samples
        reference: DataFrame,
        ref_rand: DataFrame,
        unknown: DataFrame,
        unk_rand: DataFrame,
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
        data: DataFrame,
        rand: DataFrame,
        *,
        desc: str | None = None
    ) -> Iterator[tuple[Interval, DataFrame, DataFrame]]:
        data_iter = data.groupby(pd.cut(data["z"], self.binning))
        rand_iter = rand.groupby(pd.cut(rand["z"], self.binning))
        iterator = zip(data_iter, rand_iter)  # guaranteed to have same length
        if desc is not None:
            iterator = tqdm(iterator, total=len(self.binning)-1, desc=desc)
        for (z_interval, data_bin), (_, rand_bin) in iterator:
            yield z_interval, data_bin, rand_bin

    def rbin_to_angle(self, z: float,) -> tuple[float, float]:
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
        ang_min, ang_max = self.rbin_to_angle(z_interval.mid)
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
        cat_unknown = self.make_catalogue(self.unknown)
        cat_unk_rand = self.make_catalogue(self.unk_rand)
        DD = []
        DR = []
        if compute_rr:  # otherwise default to Davis-Peebles
            RD = []
            RR = []
        for z_edges, data_bin, rand_bin in self.bin_iter(
                self.reference, self.ref_rand,
                desc=("w_sp" if progress else None)):
            cat_reference = self.make_catalogue(data_bin)
            cat_ref_rand = self.make_catalogue(rand_bin)
            # correlate
            DD.append(self.correlate(z_edges, cat_reference, cat_unknown))
            DR.append(self.correlate(z_edges, cat_reference, cat_unk_rand))
            if compute_rr:
                RD.append(self.correlate(z_edges, cat_ref_rand, cat_unknown))
                RR.append(self.correlate(z_edges, cat_ref_rand, cat_unk_rand))
        DD = TreeCorrData.from_bins(DD)
        DR = TreeCorrData.from_bins(DR)
        if compute_rr:
            RD = TreeCorrData.from_bins(RD)
            RR = TreeCorrData.from_bins(RR)
        else:
            RD = None
            RR = None
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
            if "z" not in self.unknown:
                raise KeyError(
                    f"missing redshift data for '{which}' autocorrelation")
            bin_iter = self.bin_iter(
                self.unknown, self.unk_rand,
                desc=("w_pp" if progress else None))
        else:
            raise ValueError("'which' must be either 'reference' or 'unknown'")
        DD = []
        DR = []
        if compute_rr:  # otherwise default to Davis-Peebles
            RR = []
        for z_edges, data_bin, rand_bin in bin_iter:
            cat_data_bin = self.make_catalogue(data_bin)
            cat_rand_bin = self.make_catalogue(rand_bin)
            # correlate
            DD.append(self.correlate(z_edges, cat_data_bin))
            DR.append(self.correlate(z_edges, cat_data_bin, cat_rand_bin))
            if compute_rr:
                RR.append(self.correlate(z_edges, cat_rand_bin))
        DD = TreeCorrData.from_bins(DD)
        DR = TreeCorrData.from_bins(DR)
        if compute_rr:
            RR = TreeCorrData.from_bins(RR)
        else:
            RR = None
        return CorrelationFunction(dd=DD, dr=DR, rr=RR)

    def true_redshifts(self) -> NzTrue:
        region_counts = DataFrame(
            index=pd.IntervalIndex.from_breaks(self.binning),
            data={
                reg_id: np.histogram(d["z"], self.binning)[0]
                for reg_id, d in self.unknown.groupby("region_idx")})
        return NzTrue(region_counts)
