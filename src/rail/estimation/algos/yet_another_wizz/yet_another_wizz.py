from __future__ import annotations

import os
from collections.abc import Iterator

import astropy.cosmology
import numpy as np
import pandas as pd
import treecorr
from astropy.cosmology.core import Cosmology
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Interval
from tqdm import tqdm

from rail.estimation.algos.yet_another_wizz.bootstrap import TreeCorrData
from rail.estimation.algos.yet_another_wizz.correlation import CorrelationFunction
from rail.estimation.algos.yet_another_wizz.redshifts import NzTrue


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
        """
        # TODO: implement 1/r weights
        """
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
