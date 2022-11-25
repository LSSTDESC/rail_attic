from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
from astropy.cosmology import FLRW, Planck15
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Interval
from treecorr import Catalog


def get_default_cosmology() -> FLRW:
    return Planck15


class CustomCosmology(ABC):
    """
    Can be used to implement a custom cosmology outside of astropy.cosmology
    """

    @abstractmethod
    def to_format(self, format: str = "mapping") -> str:
        raise NotImplementedError

    @abstractmethod
    def comoving_distance(self, z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def comoving_transverse_distance(self, z: ArrayLike) -> ArrayLike:
        raise NotImplementedError


class UniformRandoms:

    def __init__(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        seed: int = 12345
    ) -> None:
        self.x_min, self.y_min = self.sky2cylinder(ra_min, dec_min)
        self.x_max, self.y_max = self.sky2cylinder(ra_max, dec_max)
        self.rng = np.random.default_rng(seed=seed)

    @classmethod
    def from_catalogue(cls, cat: BinnedCatalog) -> UniformRandoms:
        return cls(
            np.rad2deg(cat.ra.min()),
            np.rad2deg(cat.ra.max()),
            np.rad2deg(cat.dec.min()),
            np.rad2deg(cat.dec.max()))

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
        names: list[str, str] | None = None,
        draw_from: dict[str, NDArray] | None = None
    ) -> DataFrame:
        # generate positions
        x = np.random.uniform(self.x_min, self.x_max, size)
        y = np.random.uniform(self.y_min, self.y_max, size)
        ra, dec = self.cylinder2sky(x, y).T
        rand = DataFrame({names[0]: ra, names[1]: dec})
        if names is None:
            names = ["ra", "dec"]
        # generate random draw
        if draw_from is not None:
            N = None
            for col in draw_from.values():
                if N is None:
                    if len(col.shape) > 1:
                        raise ValueError("data to draw from must be 1-dimensional")
                    N = len(col)
                else:
                    if len(col) != N:
                        raise ValueError(
                            "length of columns to draw from does not match")
            draw_idx = self.rng.integers(0, N, size=size)
            # draw and insert data
            for key, col in draw_from.items():
                rand[key] = col[draw_idx]
        return rand


def iter_bin_masks(
    data: NDArray,
    bins: NDArray,
    closed: str = "right"
) -> Iterator[tuple[Interval, NDArray[np.bool_]]]:
    if closed not in ("left", "right"):
        raise ValueError("'closed' must be either of 'left', 'right'")
    intervals = pd.IntervalIndex.from_breaks(bins, closed=closed)
    bin_ids = np.digitize(data, bins, right=(closed=="right"))
    for i, interval in zip(range(1, len(bins)), intervals):
        yield interval, bin_ids==i


class BinnedCatalog(Catalog):

    @classmethod
    def from_dataframe(
        cls,
        data: DataFrame,
        patches: int | BinnedCatalog,
        ra: str,
        dec: str,
        z: str | None = None,
        **kwargs
    ) -> BinnedCatalog:
        if isinstance(patches, int):
            kwargs.update(dict(npatch=patches))
        else:
            kwargs.update(dict(patch_centers=patches.patch_centers))
        redshift = None if z is None else data[z]
        new = cls(
            ra=data[ra], ra_units="degrees",
            dec=data[dec], dec_units="degrees",
            r=redshift, **kwargs)
        return new

    @classmethod
    def from_catalog(cls, cat: Catalog) -> BinnedCatalog:
        new = cls.__new__(cls)
        new.__dict__ = cat.__dict__
        return new

    def bin_iter(
        self,
        z_bins: NDArray[np.float_],
    ) -> Iterator[tuple[Interval, Catalog]]:
        if self.r is None:
            raise ValueError("no redshifts for iteration provided")
        for interval, bin_mask in iter_bin_masks(self.r, z_bins):
            new = self.copy()
            new.select(bin_mask)
            yield interval, BinnedCatalog.from_catalog(new)

    def patch_iter(self) -> Iterator[tuple[int, Catalog]]:
        patch_ids = sorted(set(self.patch))
        patches = self.get_patches()
        for patch_id, patch in zip(patch_ids, patches):
            yield patch_id, BinnedCatalog.from_catalog(patch)


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
