from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from astropy.cosmology import FLRW, Planck15
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Interval, Series
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


@dataclass(frozen=True)
class CatalogWrapper:
    data: DataFrame = field(repr=False)
    ra_name: str
    dec_name: str
    patch_name: str
    z_name: str | None = field(default=None)

    def __post_init__(self) -> None:
        # check if the columns exist
        for kind in ("ra", "dec", "z", "patch"):
            name = getattr(self, f"{kind}_name")
            if name is None:
                continue
            if name not in self.data:
                raise KeyError(f"'{name}' not in data")

    @property
    def ra(self) -> Series:
        return self.data[self.ra_name]

    @property
    def dec(self) -> Series:
        return self.data[self.dec_name]

    @property
    def patch(self) -> Series:
        return self.data[self.patch_name]

    @property
    def z(self) -> Series:
        try:
            return self.data[self.z_name]
        except KeyError:
            return None

    @property
    @lru_cache(maxsize=1)
    def npatch(self) -> int:
        return len(np.unique(self.patch))

    def get_catalogue(self,) -> Catalog:
        return Catalog(
            ra=self.ra, ra_units="degrees",
            dec=self.dec, dec_units="degrees",
            patch=self.patch)

    def bin_iter(
        self,
        z_bins: NDArray[np.float_],
    ) -> Iterator[tuple[Interval, CatalogWrapper]]:
        if self.z is None:
            raise ValueError("no redshifts for iteration provided")
        iterator = self.data.groupby(pd.cut(self.z), z_bins)
        for interval, bin_data in iterator:
            yield interval, CatalogWrapper(
                bin_data, self.ra_name, self.dec_name,
                self.patch_name, self.z_name)

    def patch_iter(self) -> Iterator[tuple[int, CatalogWrapper]]:
        iterator = self.data.groupby(self.patch)
        for patch_id, patch_data in iterator:
            yield patch_id, CatalogWrapper(
                patch_data, self.ra_name, self.dec_name,
                self.patch_name, self.z_name)


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
