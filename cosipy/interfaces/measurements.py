import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Tuple, Type, TypeVar, Generic, ClassVar, Union, Generator, Iterable

import numpy as np
from astropy.coordinates import SkyCoord, Angle, BaseCoordinateFrame, UnitSphericalRepresentation, \
    CartesianRepresentation
from astropy.units import Quantity, Unit
from numpy.typing import NDArray

class MeasurementIterator(Iterator):

    def __next__(self) -> Union[int, float, Tuple[Union[int, float]]]:...

class Measurement(Sequence):

    def  __init__(self, label:str, *args, **kwargs):
        self._label = label

    # Needs __len__ and either __iter__ or __getitem__ (or both)

    @property
    def label(self) -> str:
        return self._label

    @property
    def size(self) -> int:
        return len(self)

    @property
    @abstractmethod
    def value_type(self) -> Union[Type, Tuple[Type]]:
        """
        Types return by __iter__ and __getitem__
        """

    @property
    def nvalues(self) -> int:
        if isinstance(self.value_type, tuple):
            return len(self.value_type)
        else:
            return 1

    def cache(self, start=None, stop=None, step=None) -> Iterable:
        values = []
        for value in itertools.islice(self, start, stop, step):
            values.append(value)

        return values

class FloatingMeasurement(Measurement, ABC):

    @property
    def value_type(self) -> Union[Type, Tuple[Type]]:
        return float

    def cache(self, start=None, stop=None, step=None) -> NDArray[float]:
        values = super().cache(start, stop, step)
        return np.asarray(values)

class CachedFloatingMeasurement(FloatingMeasurement):

    def __init__(self, label:str, array: np.ndarray[float]):
        if array.ndim != 1:
            raise ValueError("This class handles 1D and only 1D arrays")

        super().__init__(label)
        self._array = array

    def __len__(self):
        return self._array.size

    def __iter__(self):
        return iter(self._array)

    def __getitem__(self, item):
        return self._array[item]

    def cache(self, start=None, stop=None, step=None) -> np.ndarray[float]:
        return self._array[start, stop, step]


class QuantityMeasurement(FloatingMeasurement, ABC):

    @property
    @abstractmethod
    def unit(self) -> Unit:...

    def fancy_iter(self, start = None, stop = None, step = None) -> Generator[Quantity, None, None]:
        for value in itertools.islice(self, start, stop, step):
            yield Quantity(value, self.unit)

    def cache(self, start = None, stop = None, step = None) -> Quantity:
        return Quantity(super().cache(start, stop, step), self.unit)


class CachedQuantityMeasurement(CachedFloatingMeasurement, QuantityMeasurement):

    def __init__(self, label:str, array: Quantity):
        if array.ndim != 1:
            raise ValueError("This class handles 1D and only 1D arrays")

        super().__init__(label, array.value)
        self._array = array.value
        self._unit = array.unit

    def unit(self) -> Unit:
        return self._unit

    def cache(self, start = None, stop = None, step = None) -> Quantity:
        return Quantity(self._array[start:stop:step], self.unit)

    def fancy_iter(self, start=None, stop=None, step=None) -> Generator[Quantity, None, None]:
        return iter(self.cache)

class AngleMeasurement(QuantityMeasurement, ABC):

    def fancy_iter(self, start=None, stop=None, step=None) -> Generator[Quantity, None, None]:
        for value in itertools.islice(self, start, stop, step):
            yield Angle(value, self.unit)

    def cache(self, start=None, stop=None, step=None) -> Angle:
        return Angle(super().cache(start, stop, step), self.unit)

class CachedAngleMeasurement(CachedQuantityMeasurement):

    def __init__(self, label:str, array: Angle):
        super().__init__(label, Quantity(array.value, array.unit))

    def cache(self, start = None, stop = None, step = None) -> Quantity:
        return Angle(self._array[start:stop:step], self.unit)

class SkyCoordMeasurement(Measurement, ABC):
    """

    """

    @property
    @abstractmethod
    def frame(self) -> BaseCoordinateFrame:...
    @property
    @abstractmethod
    def unit(self) -> Unit:...

    def as_unit_spherical(self) -> 'SkyCoordUnitSphericalMeasurement':...
    def as_cartesian(self) -> 'SkyCoordCartesianMeasurement':...

class CachedSkyCoordMeasurement(SkyCoordMeasurement, ABC):

    def __init__(self, label: str, coord: SkyCoord):

        super().__init__(label)

        if coord.ndim != 1:
            raise ValueError("This class handles 1D and only 1D SkyCoord arrays")

        self._frame = coord.frame

        self._unit = None # Set by child class. Type Unit
        self._data = None # Set by child class. array of shape (self.size,self.nvalues)

    def __len__(self):
        return self._data.shape[0]

    @property
    def frame(self) -> BaseCoordinateFrame:
        return self._unit

    @property
    def unit(self) -> Unit:
        return self._unit

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]

    @abstractmethod
    def cache(self, start=None, stop=None, step=None) -> SkyCoord:...

    def fancy_iter(self, start=None, stop=None, step=None) -> Generator[SkyCoord, None, None]:
        return iter(self.cache)


class SkyCoordUnitSphericalMeasurement(SkyCoordMeasurement, ABC):

    @property
    def value_type(self) -> Union[Type, Tuple[Type]]:
        return (float, float)

    def fancy_iter(self, start=None, stop=None, step=None) -> Generator[SkyCoord, None, None]:
        for lon,lat in itertools.islice(self, start, stop, step):
            yield SkyCoord(lon, lat, unit = self.unit, frame = self.frame)

    def cache(self, start=None, stop=None, step=None) -> SkyCoord:
        lon = []
        lat = []
        for lon_i,lat_i in itertools.islice(self, start, stop, step):
            lon.append(lon_i)
            lat.append(lat_i)

        return SkyCoord(lon, lat, unit = self.unit, frame = self.frame)

    def as_unit_spherical(self) -> 'SkyCoordUnitSphericalMeasurement':
        return self

    def as_cartesian(self) -> 'SkyCoordCartesianMeasurement':...


class CachedSkyCoordUnitSphericalMeasurement(CachedSkyCoordMeasurement, SkyCoordUnitSphericalMeasurement):

    def __init__(self, label:str, coord: SkyCoord):

        super().__init__(label, coord)

        rep = coord.represent_as('unitspherical')

        self._unit = rep.unit
        self._data = np.asarray([rep.lon.value, rep.lat.value]).transpose()

    def cache(self, start=None, stop=None, step=None) -> SkyCoord:
        return SkyCoord(self._data[start:stop:step,0], self._data[start:stop:step,1], unit = self.unit, frame = self.frame)

    def as_cartesian(self) -> 'SkyCoordCartesianMeasurement':
        rep = UnitSphericalRepresentation(Quantity(self._data[:,0], self.unit), Quantity(self._data[:,1], self.unit))
        cart_rep = rep.represent_as('cartesian')

        coord = SkyCoord(x = cart_rep.x, y = cart_rep.y, z = cart_rep.z, frame = self.frame, representation_type = 'cartesian')

        return CachedSkyCoordCartesianMeasurement(self.label, coord)


class SkyCoordCartesianMeasurement(SkyCoordMeasurement, ABC):

    @property
    def value_type(self) -> Union[Type, Tuple[Type]]:
        return (float, float, float)

    def fancy_iter(self, start=None, stop=None, step=None) -> Generator[SkyCoord, None, None]:
        for x,y,z in itertools.islice(self, start, stop, step):
            yield SkyCoord(x=x,y=y,z=z, unit = self.unit, frame = self.frame(), representation_type='cartesian')

    def cache(self, start=None, stop=None, step=None) -> SkyCoord:
        x = []
        y = []
        z = []
        for x_i,y_i,z_i in itertools.islice(self, start, stop, step):
            x.append(x_i)
            y.append(y_i)
            z.append(z_i)

        return SkyCoord(x=x,y=y,z=z, unit = self.unit, frame = self.frame(), representation_type='cartesian')

    def as_unit_spherical(self) -> 'SkyCoordUnitSphericalMeasurement':...

    def as_cartesian(self) -> 'SkyCoordCartesianMeasurement':
        return self


class CachedSkyCoordCartesianMeasurement(SkyCoordCartesianMeasurement):

    def __init__(self, label:str, coord: SkyCoord):
        super().__init__(label, coord)

        rep = coord.represent_as('cartesian')

        self._unit = rep.unit

        self._data = np.asarray([rep.x.value, rep.y.value, rep.z.value]).transpose()

    def cache(self, start=None, stop=None, step=None) -> SkyCoord:
        return SkyCoord(x = self._data[start:stop:step, 0], y = self._data[start:stop:step, 1], z = self._data[start:stop:step, 2],
                        unit = self.unit, frame = self.frame, representation_type = 'cartesian')

    def as_unit_spherical(self) -> 'SkyCoordUnitSphericalMeasurement':
        rep = CartesianRepresentation(self._data[:,0], self._data[:,1], self._data[:,2])
        usph_rep = rep.represent_as('unitspherical')

        coord = SkyCoord(usph_rep.lon, usph_rep.lat, frame = self.frame)

        return CachedSkyCoordUnitSphericalMeasurement(self.label, coord)