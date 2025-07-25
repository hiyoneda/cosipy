from abc import ABC, abstractmethod
from typing import Tuple, Type, TypeVar, Generic, ClassVar

import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Quantity

class Measurement(ABC):

    def  __init__(self, label:str, *args, **kwargs):
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @property
    @abstractmethod
    def size(self) -> int:...

    @property
    @abstractmethod
    def value_type(self) -> Type:...

    @abstractmethod
    def __getitem__(self, item:int):...

    @abstractmethod
    def __iter__(self):...

T = TypeVar('T')
t = TypeVar('t')

class ArrayLikeMeasurement(Measurement, Generic[T,t]):
    """
    Data already implements and iterable, [] and size
    """

    _value_type = ClassVar[type]

    def __init__(self, data:T, label:str, *args, **kwargs):

        self._data = data
        super().__init__(label)

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def value_type(self) -> Type:
        return self._value_type

    def __getitem__(self, item:int) -> t:
        return self._data[item]

    def __iter__(self) -> t:
        return self._data.__iter__()

    @property
    def data(self) -> T:
        return self._data


class QuantityMeasurement(ArrayLikeMeasurement[Quantity, Quantity]):
    """
    """
    _value_type = Quantity

class SkyCoordMeasurement(ArrayLikeMeasurement[SkyCoord, SkyCoord]):
    """

    """
    _value_type = SkyCoord

class AngleMeasurement(ArrayLikeMeasurement[Angle, Angle]):
    """

    """
    _value_type = Angle

class FloatingMeasurement(ArrayLikeMeasurement[np.ndarray, np.floating]):

    _value_type = np.floating

    def __init__(self, data: np.ndarray, label: str, *args, **kwargs):

        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError("This class expect float or double types")

        self._data = data
        super().__init__(data, label)
