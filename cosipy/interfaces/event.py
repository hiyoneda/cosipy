from abc import ABC, abstractmethod
from typing import Sequence, Union

from astropy.time import Time
from astropy.units import Quantity, Unit

__all__ = [
    "Event",
    "TimetaggedEvent",
    "EventWithEnergy",
]

class Event(Sequence, ABC):
    """
    Need to implement __getitem__
    """

    @classmethod
    @abstractmethod
    def size(cls) -> int:
        """
        Number of values
        """

    def __len__(self):
        return self.size

class FancyTimeDataMixin(ABC):

    @property
    @abstractmethod
    def jd1(self) -> Union[float, Sequence[float]]:...

    @property
    @abstractmethod
    def jd2(self) -> Union[float, Sequence[float]]:...

    @property
    def time(self) -> Time:
        """
        Add fancy time
        """
        return Time(self.jd1, self.jd2, format = 'jd')

class TimetaggedEvent(FancyTimeDataMixin, Sequence):

    @property
    @abstractmethod
    def jd1(self) -> float:...

    @property
    @abstractmethod
    def jd2(self) -> float:...


class FancyEnergyDataMixin(ABC):

    @property
    @abstractmethod
    def energy_value(self) -> Union[float, Sequence[float]]:...

    @property
    @abstractmethod
    def energy_unit(self) -> Unit:...

    @property
    def energy(self) -> Quantity:
        """
        Add fancy energy quantity
        """
        return Quantity(self.energy_value, self.energy_unit)


class EventWithEnergy(FancyEnergyDataMixin, Sequence):

    @property
    @abstractmethod
    def energy_value(self) -> float:...

    @property
    @abstractmethod
    def energy_unit(self) -> Unit:...

