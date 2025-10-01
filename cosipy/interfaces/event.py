from abc import ABC, abstractmethod
from typing import Sequence, Union, Protocol
from typing_extensions import runtime_checkable

from astropy.time import Time
from astropy.units import Quantity, Unit

__all__ = [
    "Event",
    "TimeTagEvent",
    "EventWithEnergy",
]

class EventMetadata:

    def __init__(self):
        self._metadata = {}

    def __getitem__(self, key):
        return self._metadata[key]

    def __setitem__(self, key, value):
        self._metadata[key] = value
        setattr(self, key, value)

    def __delitem__(self, key):
        if key in self._metadata:
            del self._metadata[key]
            delattr(self, key)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._metadata})"

@runtime_checkable
class Event(Protocol):
    """
    Derived classes implement all accessors
    """

    @property
    def metadata(self) -> EventMetadata:...

@runtime_checkable
class TimeTagEvent(Event, Protocol):

    @property
    def jd1(self) -> float:...

    @property
    def jd2(self) -> float:...

    @property
    def time(self) -> Time:
        """
        Add fancy time
        """
        return Time(self.jd1, self.jd2, format = 'jd')

@runtime_checkable
class EventWithEnergy(Event, Protocol):

    @property
    def energy_value(self) -> float:...

    @property
    def energy_unit(self) -> Unit:...

    @property
    def energy(self) -> Quantity:
        """
        Add fancy energy quantity
        """
        return Quantity(self.energy_value, self.energy_unit)

