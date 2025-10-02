import itertools
from typing import Protocol, runtime_checkable, Dict, Type, Any, Tuple, Iterator, Union, Sequence, Iterable, ClassVar

import numpy as np
from astropy.units import Unit, Quantity

from . import EventWithEnergy
from .event import Event, TimeTagEvent
from histpy import Histogram, Axes

from astropy.time import Time

# Guard to prevent circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .event_selection import EventSelectorInterface

import histpy

__all__ = ["DataInterface",
           "EventDataInterface",
           "BinnedDataInterface",
           "TimeTagEventData",
           "EventDataWithEnergy"
          ]

@runtime_checkable
class DataInterface(Protocol):

    # Type returned by __iter__ in the event data case
    event_type = ClassVar[Type]

@runtime_checkable
class BinnedDataInterface(DataInterface, Protocol):
    @property
    def data(self) -> Histogram:...
    @property
    def axes(self) -> Axes:...

@runtime_checkable
class EventDataInterface(DataInterface, Protocol):

    def __iter__(self) -> Iterator[Event]:
        """
        Return one Event at a time
        """

    def __getitem__(self, item: int) -> Event:
        """
        Convenience method. Pretty slow in general. It's suggested that
        the implementations override it
        """
        return next(itertools.islice(self, item, None))

    @property
    def nevents(self) -> int:
        """
        Total number of events yielded by __iter__

        Convenience method. Pretty slow in general. It's suggested that
        the implementations override it
        """
        return sum(1 for _ in iter(self))

@runtime_checkable
class TimeTagEventData(EventDataInterface, Protocol):

    def __iter__(self) -> Iterator[TimeTagEvent]:...

    @property
    def jd1(self) -> Iterable[float]: ...

    @property
    def jd2(self) -> Iterable[float]: ...

    @property
    def time(self) -> Time:
        """
        Add fancy time
        """
        return Time(self.jd1, self.jd2, format = 'jd')

@runtime_checkable
class EventDataWithEnergy(EventDataInterface, Protocol):

    def __iter__(self) -> Iterator[EventWithEnergy]:...

    @property
    def energy_value(self) -> Iterable[float]:...

    @property
    def energy_unit(self) -> Unit:...

    @property
    def energy(self) -> Quantity:
        """
        Add fancy energy quantity
        """
        return Quantity(self.energy_value, self.energy_unit)

