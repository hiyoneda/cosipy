import itertools
from abc import abstractmethod
from typing import Protocol, runtime_checkable, Dict, Type, Any, Tuple, Iterator, Union, Sequence, Iterable

import numpy as np
from astropy.units import Unit

from . import EventWithEnergy
from .event import Event, FancyEnergyDataMixin, FancyTimeDataMixin, TimeTagEvent
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
]

class DataInterface:

    @property
    def event_type(self) -> Type[Event]:
        """
        Type returned by __iter__ in the event data case
        """

class BinnedDataInterface(DataInterface):
    @property
    def data(self) -> Histogram:...
    @property
    def axes(self) -> Axes:...


class EventDataInterface(DataInterface, Iterable):

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
    @abstractmethod
    def nevents(self) -> int:
        """
        After selection
        """

    def __len__(self):
        return self.nevents

    def set_selection(self, selection: Union["EventSelectorInterface", None]) -> None:
        """
        None would drop the selection. Implementation might not implement the ability to change or drop
        a selection --e.g. the underlying data was discarded for efficiency reasons.
        """

    @property
    def selection(self) -> Union["EventSelectorInterface", None]:
        """
        The current selection set
        """

    def get_binned_data(self, axes:Axes, *args, **kwargs) -> BinnedDataInterface:
        raise NotImplementedError

class TimeTagEventData(FancyTimeDataMixin, EventDataInterface):

    def __iter__(self) -> Iterator[TimeTagEvent]:...

    @property
    @abstractmethod
    def jd1(self) -> Iterable[float]: ...

    @property
    @abstractmethod
    def jd2(self) -> Iterable[float]: ...

class EventDataWithEnergy(FancyEnergyDataMixin, EventDataInterface):

    def __iter__(self) -> Iterator[EventWithEnergy]:...

    @property
    @abstractmethod
    def energy_value(self) -> Iterable[float]:...

    @property
    @abstractmethod
    def energy_unit(self) -> Unit:...

