import itertools
from abc import abstractmethod
from typing import Protocol, runtime_checkable, Dict, Type, Any, Tuple, Iterator, Union, Sequence, Iterable

import numpy as np
from astropy.units import Unit

from . import EventWithEnergy
from .event import Event, FancyEnergyDataMixin, FancyTimeDataMixin, TimetaggedEvent
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
    def tstart(self) -> Union[Time, None]:
        """
        Start time of data taking
        """
        return None

    @property
    def tstop(self) -> Union[Time, None]:
        """
        Start time of data taking
        """
        return None

    @property
    def event_type(self) -> Type[Event]:
        """
        Type returned by __getitem__
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

    def get_binned_data(self, *args, **kwargs) -> BinnedDataInterface:
        raise NotImplementedError

class TimeTagEventDataInterface(FancyTimeDataMixin, EventDataInterface):

    def __getitem__(self, item: int) -> TimetaggedEvent:...

    @property
    @abstractmethod
    def jd1(self) -> Iterable[float]: ...

    @property
    @abstractmethod
    def jd2(self) -> Iterable[float]: ...

class EventDataWithEnergyInterface(FancyEnergyDataMixin, EventDataInterface):

    def __getitem__(self, item: int) -> EventWithEnergy:...

    @property
    @abstractmethod
    def energy_value(self) -> Iterable[float]:...

    @property
    @abstractmethod
    def energy_unit(self) -> Unit:...

