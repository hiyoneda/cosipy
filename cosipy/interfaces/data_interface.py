import itertools
from typing import Protocol, runtime_checkable, Dict, Type, Any, Tuple, Iterator, Union, Sequence, Iterable, ClassVar

import numpy as np
from astropy.coordinates import BaseCoordinateFrame, Angle, SkyCoord
from astropy.units import Unit, Quantity
import astropy.units as u
from scoords import SpacecraftFrame

from . import EventWithEnergyInterface
from .event import EventInterface, TimeTagEventInterface, ComptonDataSpaceEventInterface, \
    ComptonDataSpaceInSCFrameEventInterface, TimeTagEmCDSEventInSCFrameInterface
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
           "TimeTagEventDataInterface",
           "EventDataWithEnergyInterface"
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
    def fill(self, event_data:Iterable[EventInterface]):
        """
        Bin the data.

        Parameters
        ----------
        event_data

        Returns
        -------

        """



@runtime_checkable
class EventDataInterface(DataInterface, Protocol):

    def __iter__(self) -> Iterator[EventInterface]:
        """
        Return one Event at a time
        """

    def __getitem__(self, item: int) -> EventInterface:
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

    @property
    def ids(self) -> Iterable[int]:
        return [e.id for e in self]

@runtime_checkable
class TimeTagEventDataInterface(EventDataInterface, Protocol):

    def __iter__(self) -> Iterator[TimeTagEventInterface]:...

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
class EventDataWithEnergyInterface(EventDataInterface, Protocol):

    def __iter__(self) -> Iterator[EventWithEnergyInterface]:...

    @property
    def energy_rad(self) -> Iterable[float]:...

    @property
    def energy(self) -> Quantity:
        """
        Add fancy energy quantity
        """
        return Quantity(self.energy_rad, u.rad)

@runtime_checkable
class ComptonDataSpaceEventDataInterface(EventDataInterface, Protocol):

    def __iter__(self) -> Iterator[ComptonDataSpaceEventInterface]:...

    @property
    def frame(self) -> BaseCoordinateFrame: ...

    @property
    def scattering_angle_rad(self) -> Iterable[float]:...

    @property
    def scattering_angle(self) -> Angle:
        """
        Add fancy energy quantity
        """
        return Angle(self.scattering_angle_rad, u.rad)

    @property
    def scattered_lon_rad(self) -> Iterable[float]: ...

    @property
    def scattered_lat_rad(self) -> Iterable[float]: ...

    @property
    def scattered_direction(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.scattered_lon_rad,
                        np.pi/2 - self.scattered_lat_rad,
                        unit = u.rad,
                        frame = self.frame)

@runtime_checkable
class EventDataInSCFrameInterface(EventDataInterface, Protocol):

    @property
    def frame(self) -> SpacecraftFrame:...

@runtime_checkable
class ComptonDataSpaceInSCFrameEventDataInterface(EventDataInSCFrameInterface,
                                                  ComptonDataSpaceEventDataInterface,
                                                  Protocol):
    def __iter__(self) -> Iterator[ComptonDataSpaceInSCFrameEventInterface]:...

class TimeTagEmCDSEventDataInSCFrameInterface(TimeTagEventDataInterface,
                                              EventDataWithEnergyInterface,
                                               ComptonDataSpaceInSCFrameEventDataInterface):
    def __iter__(self) -> Iterator[TimeTagEmCDSEventInSCFrameInterface]:...
