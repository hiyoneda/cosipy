from abc import ABC, abstractmethod
from typing import Sequence, Union, Protocol

import numpy as np
from astropy.coordinates import Angle, SkyCoord, BaseCoordinateFrame
from scoords import SpacecraftFrame
from typing_extensions import runtime_checkable

from astropy.time import Time
from astropy.units import Quantity, Unit
import astropy.units as u

__all__ = [
    "EventInterface",
    "TimeTagEventInterface",
    "EventWithEnergyInterface",
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
class EventInterface(Protocol):
    """
    Derived classes implement all accessors
    """

    @property
    def id(self) -> int:
        """
        Typically set by the main data loader or source.

        No necessarily in sequential order
        """

    @property
    def metadata(self) -> EventMetadata:...

@runtime_checkable
class TimeTagEventInterface(EventInterface, Protocol):

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
class EventWithEnergyInterface(EventInterface, Protocol):

    @property
    def energy_keV(self) -> float:...

    @property
    def energy(self) -> Quantity:
        """
        Add fancy energy quantity
        """
        return Quantity(self.energy_keV, u.keV)

@runtime_checkable
class ComptonDataSpaceEventInterface(EventInterface, Protocol):

    @property
    def frame(self) -> BaseCoordinateFrame:...

    @property
    def scattering_angle_rad(self) -> float: ...

    @property
    def scattering_angle(self) -> Angle:
        """
        Add fancy energy quantity
        """
        return Angle(self.scattering_angle_rad, u.rad)

    @property
    def scattered_lon_rad(self) -> float: ...

    @property
    def scattered_lat_radians(self) -> float: ...

    @property
    def scattered_direction(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.scattered_lon_rad,
                        np.pi/2 - self.scattered_lat_radians,
                        unit=u.rad,
                        frame=self.frame)


@runtime_checkable
class EventInSCFrameInterface(EventInterface, Protocol):

    @property
    def frame(self) -> SpacecraftFrame:...

@runtime_checkable
class ComptonDataSpaceInSCFrameEventInterface(EventInSCFrameInterface,
                                              ComptonDataSpaceEventInterface,
                                              Protocol):
    pass

class TimeTagEmCDSEventInSCFrameInterface(TimeTagEventInterface,
                                          EventWithEnergyInterface,
                                          ComptonDataSpaceInSCFrameEventInterface):
    pass





