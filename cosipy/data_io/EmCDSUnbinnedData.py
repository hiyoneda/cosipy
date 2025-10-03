from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
from astropy.coordinates import BaseCoordinateFrame, Angle, SkyCoord, UnitSphericalRepresentation
from astropy.time import Time
from astropy.units import Quantity
from numpy._typing import ArrayLike
from scoords import SpacecraftFrame

from cosipy import UnBinnedData
from cosipy.interfaces import EventWithEnergyInterface, EventDataInterface, EventDataWithEnergyInterface
from cosipy.interfaces.data_interface import ComptonDataSpaceEventDataInterface, TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import ComptonDataSpaceEventInterface, TimeTagEmCDSEventInSCFrameInterface

import astropy.units as u

from cosipy.interfaces.event_selection import EventSelectorInterface


class TimeTagEmCDSEventInSCFrame(TimeTagEmCDSEventInSCFrameInterface):

    _frame = SpacecraftFrame()

    def __init__(self, id, jd1, jd2, energy, phi, psi, chi):
        """
        Parameters
        ----------
        jd1: julian days
        jd2: julian days
        energy: keV
        phi: scattering angle radians
        psi: scattering latitude radians
        chi: scattering longitude radians
        """
        self._id = id
        self._jd1 = jd1
        self._jd2 = jd2
        self._energy = energy
        self._phi = phi
        self._psi = psi
        self._chi = chi

    @property
    def id(self) -> int:
        return self._id

    @property
    def frame(self):
        return self._frame

    @property
    def jd1(self):
        return self._jd1

    @property
    def jd2(self):
        return self._jd2

    @property
    def energy_keV(self) -> float:
        return self._energy

    @property
    def scattering_angle_rad(self) -> float:
        return self._phi

    @property
    def scattered_lon_rad(self) -> float:
        return self._chi

    @property
    def scattered_lat_rad(self) -> float:
        return self._psi

class TimeTagEmCDSEventDataInSCFrameFromArrays(TimeTagEmCDSEventDataInSCFrameInterface):
    """

    """

    _frame = SpacecraftFrame()
    event_type = TimeTagEmCDSEventInSCFrame

    def __init__(self,
                 time:Time,
                 energy:Quantity,
                 scattering_angle:Angle,
                 scattered_direction:SkyCoord,
                 event_id:Optional[Iterable[int]] = None,
                 selection:EventSelectorInterface = None):
        """

        Parameters
        ----------
        time
        energy: keV
        scattering_angle: scattering angle radians
        psi: scattering latitude radians
        chi: scattering longitude radians
        id: range(size) by default
        selection: Optional selection for TimeTagEmCDSEventInSCFrame events
        """
        self._jd1 = time.jd1
        self._jd2 = time.jd2
        self._energy = energy.to_value(u.keV)
        self._phi = scattering_angle.to_value(u.rad)

        if not isinstance(scattered_direction.frame, SpacecraftFrame):
            raise ValueError("Coordinates need to be in SC frame")

        scattered_direction = scattered_direction.represent_as(UnitSphericalRepresentation)

        self._psi = scattered_direction.lat.rad
        self._chi = scattered_direction.lon.rad
        if event_id is None:
            self._id = np.arange(self._jd1.size)
        else:
            self._id = np.asarray(event_id)

        # Check size
        self._id, self._jd1, self._jd2, self._energy, self._phi, self._psi, self._chi = np.broadcast_arrays(self._id, self._jd1, self._jd2, self._energy, self._phi, self._psi, self._chi)

        self._nevents = self._id.size

        if selection is not None:
            # Apply selection once and for all
            new_id = []
            new_jd1 = []
            new_jd2 = []
            new_energy = []
            new_phi = []
            new_psi = []
            new_chi = []

            nevents = 0
            for event in selection(self):
                new_id.append(event.id)
                new_jd1.append(event.jd1)
                new_jd2.append(event.jd2)
                new_energy.append(event.energy_keV)
                new_phi.append(event.scattering_angle_rad)
                new_psi.append(event.scattered_lat_rad)
                new_chi.append(event.scattered_lon_rad)
                nevents +=  1

            self._nevents = nevents

            self._id = np.asarray(new_id)
            self._jd1 = np.asarray(new_jd1)
            self._jd2 = np.asarray(new_jd2)
            self._energy = np.asarray(new_energy)
            self._phi = np.asarray(new_phi)
            self._psi = np.pi/2 - np.asarray(new_psi) #Psi is colatitude
            self._chi = np.asarray(new_chi)

    def __getitem__(self, i: int) -> TimeTagEmCDSEventInSCFrameInterface:
        return TimeTagEmCDSEventInSCFrame(self._id[i], self._jd1[i], self._jd2[i], self._energy[i], self._phi[i], self._psi[i], self._chi[i])

    @property
    def nevents(self) -> int:
        return self._nevents

    def __iter__(self) -> Iterator[TimeTagEmCDSEventInSCFrameInterface]:
        for id, jd1, jd2, energy, phi, psi, chi in zip(self._id, self._jd1, self._jd2, self._energy, self._phi, self._psi, self._chi):
            yield TimeTagEmCDSEventInSCFrame(id, jd1, jd2, energy, phi, psi, chi)

    @property
    def frame(self) -> SpacecraftFrame:
        return self._frame

    @property
    def ids(self) -> Iterable[int]:
        return self._id

    @property
    def jd1(self) -> Iterable[float]:
        return self._jd1

    @property
    def jd2(self) -> Iterable[float]:
        return self._jd2

    @property
    def energy_rad(self) -> Iterable[float]:
        return self._energy

    @property
    def scattering_angle_rad(self) -> Iterable[float]:
        return self._phi

    @property
    def scattered_lon_rad(self) -> Iterable[float]:
        return self._chi

    @property
    def scattered_lat_rad(self) -> Iterable[float]:
        return self._phi

class TimeTagEmCDSEventDataInSCFrameFromDC3Fits(TimeTagEmCDSEventDataInSCFrameFromArrays):

    def __init__(self, *data_path: Tuple[Path],
                 selection:EventSelectorInterface = None):

        time = np.empty(0)
        energy = np.empty(0)
        phi = np.empty(0)
        psi = np.empty(0)
        chi = np.empty(0)

        for file in data_path:
            # get_dict_from_fits is really a static method, no config file needed
            data_dict = UnBinnedData.get_dict_from_fits(None, file)

            time = np.append(time, data_dict['TimeTags'])
            energy = np.append(energy, data_dict['Energies'])
            phi = np.append(phi, data_dict['Phi'])
            psi = np.append(psi, data_dict['Psi local'])
            chi = np.append(psi, data_dict['Chi local'])

        # Time sort
        tsort = np.argsort(time)

        time = time[tsort]
        energy = energy[tsort]
        phi = phi[tsort]
        psi = psi[tsort]
        chi = chi[tsort]

        time = Time(time, format='unix')
        energy = u.Quantity(energy, u.keV)
        phi = Angle(phi, u.rad)
        psichi = SkyCoord(chi, np.pi / 2 - psi, unit=u.rad,
                          frame=SpacecraftFrame())

        super().__init__(time, energy, phi, psichi, selection = selection)




