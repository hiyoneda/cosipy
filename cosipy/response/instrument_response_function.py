import itertools
from typing import Iterable, Tuple

import numpy as np
from astropy.coordinates import SkyCoord

from astropy import units as u
from astropy.units import Quantity

from histpy import Histogram
from scoords import SpacecraftFrame

from cosipy.interfaces import EventInterface
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface, EmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_list import PhotonListWithDirectionInterface
from cosipy.interfaces.photon_parameters import PhotonInterface, PhotonWithDirectionAndEnergyInSCFrameInterface
from cosipy.response import FullDetectorResponse
from cosipy.util.iterables import itertools_batched


class UnpolarizedDC3InterpolatedFarFieldInstrumentResponseFunction(FarFieldInstrumentResponseFunctionInterface):

    def __init__(self, response: FullDetectorResponse,
                 batch_size = 100000):

        self._prob = response.to_dr().project('NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi')

        self._area_eff = self._prob.project('NuLambda', 'Ei')

        # expand_dims removes units
        self._prob /= Quantity(self._prob.axes.expand_dims(self._area_eff, ('NuLambda', 'Ei')), self._area_eff.unit, copy=False)

        self._prob.to('', copy=False)
        self._area_eff = self._area_eff.to(u.cm*u.cm, copy=False)

        self._batch_size = batch_size

    def effective_area_cm2(self, photons: Iterable[PhotonWithDirectionAndEnergyInSCFrameInterface]) -> Iterable[float]:
        """

        """

        for photon_chunk in itertools_batched(photons, self._batch_size):

            lon, lat, energy = np.asarray([[photon.direction_lon_radians,
                                             photon.direction_lat_radians,
                                             photon.energy_keV] for photon in photon_chunk], dtype=float).transpose()

            direction = SkyCoord(lon, lat, unit = u.rad, frame = SpacecraftFrame())
            energy = Quantity(energy, u.keV)

            for area_eff in self._area_eff.interp(direction, energy):
                yield area_eff.value


    def event_probability(self, query: Iterable[Tuple[PhotonWithDirectionAndEnergyInSCFrameInterface, EmCDSEventInSCFrameInterface]]) -> Iterable[float]:
        """
        Return the probability density of measuring a given event given a photon.
        """

        for query_chunk in itertools_batched(query, self._batch_size):

            # Psi is colatitude (complementary angle)
            lon_ph, lat_ph, energy_i, energy_m, phi, psi_comp, chi  = \
                np.asarray([[photon.direction_lon_radians,
                             photon.direction_lat_radians,
                             photon.energy_keV,
                             event.energy_keV,
                             event.scattering_angle_rad,
                             event.scattered_lat_rad,
                             event.scattered_lon_rad,
                            ] for photon,event in query_chunk], dtype=float).transpose()

            direction_ph = SkyCoord(lon_ph, lat_ph, unit = u.rad, frame = SpacecraftFrame())
            energy_i = Quantity(energy_i, u.keV)
            energy_m = Quantity(energy_m, u.keV)
            phi = Quantity(phi, u.rad)
            psichi = SkyCoord(chi, psi_comp, unit=u.rad, frame=SpacecraftFrame())

            # Prob not guaranteed to sum up to 1. We should take self._prob.slice instead.
            # I think this is faster though, and a good approximation.
            for prob in self._prob.interp(direction_ph, energy_i, energy_m, phi, psichi):
                yield prob