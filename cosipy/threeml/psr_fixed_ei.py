import copy
from typing import Optional, Iterable

import numpy as np
from astromodels import PointSource
from astropy.coordinates import UnitSphericalRepresentation, CartesianRepresentation
from astropy.units import Quantity
from executing import Source
from histpy import Axis

from cosipy import SpacecraftHistory
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventInSCFrame
from cosipy.interfaces import UnbinnedThreeMLSourceResponseInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldInstrumentResponseFunctionInterface
from cosipy.response.photon_types import PhotonWithDirectionAndEnergyInSCFrame

from astropy import units as u

class UnbinnedThreeMLPointSourceResponseTrapz(UnbinnedThreeMLSourceResponseInterface):

    def __init__(self,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 irf:FarFieldInstrumentResponseFunctionInterface,
                 sc_history: SpacecraftHistory,
                 energies:Quantity):
        """
        Will integrate the spectrum by evaluation the IRF at fixed Ei position and using a simple
        trapezoidal rule

        All IRF queries are cached

        Parameters
        ----------
        irf
        energies: evaluation points
        """

        # Interface inputs
        self._source = None

        # Other implementation inputs
        self._data = data
        self._irf = irf
        self._energies_keV = energies.to_value(u.keV)
        self._sc_ori = sc_history

        # This can be computed once and for all
        # Trapezoidal rule weights to integrate in Ei
        ewidths = np.diff(self._energies_keV)
        self._trapz_weights = np.zeros_like(self._energies_keV)
        self._trapz_weights[:-1] = ewidths
        self._trapz_weights[1:] = ewidths
        self._trapz_weights /= 2

        self._attitude_at_event_times = self._sc_ori.interp_attitude(self._data.time)

        # Caches

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_source_dict = None

        # The IRF values change for each direction, but it's the same for all spectrum parameters

        # Source location cached separately since changing the response
        # for a given direction is expensive
        self._last_convolved_source_skycoord = None

        # For integral for nevents
        # int Aeff(t, Ei) F(Ei) dt dEi
        # Will need to multiply by F(Ei) and sum.
        # It includes the trapezoidal rule weights
        # and the time integral based on source position
        # and SC history
        # Once per Ei
        self._nevents_weights = None # In cm2*s*keV

        # axis 0: events
        # axis 1: energy_i samples
        self._event_prob_weights = None

        # Integrated over Ei
        self._nevents = None
        self._event_prob = None

    def set_source(self, source: Source):
        """
        The source is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """
        if not isinstance(source, PointSource):
            raise TypeError("I only know how to handle point sources!")

        self._source = source

    def clear_cache(self):

        self._last_convolved_source_dict = None
        self._last_convolved_source_skycoord = None
        self._nevents = None
        self._nevents_weights = None
        self._event_prob = None
        self._event_prob_weights = None

    def copy(self) -> "ThreeMLSourceResponseInterface":
        """
        This method is used to re-use the same object for multiple
        sources.
        It is expected to return a copy of itself, but deepcopying
        any necessary information such that when
        a new source is set, the expectation calculation
        are independent.

        psr1 = ThreeMLSourceResponse()
        psr2 = psr.copy()
        psr1.set_source(source1)
        psr2.set_source(source2)
        """

        new = copy.copy(self)
        new.clear_cache()
        return new

    def _update_cache(self):
        """
        Performs all calculation as needed depending on the current source location

        Returns
        -------
        """
        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        source_dict = self._source.to_dict()
        coord = self._source.position.sky_coord

        if (self._nevents is not None) and (self._event_prob is not None) and self._last_convolved_source_dict == source_dict:
            # Nothing has changed
            return

        if (self._nevents_weights is None) or (self._event_prob_weights is None) or coord != self._last_convolved_source_skycoord:
            # Updating the location is very cost intensive. Only do if necessary

            # Compute nevents integral by integrating though the SC history
            # This only computes the weights based on the source location.
            # Once we know the source source spectrum, we can integrate over Ei
            coord_vec = coord.transform_to(self._sc_ori.attitude.frame).cartesian.xyz.value
            sc_coord_vec = self._sc_ori.attitude.rot[:-1].apply(coord_vec)
            sc_coord_sph = UnitSphericalRepresentation.from_cartesian(CartesianRepresentation(*sc_coord_vec.transpose()))

            # For each SC timestamp, get the effective area for each energy point, store it as temporary array,
            # and multiply by livetime.
            # Sum up the exposure (one per energy point) without saving it to memory
            exposure = sum([dt*np.fromiter(self._irf.effective_area_cm2([PhotonWithDirectionAndEnergyInSCFrame(c.lon.rad, c.lat.rad, e)
                                                                         for e in self._energies_keV]), dtype = float)
                            for c,dt in zip(sc_coord_sph,self._sc_ori.livetime.to_value(u.s))])

            self._nevents_weights = exposure * self._trapz_weights

            # Get the probability for each event for the source location and each Ei
            sc_coord_vec = self._attitude_at_event_times.rot[:-1].apply(coord_vec)
            sc_coord_sph = UnitSphericalRepresentation.from_cartesian(CartesianRepresentation(*sc_coord_vec.transpose()))
            self._event_prob_weights = np.fromiter(self._irf.event_probability([(PhotonWithDirectionAndEnergyInSCFrame(coord.lon.rad, coord.lat.rad, energy), event)
                                                                                for coord,event in zip(sc_coord_sph, self._data) \
                                                                                for energy in self._energies_keV]),
                                                   dtype = float)

            self._event_prob_weights = self._event_prob_weights.reshape((sc_coord_sph.size, self._energies_keV.size))

        # 3ML default units as cm, s and keV
        flux_values = self._source(self._energies_keV)
        self._nevents = np.sum(self._nevents_weights * flux_values)
        self._event_prob = np.sum((self._event_prob_weights * flux_values[None, :]), axis=1)

        self._last_convolved_source_dict = source_dict
        self._last_convolved_source_skycoord = coord.copy()

    @property
    def ncounts(self) -> float:
        """
        Total expected counts
        """

        self._update_cache()

        return self._nevents


    def expectation_density(self, start:Optional[int] = None, stop:Optional[int] = None) -> Iterable[float]:
        """
        Return the expected number of counts density from the start-th event
        to the stop-th event.

        Parameters
        ----------
        start : None | int
            From beginning by default
        stop: None|int
            Until the end by default
        """

        self._update_cache()

        return self._event_prob[start:stop]
