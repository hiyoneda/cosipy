import logging
from pathlib import Path
from typing import Union

from mhealpy import HealpixBase

from cosipy.data_io import EmCDSBinnedData
from cosipy.interfaces.instrument_response_interface import BinnedInstrumentResponseInterface
from cosipy.polarization.polarization_axis import PolarizationAxis
from cosipy.threeml.util import to_linear_polarization

logger = logging.getLogger(__name__)

import copy

from astromodels.sources import Source, PointSource
from scoords import SpacecraftFrame
from histpy import Axes, Histogram, Axis, HealpixAxis
from cosipy.interfaces import BinnedThreeMLSourceResponseInterface, BinnedDataInterface

from cosipy.response import FullDetectorResponse, PointSourceResponse
from cosipy.spacecraftfile import SpacecraftHistory, SpacecraftAttitudeMap

from mhealpy import HealpixMap

__all__ = ["BinnedThreeMLPointSourceResponse"]

class BinnedThreeMLPointSourceResponse(BinnedThreeMLSourceResponseInterface):
    """
    COSI 3ML plugin.

    Parameters
    ----------
    dr:
        Full detector response handle, or the file path
    sc_history:
        Contains the information of the orientation: timestamps (astropy.Time) and attitudes (scoord.Attitude) that describe
        the spacecraft for the duration of the data included in the analysis
    """

    def __init__(self,
                 instrument_response: BinnedInstrumentResponseInterface,
                 sc_history: SpacecraftHistory,
                 direction_axis: HealpixAxis,
                 energy_axis:Axis,
                 polarization_axis:PolarizationAxis = None,
                 ):

        # TODO: FullDetectorResponse -> BinnedInstrumentResponseInterface

        self._sc_ori = sc_history

        # Use setters for these
        self._source = None

        # Prevent unnecessary calculations and new memory allocations

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_source_dict = None

        self._expectation = None

        # The PSR change for each direction, but it's the same for all spectrum parameters

        # Source location cached separately since changing the response
        # for a given direction is expensive
        self._last_convolved_source_skycoord = None

        self._psr = None

        self._response = instrument_response
        self._direction_axis = direction_axis
        self._energy_axis = energy_axis
        self._polarization_axis = polarization_axis

    @property
    def coordsys(self):
        return self._direction_axis.coordsys

    def clear_cache(self):

        self._source = None
        self._last_convolved_source_dict = None
        self._expectation = None
        self._last_convolved_source_skycoord = None
        self._psr = None

    def copy(self) -> "BinnedThreeMlPointSourceResponse":
        """
        Safe copy to use for multiple sources
        Returns
        -------
        A copy than can be used safely to convolve another source
        """
        new = copy.copy(self)
        new.clear_cache()
        return new

    def set_source(self, source: Source):
        """
        The source is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """
        if not isinstance(source, PointSource):
            raise TypeError("I only know how to handle point sources!")

        if (to_linear_polarization(source.spectrum.main.polarization) is not None and
                self._polarization_axis is None):
            raise RuntimeError("This response can't handle a polarized source.")

        self._source = source

    def expectation(self, data:BinnedDataInterface, copy = True)-> Histogram:
        # TODO: check coordsys from axis
        # TODO: Earth occ always true in this case

        if not isinstance(data, EmCDSBinnedData):
            raise TypeError(f"Wrong data type '{type(data)}', expected {EmCDSBinnedData}.")

        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        if self._sc_ori is None:
            raise RuntimeError("Call set_spacecraft_history() first.")

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        source_dict = self._source.to_dict()

        coord = self._source.position.sky_coord

        # Use cached expectation if nothing has changed
        if self._expectation is not None and self._last_convolved_source_dict == source_dict:
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation

        # Expectation calculation

        # Check if the source position change, since these operations
        # are expensive
        if self._psr is None or coord != self._last_convolved_source_skycoord:

            coordsys = data.axes["PsiChi"].coordsys

            if coordsys != self.coordsys:
                raise ValueError(f"Coordinate system mismatch. Data has {coordsys} while this class has {self.coordsys}.")

            logger.info("... Calculating point source response ...")

            if isinstance(self.coordsys, SpacecraftFrame):
                # Local coordinates

                dwell_time_map = self._sc_ori.get_dwell_map(coord, base=self._direction_axis)

                self._psr = PointSourceResponse.from_dwell_time_map(data.axes, self._response,
                                                                    dwell_time_map, self._energy_axis,
                                                                    self._polarization_axis)

            else:
                # Inertial e/.g. galactic
                raise NotImplementedError("Only local coordinates are supported for now.")

                # WiP
                # TODO: Move these lines to inertial version.
                scatt_map = self._sc_ori.get_scatt_map(nside=self._direction_axis.nside * 2,
                                                       target_coord=coord,
                                                       coordsys=self._direction_axis.coordsys,
                                                       earth_occ=True)

            logger.info(f"--> done (source name : {self._source.name})")

        # Convolve with spectrum
        self._expectation = self._psr.get_expectation(self._source.spectrum.main.shape,
                                                      self._source.spectrum.main.polarization)

        # Check if axes match
        if data.axes != self._expectation.axes:
            raise ValueError(
                "Currently, the expectation axes must exactly match the detector response measurement axes")

        # Cache. Use dict and copy since the internal variables can change
        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_source_dict = source_dict
        self._last_convolved_source_skycoord = coord.copy()

        # Copy to prevent user to modify our cache
        if copy:
            return self._expectation.copy()
        else:
            return self._expectation
