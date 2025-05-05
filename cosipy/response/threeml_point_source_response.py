import logging
logger = logging.getLogger(__name__)

import copy

from astromodels.sources import Source, PointSource
from scoords import SpacecraftFrame
from histpy import Axes, Histogram
from cosipy.interfaces import BinnedThreeMLSourceResponseInterface

from cosipy.response import FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftHistory, SpacecraftAttitudeMap

from mhealpy import HealpixMap

__all__ = ["BinnedThreeMlPointSourceResponse"]

class BinnedThreeMlPointSourceResponse(BinnedThreeMLSourceResponseInterface):
    """
    COSI 3ML plugin.

    Parameters
    ----------
    dr:
        Full detector response handle (**not** the file path)
    sc_orientation:
        Contains the information of the orientation: timestamps (astropy.Time) and attitudes (scoord.Attitude) that describe
        the spacecraft for the duration of the data included in the analysis
    """

    def __init__(self,
                 dr: FullDetectorResponse,
                 sc_orientation: SpacecraftHistory,
                 ):

        # TODO: FullDetectorResponse -> BinnedInstrumentResponseInterface

        self._dr = dr
        self._sc_ori = sc_orientation

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

        self._source = source

    def expectation(self, axes:Axes, copy = True)-> Histogram:
        # TODO: check coordsys from axis
        # TODO: Earth occ always true in this case

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        source_dict = self._source.to_dict()

        coord = self._source.position.sky_coord

        # Check if we can use these axes
        if 'PsiChi' not in axes.labels:
            raise ValueError("PsiChi axes not present")

        if axes["PsiChi"].coordsys is None:
            raise ValueError("PsiChi axes doesn't have a coordinate system")

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

            coordsys = axes["PsiChi"].coordsys

            logger.info("... Calculating point source response ...")

            if isinstance(coordsys, SpacecraftFrame):
                dwell_time_map = self._sc_ori.get_dwell_map(coord, base = self._dr)
                self._psr = self._dr.get_point_source_response(exposure_map=dwell_time_map)
            else:
                scatt_map = self._sc_ori.get_scatt_map(nside=self._dr.nside * 2,
                                                       target_coord=coord,
                                                       coordsys=coordsys,
                                                       earth_occ = True)
                self._psr = self._dr.get_point_source_response(coord=coord, scatt_map=scatt_map)

            logger.info(f"--> done (source name : {self._source.name})")



        # Convolve with spectrum
        self._expectation = self._psr.get_expectation(self._source.spectrum.main.shape,
                                                      self._source.spectrum.main.polarization)

        # Check if axes match
        if axes != self._expectation.axes:
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


