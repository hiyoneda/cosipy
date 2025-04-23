import copy

from astromodels.sources import Source, PointSource
from astropy.coordinates import SkyCoord
from histpy import Axes, Histogram
from cosipy.interfaces import BinnedThreeMLSourceResponseInterface

from cosipy.response import FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftFile, SpacecraftAttitudeMap

__name__ = []

from mhealpy import HealpixMap


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
                 sc_orientation: SpacecraftFile,
                 ):

        # TODO: FullDetectorResponse -> BinnedInstrumentResponseInterface

        self._dr = dr
        self._sc_orientation = sc_orientation

        self._init_cache()

    def _init_cache(self):

        # Prevent unnecessary calculations and new memory allocations
        self._expectation = None
        self._scatt_map = None

        self._source = None

        # TODO: currently Model.__eq__ seems broken. It returns True even
        #  if the internal parameters changed. Currently, caching only work
        #  for the source position, but everything related to spectral and
        #  polarization is recalculated even if it's still the same
        self._last_convolved_source = None

    def copy(self) -> "BinnedThreeMlPointSourceResponse":
        """
        Safe copy to use for multiple sources
        Returns
        -------
        A copy than can be used safely to convolve another source
        """
        new = copy.copy(self)
        new._init_cache()
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

    def expectation(self, axes:Axes)-> Histogram:
        # TODO: check coordsys from axis
        # TODO: Earth occ always true in this case

        # Check if we can use these axes
        if 'PsiChi' not in axes.labels:
            raise ValueError("PsiChi axes not present")

        if axes["PsiChi"].coordsys is None:
            raise ValueError("PsiChi axes doesn't have a coordinate system")

        # Check what we can use from the cache
        if self._expectation is None or self._expectation.axes != axes:
            # Needs new memory allocation, and recompute everything
            self._expectation = Histogram(axes)
        else:
            # If nothing has changed in the source, we can use the cached expectation
            # as is.
            # If the source has changed but the axes haven't, we can at least reuse
            # it and prevent new memory allocation, we just need to zero it out

            # TODO: currently Source.__eq__ seems broken. It returns True even
            #  if some of the internal parameters changed. Caching the expected
            #  value is not implemented. Remove the "False and" when fixed
            #  Getting the source position explicitly does seem to work though
            if False and (self._last_convolved_source == self._source):
                return self._expectation
            else:
                self._expectation.clear()

        # Expectation calculation

        # Check if the source position change, since these operations
        # are expensive
        coord = self._source.position.sky_coord
        if coord != self._last_convolved_source.position.sky_coord:

            coordsys = axes["PsiChi"].coordsys

            if coordsys == 'spacecraftframe':
                dwell_time_map = self._get_dwell_time_map(coord)
                self._psr[name] = self._dr.get_point_source_response(exposure_map=dwell_time_map)
            elif self._coordsys == 'galactic':
                scatt_map = self._get_scatt_map(coord)
                self._psr[name] = self._dr.get_point_source_response(coord=coord, scatt_map=scatt_map)
            else:
                raise RuntimeError("Unknown coordinate system")

        return self._expectation

        coord = self._source.position.sky_coord


        if self._last_convolved_source.position != :


        self._last_convolved_source = copy.deepcopy(self._source)

    def _get_scatt_map(self, coord:SkyCoord)->SpacecraftAttitudeMap:
        """
        Get the spacecraft attitude map of the source.

        Since we're accounting for Earth occultation, this is specific
        to this coordinate

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            The coordinates of the target object.

        Returns
        -------
        scatt_map : SpacecraftAttitudeMap
        """

        scatt_map = self._sc_orientation.get_scatt_map(nside=self._dr.nside * 2, target_coord=coord,
                                                       coordsys='galactic', earth_occ = True)

        return scatt_map

    def _get_dwell_time_map(self, coord: SkyCoord) -> HealpixMap:
        """
        Get the dwell time map of the source.

        This is always specific to a coordinate.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Coordinates of the target source

        Returns
        -------
        dwell_time_map : mhealpy.containers.healpix_map.HealpixMap
            Dwell time map
        """

        self._sc_orientation.get_target_in_sc_frame(target_name=self._name, target_coord=coord)
        dwell_time_map = self._sc_orientation.get_dwell_map(response=self._rsp_path)

        return dwell_time_map



