import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from cosipy.interfaces.instrument_response_interface import BinnedInstrumentResponseInterface

from cosipy.polarization import PolarizationAngle
from cosipy.response import FullDetectorResponse

from histpy import Axes, Histogram


__all__ = ["BinnedInstrumentResponse"]

class BinnedInstrumentResponse(BinnedInstrumentResponseInterface):

    def __init__(self, response:FullDetectorResponse):

        self._dr = response

    def differential_effective_area(self, axes:Axes, direction: SkyCoord, energy:u.Quantity, polarization:PolarizationAngle = None) -> Histogram:
        """

        Parameters
        ----------
        axes:
            Measured axes
        direction:
            Photon incoming direction in SC coordinates
        energy:
            Photon energy
        polarization
            Photon polarization angle

        Returns
        -------
        The effective area times the event measurement probability distribution integrated on each of the bins
        of the provided axes
        """

        # Check if we can use these axes

        if self._dr.measurement_axes != axes:
            raise ValueError("This implementation can only handle a fixed set of measurement axes equal to the underlying response file.")

        if 'PsiChi' not in axes.labels:
            raise ValueError("PsiChi axis not present")

        if axes["PsiChi"].coordsys is None:
            raise ValueError("PsiChi axes doesn't have a coordinate system")

        if polarization is not None:
            if 'Pol' not in self._dr.axes.labels:
                raise RuntimeError("The FullDetectorResponse does not contain polarization information")
            elif not np.array_equal(polarization, self._dr.axes['Pol'].centers):
                # Matches the v0.3 behaviour
                raise RuntimeError(
                    "Currently, the probed polarization angles need to match the underlying response matrix centers.")

        if not np.array_equal(energy, self._dr.axes['Ei'].centers):
            # Matches the v0.3 behaviour
            raise RuntimeError("Currently, the probed energy values need to match the underlying response matrix centers.")

        # Get the pixel as is since we already checked that the requested
        # energy and polarization points match the underlying response centers
        # Matches the v0.3 behaviour
        pix = self._dr.ang2pix(direction)

        return self._dr[pix]


