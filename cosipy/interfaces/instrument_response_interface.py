from typing import Protocol

from astropy.coordinates import SkyCoord
from histpy import Axes, Histogram

from astropy import units as u

from cosipy.polarization import PolarizationAngle

__all__ = ["BinnedInstrumentResponseInterface"]

class BinnedInstrumentResponseInterface(Protocol):

    def differential_effective_area(self, axes:Axes, direction: SkyCoord, energy:u.Quantity, polarization:PolarizationAngle) -> Histogram:
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
