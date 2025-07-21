from typing import Protocol, Union

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from histpy import Axes, Histogram

from astropy import units as u
from scoords import Attitude

from cosipy.interfaces import BinnedDataInterface
from cosipy.polarization import PolarizationAngle

__all__ = ["BinnedInstrumentResponseInterface"]

class BinnedInstrumentResponseInterface(Protocol):

    def differential_effective_area(self,
                                    data: BinnedDataInterface,
                                    direction: SkyCoord,
                                    energy:u.Quantity,
                                    polarization:PolarizationAngle,
                                    attitude:Attitude,
                                    weight: Union[Quantity, float],
                                    out: Quantity,
                                    add_inplace: bool) -> Quantity:
        """

        Parameters
        ----------
        data:
            Binned data
        direction:
            Photon incoming direction. If not in a SpacecraftFrame, then provide an attitude for the transformation
        energy:
            Photon energy
        polarization
            Photon polarization angle. If the coordinate frame of the polarization convention is not a
            SpacecraftFrame, then provide an attitude for the transformation
        attitude
            Attitude defining the orientation of the SC in an inertial coordinate system.
        weight
            Optional. Weighting the result by a given weight. Providing the weight at this point as opposed to
            apply it to the output can result in greater efficiency.
        out
            Optional. Histogram to store the output. If possible, the implementation should try to avoid allocating
            new memory.
        add_inplace
            Optional. If True and a Histogram output was provided, the implementation should try to avoid allocating new
            memory and add --not set-- the result of this operation to the output.

        Returns
        -------
        The effective area times the event measurement probability distribution integrated on each of the bins
        of the provided axes. It has the shape (direction.shape, energy.shape, polarization.shape, axes.shape)
        """
