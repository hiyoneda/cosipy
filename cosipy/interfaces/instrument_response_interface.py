from typing import Protocol, Union, Optional, Iterable, Tuple, runtime_checkable

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from histpy import Axes, Histogram

from astropy import units as u
from scoords import Attitude

from cosipy.interfaces import BinnedDataInterface, ExpectationDensityInterface, BinnedExpectationInterface, EventInterface
from cosipy.interfaces.photon_list import PhotonListWithDirectionInterface
from cosipy.interfaces.photon_parameters import PhotonInterface, PhotonWithDirectionInterface
from cosipy.polarization import PolarizationAngle

__all__ = ["BinnedInstrumentResponseInterface"]

class BinnedInstrumentResponseInterface(BinnedExpectationInterface, Protocol):

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

@runtime_checkable
class InstrumentResponseFunctionInterface(Protocol):

    def event_probability(self, query: Iterable[Tuple[PhotonInterface, EventInterface]]) -> Iterable[float]:
        """
        Return the probability density of measuring a given event given a photon.
        """

    def random_events(self, photons:Iterable[PhotonInterface]) -> Iterable[EventInterface]:
        """
        Return a stream of random events, one per photon
        """

@runtime_checkable
class FarFieldInstrumentResponseFunctionInterface(InstrumentResponseFunctionInterface, Protocol):

    def effective_area_cm2(self, photons: Iterable[PhotonWithDirectionInterface]) -> Iterable[float]:
        """

        """

    def effective_area(self, photons: Iterable[PhotonWithDirectionInterface]) -> Iterable[u.Quantity]:
        """
        Convenience function
        """
        for area_cm2 in self.effective_area_cm2(photons):
            yield u.Quantity(area_cm2, u.cm*u.cm)





















