from typing import Protocol, runtime_checkable

from astropy import units as u
from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from scoords import SpacecraftFrame

from cosipy.polarization import PolarizationConvention, PolarizationAngle, StereographicConvention


@runtime_checkable
class PhotonInterface(Protocol):
    """
    Derived classes have all access methods
    """

@runtime_checkable
class PhotonWithEnergyInterface(PhotonInterface, Protocol):

    @property
    def energy_keV(self) -> float:...

    @property
    def energy(self) -> u.Quantity:
        """
        Add fancy energy quantity
        """
        return u.Quantity(self.energy_keV, u.keV)

@runtime_checkable
class PhotonWithDirectionInterface(PhotonInterface, Protocol):

    @property
    def frame(self) -> BaseCoordinateFrame:...

    @property
    def direction_lon_radians(self) -> float: ...

    @property
    def direction_lat_radians(self) -> float: ...

    @property
    def direction(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.direction_lon_radians,
                        self.direction_lat_radians,
                        unit=u.rad,
                        frame=self.frame)

@runtime_checkable
class PhotonInSCFrameInterface(PhotonInterface, Protocol):

    @property
    def frame(self) -> SpacecraftFrame:...

@runtime_checkable
class PhotonWithDirectionInSCFrameInterface(PhotonWithDirectionInterface,
                                            PhotonInSCFrameInterface, Protocol):
    pass

@runtime_checkable
class PhotonWithDirectionAndEnergyInSCFrameInterface(PhotonWithDirectionInSCFrameInterface,
                                                     PhotonWithEnergyInterface, Protocol):
    pass

@runtime_checkable
class PolarizedPhotonInterface(Protocol):

    def polarization_angle_rad(self) -> float: ...

    def polarization_convention(self) -> PolarizationConvention:...

    def polarization_angle(self) -> PolarizationAngle:
        """
        This convenience function only makes sense for implementations
        that couple with PhotonWithDirectionInterface
        """
        raise NotImplementedError("This class does not implement the polarization_angle() convenience method.")

@runtime_checkable
class PolarizedPhotonStereographicConventionInSCInterface(PolarizedPhotonInterface, PhotonInSCFrameInterface, Protocol):

    def polarization_convention(self) -> PolarizationConvention:
        return StereographicConvention()

@runtime_checkable
class PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface(PhotonWithDirectionAndEnergyInSCFrameInterface, PolarizedPhotonStereographicConventionInSCInterface, Protocol):

    def polarization_angle(self) -> PolarizationAngle:
        return PolarizationAngle(self._pa * u.rad, self.direction, 'stereographic')

