from astropy.coordinates import SkyCoord
from scoords import SpacecraftFrame

from cosipy.interfaces.photon_parameters import PhotonWithDirectionAndEnergyInSCFrameInterface, \
    PolarizedPhotonStereographicConventionInSCInterface, \
    PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface, PhotonWithEnergyInterface
from cosipy.polarization import PolarizationAngle

from astropy import units as u

class PhotonWithEnergy(PhotonWithEnergyInterface):

    def __init__(self, energy_keV):
        self._energy = energy_keV

    @property
    def energy_keV(self) -> float:
        return self._energy

class PhotonWithDirectionAndEnergyInSCFrame(PhotonWithEnergy, PhotonWithDirectionAndEnergyInSCFrameInterface):

    frame = SpacecraftFrame()

    def __init__(self, direction_lon_radians, direction_lat_radians, energy_keV):

        super().__init__(energy_keV)

        self._lon = direction_lon_radians
        self._lat = direction_lat_radians

    @property
    def direction_lon_radians(self) -> float:
        return self._lon

    @property
    def direction_lat_radians(self) -> float:
        return self._lat

class PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention(PhotonWithDirectionAndEnergyInSCFrame, PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface):

    def __init__(self, direction_lon_radians, direction_lat_radians, energy_keV, polarization_angle_radians):

        super().__init__(direction_lon_radians, direction_lat_radians, energy_keV)

        self._pa = polarization_angle_radians

    @property
    def polarization_angle_rad(self) -> float:
        return self._pa


