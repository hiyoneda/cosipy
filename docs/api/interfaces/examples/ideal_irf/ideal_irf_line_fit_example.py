import cProfile
import itertools
import time
from typing import Iterator, Union, Iterable

import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Quantity
from cosipy.interfaces import EventInterface, ExpectationDensityInterface
from cosipy.interfaces.data_interface import EventDataInSCFrameInterface, EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EmCDSEventInSCFrameInterface

from astropy import units as u
from cosipy.interfaces.instrument_response_interface import InstrumentResponseFunctionInterface, \
    FarFieldInstrumentResponseFunctionInterface
from cosipy.polarization import PolarizationConvention, PolarizationAngle, StereographicConvention
from cosipy.response.ideal_response import IdealComptonIRF, UnpolarizedIdealComptonIRF, MeasuredEnergyDist, \
    ThresholdKleinNishinaPolarScatteringAngleDist
from cosipy.response.photon_types import PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention, \
    PhotonWithDirectionAndEnergyInSCFrame
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.statistics import UnbinnedLikelihood
from histpy import Histogram, Axis, HealpixAxis
from matplotlib import pyplot as plt
from scipy.stats import poisson
from scoords import SpacecraftFrame


class RandomEventDataFromLineInSCFrame(EmCDSEventDataInSCFrameInterface):

    def __init__(self,
                 irf:FarFieldInstrumentResponseFunctionInterface,
                 flux:Quantity,
                 duration:Quantity,
                 energy:Quantity,
                 direction:SkyCoord,
                 polarized_irf:FarFieldInstrumentResponseFunctionInterface,
                 polarization_degree:float = None,
                 polarization_angle:Union[Angle, Quantity] = None,
                 polarization_convention:PolarizationConvention = None):
        """

        Parameters
        ----------
        irf: Must handle PhotonWithDirectionAndEnergyInSCFrameInterface
        flux: Source flux in unit of 1/area/time
        duration: Integration time
        energy: Source energy (a line)
        direction: Source direction (in SC coordinates)
        polarized_irf: Must handle PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface
        polarization_degree
        polarization_angle
        polarization_convention
        """

        unpolarized_irf = irf

        self.event_type = unpolarized_irf.event_type

        flux_cm2_s = flux.to_value(1/u.cm/u.cm/u.s)
        duration_s = duration.to_value(u.s)

        energy_keV = energy.to_value(u.keV)
        direction = direction.transform_to('spacecraftframe')
        source_direction_lon_rad = direction.lon.rad
        source_direction_lat_rad = direction.lat.rad

        unpolarized_photon = PhotonWithDirectionAndEnergyInSCFrame(source_direction_lon_rad,
                                                                         source_direction_lat_rad,
                                                                         energy_keV)

        unpolarized_expected_counts = next(iter(irf.effective_area_cm2([unpolarized_photon]))) * flux_cm2_s * duration_s

        if polarization_degree is None:
            polarization_degree = 0

        if polarization_degree < 0 or polarization_degree > 1:
            raise ValueError(f"polarization_degree must lie between 0 and 1. Got {polarization_degree}")

        if polarization_degree == 0:
            polarized_irf = None
            polarized_expected_counts = 0

        else:

            polarized_irf = polarized_irf

            if polarized_irf.event_type is not unpolarized_irf.event_type:
                raise TypeError(f"Both IRF need to have the same event type. Got {unpolarized_irf.event_type} and {polarized_irf.event_type}")

            polarization_angle_rad = PolarizationAngle(polarization_angle, direction, polarization_convention).transform_to('stereographic').angle.rad

            polarized_photon =  PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention(source_direction_lon_rad,
                                                                                                            source_direction_lat_rad,
                                                                                                            energy_keV,
                                                                                                            polarization_angle_rad)

            unpolarized_expected_counts *= (1 - polarization_degree)
            polarized_expected_counts = polarization_degree * next(iter(polarized_irf.effective_area_cm2([polarized_photon]))) * flux_cm2_s * duration_s

        unpolarized_counts = poisson(unpolarized_expected_counts).rvs()
        polarized_counts = poisson(polarized_expected_counts).rvs()

        self._events = []

        unpolarized_events = iter(unpolarized_irf.random_events(itertools.repeat(unpolarized_photon, unpolarized_counts)))

        polarized_events = None
        if polarized_counts > 0:
            polarized_events = iter(polarized_irf.random_events(itertools.repeat(polarized_photon, polarized_counts)))

        nthrown_unpolarized = 0
        nthrown_polarized = 0

        while nthrown_unpolarized < unpolarized_counts or nthrown_polarized < polarized_counts:

            if np.random.uniform() < polarization_degree:
                # Polarized component
                if nthrown_polarized < polarized_counts:
                    self._events.append(next(polarized_events))
                    nthrown_polarized += 1
            else:
                # Unpolarized component
                if nthrown_unpolarized < unpolarized_counts:
                    self._events.append(next(unpolarized_events))
                    nthrown_unpolarized += 1

    def __iter__(self) -> Iterator[EmCDSEventInSCFrameInterface]:
        """
        Return one Event at a time
        """
        yield from self._events


irf_pol = IdealComptonIRF.cosi_like()
irf_unpol = UnpolarizedIdealComptonIRF.cosi_like()

ei = 500*u.keV
source_direction = SkyCoord(lon = 0, lat = 80, unit = 'deg', frame = SpacecraftFrame())
flux0 = 1/u.cm/u.cm/u.s
duration = 10*u.s
pol_degree = 0
pol_angle = 80*u.deg
pol_convention = StereographicConvention()

profile = cProfile.Profile()
profile.enable()
tstart = time.perf_counter()
data = RandomEventDataFromLineInSCFrame(irf = irf_unpol,
                                        flux = flux0,
                                        duration = duration,
                                        energy=ei,
                                        direction =  source_direction,
                                        polarized_irf= irf_pol,
                                        polarization_degree=pol_degree,
                                        polarization_angle=pol_angle,
                                        polarization_convention=pol_convention)

# energy = []
# phi = []
# psi = [] # latitude
# chi = [] # longitude
#
# for event in data:
#
#     energy.append(event.energy_keV)
#     phi.append(event.scattering_angle_rad)
#     psi.append(event.scattered_lat_rad_sc)
#     chi.append(event.scattered_lon_rad_sc)
#
# print(time.perf_counter() - tstart)
# profile.disable()
# profile.dump_stats("/Users/imartin5/tmp/prof_gen.prof")
#
# print(data.nevents)
#
# energy = Quantity(energy, u.keV)
# phi = Quantity(phi, u.rad)
# psichi = SkyCoord(chi, psi, unit = u.rad, frame = 'spacecraftframe')
#
# binned_data = Histogram([Axis(np.geomspace(10,10000,50)*u.keV, scale = 'log', label='Em'),
#                          Axis(np.linspace(0, 180, 180)*u.deg, scale='linear', label='Phi'),
#                          HealpixAxis(nside = 64, label = 'PsiChi', coordsys='spacecraftframe')])
#
# binned_data.fill(energy, phi, psichi)
#
# rel_binned_data = Histogram([Axis(np.linspace(-1,1.1,200), scale = 'linear', label='eps'),
#                              Axis(np.linspace(0, 180, 180)*u.deg, scale='linear', label='phi'),
#                              Axis(np.linspace(-180, 180, 180)*u.deg, scale='linear', label='arm'),
#                              Axis(np.linspace(-180, 180, 180) * u.deg, scale='linear', label='az')])
#
# eps = ((energy - ei)/ei).to_value('')
# phi_geom,az = RelativeCDSCoordinates(source_direction, pol_convention).to_relative(psichi)
# arm = phi_geom - phi
#
# rel_binned_data.fill(eps, phi, arm, az)

#binned_data.project('PsiChi').plot()
#plt.show()

class ExpectationFromLineInSCFrame(ExpectationDensityInterface):

    def __init__(self,
                 data:EmCDSEventDataInSCFrameInterface,
                 irf:FarFieldInstrumentResponseFunctionInterface,
                 flux:Quantity,
                 duration:Quantity,
                 energy:Quantity,
                 direction:SkyCoord,
                 polarized_irf:FarFieldInstrumentResponseFunctionInterface,
                 polarization_degree:float = None,
                 polarization_angle:Union[Angle, Quantity] = None,
                 polarization_convention:PolarizationConvention = None):

        self._unpolarized_irf = irf

        self.set_flux(flux)
        self._duration_s = duration.to_value(u.s)
        self._data = data
        direction = direction.transform_to('spacecraftframe')
        self._source_direction_lon_rad = direction.lon.rad
        self._source_direction_lat_rad = direction.lat.rad

        self._polarization_degree = polarization_degree

        if self._polarization_degree is None:
            self._polarization_degree = 0

        if self._polarization_degree < 0 or self._polarization_degree > 1:
            raise ValueError(f"polarization_degree must lie between 0 and 1. Got {self._polarization_degree}")

        if self._polarization_degree == 0:
            self._polarized_irf = None
            self._polarization_angle_rad = None
            self._polarization_convention = None
        else:

            self._polarized_irf = polarized_irf

            self._polarization_angle_rad = PolarizationAngle(polarization_angle, direction,
                                                             polarization_convention).transform_to('stereographic').angle.rad


        # Build the Photon query as well
        self.set_energy(energy)

        # Cache
        self._cached_energy_keV = None
        self._cached_diff_aeff = None # Per flux unit
        self._cached_event_probability = None

    def set_flux(self, flux:Quantity):
        self._flux_cm2_s = flux.to_value(1 / u.cm / u.cm / u.s)

    def set_energy(self, energy:Quantity):
        self._energy_keV = energy.to_value(u.keV)
        self._unpolarized_photon = PhotonWithDirectionAndEnergyInSCFrame(self._source_direction_lon_rad,
                                                                         self._source_direction_lat_rad,
                                                                         self._energy_keV)

        self._polarized_photon = PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention(
                                            self._source_direction_lon_rad,
                                            self._source_direction_lat_rad,
                                            self._energy_keV,
                                            self._polarization_angle_rad)

    def _update_cache(self):

        if self._cached_energy_keV is None or self._energy_keV != self._cached_energy_keV:
            #Either it's the first time or the energy changed

            flux_dur = self._flux_cm2_s * self._duration_s

            unpolarized_diff_aeff = (1 - self._polarization_degree) * next(
                iter(self._unpolarized_irf.effective_area_cm2([self._unpolarized_photon])))

            if self._polarization_degree > 0:

                polarized_diff_aeff = self._polarization_degree * next(iter(self._polarized_irf.effective_area_cm2([self._polarized_photon])))

                self._cached_diff_aeff = unpolarized_diff_aeff + polarized_diff_aeff

                data1, data2 = itertools.tee(self._data, 2)

                unpol_frac = 1 - self._polarization_degree

                self._cached_event_probability = [unpol_frac * unpol_prob + self._polarization_degree * pol_prob \
                                                  for unpol_prob, pol_prob in \
                                                  zip(self._unpolarized_irf.event_probability([(self._unpolarized_photon,e) for e in data1]), self._polarized_irf.event_probability([(self._polarized_photon,e) for e in data2]))]

            else:

                self._cached_diff_aeff = unpolarized_diff_aeff

                self._cached_event_probability = np.fromiter(self._unpolarized_irf.event_probability([(self._unpolarized_photon, e) for e in self._data]), dtype=float)

            self._cached_energy_keV = self._energy_keV

    def expected_counts(self) -> float:

        self._update_cache()

        return self._cached_diff_aeff * (self._flux_cm2_s * self._duration_s)

    def event_probability(self) -> Iterable[float]:

        self._update_cache()

        yield from self._cached_event_probability

expectation = ExpectationFromLineInSCFrame(data,
                                           irf=irf_unpol,
                                           flux=flux0,
                                           duration=duration,
                                           energy=ei,
                                           direction=source_direction,
                                           polarized_irf=irf_pol,
                                           polarization_degree=pol_degree,
                                           polarization_angle=pol_angle,
                                           polarization_convention=pol_convention)


# Check density
# weighted_rel_binned_data = Histogram(rel_binned_data.axes)
# weighted_rel_binned_data.fill(eps, phi, arm, az, weight = list(expectation.expectation_density()))
#
# phase_space = RelativeCDSCoordinates.get_relative_cds_phase_space(rel_binned_data.axes['phi'].lower_bounds[:,None,None], rel_binned_data.axes['phi'].upper_bounds[:,None,None],
#                                                                   rel_binned_data.axes['arm'].lower_bounds[None,:,None], rel_binned_data.axes['arm'].upper_bounds[None,:,None],
#                                                                   rel_binned_data.axes['az'].lower_bounds[None,None,:], rel_binned_data.axes['az'].upper_bounds[None,None,:])
#
# mean_rel_binned_data = weighted_rel_binned_data * phase_space[None] / rel_binned_data
# mean_rel_binned_data[np.isnan(mean_rel_binned_data.contents)] = 0
#
# rel_binned_data.project('eps').plot()
# mean_rel_binned_data.project('eps').plot()
#
# plt.show()


likelihood = UnbinnedLikelihood(expectation)

loglike = Histogram([Axis(np.linspace(.5, 1.5, 11)/u.cm/u.cm/u.s, label = 'flux'),
                        Axis(np.linspace(498, 502, 10)*u.keV, label = 'Ei')])

profile = cProfile.Profile()
profile.enable()
tstart = time.perf_counter()
for j, ei_j in enumerate(loglike.axes['Ei'].centers):
    print(j)
    expectation.set_energy(ei_j)
    for i,flux_i in enumerate(loglike.axes['flux'].centers):

        expectation.set_flux(flux_i)

        loglike[i,j] = likelihood.get_log_like()


print(time.perf_counter() - tstart)
profile.disable()
profile.dump_stats("/Users/imartin5/tmp/prof_eval.prof")

(loglike - np.max(loglike)).plot(vmin = -25)

plt.show()