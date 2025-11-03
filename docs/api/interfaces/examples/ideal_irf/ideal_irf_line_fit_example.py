import cProfile
import time

import numpy as np
from astropy.coordinates import SkyCoord, Angle

from astropy import units as u
from cosipy.polarization import StereographicConvention
from cosipy.response.ideal_response import IdealComptonIRF, UnpolarizedIdealComptonIRF, ExpectationFromLineInSCFrame, RandomEventDataFromLineInSCFrame
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.statistics import UnbinnedLikelihood
from histpy import Histogram, Axis, HealpixAxis
from matplotlib import pyplot as plt
from scoords import SpacecraftFrame

from mhealpy import HealpixMap

irf_pol = IdealComptonIRF.cosi_like()
irf_unpol = UnpolarizedIdealComptonIRF.cosi_like()

ei = 500*u.keV
source_direction = SkyCoord(lon = 0, lat = 60, unit = 'deg', frame = SpacecraftFrame())
flux0 = 1/u.cm/u.cm/u.s
duration = 10*u.s
pol_degree = .7
pol_angle = 80*u.deg
pol_convention = StereographicConvention()

# profile = cProfile.Profile()
# profile.enable()
# tstart = time.perf_counter()
data = RandomEventDataFromLineInSCFrame(irf = irf_unpol,
                                        flux = flux0,
                                        duration = duration,
                                        energy=ei,
                                        direction =  source_direction,
                                        polarized_irf= irf_pol,
                                        polarization_degree=pol_degree,
                                        polarization_angle=pol_angle,
                                        polarization_convention=pol_convention)

measured_energy = data.energy
phi = data.scattering_angle
psichi = data.scattered_direction_sc

fig,ax = plt.subplots(subplot_kw = {'projection':'mollview'})
sc = ax.scatter(psichi.lon.deg, psichi.lat.deg, transform = ax.get_transform('world'), c = phi*180/np.pi, cmap = 'inferno',
                s = 2, vmin = 0, vmax = 180)
ax.scatter(source_direction.lon.deg, source_direction.lat.deg, transform = ax.get_transform('world'), marker = 'x', s = 100, c = 'red')
fig.colorbar(sc, orientation="horizontal", fraction = .02, label = "$\phi$ [deg]")

eps = ((measured_energy - ei)/ei).to_value('')
phi_geom,az = RelativeCDSCoordinates(source_direction, pol_convention).to_relative(psichi)
theta_arm = phi_geom - phi

rel_binned_data = Histogram([Axis(np.linspace(-1,1.1,200), scale = 'linear', label='eps'),
                             Axis(np.linspace(0, 180, 180)*u.deg, scale='linear', label='phi'),
                             Axis(np.linspace(-180, 180, 180)*u.deg, scale='linear', label='arm'),
                             Axis(np.linspace(-180, 180, 180) * u.deg, scale='linear', label='az')])

rel_binned_data.fill(eps, phi, theta_arm, az)

fig,ax = plt.subplots(2, 3, figsize = [18,8])

rel_binned_data.project('eps').plot(ax[0,0],errorbars = True)
rel_binned_data.slice[{'phi':slice(30,120)}].project('az').rebin(5).plot(ax[1,0],errorbars = True)

rel_binned_data.project(['arm','phi']).rebin(3,5).plot(ax[0,1])
rel_binned_data.project('phi').rebin(5).plot(ax[0,2],errorbars = True)
rel_binned_data.project('arm').rebin(3).plot(ax[1,1],errorbars = True)


plt.show()


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



likelihood = UnbinnedLikelihood(expectation)

# ==== Free the source energy ====
if False:
    # Set everything to the injection values
    expectation.set_model(flux=flux0,
                           energy=ei,
                           direction=source_direction,
                           polarization_degree=pol_degree,
                           polarization_angle=pol_angle)

    loglike = Histogram([Axis(np.linspace(.5, 1.5, 11)/u.cm/u.cm/u.s, label = 'flux'),
                            Axis(np.linspace(498, 502, 10)*u.keV, label = 'Ei')])

    profile = cProfile.Profile()
    profile.enable()
    tstart = time.perf_counter()
    for j, ei_j in enumerate(loglike.axes['Ei'].centers):
        print(j)
        for i,flux_i in enumerate(loglike.axes['flux'].centers):

            expectation.set_model(flux = flux_i, energy = ei_j)

            loglike[i,j] = likelihood.get_log_like()


    print(time.perf_counter() - tstart)
    profile.disable()
    profile.dump_stats("/Users/imartin5/tmp/prof_eval.prof")

    (loglike - np.max(loglike)).plot(vmin = -25)

    plt.show()


# ==== Free the source direction ====
if False:
    # Set everything to the injection values
    expectation.set_model(flux=flux0,
                           energy=ei,
                           direction=source_direction,
                           polarization_degree=pol_degree,
                           polarization_angle=pol_angle)

    loglike = Histogram([Axis(np.linspace(.5, 1.5, 11)/u.cm/u.cm/u.s, label = 'flux'),
                         HealpixAxis(nside = 32, label = 'direction', coordsys=SpacecraftFrame())])

    loglike[:] = np.nan

    profile = cProfile.Profile()
    profile.enable()
    tstart = time.perf_counter()
    sample_pixels = loglike.axes['direction'].query_disc(source_direction.cartesian.xyz, np.deg2rad(10))
    for j, pix in enumerate(sample_pixels):

        print(j,len(sample_pixels))

        coord_pix = loglike.axes['direction'].pix2skycoord(pix)

        for i,flux_i in enumerate(loglike.axes['flux'].centers):

            expectation.set_model(flux = flux_i, direction = coord_pix)

            loglike[i,pix] = likelihood.get_log_like()


    print(time.perf_counter() - tstart)
    profile.disable()
    profile.dump_stats("/Users/imartin5/tmp/prof_eval.prof")

    direction_profile_loglike = HealpixMap(np.nanmax(loglike, axis = 0))
    direction_profile_loglike.plot()

    flux_prof_loglike = Histogram(loglike.axes['flux'], contents = np.nanmax(loglike, axis = 1))
    (flux_prof_loglike - np.nanmin(flux_prof_loglike)).plot()

    plt.show()

# ==== Free PD and PA ====
if True:
    # Set everything to the injection values
    expectation.set_model(flux=flux0,
                          energy=ei,
                          direction=source_direction,
                          polarization_degree=pol_degree,
                          polarization_angle=pol_angle)

    loglike = Histogram([Axis(np.linspace(.5, 1.5, 11) / u.cm / u.cm / u.s, label='flux'),
                         Axis(np.linspace(40,120,10)*u.deg, label='PA'),
                         Axis(np.linspace(0, 1, 11), label='PD'),
                         ])

    for j, pa_j in enumerate(loglike.axes['PA'].centers):
        print(j)
        for k, pd_k in enumerate(loglike.axes['PD'].centers):
            for i,flux_i in enumerate(loglike.axes['flux'].centers):

                expectation.set_model(flux = flux_i,
                                      polarization_degree = pd_k,
                                      polarization_angle = pa_j)

                loglike[i,j,k] = likelihood.get_log_like()

    flux_prof_loglike = Histogram(loglike.axes['flux'], contents = np.nanmax(loglike, axis = (1,2)))
    ts_flux = 2 * (flux_prof_loglike - np.nanmin(flux_prof_loglike))
    ax,_ = ts_flux.plot()
    ax.set_ylabel("TS")

    pol_prof_loglike = Histogram([loglike.axes['PA'], loglike.axes['PD']], contents=np.nanmax(loglike, axis=0))

    ts_pol = 2 * (pol_prof_loglike - np.nanmax(pol_prof_loglike))
    ax, _ = ts_pol.plot(vmin=-4.61)
    ax.scatter(pol_angle.to_value(u.deg), pol_degree, color='red')
    ax.get_figure().axes[-1].set_ylabel("TS")

    plt.show()