import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from scoords import SpacecraftFrame

from cosipy.polarization.polarization_stokes import PolarizationStokes
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import UnBinnedData
from cosipy.threeml.custom_functions import Band_Eflux
from cosipy import test_data

analysis = UnBinnedData(test_data.path / 'polarization_data.yaml')
data = analysis.get_dict_from_hdf5(test_data.path / 'polarization_data.hdf5')
response_path = test_data.path / 'test_polarization_response_dense.h5'
sc_orientation = SpacecraftFile.parse_from_file(test_data.path / 'polarization_ori.ori')
attitude = sc_orientation.get_attitude()[0]

a = 10. * u.keV
b = 10000. * u.keV
alpha = -1.
beta = -2.
ebreak = 350. * u.keV 
K = 50. / u.cm / u.cm / u.s
spectrum = Band_Eflux(a = a.value,
                      b = b.value,
                      alpha = alpha,
                      beta = beta,
                      E0 = ebreak.value,
                      K = K.value)
spectrum.a.unit = a.unit
spectrum.b.unit = b.unit
spectrum.E0.unit = ebreak.unit
spectrum.K.unit = K.unit

source_direction = SkyCoord(0, 70, representation_type='spherical', frame=SpacecraftFrame(attitude=attitude), unit=u.deg)

def test_stokes_polarization():

    bin_edges = Angle(np.linspace(-np.pi, np.pi, 10), unit=u.rad)
    source_photons = PolarizationStokes(source_direction, spectrum, data, 
                                        response_path, sc_orientation, background=None, 
                                        response_convention='RelativeZ', asad_bin_edges=bin_edges, show_plots=False)
    
    average_mu = source_photons._mu100['mu']
    mdp99 = source_photons._mdp99

    qs, us = source_photons.compute_data_pseudo_stokes(show_plots=False)
    polarization = source_photons.calculate_polarization(qs, us, average_mu, mdp=mdp99,
                                                         bkg_qs=None, bkg_us=None, show_plots=True)
    Pol_frac = polarization['fraction'] * 100
    Pol_angl = polarization['angle'].angle.degree

    assert np.allclose([average_mu, mdp99, Pol_frac, Pol_angl], [0.22, 0.20, 178, 82], atol=[0.1, 3.0, 5, 10])

#########################################
print('Expected values for polarization:')
print('Fraction:', 13.73038868282377, '%')
print('Fraction uncertainty:', 2.1295224814008353, '%')
print('Angle:', np.degrees(1.4851296518928818), 'degrees')
print('Angle uncertainty:', np.degrees(0.07562763316088744), 'degrees')

import matplotlib.pyplot as plt
chi_gal = data['Chi galactic']
psi_gal = data['Psi galactic']
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot()
ax.set_title('Polarized source')
ax.hist2d(chi_gal, psi_gal, bins=40, cmap='viridis', cmin=1)
ax.set_xlabel('Chi galactic (deg)')
ax.set_ylabel('Psi galactic (deg)')
print(source_direction.galactic.l.deg, source_direction.galactic.b.deg)
ax.scatter(source_direction.galactic.l.deg, source_direction.galactic.b.deg, color='red', label='Source direction')

source_photons = PolarizationStokes(source_direction, spectrum, data, 
                                        response_path, sc_orientation, background=None,
                                        response_convention='RelativeX')

data_duration = source_photons.get_data_duration()
data_counts = source_photons.get_data_counts()
print('\nData duration:', str(round(data_duration, 3)), 's')
print('Data counts:', data_counts)
print('Count rate:', round(data_counts / data_duration, 3), 'counts/s')


average_mu = source_photons._mu100['mu']
print('Average mu100:', average_mu)

mdp99 = source_photons._mdp99
print('MDP99:', mdp99 * 100)

# _bkg_qs_, _bkg_us_ = source_photons.simulate_unpolarized_stokes(n_samples=100, show_plots=True)
# _bkg_qs_, _bkg_us_ = np.load(test_data.path / 'simulated_unpolarized_stokes.npz')['qs'], np.load(test_data.path / 'simulated_unpolarized_stokes.npz')['us']

qs, us = source_photons.compute_data_pseudo_stokes(show_plots=True)

polarization = source_photons.calculate_polarization(qs, us, average_mu, 
                                                     bkg_qs=None, bkg_us=None, show_plots=True, 
                                                     mdp=mdp99)
QN = polarization['QN']
UN = polarization['UN']
QN_ERR = polarization['QN_ERR']
UN_ERR = polarization['UN_ERR']
print('Normalized Q: %.3f +/- %.3f'%(QN, QN_ERR))
print('Normalized U: %.3f +/- %.3f'%(UN, UN_ERR))
