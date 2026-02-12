import sys

from cosipy import test_data, BinnedData
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.data_io import EmCDSBinnedData
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, FullDetectorResponse, BinnedInstrumentResponse, \
    BinnedThreeMLPointSourceResponse
from cosipy.spacecraftfile import SpacecraftHistory
import astropy.units as u
import numpy as np
from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter
from astropy.coordinates import SkyCoord

from cosipy.statistics import PoissonLikelihood

data_path = test_data.path

sc_orientation = SpacecraftHistory.open(data_path / "20280301_2s.ori")
dr_path = str(data_path / "test_full_detector_response.h5") # path to detector response

crab = BinnedData(data_path / "test_spectral_fit.yaml")
bkg_dist = BinnedData(data_path / "test_spectral_fit.yaml")

crab.load_binned_data_from_hdf5(binned_data=data_path / "test_spectral_fit_data.h5")
bkg_dist.load_binned_data_from_hdf5(binned_data=data_path / "test_spectral_fit_background.h5")

bkg_par = Parameter("background_cosi",                                         # background parameter
                    1,                                                         # initial value of parameter
                    min_value=0,                                               # minimum value of parameter
                    max_value=50,                                              # maximum value of parameter
                    delta=0.05,                                                # initial step used by fitting engine
                    desc="Background parameter for cosi")

l = 50
b = -45

alpha = -1
beta = -2
xp = 500. * u.keV
piv = 500. * u.keV
K = 1 / u.cm / u.cm / u.s / u.keV

spectrum = Band()

spectrum.alpha.value = alpha
spectrum.beta.value = beta
spectrum.xp.value = xp.value
spectrum.K.value = K.value
spectrum.piv.value = piv.value

spectrum.xp.unit = xp.unit
spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit

source = PointSource("source",                     # Name of source (arbitrary, but needs to be unique)
                     l = l,                        # Longitude (deg)
                     b = b,                        # Latitude (deg)
                     spectral_shape = spectrum)    # Spectral model

model = Model(source)

def test_point_source_spectral_fit(background=None):

    dr = FullDetectorResponse.open(dr_path)
    instrument_response = BinnedInstrumentResponse(dr)

    # Workaround to avoid inf values. Out bkg should be smooth, but currently it's not.
    # Reproduces results before refactoring. It's not _exactly_ the same, since this fudge value was 1e-12, and
    # it was added to the expectation, not the normalized bkg
    bkg_dist_proj = bkg_dist.binned_data.project('Em', 'Phi', 'PsiChi')
    bkg_dist_proj += sys.float_info.min



    data = EmCDSBinnedData(crab.binned_data.project('Em', 'Phi', 'PsiChi') + bkg_dist_proj)
    bkg = FreeNormBinnedBackground(bkg_dist_proj,
                                   sc_history=sc_orientation,
                                   copy=False)

    psr = BinnedThreeMLPointSourceResponse(data=data,
                                           instrument_response=instrument_response,
                                           sc_history=sc_orientation,
                                           energy_axis=dr.axes['Ei'],
                                           polarization_axis=dr.axes['Pol'] if 'Pol' in dr.axes.labels else None,
                                           nside=2 * data.axes['PsiChi'].nside)

    response = BinnedThreeMLModelFolding(data=data, point_source_response=psr)

    like_fun = PoissonLikelihood(data, response, bkg)

    cosi = ThreeMLPluginInterface('cosi',
                                  like_fun,
                                  response,
                                  bkg)

    plugins = DataList(cosi)

    like = JointLikelihood(model, plugins, verbose = False)

    like.fit(compute_covariance = False) # avoid sampling-related threeML crashes

    sp = source.spectrum.main.Band

    assert np.allclose([sp.K.value, sp.alpha.value, sp.beta.value, sp.xp.value, bkg_par.value],
                       [1.0522695866399103, 2.6276132958523926, -2.909795888815157, 18.19702619330248, 2.3908438191547012],
                       atol=[0.1, 0.1, 0.1, 1.0, 0.1])

    assert np.allclose([cosi.get_log_like()],
                       [213.14242014103897],
                       atol=[1.0])
