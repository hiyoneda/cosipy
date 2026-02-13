from cosipy import COSILike, test_data, BinnedData
from cosipy.response import FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftFile
import astropy.units as u
import numpy as np
from threeML import Powerlaw, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter
from astropy.coordinates import SkyCoord

data_path = test_data.path

sc_orientation = SpacecraftFile.parse_from_file(data_path / "20280301_2s.ori")
dr = str(data_path / "test_full_detector_response.h5") # path to detector response

bkg_par_value = 1
bkg_par = Parameter("background_cosi",                                         # background parameter
                    bkg_par_value,                                                         # initial value of parameter
                    min_value=0,                                               # minimum value of parameter
                    max_value=50,                                              # maximum value of parameter
                    delta=0.05,                                                # initial step used by fitting engine
                    desc="Background parameter for cosi")

l = 50
b = -45

index = -2
piv = 1. * u.MeV
K = 1 / u.cm / u.cm / u.s / u.keV

spectrum = Powerlaw()

spectrum.index.value = index
spectrum.K.value = K.value
spectrum.piv.value = piv.value

spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit

source = PointSource("source",                     # Name of source (arbitrary, but needs to be unique)
                     l = l,                        # Longitude (deg)
                     b = b,                        # Latitude (deg)
                     spectral_shape = spectrum)    # Spectral model

model = Model(source)

def test_point_source_spectral_fit():

    # Create fake data and background using the same
    # response as the fit (for a circular test)
    fdr = FullDetectorResponse.open(dr)
    psr = fdr.get_point_source_response(coord=source.position.sky_coord,
                                        scatt_map=sc_orientation.get_scatt_map(fdr.nside * 2,  # Use same nside as hardcoded in COSILke
                                                                               source.position.sky_coord))

    data = psr.get_expectation(source.spectrum.main.shape)
    bkg = data.copy()
    bkg[:] = np.mean(bkg) # Flat background
    data += bkg_par.value * bkg

    # Set plugin
    cosi = COSILike("cosi",                                                        # COSI 3ML plugin
                    dr = dr,                                                       # detector response
                    data = data,        # data (source+background)
                    bkg = bkg,   # background model
                    sc_orientation = sc_orientation,                               # spacecraft orientation
                    nuisance_param = bkg_par)                                      # background parameter

    plugins = DataList(cosi)

    like = JointLikelihood(model, plugins, verbose = False)

    like.fit(compute_covariance = False) # avoid sampling-related threeML crashes

    sp = source.spectrum.main.shape

    assert np.allclose([sp.K.value, sp.index.value, bkg_par.value],
                       [K.value, index, bkg_par_value])

    assert np.allclose([cosi.get_log_like()],
                       [6377269.127606418])
