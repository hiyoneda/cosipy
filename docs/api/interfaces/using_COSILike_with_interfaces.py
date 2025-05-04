from cosipy.statistics import PoissonLikelihood
from histpy import Histogram

from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLResponse, BinnedThreeMlPointSourceResponse
from threeML import Band, PointSource, Model, JointLikelihood, DataList
from cosipy.util import fetch_wasabi_file
from cosipy.spacecraftfile import SpacecraftFile
from astropy import units as u

from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftFile
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.util import fetch_wasabi_file

from scoords import SpacecraftFrame

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import poisson_conf_interval

import numpy as np
import matplotlib.pyplot as plt

from threeML import Band, PointSource, Model, JointLikelihood, DataList
from cosipy import Band_Eflux
from astromodels import Parameter

from pathlib import Path

import os

def main():

    # Set model to fit
    l = 93.
    b = -53.

    alpha = -1
    beta = -3
    xp = 450. * u.keV
    piv = 500. * u.keV
    K = 1 / u.cm / u.cm / u.s / u.keV

    spectrum = Band()

    spectrum.beta.min_value = -15.0

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

    model = Model(source)                              # Model with single source. If we had multiple sources, we would do Model(source1, source2, ...)

    # Data preparation
    data_path = Path("") # /path/to/files. Current dir by default
    # fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_bkg_binned_data.hdf5', output=str(data_path / 'grb_bkg_binned_data.hdf5'), checksum = 'fce391a4b45624b25552c7d111945f60')
    # fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_binned_data.hdf5', output=str(data_path / 'grb_binned_data.hdf5'), checksum = 'fcf7022369b6fb378d67b780fc4b5db8')
    # fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/bkg_binned_data_1s_local.hdf5', output=str(data_path / 'bkg_binned_data_1s_local.hdf5'), checksum = 'b842a7444e6fc1a5dd567b395c36ae7f')


    grb = BinnedData(data_path / "grb.yaml")
    grb_bkg = BinnedData(data_path / "grb.yaml")
    bkg = BinnedData(data_path / "background.yaml")

    grb.load_binned_data_from_hdf5(binned_data=data_path / "grb_binned_data.hdf5")
    grb_bkg.load_binned_data_from_hdf5(binned_data=data_path / "grb_bkg_binned_data.hdf5")
    bkg.load_binned_data_from_hdf5(binned_data=data_path / "bkg_binned_data_1s_local.hdf5")

    # Generate interface on the fly. All we need is to implement this method
    # @property
    #     def data(self) -> histpy.Histogram:...

    # We can move this to BinnedData later, but this showed the flexibility of using Protocols over abstract classes
    data_hist = grb_bkg.binned_data.project('Em', 'Phi', 'PsiChi')

    class BinnedDataAux:
        @property
        def data(self) -> Histogram:
            return data_hist

    data = BinnedDataAux()

    bkg_tmin = 1842597310.0
    bkg_tmax = 1842597550.0
    bkg_min = np.where(bkg.binned_data.axes['Time'].edges.value == bkg_tmin)[0][0]
    bkg_max = np.where(bkg.binned_data.axes['Time'].edges.value == bkg_tmax)[0][0]
    bkg = FreeNormBinnedBackground(bkg.binned_data.slice[{'Time':slice(bkg_min,bkg_max)}].project('Em', 'Phi', 'PsiChi'))

    # Response preparation
    # fetch_wasabi_file('COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori', output=str(data_path / '20280301_3_month_with_orbital_info.ori'), checksum = '416fcc296fc37a056a069378a2d30cb2')
    # fetch_wasabi_file('COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip', output=str(data_path / 'SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip'), unzip = True, checksum = 'e8ff763c5d9e63d3797567a4a51d9eda')

    tmin = Time(1842597410.0, format='unix')
    tmax = Time(1842597450.0, format='unix')
    ori = SpacecraftFile.open(data_path / "20280301_3_month_with_orbital_info.ori", tmin, tmax)
    sc_orientation = ori.select_interval(tmin, tmax)

    dr = FullDetectorResponse.open(data_path / "SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5")

    # Options for point sources
    psr = BinnedThreeMlPointSourceResponse(dr, ori)

    # Option for extended sources
    # Not yet implemented
    #esr = BinnedThreeMLExtendedSourceResponse()
    esr = None

    response = BinnedThreeMLResponse(point_source_response = psr,
                                     extended_source_response = esr)



    # Optional: if you want to call get_log_like manually, then you also need to set the model manually
    # 3ML does this internally during the fit though
    cosi = ThreeMLPluginInterface('cosi', PoissonLikelihood(data, response, bkg))
    plugins = DataList(cosi)
    like = JointLikelihood(model, plugins)
    like.fit()

    results = like.results

    print(results.display())

if __name__ == "__main__":

    import cProfile
    cProfile.run('main()', filename = "prof.prof")
    exit()

    main()