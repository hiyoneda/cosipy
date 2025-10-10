#!/usr/bin/env python
# coding: utf-8

import logging

from histpy import Histogram

from cosipy.background_estimation.free_norm_threeml_binned_bkg import FreeNormBackgroundInterpolatedDensityTimeTagEmCDS
from cosipy.threeml.unbinned_model_folding import UnbinnedThreeMLModelFolding

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import cProfile

from cosipy import test_data, BinnedData, UnBinnedData
from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays, \
    TimeTagEmCDSEventDataInSCFrameFromDC3Fits, TimeTagEmCDSEventInSCFrame
from cosipy.event_selection.time_selection import TimeSelector
from cosipy.interfaces.photon_parameters import PhotonWithDirectionAndEnergyInSCFrameInterface
from cosipy.response.instrument_response_function import UnpolarizedDC3InterpolatedFarFieldInstrumentResponseFunction
from cosipy.response.photon_types import PhotonWithDirectionAndEnergyInSCFrame
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.threeml.psr_fixed_ei import UnbinnedThreeMLPointSourceResponseTrapz
from cosipy.util import fetch_wasabi_file

from cosipy.statistics import PoissonLikelihood, UnbinnedLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse

import sys

from scoords import SpacecraftFrame

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic, Angle

import numpy as np
import matplotlib.pyplot as plt

from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter, Powerlaw

from pathlib import Path

import os

def main():

    use_bkg = False

    profile = cProfile.Profile()

    # Download all data
    data_path = Path("")  # /path/to/files. Current dir by default

    crab_data_path = data_path / "crab_standard_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
    fetch_wasabi_file('COSI-SMEX/DC3/Data/Sources/crab_standard_3months_unbinned_data_filtered_with_SAAcut.fits.gz',
                      output=str(crab_data_path), checksum='1d73e7b9e46e51215738075e91a52632')

    bkg_data_path = data_path / "AlbedoPhotons_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
    fetch_wasabi_file('COSI-SMEX/DC3/Data/Backgrounds/Ge/AlbedoPhotons_3months_unbinned_data_filtered_with_SAAcut.fits.gz',
                      output=str(bkg_data_path), checksum='191a451ee597fd2e4b1cf237fc72e6e2')

    dr_path = data_path / "SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5"  # path to detector response
    fetch_wasabi_file(
        'COSI-SMEX/develop/Data/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5',
        output=str(dr_path),
        checksum='eb72400a1279325e9404110f909c7785')

    sc_orientation_path = data_path / "20280301_3_month_with_orbital_info.ori"
    fetch_wasabi_file('COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori',
                      output=str(sc_orientation_path), checksum='416fcc296fc37a056a069378a2d30cb2')

    binned_bkg_data_path = data_path / "bkg_binned_data.hdf5"
    fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/crab_spectral_fit_galactic_frame/bkg_binned_data.hdf5',
                    output=str(binned_bkg_data_path), checksum = '54221d8556eb4ef520ef61da8083e7f4')

    profile.enable()
    # orientation history
    tstart = Time("2028-03-01 01:35:00.117")
    tstop = Time("2028-03-01 02:35:00.117")
    sc_orientation = SpacecraftHistory.open(sc_orientation_path)
    sc_orientation = sc_orientation.select_interval(tstart, tstop)

    # Prepare instrument response function
    logger.info("Loading response....")
    dr = FullDetectorResponse.open(dr_path)
    irf = UnpolarizedDC3InterpolatedFarFieldInstrumentResponseFunction(dr)
    logger.info("Loading response DONE")

    # Prepare data
    selector = TimeSelector(tstart = sc_orientation.tstart, tstop = sc_orientation.tstop)

    logger.info("Loading data...")
    if use_bkg:
        data = TimeTagEmCDSEventDataInSCFrameFromDC3Fits([crab_data_path, bkg_data_path],
                                                      selection=selector)
    else:
        data = TimeTagEmCDSEventDataInSCFrameFromDC3Fits(crab_data_path,
                                                         selection=selector)

    logger.info("Loading data DONE")

    # Set background

    if use_bkg:
        bkg = BinnedData(data_path / "background.yaml")
        bkg.load_binned_data_from_hdf5(binned_data=str(binned_bkg_data_path))
        bkg_dist = bkg.binned_data.project('Em', 'Phi', 'PsiChi')

        # Workaround to avoid inf values. Our bkg should be smooth, but currently it's not.
        bkg_dist += sys.float_info.min

        logger.info("Setting bkg...")
        bkg = FreeNormBackgroundInterpolatedDensityTimeTagEmCDS(data, bkg_dist, sc_orientation, copy = False)
        logger.info("Setting bkg DONE")
    else:
        bkg = None

    # Prepare point source response, which convolved the IRF with the SC orientation
    psr = UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc_orientation, dr.axes['Ei'].centers)

    # Prepare the model
    l = 184.56
    b = -5.78

    index = -1.99
    piv = 1 * u.MeV
    K = 0.048977e-3 / u.cm / u.cm / u.s / u.keV

    spectrum = Powerlaw()

    spectrum.index.min_value = -3
    spectrum.index.max_value = -1

    # Fix it for testing purposes
    # spectrum.index.value = -2
    # spectrum.index.free = False

    spectrum.K.value = K.value
    spectrum.piv.value = piv.value

    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    spectrum.index.delta = 0.01

    source = PointSource("source",  # Name of source (arbitrary, but needs to be unique)
                         l=l,  # Longitude (deg)
                         b=b,  # Latitude (deg)
                         spectral_shape=spectrum)  # Spectral model

    model = Model(
        source)  # Model with single source. If we had multiple sources, we would do Model(source1, source2, ...)


    # Set model folding
    response = UnbinnedThreeMLModelFolding(psr)

    # response.set_model(model) # optional. Will be called by likelihood
    # print(response.ncounts())
    # print(np.fromiter(response.expectation_density(), dtype = float))

    # Setup likelihood
    like_fun = UnbinnedLikelihood(response, bkg)

    cosi = ThreeMLPluginInterface('cosi', like_fun, response, bkg)

    # Nuisance parameter guess, bounds, etc.
    if use_bkg:
        cosi.bkg_parameter['bkg_norm'] = Parameter("bkg_norm",  # background parameter
                                          1,  # initial value of parameter
                                          unit = u.Hz,
                                          min_value=0,  # minimum value of parameter
                                          max_value=100,  # maximum value of parameter
                                          delta=0.05,  # initial step used by fitting engine
                                          )

    plugins = DataList(cosi) # If we had multiple instruments, we would do e.g. DataList(cosi, lat, hawc, ...)

    like = JointLikelihood(model, plugins, verbose = False)


    # Grid
    if use_bkg:
        loglike = Histogram([np.geomspace(2e-6, 2e-4, 30),
                                   np.geomspace(.1, 10, 31)], labels=['K', 'B'])

        for i, k in enumerate(loglike.axes['K'].centers):
            for j, b in enumerate(loglike.axes['B'].centers):
                spectrum.K.value = k
                cosi.bkg_parameter['bkg_norm'].value = b

                loglike[i, j] = cosi.get_log_like()

        loglike.plot()
    else:
        loglike = Histogram([np.geomspace(2e-6, 2e-4, 30)], labels=['K'], axis_scale='log')

        for i, k in enumerate(loglike.axes['K'].centers):
            spectrum.K.value = k

            loglike[i] = cosi.get_log_like()

        loglike.plot()

    plt.show()

    # Run
    like.fit()

    profile.disable()
    profile.dump_stats("prof_interfaces.prof")

    return


if __name__ == "__main__":

    main()