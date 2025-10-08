#!/usr/bin/env python
# coding: utf-8

import logging

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

    bkg_data_path = data_path / "Total_BG_3months_binned_data_filtered_with_SAAcut_SAAreducedHEPD01_DC3binning.hdf5"
    fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/crab_spectral_fit_galactic_frame/bkg_binned_data.hdf5',
                    output=str(bkg_data_path), checksum = '54221d8556eb4ef520ef61da8083e7f4')

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
    data = TimeTagEmCDSEventDataInSCFrameFromDC3Fits(crab_data_path, bkg_data_path,
                                                     selection=selector)
    logger.info("Loading data DONE")

    # Prepare point source response, which convolved the IRF with the SC orientation
    psr = UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc_orientation, dr.axes['Ei'].centers)

    # Prepare the model
    l = 184.56
    b = -5.78

    index = -1.99
    piv = 500. * u.keV
    K = 0.048977e-3 / u.cm / u.cm / u.s / u.keV

    spectrum = Powerlaw()

    spectrum.index.min_value = -3
    spectrum.index.max_value = -1

    spectrum.index.value = index
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
    response = UnbinnedThreeMLModelFolding(data, psr)

    # response.set_model(model) # optional. Will be called by likelihood
    # print(response.ncounts())
    # print(np.fromiter(response.expectation_density(), dtype = float))

    # Set background
    bkg = BinnedData(data_path / "background.yaml")
    bkg.load_binned_data_from_hdf5(binned_data=bkg_data_path)


    like_fun = UnbinnedLikelihood(response, bkg)

    cosi = ThreeMLPluginInterface('cosi', like_fun)

    plugins = DataList(cosi) # If we had multiple instruments, we would do e.g. DataList(cosi, lat, hawc, ...)

    like = JointLikelihood(model, plugins, verbose = False)

    like.fit()

    profile.disable()
    profile.dump_stats("prof_interfaces.prof")

    return


if __name__ == "__main__":

    main()