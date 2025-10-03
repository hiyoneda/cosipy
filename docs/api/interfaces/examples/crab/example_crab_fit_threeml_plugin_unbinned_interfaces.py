#!/usr/bin/env python
# coding: utf-8




from cosipy import test_data, BinnedData, UnBinnedData
from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays, \
    TimeTagEmCDSEventDataInSCFrameFromDC3Fits
from cosipy.event_selection.time_selection import TimeSelector
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.util import fetch_wasabi_file

from cosipy.statistics import PoissonLikelihood
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
from astromodels import Parameter

from pathlib import Path

import os

def main():

    # Download all data
    data_path = Path("")  # /path/to/files. Current dir by default

    crab_data_path = data_path / "crab_standard_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
    fetch_wasabi_file('COSI-SMEX/DC3/Data/Sources/crab_standard_3months_unbinned_data_filtered_with_SAAcut.fits.gz',
                      output=str(crab_data_path), checksum='1d73e7b9e46e51215738075e91a52632')

    bkg_data_path = data_path / "AlbedoPhotons_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
    fetch_wasabi_file('COSI-SMEX/DC3/Data/Backgrounds/Ge/AlbedoPhotons_3months_unbinned_data_filtered_with_SAAcut.fits.gz',
                      output=str(bkg_data_path), checksum='191a451ee597fd2e4b1cf237fc72e6e2')

    selector = TimeSelector(tstart = Time("2028-03-01 01:35:00.117"), tstop = Time("2028-03-03 01:35:00.117")) #About 3 days

    data = TimeTagEmCDSEventDataInSCFrameFromDC3Fits(crab_data_path, bkg_data_path,
                                                     selection=selector)

    return


if __name__ == "__main__":

    import cProfile
    cProfile.run('main()', filename = "prof_interfaces.prof")
    exit()

    main()