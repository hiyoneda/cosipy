import logging

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

import sys

from mhealpy import HealpixBase

from matplotlib import pyplot as plt

from cosipy.statistics import PoissonLikelihood

from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse

from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.response.FullDetectorResponse import FullDetectorResponse

from astropy.time import Time
import astropy.units as u

import numpy as np

from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter

from pathlib import Path

import os

def main():

    # Download data
    data_path = Path("")  # /path/to/files. Current dir by default
    # fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_bkg_binned_data.hdf5', output=str(data_path / 'grb_bkg_binned_data.hdf5'), checksum = 'fce391a4b45624b25552c7d111945f60')
    # fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_binned_data.hdf5', output=str(data_path / 'grb_binned_data.hdf5'), checksum = 'fcf7022369b6fb378d67b780fc4b5db8')
    # fetch_wasabi_file('COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/bkg_binned_data_1s_local.hdf5', output=str(data_path / 'bkg_binned_data_1s_local.hdf5'), checksum = 'b842a7444e6fc1a5dd567b395c36ae7f')
    # fetch_wasabi_file('COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori', output=str(data_path / '20280301_3_month_with_orbital_info.ori'), checksum = '416fcc296fc37a056a069378a2d30cb2')
    # fetch_wasabi_file('COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip', output=str(data_path / 'SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip'), unzip = True, checksum = 'e8ff763c5d9e63d3797567a4a51d9eda')


    case = 'grb'
    case = 'crab'

    if case == 'grb':

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

        source = PointSource("source",  # Name of source (arbitrary, but needs to be unique)
                             l=l,  # Longitude (deg)
                             b=b,  # Latitude (deg)
                             spectral_shape=spectrum)  # Spectral model

        # Date preparation
        binned_data = BinnedData(data_path / "grb.yaml")
        binned_data.load_binned_data_from_hdf5(binned_data=data_path / "grb_bkg_binned_data.hdf5")

        bkg = BinnedData(data_path / "background.yaml")

        bkg.load_binned_data_from_hdf5(binned_data=data_path / "bkg_binned_data_1s_local.hdf5")

        bkg_tmin = 1842597310.0
        bkg_tmax = 1842597550.0
        bkg_min = np.where(bkg.binned_data.axes['Time'].edges.value == bkg_tmin)[0][0]
        bkg_max = np.where(bkg.binned_data.axes['Time'].edges.value == bkg_tmax)[0][0]
        bkg_dist = bkg.binned_data.slice[{'Time': slice(bkg_min, bkg_max)}].project('Em', 'Phi', 'PsiChi')

        tmin = Time(1842597410.0, format='unix')
        tmax = Time(1842597450.0, format='unix')

    elif case == 'crab':

        # Set model to fit
        l = 184.56
        b = -5.78

        alpha = -1.99
        beta = -2.32
        E0 = 531. * (alpha - beta) * u.keV
        xp = E0 * (alpha + 2) / (alpha - beta)
        piv = 500. * u.keV
        K = 3.07e-5 / u.cm / u.cm / u.s / u.keV

        spectrum = Band()

        spectrum.alpha.min_value = -2.14
        spectrum.alpha.max_value = 3.0
        spectrum.beta.min_value = -5.0
        spectrum.beta.max_value = -2.15
        spectrum.xp.min_value = 1.0

        spectrum.alpha.value = alpha
        spectrum.beta.value = beta
        spectrum.xp.value = xp.value
        spectrum.K.value = K.value
        spectrum.piv.value = piv.value

        spectrum.xp.unit = xp.unit
        spectrum.K.unit = K.unit
        spectrum.piv.unit = piv.unit

        spectrum.alpha.delta = 0.01
        spectrum.beta.delta = 0.01

        source = PointSource("source",  # Name of source (arbitrary, but needs to be unique)
                             l=l,  # Longitude (deg)
                             b=b,  # Latitude (deg)
                             spectral_shape=spectrum)  # Spectral model

        # Data preparation
        binned_data = BinnedData(data_path / "crab.yaml")
        bkg = BinnedData(data_path / "background.yaml")

        binned_data.load_binned_data_from_hdf5(binned_data=data_path / "crab_bkg_binned_data.hdf5")
        bkg.load_binned_data_from_hdf5(binned_data=data_path / "bkg_binned_data.hdf5")

        bkg_dist = bkg.binned_data.project('Em', 'Phi', 'PsiChi')

        # SC attitude and orbit
        ori = SpacecraftHistory.open(data_path / "20280301_3_month_with_orbital_info.ori")

    else:
        raise ValueError(r"Unknown case '{case}'")

    # Prepare instrument response
    dr_path = data_path / "SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5"
    dr = FullDetectorResponse.open(dr_path)

    # Workaround to avoid inf values. Out bkg should be smooth, but currently it's not.
    # Reproduces results before refactoring. It's not _exactly_ the same, since this fudge value was 1e-12, and
    # it was added to the expectation, not the normalized bkg
    bkg_dist += sys.float_info.min

    # ============ Interfaces ==============
    data = binned_data.get_em_cds()

    bkg = FreeNormBinnedBackground(bkg_dist)

    instrument_response = BinnedInstrumentResponse(dr)

    # Currently using the same NnuLambda, Ei and Pol axes as the underlying FullDetectorResponse,
    # matching the behavior of v0.3. This is all the current BinnedInstrumentResponse can do.
    # In principle, this can be decoupled, and a BinnedInstrumentResponseInterface implementation
    # can provide the response for an arbitrary directions, Ei and Pol values.
    # NOTE: this is currently only implemented for data in local coords
    psr = BinnedThreeMLPointSourceResponse(instrument_response,
                                               sc_history=ori,
                                               direction_axis = data.axes['PsiChi'],
                                               energy_axis = dr.axes['Ei'],
                                               polarization_axis = dr.axes['Pol'] if 'Pol' in dr.axes.labels else None)

    response = BinnedThreeMLModelFolding(point_source_response = psr)

    like_fun = PoissonLikelihood()
    like_fun.set_data(data)
    like_fun.set_response(response)
    like_fun.set_background(bkg)

    cosi = ThreeMLPluginInterface('cosi', like_fun)

    # Nuisance parameter guess, bounds, etc.
    cosi.bkg_parameter['bkg_norm'] = Parameter("bkg_norm",  # background parameter
                                      0.1,  # initial value of parameter
                                      min_value=0,  # minimum value of parameter
                                      max_value=5,  # maximum value of parameter
                                      delta=1e-3,  # initial step used by fitting engine
                                      )

    # ======== Interfaces end ==========

    # 3Ml fit. Same as before
    plugins = DataList(cosi)
    model = Model(source)  # Model with single source. If we had multiple sources, we would do Model(source1, source2, ...)
    like = JointLikelihood(model, plugins)
    like.fit()
    results = like.results
    print(results.display())


    # plot
    if case == 'crab':

        alpha_inj = -1.99
        beta_inj = -2.32
        E0_inj = 531. * (alpha_inj - beta_inj) * u.keV
        xp_inj = E0_inj * (alpha_inj + 2) / (alpha_inj - beta_inj)
        piv_inj = 100. * u.keV
        K_inj = 7.56e-4 / u.cm / u.cm / u.s / u.keV

        spectrum_inj = Band()

        spectrum_inj.alpha.min_value = -2.14
        spectrum_inj.alpha.max_value = 3.0
        spectrum_inj.beta.min_value = -5.0
        spectrum_inj.beta.max_value = -2.15
        spectrum_inj.xp.min_value = 1.0

        spectrum_inj.alpha.value = alpha_inj
        spectrum_inj.beta.value = beta_inj
        spectrum_inj.xp.value = xp_inj.value
        spectrum_inj.K.value = K_inj.value
        spectrum_inj.piv.value = piv_inj.value

        spectrum_inj.xp.unit = xp_inj.unit
        spectrum_inj.K.unit = K_inj.unit
        spectrum_inj.piv.unit = piv_inj.unit

        results = like.results

        print(results.display())

        parameters = {par.name: results.get_variates(par.path)
                      for par in results.optimized_model["source"].parameters.values()
                      if par.free}

        results_err = results.propagate(results.optimized_model["source"].spectrum.main.shape.evaluate_at, **parameters)

        print(results.optimized_model["source"])

        energy = np.geomspace(100 * u.keV, 10 * u.MeV).to_value(u.keV)

        flux_lo = np.zeros_like(energy)
        flux_median = np.zeros_like(energy)
        flux_hi = np.zeros_like(energy)
        flux_inj = np.zeros_like(energy)

        for i, e in enumerate(energy):
            flux = results_err(e)
            flux_median[i] = flux.median
            flux_lo[i], flux_hi[i] = flux.equal_tail_interval(cl=0.68)
            flux_inj[i] = spectrum_inj.evaluate_at(e)

        binned_energy_edges = binned_data.binned_data.axes['Em'].edges.value
        binned_energy = np.array([])
        bin_sizes = np.array([])

        for i in range(len(binned_energy_edges) - 1):
            binned_energy = np.append(binned_energy, (binned_energy_edges[i + 1] + binned_energy_edges[i]) / 2)
            bin_sizes = np.append(bin_sizes, binned_energy_edges[i + 1] - binned_energy_edges[i])

        fig, ax = plt.subplots()

        ax.plot(energy, energy * energy * flux_median, label="Best fit")
        ax.fill_between(energy, energy * energy * flux_lo, energy * energy * flux_hi, alpha=.5,
                        label="Best fit (errors)")
        ax.plot(energy, energy * energy * flux_inj, color='black', ls=":", label="Injected")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(r"$E^2 \frac{dN}{dE}$ (keV cm$^{-2}$ s$^{-1}$)")

        ax.legend()

        plt.show()

        here

    else:
        raise ValueError(r"Unknown case '{case}'")


if __name__ == "__main__":

    import cProfile
    cProfile.run('main()', filename = "prof.prof")
    exit()

    main()