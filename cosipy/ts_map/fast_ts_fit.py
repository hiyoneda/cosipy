from pathlib import Path

import numpy as np
import numba

import matplotlib.pyplot as plt

from astropy.units import Unit

import healpy as hp
from mhealpy import HealpixBase

from histpy import Histogram, Axis, Axes

from cosipy import SpacecraftFile
from cosipy.response import FullDetectorResponse, PointSourceResponse
from cosipy.response.functions import get_integrated_spectral_model

from .fast_norm_fit import FastNormFit as fnf

import logging
logger = logging.getLogger(__name__)

class GalacticResponse:

    def __init__(self, response_path):
        """
        Load a galactic-frame response from a specified path.  The
        response is stored as a standard Histogram; we selectively
        load just enough information to retrieve slices for individual
        source directions from the file as needed, rather than loading
        the whole response.

        Parameters
        ----------
        response_path : string or Path
          file containing response

        """
        import h5py as h5

        self._file = h5.File(response_path, mode='r')

        axes_group = self._file['hist/axes']
        axes = Axes.open(axes_group)

        self._axes = axes
        self.hpbase = axes[0]
        self.rest_axes = axes[1:]

        self.unit = Unit(self._file['hist'].attrs['unit'])
        self.contents = self._file['hist/contents']

    @property
    def axes(self):
        """
        List of axes.

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        return self._axes

    def get_point_source_response(self, source):

        """
        Get point source response (psr) corresponding to a
        given source direction in the galactic frame.

        Parameters
        ----------
        source : astropy.coordinates.SkyCoord
            Source direction in galactic frame.

        Returns
        -------
        psr : histpy.Histogram
            Point source response for source direction.

        """

        pix = self.hpbase.ang2pix(source)

        return PointSourceResponse(self.rest_axes, self.contents[pix+1], unit = self.unit)


class FastTSMap():

    def __init__(self, data, bkg_model, response_path, orientation = None,
                 cds_frame = "local"):

        """Initialize the instance of a TS map fit.

        Parameters
        ----------
        data : histpy.Histogram
            Observed data, which includes counts from both signal and
            background.
        bkg_model : histpy.Histogram
            Model used to estimate background counts in observed data.
        response_path : str or pathlib.Path
            Path to response file.
        orientation : cosipy.SpacecraftFile, optional
            Orientation history of spacecraft; required for "local"
            cds_frame, not used if frame is "galactic"
        cds_frame : str, optional
            frame of directions used for PsiChi axis of CDS.  One of
            "local" (frame attached to spacecraft) or "galactic".
            Default is local.

        """

        if cds_frame not in ("local", "galactic"):
            raise ValueError("cds_frame must be one of local or galactic")

        if cds_frame == "local" and orientation is None:
            raise TypeError("When data are binned in local frame, "
                            "orientation must be provided")

        self._orientation = orientation
        self._cds_frame = cds_frame

        # open the response file
        self._response_path = Path(response_path)
        if cds_frame == "local":
            self._response = FullDetectorResponse.open(response_path)
        else:
            self._response = GalacticResponse(response_path)

        self._data = data.todense().project(["Em", "Phi", "PsiChi"])
        self._bkg_model = bkg_model.todense().project(["Em", "Phi", "PsiChi"])

        self._fnf = fnf(max_iter=1000)


    @staticmethod
    def get_hypothesis_coords(nside, pixels = None,
                              scheme = "ring", coordsys = "galactic"):

        """
        Get directions corresponding to pixels of a HEALPix map of a
        given resolution and scheme.

        Parameters
        ----------
        nside : int
            Nside of HEALPix map
        pixels : array-like of int, optional
            Array of pixels to convert to directions; if not
            specified, directions will be generated for every pixel in
            map
        scheme : str, optional
            Scheme of HEALPix map (default: ring)
        coordsys : str, optional
            Coordinate system of HEALPix map (default: galactic)

        Returns
        -------
        hypothesis_coords : SkyCoord
            Vector SkyCoord containing coordinates for each pixel

        """

        if pixels is None:
            npix = hp.nside2npix(nside)
            pixels = np.arange(npix, dtype=int)

        hpbase = HealpixBase(nside = nside, scheme = scheme, coordsys = coordsys)
        coords = hpbase.pix2skycoord(pixels)

        return coords

    @staticmethod
    def slice_energy_channel(hist, channel_start, channel_stop):
        """
        Slice one or more bins along first axis of the `histogram`.

        Parameters
        ----------
        hist : histpy.Histogram
            The histogram object to be sliced.
        channel_start : int
            The start of the slice (inclusive).
        channel_stop : int
            The stop of the slice (exclusive).

        Returns
        -------
        sliced_hist : histpy.Histogram
            The sliced histogram.
        """

        sliced_hist = hist.slice[{"Em" : slice(channel_start, channel_stop)}]

        return sliced_hist

    @staticmethod
    def get_cds_array(hist, energy_channel):

        """
        Convert a CDS histogram to a flattened array, enforcing canonical
        order for dimensions and projecting over just the selected channels
        of the Em dimension.

        Parameters
        -----------
        hist : histpy.Histogram
           A CDS count Histogram
        energy_channel : 2-element list of form [lower_channel, upper_channel]
            energy (Em) channels to use in fitting (Python range lower_channel:upper_channel)

        Returns
        -------
        cds_array : numpy.ndarray
            Flattened CDS array

        """

        hist_cds_sliced = FastTSMap.slice_energy_channel(hist, energy_channel[0], energy_channel[1])
        hist_cds = hist_cds_sliced.project(["Phi", "PsiChi"]) # project out Em

        cds_array = hist_cds.contents
        if hist_cds.unit is not None:
            cds_array = cds_array.value

        return cds_array.ravel()

    def get_ei_cds_array(self, source, energy_channel, flux):

        """
        Get the expected counts in CDS in local or galactic frame.

        Parameters
        ----------
        source : astropy.coordinates.SkyCoord
            source direction
        energy_channel : 2-element list of form [lower_channel, upper_channel]
            energy (Em) channels to use in fitting (Python range lower_channel:upper_channel)
        flux: Histogram
            The integrated spectral flux of the source, binned according to the response

        Returns
        -------
        cds_array : numpy.ndarray
            Flattended Compton data space (CDS) array
        """

        if self._cds_frame == "local":

            # convert source direction to local frame
            source_in_sc_frame = self._orientation.get_target_in_sc_frame(source)

            # get map of the time spent at each pixel in the local frame
            dwell_time_map = self._orientation.get_dwell_map(response = self._response_path,
                                                             src_path = source_in_sc_frame)

            # convolve response with dwell_time_map to get point source response
            psr = self._response.get_point_source_response(dwell_time_map)

        else: # galactic frame

            psr = self._response.get_point_source_response(source)

        # convolve PSR with spectral flux to get expected counts
        expectation = psr.get_expectation(spectrum = None, flux = flux)

        # slice energy channals and project it to CDS
        ei_cds_array = self.get_cds_array(expectation, energy_channel)

        return ei_cds_array

    def fast_ts_fit(self, source, energy_channel, flux,
                    data_cds_array, bkg_model_cds_array,
                    valid_cells):
        """
        Perform a TS fit of data for a single source direction

        Parameters
        ----------
        source : astropy.coordinates.SkyCoord
            source direction
        energy_channel : 2-element list of form [lower_channel, upper_channel]
            energy (Em) channels to use in fitting (Python range lower_channel:upper_channel)
        flux: Histogram
            The integrated spectral flux of the source, binned according to the response
        data_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the data.
        bkg_model_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the background model.
        valid_cells : np.ndarray of bool
            Mask indicating which cells of CDS were preserved in data, bkg_model

        Returns
        -------
        result of TS fitting: [ts value, norm, norm_err, failed, # iterations]

        """

        ei_cds_array = self.get_ei_cds_array(source, energy_channel, flux)
        ei_sum = np.sum(ei_cds_array)
        ei_cds_array = ei_cds_array[valid_cells]

        return self._fnf.solve(data_cds_array, bkg_model_cds_array, ei_cds_array, ei_sum)

    def parallel_ts_fit(self, hypothesis_coords, energy_channel, spectrum,
                        cpu_cores = None):

        """
        Perform parallel computation on all the hypothesis coordinates.

        Parameters
        ----------
        hypothesis_coords : list
            List of hypothesis coordinates to fit
        energy_channel : 2-element list of form [lower_channel, upper_channel]
            energy (Em) channels to use in fitting (Python range lower_channel:upper_channel)
        spectrum : astromodels.functions
            Spectrum of the source.
        cpu_cores : int, optional
            Number of processors to use (default: do not restrict)
        Returns
        -------
        results : numpy.ndarray
            Fitted ts values for each hypothesis coordinate

        """

        if cpu_cores is not None:
            numba.set_num_threads(cpu_cores)

        # get the flattened data_cds_array
        data_cds_array = self.get_cds_array(self._data, energy_channel)
        bkg_model_cds_array = self.get_cds_array(self._bkg_model, energy_channel)

        # eliminate CDS cells with no counts in data (due to data sparsity) or
        # in bkg model (lack of pseudocounts in bkg model -- could be considered
        # a bug, may cause divide-by-zero error in fitting)
        valid_cells = np.logical_and(data_cds_array > 0, bkg_model_cds_array > 0)
        data_cds_array = data_cds_array[valid_cells]
        bkg_model_cds_array = bkg_model_cds_array[valid_cells]

        flux = get_integrated_spectral_model(spectrum, self._response.axes["Ei"])

        results = [
            self.fast_ts_fit(source, energy_channel, flux,
                             data_cds_array, bkg_model_cds_array,
                             valid_cells)[0]
            for source in hypothesis_coords
        ]

        return np.array(results)


    @staticmethod
    def plot_ts(m_ts, skycoord = None, containment = None,
                save_plot = False, save_dir = "",
                save_name = "ts_map.png", dpi = 300):

        """
        Plot a TS map.

        Parameters
        ----------
        m_ts : numpy.ndarray
            The array of ts values from a ts fit.
        skycoord : astropy.coordinates.SkyCoord, optional
            The true location of the source (the default is `None`,
            which implies that there are no coordiantes to be printed on
            the TS map).
        containment : float, optional
            The containment level of the source (the default is `None`, which will plot
            raw TS values).
        save_plot : bool, optional
            Set `True` to save the plot (the default is `False`, which means it won't save
            the plot.
        save_dir : str or pathlib.Path, optional
            The directory to save the plot.
        save_name : str, optional
            The file name of the plot to be saved.
        dpi : int, optional
            The dpi for plotting and saving.

        """

        fig, ax = plt.subplots(dpi = dpi)

        if containment is not None:
            critical = FastTSMap.get_chi_critical_value(containment = containment)
            max_ts = np.max(m_ts)
            min_ts = np.min(m_ts)
            hp.mollview(m_ts, max = max_ts, min = max_ts - critical,
                        title = f"Containment {containment*100}%", coord = "G", hold = True)
        else:
            hp.mollview(m_ts, coord = "G", hold = True)

        if skycoord is not None:
            lon = skycoord.l.deg
            lat = skycoord.b.deg
            hp.projscatter(lon, lat, marker = "x", linewidths = 0.5, lonlat=True,
                           coord = "G", label = f"True location at l={lon}, b={lat}", color = "fuchsia")

        hp.projscatter(0, 0, marker = "o", linewidths = 0.5, lonlat=True, coord = "G", color = "red")
        hp.projtext(350, 0, "(l=0, b=0)", lonlat=True, coord = "G", color = "red")

        if save_plot:
            fig.savefig(Path(save_dir)/save_name, dpi = dpi)

    @staticmethod
    def get_chi_critical_value(containment = 0.90):

        """
        Get the critical value of the chi^2 distribution based on the
        confidence level.

        Parameters
        ----------
        containment : float, optional
          The confidence level of the chi^2 distribution (the default is
          `0.9`, which implies that the 90% containment region).

        Returns
        -------
        float
            The critical value corresponding to the confidence level.

        """

        from scipy.stats import chi2

        return chi2.ppf(containment, df=2)
