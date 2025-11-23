from pathlib import Path

import numpy as np
import numba

import matplotlib.pyplot as plt

import astropy.units as u
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

        pix = self.hpbase.vec2pix(source)

        return PointSourceResponse(self.rest_axes, self.contents[pix+1], unit = self.unit)


class PSRCache:
    """
    A cached reader for PSR data from a response file, designed for
    use with FastTSMap.  For a given NuLambda pixel p, we fetch the
    pixel's data from the underlying response file and do all the data
    reduction needed to compute a PSR for pixel p averaged over the
    input flux.  The result is cached so that, when different source
    directions require a PSR for the same pixel p, we don't do the
    fetching and reduction more than once.

    If memory usage is a concern, the cache can be set to a given max
    size with LRU replacement.  But the reduced PSR sums away the Ei
    and Em dimensions *and* removes CDS voxels that do not matter for
    the ts_map computation, so it is much smaller than a raw chunk of
    the response file. Hence, it is likely not necessary to limit the
    cache size in practice.

    """

    def __init__(self, response, em_slice, valid_cells, flux,
                 maxSize = None):
        """
        Create a new PSRCache, providing the information needed
        to fetch and reduce PSRs from the response file on demand.

        Parameters
        ----------
        response : FullDetectorResponse
          The response from which to read slices for PSR computation
        em_slice : Slice object
          The slice of the Em axis used to compute PSRs
        valid_cells : np.ndarray of int
          CDS voxels on the linearized Phi/PsiChi axis that are actually
          used in in the ts_map computation
        flux : Histogram
          Integrated spectral flux, binned according to response's Ei axis
        maxSize: int (optional)
          If not None, maximum number of NuLambda pixels for which we will
          cache PSRs.  The cache is managed according to an LRU policy.

        """

        from collections import OrderedDict

        self.cache = OrderedDict()
        self.maxSize = maxSize

        self.response = response
        self.em_slice = em_slice
        self.valid_cells = valid_cells

        self.ei_weights = flux.contents.value * response.eff_area

        self.nLookups = 0
        self.nMisses = 0

    def get_psr(self, p):
        """
        Get the reduced PSR for NuLambda pixel p.

        Parameters
        ----------
        p : int
          NuLambda value of requested PSR

        Returns
        -------
        psr : np.ndarray of float (length = |valid_cells|)
          PSR for pixel p, summed over requested Em and
          convolved with spectral flux.  The result gives
          one value per valid voxel.
        psr_sum
          Sum of PSR for pixel p over *all* voxels, not just
          the valid ones.

        """

        self.nLookups += 1
        v = self.cache.get(p)
        if v is None: # cache miss
            self.nMisses += 1
            v = self.compute_psr(p)
            self.cache[p] = v

            # implement LRU policy if requested
            if self.maxSize is not None and len(self.cache) > self.maxSize:
                self.cache.popitem(last=False)
        else:
            # move MRU value to end to support LRU policy if requested
            if self.maxSize is not None:
                self.cache.move_to_end(p)

        return v

    def print_stats(self):
        """
        Print cache miss statistics
        """

        missRate = 0. if self.nLookups == 0 else self.nMisses/self.nLookups

        print(f"Cache size: {len(self.cache)} (out of {self.maxSize})")
        print(f"Misses: {self.nMisses} / {self.nLookups} = {missRate:0.3f}")

    def compute_psr(self, p):
        """
        Compute a reduced PSR for NuLambda pixel p.

        Returns
        -------
        psr : np.ndarray of float (length = |valid_cells|)
          PSR for pixel p, summed over requested Em and
          convolved with spectral flux.  The result gives
          one value per valid voxel.
        psr_sum
          Sum of PSR for pixel p over *all* voxels, not just
          the valid ones.

        """

        # FIXME? Following assumes response has rest_axes = [ Ei, Em,
        # Phi/PsiChi ]. We should probably verify this in __init__.

        # get raw CDS counts for pixel, trimmed by Em slice
        # size is Ei x Em x Phi/PsiChi
        counts = self.response.get_counts(p, self.em_slice)

        # linearize CDS : Ei x Em x CDS voxels. Note that we ensure
        # in FastTSMap that data and bkg will use the same dimension
        # ordering as the response for the CDS, so there is no need to
        # re-order dimensions here.
        counts = counts.reshape(*counts.shape[:2], -1)

        # sum over Em dimension and convert to float : Ei x CDS voxels
        counts = np.sum(counts, axis=1, dtype=self.response.dtype)

        # extract valid CDS voxels of psr after capturing sum of
        # *all* voxels : Ei x valid CDS voxels
        psr_sum = np.sum(counts, axis=1)
        psr = counts[:, self.valid_cells]

        # convolve psr with flux (and also eff_area weights, which
        # have not yet been applied) to remove Ei dimension
        psr_sum = np.dot(psr_sum, self.ei_weights)
        psr = np.tensordot(psr, self.ei_weights, axes=(0,0))

        return psr, psr_sum


class FastTSMap():

    def __init__(self, data, bkg_model, response_path, orientation = None,
                 cds_frame = "local"):

        """
        Initialize the instance of a TS map fit.

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

        self._cds_frame = cds_frame

        if cds_frame == "local":

            if orientation is None:
                raise TypeError("When data are binned in local frame, "
                                "orientation must be provided")

            self._orientation = orientation

            # open the response file
            self._response = FullDetectorResponse.open(response_path)

        else:

            self._response = GalacticResponse(response_path)

        # record order of response's CDS physical dimensions for
        # linearization of data, bkg
        axes = self._response.axes
        if axes.label_to_index("Phi") < axes.label_to_index("PsiChi"):
            self._cds_order = ("Phi", "PsiChi")
        else:
            self._cds_order = ("PsiChi", "Phi")

        self._data = data.todense().project(["Em", "Phi", "PsiChi"])
        self._bkg_model = bkg_model.todense().project(["Em", "Phi", "PsiChi"])

        self._fnf = fnf(max_iter=1000)

    @staticmethod
    def get_hypothesis_coords(nside, pixels = None,
                              scheme = "ring",
                              coordsys = "galactic"):
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
        hypothesis_coords : np.ndarray of (# pixels x 3)
            Cartesian 3-vectors for each pixel's direction

        """

        if pixels is None:
            npix = hp.nside2npix(nside)
            pixels = np.arange(npix, dtype=int)

        hpbase = HealpixBase(nside = nside, scheme = scheme,
                             coordsys = coordsys)

        return np.column_stack(hpbase.pix2vec(pixels))

    @staticmethod
    def get_cds_array(hist, em_slice, cds_order):

        """
        Convert a CDS histogram to a flattened array, enforcing
        canonical order for dimensions and projecting over just the
        selected channels of the Em dimension.

        Parameters
        -----------
        hist : histpy.Histogram
           A CDS count Histogram
        em_slice : Slice object
           Energy (Em) channels to use in fitting
        cds_order : list-like
           order of Psi and PhiChi dimensions for linearization

        Returns
        -------
        cds_array : numpy.ndarray
            Flattened CDS array

        """

        hist_cds_sliced = hist.slice[{"Em" : em_slice}]
        hist_cds = hist_cds_sliced.project(cds_order) # project out Em

        cds_array = hist_cds.contents
        if hist_cds.unit is not None:
            cds_array = cds_array.value

        return cds_array.ravel()

    def get_ei_cds_array(self, source, em_slice, valid_cells, flux):

        """
        Get the expected counts in CDS in local or galactic frame.

        Parameters
        ----------
        source : astropy.coordinates.SkyCoord
            source direction
        em_slice : Slice object
           Energy (Em) channels to use in fitting
        valid_cells : array
           valid Phi/PsiChi voxels of CDS
        flux: Histogram
            The integrated spectral flux of the source, binned
            according to the response

        Returns
        -------
        cds_array : numpy.ndarray
            Flattended Compton data space (CDS) array

        """

        if self._cds_frame == "local":

            # convert source direction to path in local frame
            lons, colats = self._orientation.get_target_in_sc_frame(source)

            # get list of HEALPix pixels with nonzero exposure on path
            pixels, exposures = self._orientation.get_exposure(base = self._response,
                                                               theta = colats,
                                                               phi = lons,
                                                               lonlat = False)

            # sum the PSRs for each NuLambda pixel according to their
            # exposure weights
            ei_cds_array = np.zeros(len(valid_cells), dtype=self._response.dtype)
            ei_sum = 0.

            for p, exposure in zip(pixels, exposures):
                psr, psr_sum = self.psr_cache.get_psr(p)
                ei_cds_array += psr * exposure
                ei_sum += psr_sum * exposure

        else: # galactic frame

            psr = self._response.get_point_source_response(source)

            # convolve PSR with spectral flux to get expected counts
            expectation = psr.get_expectation(spectrum = None, flux = flux)

            # slice energy channals and project it to CDS
            ei_cds_array = self.get_cds_array(expectation, em_slice, self._cds_order)

            ei_sum = np.sum(ei_cds_array)
            ei_cds_array = ei_cds_array[valid_cells]

        return ei_cds_array, ei_sum

    def fast_ts_fit(self, source, em_slice, flux,
                    data_cds_array, bkg_model_cds_array,
                    valid_cells):
        """
        Perform a TS fit of data for a single source direction

        Parameters
        ----------
        source : astropy.coordinates.SkyCoord
            source direction
        em_slice : Slice object
           Energy (Em) channels to use in fitting
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

        ei_cds_array, ei_sum = self.get_ei_cds_array(source, em_slice, valid_cells, flux)

        return self._fnf.solve(data_cds_array, bkg_model_cds_array, ei_cds_array, ei_sum)

    def parallel_ts_fit(self, hypothesis_coords, energy_channel, spectrum,
                        cpu_cores = None):

        """Perform parallel computation on all the hypothesis coordinates.

        Parameters
        ----------
        hypothesis_coords : np.ndarray (N x 3)
            Hypothesis coordinates to fit
        energy_channel : 2-element list of form [lower_channel, upper_channel]
            Energy (Em) channels to use in fitting (Python range
            lower_channel:upper_channel)
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

        if energy_channel is None:
            em_slice = slice(None)
        else:
            em_slice = slice(energy_channel[0], energy_channel[1])

        # get the flattened data and background CDS arrays
        data_cds_array = self.get_cds_array(self._data, em_slice,
                                            self._cds_order)
        bkg_model_cds_array = self.get_cds_array(self._bkg_model, em_slice,
                                                 self._cds_order)

        # eliminate CDS cells with no counts in data (due to data
        # sparsity) or in bkg model (lack of pseudocounts in bkg model
        # -- could be considered a bug, may cause divide-by-zero error
        # in fitting)
        valid_cells = np.where(np.logical_and(data_cds_array > 0,
                                              bkg_model_cds_array > 0))[0]

        data_cds_array = data_cds_array[valid_cells]
        bkg_model_cds_array = bkg_model_cds_array[valid_cells]

        flux = get_integrated_spectral_model(spectrum, self._response.axes["Ei"])

        self.psr_cache = PSRCache(self._response, em_slice, valid_cells, flux)

        results = [
            self.fast_ts_fit(source, em_slice, flux,
                             data_cds_array, bkg_model_cds_array,
                             valid_cells)[0]
            for source in hypothesis_coords
        ]

        del self.psr_cache

        return np.array(results)


    @staticmethod
    def plot_ts(m_ts, skycoord = None, containment = None, scheme="ring",
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
        scheme : string, optional
            HEALPix scheme of TS map values ("ring" or "nested"; default = "ring")
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
        nest = scheme.startswith("nest")

        if containment is not None:
            critical = FastTSMap.get_chi_critical_value(containment = containment)
            max_ts = np.max(m_ts)
            min_ts = np.min(m_ts)
            hp.mollview(m_ts, max = max_ts, min = max_ts - critical, nest=nest,
                        title = f"Containment {containment*100}%", coord = "G", hold = True)
        else:
            hp.mollview(m_ts, nest=nest, coord = "G", hold = True)

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
