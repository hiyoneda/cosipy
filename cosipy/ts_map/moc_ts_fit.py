from pathlib import Path

import numpy as np
import numba

import mhealpy as hp

import matplotlib.pyplot as plt

from .fast_ts_fit import FastTSMap

import logging
logger = logging.getLogger(__name__)

class MOCTSMap(FastTSMap):
    """
    Multi-resolution source mapping.
    """

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

        super().__init__(data, bkg_model, response_path,
                         orientation = orientation,
                         cds_frame = cds_frame)

    def _choose_pix_to_refine(self, ts):
        """
        Decide which pixels to refine based on their ts scores.

        Parameters
        ----------
        ts : np.ndarray
          ts scores for one or more pixels

        Returns
        -------
        hi_mask : np.ndarray of bool
          Boolean mask of size equal to ts that is True if pixel
          should be refined.

        """

        # mask identifies the top self._refine_rank pixels
        top_ts_indices = np.argpartition(ts, -self._top_rank)[-self._top_rank:]
        hi_idx = top_ts_indices[-self._top_rank:]
        hi_mask = np.zeros(len(ts), dtype=bool)
        hi_mask[hi_idx] = True

        """
        # mask identifies pixels above a threshold
        hi_mask = (ts >= ts.max() - self._cv_chi)
        """

        return hi_mask

    def _pad_refined_pixels(self, pixels, nside, hi_mask):
        """
        Given a mask specifying which of a set of pixels to retain,
        expand the mask to include adjacent pixels as well.

        hi_mask is modified in place.

        Parameters
        ----------
        pixels : np.ndarray of int
          A set of pixels, some of which are designated for refinement.
        nside : int
          nside of pixels in array
        hi_mask : np.ndarray of bool
          Boolean mask of size equal to pixels that is True if pixel
          should be refined
        """

        hi_adj = hp.get_all_neighbours(nside, pixels[hi_mask], nest=True)
        hi_adj = np.unique(hi_adj)
        adj_mask = np.isin(pixels, hi_adj, assume_unique=True)
        hi_mask[adj_mask] = True

    def moc_ts_fit(self, max_nside, top_rank, energy_channel, spectrum,
                   cpu_cores = None, init_nside = 1):
        """
        Construct a multi-resolution map of ts statistics, selectively
        refining the highest-scoring pixels.

        Parameters
        ----------
        max_nside : int
          highest possible nside reached during refinement
        top_rank : int
          ...
        energy_channel : 2-element list of form [lower_channel, upper_channel]
          energy (Em) channels to use in fitting (Python range
          lower_channel:upper_channel)
        spectrum : astromodels.functions
          spectrum of the source.
        cpu_cores : int, optional
          number of processors to use (default: do not restrict)
        init_nside : int, optional
          lowest nside used in map

        Returns
        -------
        ts : np.ndarray
          ts statistics for each pixel in map
        uniqs : np.ndarray of int
          uniq pixel IDs for each output pixel in map

        """

        def refine(pix):
            """
            Given pixels in NEST format, expand each to
            its four sub-pixels at the next nside.
            """

            res = np.tile(pix, 4)
            res *= 4

            n = len(pix)
            res[n:2*n]   += 1
            res[2*n:3*n] += 2
            res[3*n:]    += 3

            return res

        self._top_rank = top_rank

        if cpu_cores is not None:
            numba.set_num_threads(cpu_cores)

        data_cds_array, bkg_model_cds_array, psr_cache = \
            self._prepare_inputs(energy_channel, spectrum)

        all_pix = []
        all_ts = []

        # initially, compute ts for all pixels at minimum nside
        nside = init_nside
        pixels = np.arange(hp.nside2npix(init_nside), dtype=int)

        while nside <= max_nside:

            src_locs = self._get_hypothesis_coords(nside, pixels)

            results = [
                self.fast_ts_fit(source,
                                 data_cds_array, bkg_model_cds_array,
                                 psr_cache)[0]
                for source in src_locs
            ]

            ts = np.array(results)

            if nside == max_nside:
                # Done -- save all remaining pixels and their values
                all_pix.append(hp.nest2uniq(nside, pixels))
                all_ts.append(ts)
                break

            hi_mask = self._choose_pix_to_refine(ts)

            # self._pad_refined_pixels(pixels, nside, hi_mask)

            # For pixels that we will *not* refine, compute their
            # unique indices and save them.
            lo_mask = ~hi_mask
            lo_pix = hp.nest2uniq(nside, pixels[lo_mask])
            lo_ts  = ts[lo_mask]

            all_pix.append(lo_pix)
            all_ts.append(lo_ts)

            # Split pixels that we *will* refine down to next nside
            pixels = refine(pixels[hi_mask])

            nside *= 2

        return np.concatenate(all_ts), np.concatenate(all_pix)

    @staticmethod
    def plot_ts(moc_ts, moc_uniq,
                skycoord = None, containment = None,
                grid_lines = True,
                save_plot = False, save_dir = "",
                save_name = "ts_map.png", dpi = 300):

        """
        Plot a multi-resolution TS map.

        Parameters
        ----------
        moc_ts : np.ndarray
            ts values of multiresolution map
        moc_uniq : np.ndarray of int
            uniq HEALPixel values of multiresolution map
        skycoord : astropy.coordinates.SkyCoord, optional
            The true location of the source (default: do not plot)
        containment : float, optional
            Restrict the plotted pixels to the specified containment
            threshold relative to the max ts value (default: plot
            *all* ts values)
        grid_lines : bool, optional
            Print lines bordering each pixel in the map (default: True)
        save_plot : bool, optional
            Save the plot to a file (default: False)
        save_dir : string, optional
            Directory in which to save the plot
        save_name : str, optional
            File name under which tos ave the plot
        dpi : int, optional
            DPI used for plotting / saving

        """

        moc_map = hp.HealpixMap(data = moc_ts, uniq = moc_uniq)

        # get plotting canvas
        fig = plt.figure(dpi = dpi)

        axMoll = fig.add_subplot(1,1,1, projection = 'mollview')

        if containment is None:
            moc_map.plot(ax = axMoll)
        else:
            critical = FastTSMap.get_chi_critical_value(containment = 0.9)
            max_ts = np.max(moc_ts)
            moc_map.plot(ax = axMoll,
                         vmin = max_ts - critical,
                         vmax = max_ts)

        if grid_lines:
            moc_map.plot_grid(ax = plt.gca(), color = 'grey',
                              linewidth = 0.1);

        # plot the source location if given
        if skycoord is not None:

            axMoll.text(skycoord.l.deg, skycoord.b.deg, "x", size = 4,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = axMoll.get_transform('world'),
                        color = "red")

        if save_plot:
            fig.savefig(Path(save_dir)/save_name, dpi = dpi)
