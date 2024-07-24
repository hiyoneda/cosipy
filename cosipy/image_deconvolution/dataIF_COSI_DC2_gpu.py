import numpy as np
from tqdm.autonotebook import tqdm
import astropy.units as u
import cupy as cp
import gc

import logging
logger = logging.getLogger(__name__)

from histpy import Histogram, Axes

from cosipy.response import FullDetectorResponse
from cosipy.data_io import BinnedData
from cosipy.image_deconvolution import CoordsysConversionMatrix, ImageDeconvolutionDataInterfaceBase

class DataIF_COSI_DC2_GPU(ImageDeconvolutionDataInterfaceBase):
    """
    A class for the interface for the COSI DC2 dataset using cupy.
    """

    def __init__(self, name = None):

        ImageDeconvolutionDataInterfaceBase.__init__(self, name)

        # None if using Galactic CDS, mandotary if using local CDS
        self._coordsys_conv_matrix = None 

        # optional
        self.is_miniDC2_format = False #should be removed in the future
        
        # for memory management
        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()

        # dtype of cp.array
        self.dtype_cp = np.float32

        # return cp.ndarray or not in some functions
        self.return_cp_array = True

    @classmethod
    def load(cls, name, event_binned_data, dict_bkg_binned_data, rsp, coordsys_conv_matrix = None, is_miniDC2_format = False, dtype_cp = np.float32, return_cp_array = True):
        """
        Load data

        Parameters
        ----------
        name : str
            The name of data
        event_binned_data : :py:class:`histpy.Histogram`
            Event histogram
        dict_bkg_binned_data : dict
            Background models as {background_model_name: :py:class:`histpy.Histogram`}
        rsp : :py:class:`histpy.Histogram` or :py:class:`cosipy.response.FullDetectorResponse`
            Response
        coordsys_conv_matrix : :py:class:`cosipy.image_deconvolution.CoordsysConversionMatrix`, default False
            Coordsys conversion matrix 
        is_miniDC2_format : bool, default False
            Whether the file format is for mini-DC2. It will be removed in the future.

        Returns
        -------
        :py:class:`cosipy.image_deconvolution.DataLoader`
            An instance of DataLoader containing the input data set
        """

        new = cls(name)

        new.dtype_cp = dtype_cp

        new.return_cp_array = return_cp_array

        new._event = event_binned_data.to_dense()

        new._bkg_models = dict_bkg_binned_data

        for key in new._bkg_models:
            if new._bkg_models[key].is_sparse:
                new._bkg_models[key] = new._bkg_models[key].to_dense()

            new._summed_bkg_models[key] = np.sum(new._bkg_models[key])

        new._bkg_models_cp = {key: cp.asarray(new.bkg_model(key).contents, dtype = new.dtype_cp) for key in new.keys_bkg_models()}
        
        if coordsys_conv_matrix is not None:
            new._coordsys_conv_matrix = coordsys_conv_matrix
            _coordsys_conv_matrix_dense = new._coordsys_conv_matrix.contents.todense()
            new._coordsys_conv_matrix_cp = cp.asarray(_coordsys_conv_matrix_dense, dtype = new.dtype_cp)
            del _coordsys_conv_matrix_dense 

#            new.mempool.free_all_blocks()
#            new.pinned_mempool.free_all_blocks()

        new.is_miniDC2_format = is_miniDC2_format
        
        logger.info('Loading the response matrix onto your computer memory...')
        new._load_full_detector_response_on_memory(rsp, is_miniDC2_format)
        logger.info('Finished')
        
        # We modify the axes in event, bkg_models, response. This is only for DC2.
        result = new._modify_axes()
        if result == False:
            logger.warning('Please rerun after checking the axes of the input histograms')
            return 
        
        new._data_axes = new._event.axes
        
        if new._coordsys_conv_matrix is None:
            axes = [new._image_response_axes['NuLambda'], new._image_response_axes['Ei']]
            axes[0].label = 'lb' 
            # The gamma-ray direction of pre-computed response in DC2 is in the galactic coordinate, not in the local coordinate.
            # Actually, it is labeled as 'NuLambda'. So I replace it with 'lb'.
            new._model_axes = Axes(axes)
        else:
            new._model_axes = Axes([new._coordsys_conv_matrix.axes['lb'], new._image_response_axes['Ei']])

        new._calc_exposure_map()

        if new.return_cp_array == True:
            new._event = cp.asarray(new._event.contents, dtype = new.dtype_cp)

        return new

    def _modify_axes(self):
        """
        Modify the axes of data. This method will be removed in the future.
        """

        logger.warning("Note that _modify_axes() in DataLoader was implemented for a temporary use. It will be removed in the future.")

        if self._coordsys_conv_matrix is None:
            axis_name = ['Em', 'Phi', 'PsiChi']

        elif self._coordsys_conv_matrix.binning_method == 'Time':
            axis_name = ['Time', 'Em', 'Phi', 'PsiChi']

        elif self._coordsys_conv_matrix.binning_method == 'ScAtt':
            axis_name = ['ScAtt', 'Em', 'Phi', 'PsiChi']

        for name in axis_name:

            logger.info(f"... checking the axis {name} of the event and background files...")
            
            event_edges, event_unit = self._event.axes[name].edges, self._event.axes[name].unit

            for key in self._bkg_models:

                bkg_edges, bkg_unit = self._bkg_models[key].axes[name].edges, self._bkg_models[key].axes[name].unit

                if np.all(event_edges == bkg_edges):
                    logger.info(f"    --> pass (edges)") 
                else:
                    logger.warning(f"Warning: the edges of the axis {name} are not consistent between the event and the background model {key}!")
                    logger.warning(f"         event      : {event_edges}")
                    logger.warning(f"         background : {bkg_edges}")
                    return False

                if event_unit == bkg_unit:
                    logger.info(f"    --> pass (unit)") 
                else:
                    logger.warning(f"Warning: the unit of the axis {name} are not consistent between the event and the background model {key}!")
                    logger.warning(f"         event      : {event_unit}")
                    logger.warning(f"         background : {bkg_unit}")
                    return False

        # check the axes of the event/response files. 
        # Note that currently (2023-08-29) no unit is stored in the binned data. So only the edges are compared. This should be modified in the future.

        axis_name = ['Em', 'Phi', 'PsiChi']
        
        for name in axis_name:

            logger.info(f"...checking the axis {name} of the event and response files...")

            event_edges, event_unit = self._event.axes[name].edges, self._event.axes[name].unit
            response_edges, response_unit = self._image_response_axes[name].edges, self._image_response_axes[name].unit
            
            if type(response_edges) == u.quantity.Quantity and self.is_miniDC2_format == True:
                response_edges = response_edges.value

            if np.all(event_edges == response_edges):
                logger.info(f"    --> pass (edges)") 
            else:
                logger.warning(f"Warning: the edges of the axis {name} are not consistent between the event and background!")
                logger.warning(f"        event      : {event_edges}")
                logger.warning(f"        response : {response_edges}")
                return False

        if self._coordsys_conv_matrix is None:
            axes_cds = Axes([self._image_response_axes["Em"], \
                             self._image_response_axes["Phi"], \
                             self._image_response_axes["PsiChi"]])
        else:
            axes_cds = Axes([self._event.axes[0], \
                             self._image_response_axes["Em"], \
                             self._image_response_axes["Phi"], \
                             self._image_response_axes["PsiChi"]])
        
        self._event = Histogram(axes_cds, unit = self._event.unit, contents = self._event.contents)

        for key in self._bkg_models:
            bkg_model = self._bkg_models[key]
            self._bkg_models[key] = Histogram(axes_cds, unit = bkg_model.unit, contents = bkg_model.contents)
            del bkg_model

        logger.info(f"The axes in the event and background files are redefined. Now they are consistent with those of the response file.")

        return True

    def _load_full_detector_response_on_memory(self, full_detector_response, is_miniDC2_format):
        """
        Load a response file on the computer memory.
        """

        self._image_response_axes = full_detector_response.axes

        if isinstance(full_detector_response, Histogram):
            self._image_response_cp = cp.asarray(full_detector_response.contents.value, dtype = self.dtype_cp)
            self._image_response_unit = full_detector_response.contents.unit
            return

        self._image_response_unit = full_detector_response.unit

        axes_image_response = Axes([full_detector_response.axes["NuLambda"], full_detector_response.axes["Ei"],
                                    full_detector_response.axes["Em"], full_detector_response.axes["Phi"], full_detector_response.axes["PsiChi"]])

        nside = full_detector_response.axes["NuLambda"].nside
        npix = full_detector_response.axes["NuLambda"].npix 
    
        if is_miniDC2_format:
            self._image_response_cp = cp.zeros(axes_image_response.nbins, dtype = np.float32)
            for ipix in tqdm(range(npix)):
                self._image_response_cp[ipix] = cp.asarray(np.sum(full_detector_response[ipix].to_dense(), axis = (4,5)), dtype = self.dtype_cp) #Ei, Em, Phi, ChiPsi
        else:
            contents = full_detector_response._file['DRM']['CONTENTS'][:]
            self._image_response_cp = cp.asarray(contents, dtype = self.dtype_cp)
            del contents

        gc.collect()

    def _calc_exposure_map(self):
        """
        Calculate exposure_map, which is an intermidiate matrix used in RL algorithm.
        """

        logger.info("Calculating an exposure map...")
        
        if self._coordsys_conv_matrix is None:
            self._exposure_map = Histogram(self._model_axes, unit = self._image_response_unit * u.sr)
            self._exposure_map[:] = cp.asnumpy(cp.sum(self._image_response_cp, axis = (2,3,4))) * self._image_response_unit * self.model_axes['lb'].pixarea()
        else:
            self._exposure_map = Histogram(self._model_axes, unit = self._image_response_unit * self._coordsys_conv_matrix.unit * u.sr)
            self._exposure_map[:] = np.tensordot(np.sum(self._coordsys_conv_matrix, axis = (0)), 
                                                 cp.asnumpy(cp.sum(self._image_response_cp, axis = (2,3,4))),
                                                 axes = ([1], [0])) * self._image_response_unit * self._coordsys_conv_matrix.unit * self.model_axes['lb'].pixarea()
            # [Time/ScAtt, lb, NuLambda] -> [lb, NuLambda]
            # [NuLambda, Ei, Em, Phi, PsiChi] -> [NuLambda, Ei]
            # [lb, NuLambda] x [NuLambda, Ei] -> [lb, Ei]

        logger.info("Finished...")

    def calc_expectation(self, model_map, dict_bkg_norm = None, almost_zero = 1e-12):
        """
        Calculate expected counts from a given model map.

        Parameters
        ----------
        model_map : :py:class:`cosipy.image_deconvolution.ModelMap`
            Model map
        dict_bkg_norm : dict, default None
            background normalization for each background model, e.g, {'albedo': 0.95, 'activation': 1.05}
        almost_zero : float, default 1e-12
            In order to avoid zero components in extended count histogram, a tiny offset is introduced.
            It should be small enough not to effect statistics.

        Returns
        -------
        :py:class:`histpy.Histogram`
            Expected count histogram

        Notes
        -----
        This method should be implemented in a more general class, for example, extended source response class in the future.
        """
        # Currenly (2024-01-12) this method can work for both local coordinate CDS and in galactic coordinate CDS.
        # This is just because in DC2 the rotate response for galactic coordinate CDS does not have an axis for time/scatt binning.
        # However it is likely that it will have such an axis in the future in order to consider background variability depending on time and pointign direction etc.
        # Then, the implementation here will not work. Thus, keep in mind that we need to modify it once the response format is fixed.

        model_map_cp = cp.asarray(model_map.contents.value, dtype = self.dtype_cp)
        
        if self._coordsys_conv_matrix is None:
            expectation_cp = cp.tensordot( model_map_cp, self._image_response_cp, axes = ([0,1],[0,1])) * model_map.axes['lb'].pixarea().value
            # ['lb', 'Ei'] x [NuLambda(lb), Ei, Em, Phi, PsiChi] -> [Em, Phi, PsiChi]
        else:
            map_rotated = cp.tensordot(self._coordsys_conv_matrix_cp, model_map_cp, axes = ([1], [0])) 
            map_rotated *= model_map.axes['lb'].pixarea().value
            # ['Time/ScAtt', 'lb', 'NuLambda'] x ['lb', 'Ei'] -> [Time/ScAtt, NuLambda, Ei]
            # the unit of map_rotated is 1/cm2 ( = s * 1/cm2/s/sr * sr)
            expectation_cp = cp.tensordot(map_rotated, self._image_response_cp, axes = ([1,2], [0,1]))
            # [Time/ScAtt, NuLambda, Ei] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, Em, Phi, PsiChi]
            del map_rotated

#            map_rotated = np.tensordot(self._coordsys_conv_matrix.contents, model_map.contents, axes = ([1], [0])) 
#            map_rotated *= model_map.axes['lb'].pixarea().value
#            map_rotated_cp = cp.asarray(map_rotated, dtype = self.dtype_cp)
#            # ['Time/ScAtt', 'lb', 'NuLambda'] x ['lb', 'Ei'] -> [Time/ScAtt, NuLambda, Ei]
#            # the unit of map_rotated is 1/cm2 ( = s * 1/cm2/s/sr * sr)
#            expectation_cp = cp.tensordot(map_rotated_cp, self._image_response_cp, axes = ([1,2], [0,1]))
#            # [Time/ScAtt, NuLambda, Ei] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, Em, Phi, PsiChi]
#            del map_rotated_cp 

        del model_map_cp

        if dict_bkg_norm is not None: 
            for key in self.keys_bkg_models():
                expectation_cp += self._bkg_models_cp[key] * dict_bkg_norm[key]
        expectation_cp += almost_zero

        if self.return_cp_array == True:
            return expectation_cp

        expectation = Histogram(self.data_axes, contents = cp.asnumpy(expectation_cp))

        del expectation_cp
        
        return expectation

    def calc_T_product(self, dataspace_histogram):
        """
        Calculate the product of the input histogram with the transonse matrix of the response function.
        Let R_{ij}, H_{i} be the response matrix and dataspace_histogram, respectively.
        Note that i is the index for the data space, and j is for the model space.
        In this method, \sum_{j} H{i} R_{ij}, namely, R^{T} H is calculated.

        Parameters
        ----------
        dataspace_histogram: :py:class:`histpy.Histogram`
            Its axes must be the same as self.data_axes

        Returns
        -------
        :py:class:`histpy.Histogram`
            The product with self.model_axes
        """
        # TODO: currently, dataspace_histogram is assumed to be a dense.

        hist_unit = self.exposure_map.unit

        if isinstance(dataspace_histogram, cp.ndarray):
            dataspace_histogram_cp = dataspace_histogram
        elif dataspace_histogram.unit is not None:
            hist_unit *= dataspace_histogram.unit
            dataspace_histogram_cp = cp.asarray(dataspace_histogram.contents.value, dtype = self.dtype_cp)
        else:
            dataspace_histogram_cp = cp.asarray(dataspace_histogram.contents, dtype = self.dtype_cp)

        if self._coordsys_conv_matrix is None:
            hist_cp = cp.tensordot(dataspace_histogram_cp, self._image_response_cp, axes = ([0,1,2], [2,3,4])) 
            # [Em, Phi, PsiChi] x [NuLambda (lb), Ei, Em, Phi, PsiChi] -> [NuLambda (lb), Ei]
            hist = Histogram(self.model_axes, contents = cp.asnumpy(hist_cp) * self.model_axes['lb'].pixarea().value, unit = hist_unit)
            del hist_cp
        else:

            # NOTE: The memory leak happened with the following lines if using dtype = np.float64
            _ = cp.tensordot(dataspace_histogram_cp, self._image_response_cp, axes = ([1,2,3], [2,3,4])) 
            # [Time/ScAtt, Em, Phi, PsiChi] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, NuLambda, Ei]
            _ = cp.tensordot(self._coordsys_conv_matrix_cp, _, axes = ([0,2], [0,1]))
            # [Time/ScAtt, lb, NuLambda] x [Time/ScAtt, NuLambda, Ei] -> [lb, Ei]
            contents = cp.asnumpy(_) * self.model_axes['lb'].pixarea().value
            hist = Histogram(self.model_axes, contents = contents, unit = hist_unit)

            del _

#            _ = cp.tensordot(dataspace_histogram_cp, self._image_response_cp, axes = ([1,2,3], [2,3,4])) 
#            _ = cp.asnumpy(_)
#            # [Time/ScAtt, Em, Phi, PsiChi] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, NuLambda, Ei]
#            hist = Histogram(self.model_axes,\
#                             contents = np.tensordot(self._coordsys_conv_matrix.contents, _, axes = ([0,2], [0,1])),\
#                             unit = hist_unit)
#            # [Time/ScAtt, lb, NuLambda] x [Time/ScAtt, NuLambda, Ei] -> [lb, Ei]
#            # note that coordsys_conv_matrix is the sparse, so the unit should be recovered.
#            del _
        
        if not isinstance(dataspace_histogram, cp.ndarray):
            del dataspace_histogram_cp

        return hist

    def calc_bkg_model_product(self, key, dataspace_histogram):
        """
        Calculate the product of the input histogram with the background model.
        Let B_{i}, H_{i} be the background model and dataspace_histogram, respectively.
        In this method, \sum_{i} B_{i} H_{i} is calculated.

        Parameters
        ----------
        key: str
            Background model name
        dataspace_histogram: :py:class:`histpy.Histogram`
            its axes must be the same as self.data_axes

        Returns
        -------
        flaot
        """
        # TODO: currently, dataspace_histogram is assumed to be a dense.

        if isinstance(dataspace_histogram, cp.ndarray):
            dataspace_histogram_cp = dataspace_histogram
        else:
            dataspace_histogram_cp = cp.asarray(dataspace_histogram.contents, dtype = self.dtype_cp)

        if self._coordsys_conv_matrix is None:

            return cp.asnumpy(cp.tensordot(dataspace_histogram_cp, self._bkg_models_cp[key], axes = ([0,1,2], [0,1,2])))

        return cp.asnumpy(cp.tensordot(dataspace_histogram_cp, self._bkg_models_cp[key], axes = ([0,1,2,3], [0,1,2,3])))

    def calc_loglikelihood(self, expectation):
        """
        Calculate log-likelihood from given expected counts or model/expectation.

        Parameters
        ----------
        expectation : :py:class:`histpy.Histogram`
            Expected count histogram.

        Returns
        -------
        float
            Log-likelood
        """

        if self.return_cp_array == True:
            loglikelood = cp.sum( self.event * cp.log(expectation) , dtype = np.float64) - cp.sum(expectation, dtype = np.float64)
            loglikelood = cp.asnumpy(loglikelood)
        else:
            loglikelood = np.sum( self.event * np.log(expectation) ) - np.sum(expectation)

        return loglikelood
