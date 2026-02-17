import numpy as np
import astropy.io.fits as fits
import functools
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

from .constants import NUMERICAL_ZERO, CHUNK_SIZE_FITS, DEFAULT_ITERATION_MAX

def _to_float(x) -> float:
    """
    Convert to float, handling astropy Quantity.
    """

    if hasattr(x, 'unit'):
        return float(x.to(''))
    return float(x)

class DeconvolutionAlgorithmBase(ABC):
    """
    A base class for image deconvolution algorithms.
    Subclasses should override these methods:

    - initialization 
    - pre_processing
    - processing_core
    - post_processing
    - register_result
    - check_stopping_criteria
    - finalization
    
    When the method run_deconvolution is called in ImageDeconvolution class, 
    the iteration method in this class is called for each iteration.

    Attributes
    ----------
    initial_model: :py:class:`cosipy.image_deconvolution.ModelBase` or its subclass
    dataset: list of :py:class:`cosipy.image_deconvolution.ImageDeconvolutionDataInterfaceBase` or its subclass
    parameter : py:class:`yayc.Configurator`
    results: list of results
    dict_bkg_norm: the dictionary of background normalizations
    dict_dataset_indexlist_for_bkg_models: the indices of data corresponding to each background model in the dataset
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        self.initial_model = initial_model
        self.dataset = dataset
        self.mask = mask 
        self.parameter = parameter 
        self.results = []

        # background normalization
        self.dict_bkg_norm = {}
        self.dict_dataset_indexlist_for_bkg_models = {}
        for data in self.dataset:
            for key in data.keys_bkg_models():
                if not key in self.dict_bkg_norm.keys():
                    self.dict_bkg_norm[key] = 1.0
                    self.dict_dataset_indexlist_for_bkg_models[key] = []
        
        for key in self.dict_dataset_indexlist_for_bkg_models.keys():
            for index, data in enumerate(self.dataset):
                if key in data.keys_bkg_models():
                    self.dict_dataset_indexlist_for_bkg_models[key].append(index)

        logger.debug(f'dict_bkg_norm: {self.dict_bkg_norm}')
        logger.debug(f'dict_dataset_indexlist_for_bkg_models: {self.dict_dataset_indexlist_for_bkg_models}')

        # parameters of the iteration
        self.iteration_count = 0
        self.iteration_max = parameter.get('iteration_max', DEFAULT_ITERATION_MAX)

    @abstractmethod
    def initialization(self):
        """
        initialization before running the image deconvolution
        """
        raise NotImplementedError

    @abstractmethod
    def pre_processing(self):
        """
        pre-processing for each iteration
        """
        raise NotImplementedError

    @abstractmethod
    def processing_core(self):
        """
        Core processing for each iteration.
        
        This method should implement the main algorithm logic.
        For EM-based algorithms, this typically includes:
        - E-step: Expectation calculation
        - M-step: Maximization/update
        
        Subclasses must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def post_processing(self):
        """
        Post-processing for each iteration. 
        In this step, if needed, you can apply some filters to self.delta_model and set it as self.processed_delta_model.
        Then, the updated model should be calculated as self.model.
        For example, Gaussian smoothing can be applied to self.delta_model in this step.
        """
        raise NotImplementedError

    @abstractmethod
    def register_result(self):
        """
        Register results at the end of each iteration. 
        Users can define what kinds of values are stored in this method.
        """
        raise NotImplementedError

    @abstractmethod
    def check_stopping_criteria(self) -> bool:
        """
        Check if the iteration process should be continued or stopped.
        When it returns True, the iteration will stopped.
        """
        raise NotImplementedError

    @abstractmethod
    def finalization(self):
        """
        finalization after running the image deconvolution
        """
        raise NotImplementedError

### A subclass should not override the methods below. ###

    def iteration(self):
        """
        Perform one iteration of image deconvolution.
        This method should not be overrided in subclasses.
        """
        self.iteration_count += 1

        logger.info(f"## Iteration {self.iteration_count}/{self.iteration_max} ##")

        logger.info("<< Pre-processing >>")
        self.pre_processing()

        logger.info("<< Processing Core>>")
        self.processing_core()
            
        logger.info("<< Post-processing >>")
        self.post_processing()

        logger.info("<< Registering Result >>")
        self.register_result()

        logger.info("<< Checking Stopping Criteria >>")
        stop_iteration = self.check_stopping_criteria()
        logger.info("--> {}".format("Stop" if stop_iteration else "Continue"))

        return stop_iteration

    def calc_expectation_list(self, model, dict_bkg_norm = None, almost_zero = NUMERICAL_ZERO):
        """
        Calculate a list of expected count histograms corresponding to each data in the registered dataset.

        Parameters
        ----------
        model: :py:class:`cosipy.image_deconvolution.ModelBase` or its subclass
            Model
        dict_bkg_norm : dict, default None
            background normalization for each background model, e.g, {'albedo': 0.95, 'activation': 1.05}
        almost_zero : float, default NUMERICAL_ZERO 
            In order to avoid zero components in extended count histogram, a tiny offset is introduced.
            It should be small enough not to effect statistics.

        Returns
        -------
        list of :py:class:`histpy.Histogram`
            List of expected count histograms
        """
        
        return [data.calc_expectation(model, dict_bkg_norm = dict_bkg_norm, almost_zero = almost_zero) for data in self.dataset]

    def calc_log_likelihood_list(self, expectation_list):
        """
        Calculate a list of log-likelihood from each data in the registered dataset and the corresponding given expected count histogram.

        Parameters
        ----------
        expectation_list : list of :py:class:`histpy.Histogram`
            List of expected count histograms

        Returns
        -------
        list of float
            List of Log-likelihood
        """

        return [_to_float(data.calc_log_likelihood(expectation)) for data, expectation in zip(self.dataset, expectation_list)]

    def calc_summed_exposure_map(self):
        """
        Calculate a list of exposure maps from the registered dataset.

        Returns
        -------
        :py:class:`histpy.Histogram`
        """

        return self._histogram_sum([ data.exposure_map for data in self.dataset ])

    def calc_summed_bkg_model(self, key):
        """
        Calculate the sum of histograms for a given background model in the registered dataset.

        Parameters
        ----------
        key: str
            Background model name

        Returns
        -------
        float
        """
        
        indexlist = self.dict_dataset_indexlist_for_bkg_models[key]

        return sum([self.dataset[i].summed_bkg_model(key) for i in indexlist])

    def calc_summed_T_product(self, dataspace_histogram_list):      # dataspace_histogram_list = ratio_list = d_i/E_i
        """
        For each data in the registered dataset, the product of the corresponding input histogram with the transpose of the response function is computed.
        Then, this method returns the sum of all of the products.

        Parameters
        ----------
        dataspace_histogram_list: list of :py:class:`histpy.Histogram`

        Returns
        -------
        :py:class:`histpy.Histogram`
        """

        return self._histogram_sum([data.calc_T_product(hist)
                                    for data, hist in zip(self.dataset, dataspace_histogram_list)])

    def calc_summed_bkg_model_product(self, key, dataspace_histogram_list):
        """
        For each data in the registered dataset, the product of the corresponding input histogram with the specified background model is computed.
        Then, this method returns the sum of all of the products.

        Parameters
        ----------
        key: str
            Background model name
        dataspace_histogram_list: list of :py:class:`histpy.Histogram`

        Returns
        -------
        flaot
        """

        indexlist = self.dict_dataset_indexlist_for_bkg_models[key]

        return sum(
            self.dataset[i].calc_bkg_model_product(key = key, dataspace_histogram = dataspace_histogram_list[i])
            for i in indexlist
        )

    @staticmethod
    def _histogram_sum(hlist):
        """
        Sum a list of Histograms.  If only one input, just return it.
        """
        if len(hlist) == 1:
            return hlist[0]
        else:
            result = hlist[0].copy()
            for h in hlist[1:]:
                result += h
            return result

    def save_histogram(self, filename, counter_name, histogram_key, only_final_result = False):

        # save last result
        self.results[-1][histogram_key].write(filename, name = 'result', overwrite = True)

        # save all results
        if not only_final_result:

            for result in self.results:

                counter = result[counter_name]

                result[histogram_key].write(filename, name = f'{counter_name}{counter}', overwrite = True)

    def save_results_as_fits(self, filename, counter_name, values_key_name_format, dicts_key_name_format, lists_key_name_format):

        hdu_list = []

        # primary HDU
        primary_hdu = fits.PrimaryHDU()

        hdu_list.append(primary_hdu)

        # counter
        col_counter = fits.Column(name=counter_name, array=[int(result[counter_name]) for result in self.results], format = 'K') #64bit integer

        # values
        for key, name, fits_format in values_key_name_format:

            col_value = fits.Column(name=key, array=[result[key] for result in self.results], format=fits_format)

            hdu = fits.BinTableHDU.from_columns([col_counter, col_value])

            hdu.name = name

            hdu_list.append(hdu)

        # dictionary
        for key, name, fits_format in dicts_key_name_format:

            dict_keys = list(self.results[0][key].keys())

            chunk_size = CHUNK_SIZE_FITS # when the number of columns >= 1000, the fits file may not be saved.
            for i_chunk, chunked_dict_keys in enumerate([dict_keys[i:i+chunk_size] for i in range(0, len(dict_keys), chunk_size)]):

                cols_dict = [fits.Column(name=dict_key, array=[result[key][dict_key] for result in self.results], format=fits_format) for dict_key in chunked_dict_keys]

                hdu = fits.BinTableHDU.from_columns([col_counter] + cols_dict)

                hdu.name = name

                if i_chunk != 0:
                    hdu.name = name + f"{i_chunk}"

                hdu_list.append(hdu)

        # list
        for key, name, fits_format in lists_key_name_format:

            cols_list = [fits.Column(name=f"{self.dataset[i].name}", array=[result[key][i] for result in self.results], format=fits_format) for i in range(len(self.dataset))]

            hdu = fits.BinTableHDU.from_columns([col_counter] + cols_list)

            hdu.name = name

            hdu_list.append(hdu)

        # write
        fits.HDUList(hdu_list).writeto(filename, overwrite=True)

    def _save_standard_results(self, counter_name, histogram_keys, fits_filename, 
                               values_key_name_format=None, dicts_key_name_format=None, lists_key_name_format=None):
        """
        Save standard results including histograms and FITS files.
        
        Parameters
        ----------
        counter_name : str
            Name of the counter (e.g., "iteration")
        histogram_keys : list of tuple
            List of (key, filename, only_final_result) for histograms to save.
        fits_filename : str
            Path to the FITS file.
        values_key_name_format : list of tuple, optional
            List of (key, name, fits_format) for single values to save in FITS.
        dicts_key_name_format : list of tuple, optional
            List of (key, name, fits_format) for dictionaries to save in FITS.
        lists_key_name_format : list of tuple, optional
            List of (key, name, fits_format) for lists to save in FITS.
        """
        # Save histograms
        for key, filename, only_final_result in histogram_keys:
            self.save_histogram(
                filename = filename,
                counter_name = counter_name,
                histogram_key = key,
                only_final_result = only_final_result
            )
        
        # Save FITS file (use default if not specified)
        self.save_results_as_fits(
            filename = fits_filename,
            counter_name = counter_name,
            values_key_name_format = values_key_name_format if values_key_name_format is not None else [],
            dicts_key_name_format = dicts_key_name_format if dicts_key_name_format is not None else [],
            lists_key_name_format = lists_key_name_format if lists_key_name_format is not None else []
        )
