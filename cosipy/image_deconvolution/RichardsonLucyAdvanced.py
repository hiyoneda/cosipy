import numpy as np
import astropy.units as u
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from .RichardsonLucy import RichardsonLucy
from .build_accelerator import build_accelerator

from .response_weighting_filter import ResponseWeightingFilter 

from .constants import DEFAULT_STOPPING_THRESHOLD, DEFAULT_RESPONSE_WEIGHTING_INDEX

class RichardsonLucyAdvanced(RichardsonLucy):
    """
    A class for the RichardsonLucy algorithm. 
    The algorithm here is based on Knoedlseder+99, Knoedlseder+05, Siegert+20.
    
    An example of parameter is as follows.

    iteration_max: 100
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    acceleration:
        activate: True
        accel_factor_max: 10.0
    response_weighting:
        activate: True
        index: 0.5
    smoothing:
        activate: True
        FWHM:
            value: 2.0
            unit: "deg"
    stopping_criteria:
        statistics: "log-likelihood"
        threshold: 1e-2
    background_normalization_optimization:
        activate: True
        range: {"albedo": [0.9, 1.1]}
    save_results:
        activate: True
        directory: "./results"
        only_final_result: True
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        super().__init__(initial_model, dataset, mask, parameter)

        # acceleration
        self.do_acceleration = parameter.get('acceleration:activate', False)
        if self.do_acceleration == True:
            self.accelerator_parameter = parameter['acceleration']

        # response_weighting
        self.do_response_weighting = parameter.get('response_weighting:activate', False)
        if self.do_response_weighting:
            self.response_weighting_index = parameter.get('response_weighting:index', DEFAULT_RESPONSE_WEIGHTING_INDEX)

        # smoothing
        self.do_smoothing = parameter.get('smoothing:activate', False)
        if self.do_smoothing:
            self.smoothing_fwhm = parameter.get('smoothing:FWHM:value') * u.Unit(parameter.get('smoothing:FWHM:unit'))
            logger.info(f"Gaussian filter with FWHM of {self.smoothing_fwhm} will be applied to delta images ...")

        # stopping criteria
        self.stopping_criteria_statistics = parameter.get('stopping_criteria:statistics', "log-likelihood")
        self.stopping_criteria_threshold  = parameter.get('stopping_criteria:threshold', DEFAULT_STOPPING_THRESHOLD)

        if not self.stopping_criteria_statistics in ["log-likelihood"]:
            raise ValueError

    def initialization(self):
        """
        Initialize before running image deconvolution.
        
        This method sets up response weighting filter based on the exposure map.
        """

        super().initialization()

        # response-weighting filter
        if self.do_response_weighting:
            self.response_weighting_filter = ResponseWeightingFilter(self.summed_exposure_map, self.response_weighting_index)

        # build accelerator
        if self.do_acceleration:
            self.accelerator = build_accelerator(self.accelerator_parameter)

    def pre_processing(self):
        """
        pre-processing for each iteration
        """

        if self.iteration_count == 1:
            self.Estep()
            logger.info("The expected count histograms were calculated with the initial model map.")

    def processing_core(self):
        """
        Core processing for each iteration.
        """

        self.Mstep()
        # Note that Estep() is performed in self.post_processing().

    def Mstep(self):
        """
        Mstep
        """
        super().Mstep()

        # apply response_weighting_filter
        if self.do_response_weighting:
            self.delta_model = self.response_weighting_filter.apply(self.delta_model)

        # apply smoothing 
        if self.do_smoothing:
            self.delta_model = self.delta_model.smoothing(fwhm = self.smoothing_fwhm)

    def post_processing(self):
        """
        perform the acceleration of RL algirithm
        """

        if self.do_acceleration:

            # save state before extra EM steps
            model_before    = self.model.copy()
            bkg_norm_before = self.dict_bkg_norm.copy()

            # run extra EM steps required by the accelerator
            em_results = []
            for _ in range(self.accelerator.n_em_steps_required):
                self.Estep()
                self.Mstep()
                self.model += self.delta_model
                self._ensure_model_constraints()
                if self.do_bkg_norm_optimization:
                    for key in self.dict_bkg_norm:
                        self.dict_bkg_norm[key] += self.dict_delta_bkg_norm[key]
                    self._ensure_bkg_norm_range()
                em_results.append((
                    self.model.copy(),
                    self.dict_bkg_norm.copy(),
                    self.expectation_list.copy(),
                ))

            result = self.accelerator.compute(
                delta_model          = self.delta_model,
                dict_delta_bkg_norm  = self.dict_delta_bkg_norm,
                model_before         = model_before,
                bkg_norm_before      = bkg_norm_before,
                dataset              = self.dataset,
                mask                 = self.mask,
                em_results           = em_results,
            )

            self.model         = result.model
            self.dict_bkg_norm = result.dict_bkg_norm
            self._accel_result = result

        else:
            self.model += self.delta_model
            if self.do_bkg_norm_optimization:
                for key in self.dict_bkg_norm:
                    self.dict_bkg_norm[key] += self.dict_delta_bkg_norm[key]
            self._accel_result = None

        self._ensure_model_constraints()

        if self.do_bkg_norm_optimization:
            self._ensure_bkg_norm_range()

        # reuse expectation/likelihood if accelerator already computed them
        if self._accel_result is not None and self._accel_result.expectation_list is not None:
            self.expectation_list = self._accel_result.expectation_list
        else:
            self.Estep()
        logger.debug("Expected count histograms updated.")

        if self._accel_result is not None and self._accel_result.log_likelihood_list is not None:
            self.log_likelihood_list = self._accel_result.log_likelihood_list
        else:
            self.log_likelihood_list = self.dataset.calc_log_likelihood_list(self.expectation_list)
        logger.debug("Log-likelihood list updated.")


    def register_result(self):
        """
        The values below are stored at the end of each iteration.
        - iteration: iteration number
        - model: updated image
        - delta_model: delta map after M-step 
        - processed_delta_model: delta map after post-processing
        - accel_factor: acceleration parameter in RL algirithm
        - background_normalization: optimized background normalization
        - log-likelihood: log-likelihood
        """
        
        this_result = {"iteration": self.iteration_count, 
                       "model": self.model.copy(),
                       "background_normalization": self.dict_bkg_norm.copy(),
                       "log-likelihood": self.log_likelihood_list.copy()}

        if self._accel_result is not None and self._accel_result.extras is not None:
            this_result.update(self._accel_result.extras)

        for key in ["accel_factor"]:
            if key in this_result:
                logger.info(f"  {key}: {this_result[key]}")

        # register this_result in self.results
        self.results.append(this_result)

    def check_stopping_criteria(self):
        """
        If iteration_count is smaller than iteration_max, the iterative process will continue.

        Returns
        -------
        bool
        """

        if self.iteration_count == 1:
            return False
        elif self.iteration_count == self.iteration_max:
            return True

        if self.stopping_criteria_statistics == "log-likelihood":

            log_likelihood = np.sum(self.results[-1]["log-likelihood"])
            log_likelihood_before = np.sum(self.results[-2]["log-likelihood"])
            delta_log_likelihood = log_likelihood - log_likelihood_before

            logger.debug(f'Delta log-likelihood: {delta_log_likelihood}')

            if delta_log_likelihood < 0:
                logger.warning(f"Log-likelihood decreased {delta_log_likelihood}. Reconstruction may be unstable.")
                return False

            elif log_likelihood - log_likelihood_before < self.stopping_criteria_threshold:
                return True

        return False

    def finalization(self):
        """
        finalization after running the image deconvolution
        """

        if not self.save_results:
            return

        logger.info(f"Saving results in {self.save_results_directory}")

        values_key_name_format = []
        if "accel_factor" in self.results[0]:
            values_key_name_format.append(("accel_factor", "ACCEL_FACTOR", "D"))

        self._save_standard_results(
            counter_name           = "iteration",
            histogram_keys         = [("model", f"{self.save_results_directory}/model.hdf5", self.save_only_final_result)],
            fits_filename          = f"{self.save_results_directory}/results.fits",
            values_key_name_format = values_key_name_format,
            dicts_key_name_format  = [("background_normalization", "BKG_NORM", "D")],
            lists_key_name_format  = [("log-likelihood", "LOG-LIKELIHOOD", "D")],
        )
