"""
accelerator_base.py

Defines AcceleratorBase: the abstract base class for RL acceleration strategies.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass 
from typing import Union
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

@dataclass
class AcceleratorResult:
    """
    Return value of AcceleratorBase.compute().

    Parameters
    ----------
    model : Histogram
        Updated model after acceleration.
    dict_bkg_norm : dict
        Updated background normalization after acceleration.
    expectation_list : list or None
        If the accelerator computed expected counts internally
        (e.g. for a line search), they are returned here so that
        the calling algorithm can reuse them without recomputing.
        None if not computed.
    log_likelihood_list : list or None
        Same idea for log-likelihood values. None if not computed.
    extras : dict or None
        Algorithm-specific diagnostics to be stored in results,
        e.g. {"alpha": 1.5} for MonotonicitySafeAccelerator.
        None if not applicable (e.g. SQUAREM).
    """

    model               : Histogram
    dict_bkg_norm       : dict
    expectation_list    : Union[list, None] = None
    log_likelihood_list : Union[list, None] = None
    extras              : Union[dict, None] = None


class AcceleratorBase(ABC):
    """
    Abstract base class for RL acceleration strategies.

    n_em_steps_required : int
        Number of additional EM steps the accelerator needs.
        0 for simple step-size accelerators (e.g. MonotonicitySafe).
    """

    n_em_steps_required : int = 0

    def __init__(self, parameter):
        self._parameter = parameter

    @abstractmethod
    def compute(
        self,
        delta_model,
        dict_delta_bkg_norm : dict,
        model_before,
        bkg_norm_before     : dict,
        em_results          : list,
        dataset,
        mask,
    ) -> AcceleratorResult:
        """
        Parameters
        ----------
        delta_model : Histogram
            Δ for model 
        dict_delta_bkg_norm : dict[str, float]
            Δ for each background normalization
        model_before : Histogram
            Model at the start of this iteration (before extra EM steps).
        bkg_norm_before : dict[str, float]
            Background normalization at the start of this iteration.
        em_results : list of tuple
            Pre-computed extra EM results, each element:
            ``(model, dict_bkg_norm, expectation_list)``
            Length equals n_em_steps_required. Empty when 0.
        dataset : DataInterfaceCollection
        mask : Histogram or None

        Returns
        -------
        AcceleratorResult
        """
        raise NotImplementedError
