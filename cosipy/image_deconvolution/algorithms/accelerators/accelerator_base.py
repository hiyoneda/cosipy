"""
accelerator_base.py
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass 
from typing import Union
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

@dataclass
class EMStepResult:
    """
    Holds the state and results of one EM step (or the initial "before" state).

    em_results passed to AcceleratorBase.compute() is a list of EMStepResult:
    - em_results[0] : "before" state
    - em_results[1] : result of 1st EM step
    - em_results[2] : result of 2nd EM step (e.g. SQUAREM)
    - ...
    """

    model                   : Histogram
    dict_bkg_norm           : dict
    expectation_list        : Union[list, None] = None
    source_expectation_list : Union[list, None] = None
    bkg_expectation_list    : Union[list, None] = None


@dataclass
class AcceleratorResult:
    """
    Return value of AcceleratorBase.compute().
    """

    model         : Histogram
    dict_bkg_norm : dict
    extras        : Union[dict, None] = None


class AcceleratorBase(ABC):

    n_em_steps_required : int = 1

    # Fields from AcceleratorResult.extras to be saved in results and logged.
    # Each entry is a (field_name, fits_format) tuple, e.g. [("accel_factor", "D")].
    # Algorithm classes (e.g. RichardsonLucyAdvanced) iterate over this list in
    # register_result and finalization, so accelerator-specific keys never need
    # to be hardcoded in the algorithm.
    logged_result_fields: list = []

    def __init__(self, parameter):
        self._parameter = parameter

    @abstractmethod
    def compute(
        self,
        em_results : list,
        dataset,
        mask,
    ) -> AcceleratorResult:
        raise NotImplementedError
