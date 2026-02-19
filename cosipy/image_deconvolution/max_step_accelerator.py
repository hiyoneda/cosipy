"""
max_step_accelerator.py

Monotonicity-safe acceleration strategy (Knoedlseder+99, Knoedlseder+05, Siegert+20).
"""

import numpy as np
import logging

from .accelerator_base import AcceleratorBase, AcceleratorResult

logger = logging.getLogger(__name__)

DEFAULT_ACCEL_FACTOR_MAX = 10.0


class MaxStepAccelerator(AcceleratorBase):
    """
    Finds the largest accel_factor in [1, accel_factor_max] such that

        model_before + accel_factor * delta_model >= 0   (element-wise)

    n_em_steps_required = 0: em_results is always empty.
    accel_factor is stored in extras={"accel_factor": accel_factor}.

    An example of parameter is as follows.

    algorithm: MaxStep
    accel_factor_max: 10.0
    """

    n_em_steps_required = 0

    def __init__(self, parameter):
        super().__init__(parameter)
        self.accel_factor_max = float(parameter.get("accel_factor_max", DEFAULT_ACCEL_FACTOR_MAX))
        logger.info(f"[MonotonicitySafeAccelerator] accel_factor_max={self.accel_factor_max}")

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

        accel_factor = self._compute_accel_factor(delta_model, model_before, mask)

        new_model = model_before + delta_model * accel_factor
        new_dict_bkg_norm = {
            key: bkg_norm_before[key] + dict_delta_bkg_norm[key] #* accel_factor
            for key in bkg_norm_before
        }

        logger.debug(f"[MonotonicitySafeAccelerator] accel_factor={accel_factor}")

        return AcceleratorResult(
            model         = new_model,
            dict_bkg_norm = new_dict_bkg_norm,
            extras        = {"accel_factor": accel_factor},
        )

    def _compute_accel_factor(self, delta_model, model, mask) -> float:

        diff = -1 * (model / delta_model).contents

        diff[(diff <= 0) | (delta_model.contents == 0)] = np.inf

        if mask is not None:
            diff[np.invert(mask.contents)] = np.inf

        accel_factor = min(np.min(diff), self.accel_factor_max)

        if accel_factor < 1.0:
            accel_factor = 1.0

        return accel_factor
