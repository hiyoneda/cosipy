"""
max_step_accelerator.py
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

    If accel_factor does not improve LH, falls back to accel_factor=1.
    accel_factor is stored in extras={"accel_factor": accel_factor}.

    algorithm: MaxStep
    accel_factor_max: 10.0
    """

    n_em_steps_required = 1

    def __init__(self, parameter):
        super().__init__(parameter)
        self.accel_factor_max = float(parameter.get("accel_factor_max", DEFAULT_ACCEL_FACTOR_MAX))
        logger.info(f"[MaxStepAccelerator] accel_factor_max={self.accel_factor_max}")

    def compute(
        self,
        em_results : list,
        dataset,
        mask,
    ) -> AcceleratorResult:

        before = em_results[0]
        after  = em_results[1]

        delta_model = after.model - before.model

        accel_factor = self._compute_accel_factor(delta_model, before.model, mask)

        if accel_factor > 1.0:
            new_model = before.model + delta_model * accel_factor
            new_dict_bkg_norm = after.dict_bkg_norm
#            new_dict_bkg_norm = {
#                key: before.dict_bkg_norm[key] + (after.dict_bkg_norm[key] - before.dict_bkg_norm[key]) * accel_factor
#                for key in before.dict_bkg_norm
#            }

            # LH check: reuse source expectation, recompute bkg only
            source_expectation_list = [
                src_before + (src_after - src_before) * accel_factor
                for src_before, src_after in zip(before.source_expectation_list, after.source_expectation_list)
            ]
            bkg_expectation_list = after.bkg_expectation_list
            expectation_list     = dataset.combine_expectation_list(source_expectation_list, bkg_expectation_list)

            ll_accel = np.sum(dataset.calc_log_likelihood_list(expectation_list))
            ll_after = np.sum(dataset.calc_log_likelihood_list(after.expectation_list))

            if ll_accel < ll_after:
                logger.debug(
                    f"[MaxStepAccelerator] accel_factor={accel_factor:.3f} did not improve LH "
                    f"({ll_accel:.6f} < {ll_after:.6f}). Falling back to accel_factor=1."
                )
                accel_factor      = 1.0
                new_model         = after.model
                new_dict_bkg_norm = after.dict_bkg_norm

        else:
            new_model         = after.model
            new_dict_bkg_norm = after.dict_bkg_norm

        logger.debug(f"[MaxStepAccelerator] accel_factor={accel_factor}")

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
