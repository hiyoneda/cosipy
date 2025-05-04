from typing import Dict

from cosipy.interfaces.likelihood_interface import LikelihoodInterface
from threeML import PluginPrototype, Parameter

__all__ = ["ThreeMLPluginInterface"]

class ThreeMLPluginInterface(PluginPrototype):

    def __init__(self,
                 name: str, likelihood: LikelihoodInterface):
        """

        Parameters
        ----------
        name
        likefun: str or LikelihoodInterface (Use at your own risk. make sure it knows about the input data, response and bkg)
        """

        # PluginPrototype.__init__ does the following:
        # Sets _name = name
        # Sets _tag = None
        # Set self._nuisance_parameters, which we do not use because
        # we're overriding nuisance_parameters() and update_nuisance_parameters()
        super().__init__(name, {})

        self._like = likelihood

    @property
    def nuisance_parameters(self) -> Dict[str, Parameter]:
        # Adds plugin name, required by 3ML code
        # See https://github.com/threeML/threeML/blob/7a16580d9d5ed57166e3b1eec3d4fccd3eeef1eb/threeML/classicMLE/joint_likelihood.py#L131
        if self._like.bkg is None:
            return {}
        else:
            return {self._name + "_" + l:p for l,p in self._like.bkg.threeml_parameters.items()}

    def update_nuisance_parameters(self, new_nuisance_parameters: Dict[str, Parameter]):
        # Remove plugin name. Opposite of the nuisance_parameters property
        if self._like.bkg is not None:
            new_nuisance_parameters = {l[len(self._name)+1:]:p for l,p in new_nuisance_parameters.items()}
            self._like.bkg.set_threeml_parameters(**new_nuisance_parameters)

    def get_number_of_data_points(self) -> int:
        return self._like.nobservations

    def set_model(self, model):
        self._like.response.set_model(model)

    def get_log_like(self):
        return self._like.get_log_like()

    def inner_fit(self):
        """
        Required for 3ML fit.

        Maybe in the future use fast norm fit to minimize the background normalization
        """
        return self.get_log_like()
