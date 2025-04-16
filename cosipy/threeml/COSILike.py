from typing import Dict
from threeML import PluginPrototype, Parameter
from cosipy.statistics import UnbinnedLikelihood, PoissonLikelihood
from cosipy.interfaces import (DataInterface,
                               ThreeMLSourceResponseInterface,
                               BackgroundInterface,
                               LikelihoodInterface)

class COSILike(PluginPrototype):

    def __init__(self,
                 name,
                 data: DataInterface,
                 response: ThreeMLSourceResponseInterface,
                 bkg: BackgroundInterface,
                 likelihood = 'poisson'):
        """

        Parameters
        ----------
        name
        data
        response
        bkg
        likefun: str or LikelihoodInterface (Use at your own risk. make sure uses data, response and bkg)
        """

        self._name = name

        class ThreeMLBackgroundWrapper:
            """
            Translate background porameters to 3ml Parameter dict
            """

            def __init__(self, bkg: BackgroundInterface):
                self.bkg = bkg

            def set_parameters(self, parameters: Dict[str, Parameter]):
                # Translate self.bkg.set_parameters
                ...

            @property
            def parameters(self) -> Dict[str, Parameter]:
                ## Translate self.bkg.parameters
                ...

        self._bkg = ThreeMLBackgroundWrapper(bkg)
        self._response = response

        if isinstance(likelihood, LikelihoodInterface):
            # Use user's likelihood at their own risk
            self._like = likelihood
        elif likelihood == 'poisson':
            self._like = PoissonLikelihood(data, response, bkg)
        elif likelihood == 'unbinned':
            self._like = UnbinnedLikelihood(data, response, bkg)
        else:
            raise ValueError(f"Likelihood function \"{likelihood}\" not supported")

    @property
    def nuisance_parameters(self) -> Dict[str, Parameter]:
        return self._bkg.parameters

    def update_nuisance_parameters(self, new_nuisance_parameters: Dict[str, Parameter]):
        self._bkg.set_parameters(new_nuisance_parameters)

    def get_number_of_data_points(self) -> int:
        return self._like.nobservations

    def set_model(self, model):
        self._response.set_model(model)

    def get_log_like(self):
        return self._like.get_log_like()

    def inner_fit(self):
        """
        Required for 3ML fit.

        Maybe in the future use fast norm fit to minimize the background normalization
        """
        return self.get_log_like()
