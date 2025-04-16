from cosipy.interfaces import BinnedLikelihoodInterface, UnbinnedLikelihoodInterface

__all__ = ['UnbinnedLikelihood',
           'PoissonLikelihood']

class UnbinnedLikelihood(UnbinnedLikelihoodInterface):
    ...

class PoissonLikelihood(BinnedLikelihoodInterface):
    ...

