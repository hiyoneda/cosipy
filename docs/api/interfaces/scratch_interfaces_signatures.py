from typing import Dict, Any

from astromodels import Model, Parameter

from cosipy.threeml import COSILike
from cosipy.interfaces import BinnedDataInterface, ThreeMLBinnedBackgroundInterface, ThreeMLBinnedSourceResponseInterface
from histpy import Axis,Axes,Histogram
import numpy as np
from scipy.stats import norm, uniform

from matplotlib import pyplot as plt

toy_axis = Axis(np.linspace(-5, 5))

class ToyData(BinnedDataInterface):
    # Random data. Normal signal on opt of uniform bkg

    def __init__(self):
        self._data = Histogram(toy_axis)

        # Signal
        self._data.fill(norm.rvs(size = 1000))

        # Bkg
        self._data.fill(uniform.rvs(-5,10, size=1000))

    @property
    def data(self) -> Histogram:
        return self._data


class ToyBkg(ThreeMLBinnedBackgroundInterface):
    def __init__(self):
        self._unit_expectation = Histogram(toy_axis)
        self._unit_expectation[:] = 1/self._unit_expectation.nbins
        self._norm = None
        self._threeml_parameters = {}

    def set_parameters(self, norm) -> None:
        self._norm = norm

    @property
    def parameters(self) -> Dict[str, Any]:
        return {'norm':self._norm}

    def expectation(self, axes:Axes)->Histogram:

        if axes != self._unit_expectation.axes:
            raise ValueError("Wrong axes. I have fixed axes.")

        if self._norm is None:
            raise RuntimeError("Set norm parameter first")

        # In case it changed
        self.set_parameters(norm= self._threeml_parameters['norm'].value)

        return self._unit_expectation*self._norm

    @property
    def threeml_parameters(self) -> Dict[str, Parameter]:
        return self._threeml_parameters

    def set_threeml_parameters(self, norm: Parameter, **kwargs):
        self._threeml_parameters['norm'] = norm
        self.set_parameters(norm.value)

class ToySourceResponse(ThreeMLBinnedSourceResponseInterface):

    def __init__(self):
        self._model = None
        self._unit_expectation = Histogram(toy_axis,
                                           contents = np.diff(norm.cdf(toy_axis.edges)))

    def set_model(self, model: Model):
        self._model = model

    def expectation(self, axes:Axes)->Histogram:
        if axes != self._unit_expectation.axes:
            raise ValueError("Wrong axes. I have fixed axes.")

        if self._model is None:
            raise RuntimeError("Set model first")

        sources = self._model.sources

        if len(sources) == 0:
            flux = 0.
        else:
            flux = self._model.sources['source'].spectrum.main.shape.k.value

        return self._unit_expectation*flux

data = ToyData()
bkg = ToyBkg()
bkg.set_threeml_parameters(norm = Parameter('norm', 1))

#bkg = None # Uncomment for not bkg fit

response = ToySourceResponse()

## 3Ml model
## We'll just use the K value in u.cm / u.cm / u.s / u.keV
from threeML import Constant, PointSource, Model, JointLikelihood, DataList
spectrum = Constant()
spectrum.k.value = 1
source = PointSource("source", # arbitrary, but needs to be unique
                     l = 0, b = 0, # Doesn't matter
                     spectral_shape = spectrum)

model = Model(source)

model = Model() # Uncomment for bkg-only hypothesis

cosi = COSILike('cosi', data, response, bkg)

plugins = DataList(cosi)

like = JointLikelihood(model, plugins, verbose = True)

like.fit()

fig,ax = plt.subplots()
data.data.plot(ax)
expectation = response.expectation(data.data.axes)
if bkg is not None:
    expectation = expectation + bkg.expectation(data.data.axes)
expectation.plot(ax)
plt.show()