from typing import Dict, Any

from astromodels import Model, Parameter

from cosipy.threeml import COSILike
from cosipy.interfaces import BinnedDataInterface, ThreeMLBinnedBackgroundInterface, ThreeMLBinnedSourceResponseInterface
from histpy import Axis,Histogram
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
        self._norm = 1
        self._threeml_parameters = {'bkg_norm': Parameter('bkg_norm', self._norm)}

    def set_parameters(self, bkg_norm) -> None:
        self._norm = bkg_norm

    @property
    def parameters(self) -> Dict[str, Any]:
        return {'bkg_norm':self._norm}

    @property
    def expectation(self)->Histogram:
        return self._norm * self._unit_expectation

    @property
    def threeml_parameters(self) -> Dict[str, Parameter]:
        return self._threeml_parameters

    def set_threeml_parameters(self, bkg_norm: Parameter, **kwargs):
        self._threeml_parameters['bkg_norm'] = bkg_norm
        self.set_parameters(bkg_norm = bkg_norm.value)

class ToySourceResponse(ThreeMLBinnedSourceResponseInterface):

    def __init__(self):
        self._model = None
        self._unit_expectation = Histogram(toy_axis,
                                           contents = np.diff(norm.cdf(toy_axis.edges)))

    def set_model(self, model: Model):
        self._flux = model.sources['source'].spectrum.main.shape.k

    @property
    def expectation(self)->Histogram:
        print(self._flux.value)
        return self._unit_expectation*self._flux.value

data = ToyData()
bkg = ToyBkg()
bkg.set_threeml_parameters(bkg_norm = Parameter('bkg_norm', 1000,
                                                min_value=0, max_value = 100000,
                                                delta = 0.01))
response = ToySourceResponse()

## 3Ml model
## We'll just use the K value in u.cm / u.cm / u.s / u.keV
from threeML import Constant, PointSource, Model, JointLikelihood, DataList
spectrum = Constant(k = 1000)
spectrum.k.min_value = 0
spectrum.k.max_value = 100000
spectrum.k.delta = 1
source = PointSource("source", # arbitrary, but needs to be unique
                     l = 0, b = 0, # Doesn't matter
                     spectral_shape = spectrum)

model = Model(source)

cosi = COSILike('cosi', data, response, bkg)

fig,ax = plt.subplots()
cosi.set_model(model)
data.data.plot(ax)
(bkg.expectation + response.expectation).plot(ax)
plt.show()

plugins = DataList(cosi)

like = JointLikelihood(model, plugins, verbose = True)

like.fit()
