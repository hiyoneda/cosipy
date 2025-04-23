from cosipy.threeml import COSILike
from cosipy.response import BinnedThreeMLResponse
from threeML import Band, PointSource, Model, JointLikelihood, DataList, Parameter

from astropy import units as u

# Options for point sources
psr = BinnedThreeMlPointSourceResponse()

psr = BinnedThreeMlPointSourceResponse

# Option for extended sources
esr = BinnedThreeMLExtendedSourceResponse()

response = BinnedThreeMLResponse(point_source_response = psr,
                                 extended_source_response = esr)

# Set model
l = 93.
b = -53.

alpha = -1
beta = -3
xp = 450. * u.keV
piv = 500. * u.keV
K = 1 / u.cm / u.cm / u.s / u.keV

spectrum = Band()

spectrum.beta.min_value = -15.0

spectrum.alpha.value = alpha
spectrum.beta.value = beta
spectrum.xp.value = xp.value
spectrum.K.value = K.value
spectrum.piv.value = piv.value

spectrum.xp.unit = xp.unit
spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit

source = PointSource("source",                     # Name of source (arbitrary, but needs to be unique)
                     l = l,                        # Longitude (deg)
                     b = b,                        # Latitude (deg)
                     spectral_shape = spectrum)    # Spectral model

model = Model(source)                              # Model with single source. If we had multiple sources, we would do Model(source1, source2, ...)

# Optional: if you want to call get_log_like manually, then you also need to set the model manually
# 3ML does this internally during the fit though
cosi = COSILike('cosi', data, response, bkg)
plugins = DataList(cosi)
like = JointLikelihood(model, plugins)
like.fit()
