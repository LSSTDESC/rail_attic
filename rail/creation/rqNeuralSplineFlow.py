"""
Implementation of a Normalizing Flow using Rational Quadratic Splines
source: https://arxiv.org/abs/1906.04032

The flow is built inside a Flax module:
https://flax.readthedocs.io/en/latest/notebooks/flax_guided_tour.html

using the RationalQuadraticSpline bijector from Tensorflow Probability:
https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RationalQuadraticSpline

and the RealNVP bijector, modified by Francois Lanusse:
original   - https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP
Francoi's  - https://github.com/EiffL/jax-nf/blob/master/jax_nf/real_nvp.py#L97-L115
literature - https://arxiv.org/abs/1605.08803
"""

import flax
from flax import nn
import jax
import jax.numpy as jnp

from tensorflow_probability.python.internal.backend import jax as tf
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

from .realNVP import RealNVP

@nn.module
def RQNeuralSplineFlow(hyperparams):
    """
    A Normalizing Flow using rational quadratic splines for the
    coupling layer. Returns a tensorflow probability transformed
    distribution wrapped in a flax module.

    Input is a dictionary of hyperparameters.
    The only required hyperparams are nfeatures and nlayers.
    """

    # unpack parameters from the hyperparameter dictionary
    nfeatures = hyperparams['nfeatures']
    nlayers = hyperparams['nlayers']

    # construct the chain of bijectors
    # each layer consists of a RQ Spline
    # followed by a permutation

    # first scale up by factor of 10
    # not sure why - this was done in Francois' original notebook
    # but it definitely helps performace ¯\_(ツ)_/¯
    chain = [tfb.Scale(10)]
    for i in range(nlayers):
        # each layer as a RQ spline
        spline = RQNeuralSplineCoupling.shared(name=f'spline{i+1}',
                                               hyperparams=hyperparams)
        chain.append(RealNVP(nfeatures//2, bijector_fn=spline))
        # and a permutation
        chain.append(tfb.Permute(jnp.roll(jnp.arange(nfeatures),1)))

    # undo the mysterious scaling
    chain.append(tfb.Scale(0.1))
    # build a bijector chain from all these layers
    chain = tfb.Chain(chain)

    # construct the flow
    flow = tfd.TransformedDistribution(
                    tfd.Normal(0,1),
                    bijector=chain,
                    event_shape=(nfeatures,))

    return flow


@nn.module
def RQNeuralSplineCoupling(x, nfeatures, hyperparams):
    """
    Rational Quadratic Neural Spline coupling Conor Durkan, Artur Bekasov, Iain Murray,
    George Papamakarios. Neural Spline Flows, 2019. http://arxiv.org/abs/1906.04032

    x is the input
    nfeatures is the number of features
    hyperparams is a dictionary that should contain the following:
        - nbins is the number of spline bins. Number of knots is nbins + 1
        - coupling_range sets the range of the splines. Outside of the range
            (-coupling_range,+coupling_range), the coupling is just the identity
        - coupling_dim is the dimension of hidden layers in the coupling function,
            which is a dense network
        - coupling_layers is the number of layers in the coupling function
        - coupling_activation is the activation function to use in the coupling
            function's hidden layers
    """

    # unpack the hyperparameter dictionary
    nbins = hyperparams['nbins']
    coupling_range = hyperparams['coupling_range']
    coupling_dim = hyperparams['coupling_dim']
    coupling_layers = hyperparams['coupling_layers']
    coupling_activation = hyperparams['coupling_activation']

    # coupling function: y = cf(x)
    # we use a dense feed forward network
    # from paper: theta_i = NN(x_1:d-1)
    y = x
    for _ in range(coupling_layers - 1):
        y = coupling_activation(nn.Dense(y, coupling_dim))
    y = nn.Dense(y, nfeatures * (3*nbins - 1))
    y = jnp.reshape(y, [-1, nfeatures, 3*nbins - 1])

    # pull out widths, heights, derivatives
    bin_widths, bin_heights, knot_slopes = jnp.split(y, [nbins, 2*nbins], axis=2)
    # apply transforms as listed in the paper
    bin_widths = 2 * coupling_range * nn.softmax(bin_widths, axis=-1)
    bin_heights = 2 * coupling_range * nn.softmax(bin_heights, axis=-1)
    knot_slopes = nn.softplus(knot_slopes)

    # return the rational quadratic spline coupling
    return tfb.RationalQuadraticSpline(bin_widths=bin_widths,
                                       bin_heights=bin_heights,
                                       knot_slopes=knot_slopes,
                                       range_min=-coupling_range)
