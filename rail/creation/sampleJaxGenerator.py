import sys
sys.path.insert(0,'/global/homes/b/brycek/desc_pz_work/jax-nf/')
import pickle
import jax
import pandas as pd
import numpy as np
from jax_nf.real_nvp import RealNVP
from baseGenerator import baseGenerator
from flax import nn

from tensorflow_probability.python.internal.backend import jax as tf
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax

tfb = tfp.bijectors
tfd = tfp.distributions

"""
Sample Model daughter class
"""

@nn.module
def NeuralSplineCoupling(x, nunits, nbins=32):

  def bin_positions(x):
    x = np.reshape(x, [-1, nunits, nbins])
    return nn.activation.softmax(x, axis=-1) *  (2. - nbins * 1e-2) + 1e-2

  def slopes(x):
    x = np.reshape(x, [-1, nunits, nbins - 1])
    return nn.activation.softplus(x) + 1e-2

  # Let's have one more non stupid layer
  net = nn.leaky_relu(nn.Dense(x, 128))
  net = nn.leaky_relu(nn.Dense(net, 128))

  bin_widths = bin_positions(nn.Dense(net, nunits*nbins))
  bin_heights = bin_positions(nn.Dense(net, nunits*nbins))
  knot_slopes = slopes(nn.Dense(net, nunits*(nbins-1)))

  return tfp.bijectors.RationalQuadraticSpline(
            bin_widths=bin_widths,
            bin_heights=bin_heights,
            knot_slopes=knot_slopes)

@nn.module
def NeuralSplineFlowSampler(key, n_samples):

    d = 7

    spline1 = NeuralSplineCoupling.shared(name='spline1')
    spline2 = NeuralSplineCoupling.shared(name='spline2')

    # Computes the likelihood of these x
    chain = tfb.Chain([
        tfb.Scale(10),
        RealNVP(d//2, bijector_fn=spline1),
        tfb.Permute(np.arange(d)[::-1]),
        RealNVP(d//2, bijector_fn=spline2),
        tfb.Permute(np.arange(d)[::-1]),
        tfb.Scale(0.1)
    ])
    nvp = tfd.TransformedDistribution(
                tfd.Normal(0,1),
                bijector=chain,
                event_shape=(d,))

    return nvp.sample(n_samples, seed=key)

@nn.module
def NeuralSplineFlow(x):

    d = 7

    spline1 = NeuralSplineCoupling.shared(name='spline1')
    spline2 = NeuralSplineCoupling.shared(name='spline2')

    # Computes the likelihood of these x
    chain = tfb.Chain([
        tfb.Scale(10),
        RealNVP(d//2, bijector_fn=spline1),
        tfb.Permute(np.arange(d)[::-1]),
        RealNVP(d//2, bijector_fn=spline2),
        tfb.Permute(np.arange(d)[::-1]),
        tfb.Scale(0.1)
    ])

    nvp = tfd.TransformedDistribution(
                tfd.Normal(0,1),
                bijector=chain,
                event_shape=(d,))

    return nvp.log_prob(x)

class jaxNFGenerator(baseGenerator):

    def load_model(self, model_param_file):

        self.n_dim = 7

        with open('jax_nf_example_opt.pkl', 'rb') as g:
            model_dict = pickle.load(g)
        self.dataset_params = model_dict['dataset_params']
        self.opt_state_dict = model_dict['optimizer']
        self.sampler = nn.Model(NeuralSplineFlowSampler,
                                self.opt_state_dict['target']['params'])
        self.log_prob_return = nn.Model(NeuralSplineFlow,
                                        self.opt_state_dict['target']['params'])

    def return_sample(self, n_samples, seed=17):

        catalog_sample = self.sampler(jax.random.PRNGKey(seed), n_samples)
        sample_df = pd.DataFrame(catalog_sample, columns=['redshift', 'r_mag_normalized',
                                                          'umg', 'gmr', 'rmi', 'imz', 'zmy'])
        sample_df['r'] = (sample_df['r_mag_normalized'] * self.dataset_params['r_mag_stdev']) + \
                                       self.dataset_params['r_mag_mean']
        sample_df['g'] = sample_df['gmr'] + sample_df['r']
        sample_df['u'] = sample_df['umg'] + sample_df['g']
        sample_df['i'] = sample_df['r'] - sample_df['rmi']
        sample_df['z'] = sample_df['i'] - sample_df['imz']
        sample_df['y'] = sample_df['z'] - sample_df['zmy']

        return sample_df

    def return_log_prob(self, z_mag_datapoints):

        datapoints = []
        for dp in z_mag_datapoints:
            new_dp = []
            new_dp.append(dp[0])
            new_dp.append((dp[3] -
                           self.dataset_params['r_mag_mean'])/self.dataset_params['r_mag_stdev'])
            for idx in range(1, 6):
                new_dp.append(dp[idx] - dp[idx+1])
            datapoints.append(new_dp)

        return self.log_prob_return(datapoints)