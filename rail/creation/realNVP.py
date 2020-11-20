# From Francois Lanusse
# Modified version of https://github.com/tensorflow/probability/blob/v0.10.0/tensorflow_probability/python/bijectors/real_nvp.py
# Includes minor modifications for compatibility with JAX, original license below
#===============================================================================
# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Real NVP bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
import jax.numpy as jnp

from tensorflow_probability.python.internal.backend import jax as tf
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax

bijector_lib = tfp.bijectors.bijector

__all__ = [
    'RealNVP'
]

class RealNVP(bijector_lib.Bijector):
  """RealNVP 'affine coupling layer' for vector-valued events.
  Real NVP models a normalizing flow on a `D`-dimensional distribution via a
  single `D-d`-dimensional conditional distribution [(Dinh et al., 2017)][1]:
  `y[d:D] = x[d:D] * tf.exp(log_scale_fn(x[0:d])) + shift_fn(x[0:d])`
  `y[0:d] = x[0:d]`
  The last `D-d` units are scaled and shifted based on the first `d` units only,
  while the first `d` units are 'masked' and left unchanged. Real NVP's
  `shift_and_log_scale_fn` computes vector-valued quantities. For
  scale-and-shift transforms that do not depend on any masked units, i.e.
  `d=0`, use the `tfb.Affine` bijector with learned parameters instead.
  Masking is currently only supported for base distributions with
  `event_ndims=1`. For more sophisticated masking schemes like checkerboard or
  channel-wise masking [(Papamakarios et al., 2016)[4], use the `tfb.Permute`
  bijector to re-order desired masked units into the first `d` units. For base
  distributions with `event_ndims > 1`, use the `tfb.Reshape` bijector to
  flatten the event shape.
  Recall that the MAF bijector [(Papamakarios et al., 2016)][4] implements a
  normalizing flow via an autoregressive transformation. MAF and IAF have
  opposite computational tradeoffs - MAF can train all units in parallel but
  must sample units sequentially, while IAF must train units sequentially but
  can sample in parallel. In contrast, Real NVP can compute both forward and
  inverse computations in parallel. However, the lack of an autoregressive
  transformations makes it less expressive on a per-bijector basis.
  A 'valid' `shift_and_log_scale_fn` must compute each `shift` (aka `loc` or
  'mu' in [Papamakarios et al. (2016)][4]) and `log(scale)` (aka 'alpha' in
  [Papamakarios et al. (2016)][4]) such that each are broadcastable with the
  arguments to `forward` and `inverse`, i.e., such that the calculations in
  `forward`, `inverse` [below] are possible. For convenience,
  `real_nvp_default_nvp` is offered as a possible `shift_and_log_scale_fn`
  function.
  NICE [(Dinh et al., 2014)][2] is a special case of the Real NVP bijector
  which discards the scale transformation, resulting in a constant-time
  inverse-log-determinant-Jacobian. To use a NICE bijector instead of Real
  NVP, `shift_and_log_scale_fn` should return `(shift, None)`, and
  `is_constant_jacobian` should be set to `True` in the `RealNVP` constructor.
  Calling `real_nvp_default_template` with `shift_only=True` returns one such
  NICE-compatible `shift_and_log_scale_fn`.
  The `bijector_fn` argument allows specifying a more general coupling relation,
  such as the LSTM-inspired activation from [5], or Neural Spline Flow [6].
  Caching: the scalar input depth `D` of the base distribution is not known at
  construction time. The first call to any of `forward(x)`, `inverse(x)`,
  `inverse_log_det_jacobian(x)`, or `forward_log_det_jacobian(x)` memoizes
  `D`, which is re-used in subsequent calls. This shape must be known prior to
  graph execution (which is the case if using tf.layers).
  #### Examples
  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors
  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution. (However, any continuous distribution would work.) E.g.,
  nvp = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(loc=[0., 0., 0.]),
      bijector=tfb.RealNVP(
          num_masked=2,
          shift_and_log_scale_fn=tfb.real_nvp_default_template(
              hidden_layers=[512, 512])))
  x = nvp.sample()
  nvp.log_prob(x)
  nvp.log_prob(0.)
  ```
  For more examples, see [Jang (2018)][3].
  #### References
  [1]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803
  [2]: Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear
       Independent Components Estimation. _arXiv preprint arXiv:1410.8516_,
       2014. https://arxiv.org/abs/1410.8516
  [3]: Eric Jang. Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows.
       _Technical Report_, 2018. http://blog.evjang.com/2018/01/nf2.html
  [4]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  [5]: Diederik P Kingma, Tim Salimans, Max Welling. Improving Variational
       Inference with Inverse Autoregressive Flow. In _Neural Information
       Processing Systems_, 2016. https://arxiv.org/abs/1606.04934
  [6]: Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
       Spline Flows, 2019. http://arxiv.org/abs/1906.04032
  """

  def __init__(self,
               num_masked=None,
               fraction_masked=None,
               shift_and_log_scale_fn=None,
               bijector_fn=None,
               is_constant_jacobian=False,
               validate_args=False,
               name=None):
    """Creates the Real NVP or NICE bijector.
    Args:
      num_masked: Python `int`, indicating the number of units of the
        event that should should be masked. Must be in the closed interval
        `[0, D-1]`, where `D` is the event size of the base distribution.
        If the value is negative, then the last `d` units of the event are
        masked instead. Must be `None` if `fraction_masked` is defined.
      fraction_masked: Python `float`, indicating the number of units of the
        event that should should be masked. Must be in the closed interval
        `[-1, 1]`, and the value represents the fraction of the values to be
        masked. The final number of values to be masked will be the input size
        times the fraction, rounded to the the nearest integer towards zero.
        If negative, then the last fraction of units are masked instead. Must
        be `None` if `num_masked` is defined.
      shift_and_log_scale_fn: Python `callable` which computes `shift` and
        `log_scale` from both the forward domain (`x`) and the inverse domain
        (`y`). Calculation must respect the 'autoregressive property' (see class
        docstring). Suggested default
        `masked_autoregressive_default_template(hidden_layers=...)`.
        Typically the function contains `tf.Variables` and is wrapped using
        `tf.make_template`. Returning `None` for either (both) `shift`,
        `log_scale` is equivalent to (but more efficient than) returning zero.
      bijector_fn: Python `callable` which returns a `tfb.Bijector` which
        transforms the last `D-d` unit with the signature `(masked_units_tensor,
        output_units, **condition_kwargs) -> bijector`. The bijector must
        operate on scalar or vector events and must not alter the rank of its
        input.
      is_constant_jacobian: Python `bool`. Default: `False`. When `True` the
        implementation assumes `log_scale` does not depend on the forward domain
        (`x`) or inverse domain (`y`) values. (No validation is made;
        `is_constant_jacobian=False` is always safe but possibly computationally
        inefficient.)
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object.
    Raises:
      ValueError: If both or none of `shift_and_log_scale_fn` and `bijector_fn`
          are specified.
    """
    parameters = dict(locals())
    name = name or 'real_nvp'
    with tf.name_scope(name) as name:
      # At construction time, we don't know input_depth.
      self._input_depth = None
      if num_masked is not None and fraction_masked is not None:
        raise ValueError('Exactly one of `num_masked` and '
                         '`fraction_masked` should be specified.')

      if num_masked is not None:
        if int(num_masked) != num_masked:
          raise TypeError('`num_masked` must be an integer. Got: {} of type {}'
                          ''.format(num_masked, type(num_masked)))
        self._num_masked = int(num_masked)
        self._fraction_masked = None
        self._reverse_mask = self._num_masked < 0
      else:
        if not jnp.issubdtype(type(fraction_masked), jnp.floating):
          raise TypeError('`fraction_masked` must be a float. Got: {} of type '
                          '{}'.format(fraction_masked, type(fraction_masked)))
        if jnp.abs(fraction_masked) >= 1.:
          raise ValueError(
              '`fraction_masked` must be in (-1, 1), but is {}.'.format(
                  fraction_masked))
        self._num_masked = None
        self._fraction_masked = float(fraction_masked)
        self._reverse_mask = self._fraction_masked < 0

      if shift_and_log_scale_fn is not None and bijector_fn is not None:
        raise ValueError('Exactly one of `shift_and_log_scale_fn` and '
                         '`bijector_fn` should be specified.')

      if shift_and_log_scale_fn:
        def _bijector_fn(x0, input_depth, **condition_kwargs):
          shift, log_scale = shift_and_log_scale_fn(x0, input_depth,
                                                    **condition_kwargs)
          return affine_scalar.AffineScalar(shift=shift, log_scale=log_scale)

        bijector_fn = _bijector_fn
      #
      # if validate_args:
      #   bijector_fn = _validate_bijector_fn(bijector_fn)

      # Still do this assignment for variable tracking.
      self._shift_and_log_scale_fn = shift_and_log_scale_fn
      self._bijector_fn = bijector_fn

      super(RealNVP, self).__init__(
          forward_min_event_ndims=1,
          is_constant_jacobian=is_constant_jacobian,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @property
  def _masked_size(self):
    masked_size = (
        self._num_masked if self._num_masked is not None else int(
            jnp.round(self._input_depth * self._fraction_masked)))
    return masked_size

  def _cache_input_depth(self, x):
    if self._input_depth is None:
      self._input_depth = x.shape[-1]
      if self._input_depth is None:
        raise NotImplementedError(
            'Rightmost dimension must be known prior to graph execution.')

      if abs(self._masked_size) >= self._input_depth:
        raise ValueError(
            'Number of masked units {} must be smaller than the event size {}.'
            .format(self._masked_size, self._input_depth))

  def _bijector_input_units(self):
    return self._input_depth - abs(self._masked_size)

  def _forward(self, x, **condition_kwargs):
    self._cache_input_depth(x)

    x0, x1 = x[..., :self._masked_size], x[..., self._masked_size:]

    if self._reverse_mask:
      x0, x1 = x1, x0

    y1 = self._bijector_fn(x0, self._bijector_input_units(),
                           **condition_kwargs).forward(x1)

    if self._reverse_mask:
      y1, x0 = x0, y1

    y = tf.concat([x0, y1], axis=-1)
    return y

  def _inverse(self, y, **condition_kwargs):
    self._cache_input_depth(y)

    y0, y1 = y[..., :self._masked_size], y[..., self._masked_size:]

    if self._reverse_mask:
      y0, y1 = y1, y0

    x1 = self._bijector_fn(y0, self._bijector_input_units(),
                           **condition_kwargs).inverse(y1)

    if self._reverse_mask:
      x1, y0 = y0, x1

    x = tf.concat([y0, x1], axis=-1)
    return x

  def _forward_log_det_jacobian(self, x, **condition_kwargs):
    self._cache_input_depth(x)

    x0, x1 = x[..., :self._masked_size], x[..., self._masked_size:]

    if self._reverse_mask:
      x0, x1 = x1, x0

    return self._bijector_fn(x0, self._bijector_input_units(),
                             **condition_kwargs).forward_log_det_jacobian(
                                 x1, event_ndims=1)

  def _inverse_log_det_jacobian(self, y, **condition_kwargs):
    self._cache_input_depth(y)

    y0, y1 = y[..., :self._masked_size], y[..., self._masked_size:]

    if self._reverse_mask:
      y0, y1 = y1, y0

    return self._bijector_fn(y0, self._bijector_input_units(),
                             **condition_kwargs).inverse_log_det_jacobian(
                                 y1, event_ndims=1)