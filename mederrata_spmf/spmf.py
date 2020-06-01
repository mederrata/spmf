#!/usr/bin/env python3
"""Sparse probabilistic PCA using the horseshoe
See main() for usage
Note that you currently have to babysit the optimization a bit
"""

from itertools import cycle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow_probability import distributions as tfd

from mederrata_spmf.model import BayesianModel
from mederrata_spmf.distributions import SqrtInverseGamma
from mederrata_spmf.util import (
    build_trainable_InverseGamma_dist, build_trainable_normal_dist,
    run_chain, clip_gradients, build_surrogate_posterior,
    fit_surrogate_posterior)

# import tensorflow_transform as tft


tfb = tfp.bijectors


class PoissonMatrixFactorization(BayesianModel):
    """Sparse (horseshoe) poisson matrix factorization
    Arguments:
        object {[type]} -- [description]
    """
    bijectors = None
    var_list = []
    s_tau_scale = 1
    def encoder_function(self, x): return tf.math.log(x+1.)
    def decoder_function(self, x): return tf.math.exp(x) - 1.

    def __init__(
            self, data, data_transform_fn=None, latent_dim=None,
            u_tau_scale=0.01, s_tau_scale=1., symmetry_breaking_decay=0.5,
            strategy=None, encoder_function=None, decoder_function=None,
            scale_rates=True, with_s=True, with_w=True,
            dtype=tf.float64, **kwargs):
        """Instantiate PMF object

        Arguments:
            data {[type]} -- a BatchDataset object that
                             we will iterate for training

        Keyword Arguments:
            data_transform_fn {[type]} -- Not currently used,
                but intended to allow for specification
                of a preprocessing function (default: {None})
            latent_dim {[type]} -- P (default: {None})
            u_tau_scale {[type]} -- Global shrinkage scale on u (default: {1.})
            s_tau_scale {int} -- Global shrinkage scale on s (default: {1})
            symmetry_breaking_decay {float} -- Decay factor along dimensions
                                                on u (default: {0.5})
            strategy {[type]} -- For multi-GPU (default: {None})
            decoder_function {[type]} -- f(x) (default: {None})
            encoder_function {[type]} -- g(x) (default: {None})
            scale_rates {bool} -- Scale the rates by the mean (default: {True})
            with_s {bool} -- [description] (default: {True})
            with_w {bool} -- [description] (default: {True})
            dtype {[type]} -- [description] (default: {tf.float64})
        """

        super(PoissonMatrixFactorization, self).__init__(
            data, data_transform_fn, strategy=strategy, dtype=dtype)
        if encoder_function is not None:
            self.encoder_function = encoder_function
        if decoder_function is not None:
            self.decoder_function = decoder_function
        self.dtype = dtype
        self.symmetry_breaking_decay = symmetry_breaking_decay
        self.with_s = with_s
        self.with_w = with_w

        record = next(iter(data))
        indices = record['indices']
        data = record['data']
        self.norm_factor = 1.
        if scale_rates:
            self.norm_factor = tf.reduce_mean(
                tf.cast(data, self.dtype), axis=0, keepdims=True)
        if 'normalization' in record.keys():
            norm = record['normalization']
        data = tf.cast(data, self.dtype)
        self.feature_dim = data.shape[-1]
        self.scale_rates = scale_rates

        self.latent_dim = self.feature_dim if (
            latent_dim) is None else latent_dim

        self.u_tau_scale = u_tau_scale
        self.s_tau_scale = s_tau_scale

        self.create_distributions()
        print(
            f"Feature dim: {self.feature_dim} -> Latent dim {self.latent_dim}")

    def log_likelihood_components(
            self, s, u, v, w,  data=None, *args, **kwargs):
        weights = s/tf.reduce_sum(s, axis=-2)[
            ..., tf.newaxis, :]
        weights_1 = tf.expand_dims(weights[..., 0, :], -1)
        weights_2 = tf.expand_dims(weights[..., 1, :], -1)

        encoding = weights_1*u
        log_likes = []

        if not isinstance(data, BatchDataset):
            data = [data]

        for record in data:
            indices = record['indices']
            batch = record['data']
            batch = tf.cast(batch, self.dtype)
            z_batch = tf.matmul(
                self.encoder_function(batch),
                encoding
            )
            L = len(weights_2.shape)
            trans = tuple(list(range(L-2)) + [L-1, L-2])
            weights_2 = tf.transpose(weights_2, trans)
            rate = self.decoder_function(tf.matmul(z_batch, v)) + weights_2*w
            if 'normalization' in record.keys():
                multiplier = record['normalization'][..., tf.newaxis]
                for _ in range(len(rate.shape)-len(multiplier.shape)):
                    multiplier = multiplier[tf.newaxis, ...]
                rate *= tf.cast(multiplier, self.dtype)
            rate *= self.norm_factor
            rv_poisson = tfd.Poisson(rate=rate)
            log_likes += [rv_poisson.log_prob(batch)]
        combined = tf.concat(log_likes, axis=-1)
        return combined

    # @tf.function
    def log_likelihood(
            self, s, u, v, w,  data=None, *args, **kwargs):
        ll = self.log_likelihood_components(
            s, u, v, w, data=data, *args, **kwargs)

        reduce_dim = len(s.shape) - 2
        if reduce_dim > 0:
            ll = tf.reduce_sum(ll, -np.arange(reduce_dim)-1)

        return ll

    def create_distributions(self):
        """Create distribution objects

        """
        self.bijectors = {
            'u': tfb.Softplus(),
            'v': tfb.Softplus(),
            'u_eta': tfb.Softplus(),
            'u_tau': tfb.Softplus(),
            's': tfb.Softplus(),
            's_eta': tfb.Softplus(),
            's_tau': tfb.Softplus(),
            'w': tfb.Softplus()
        }
        symmetry_breaking_decay = self.symmetry_breaking_decay**tf.cast(
            tf.range(self.latent_dim), self.dtype)[tf.newaxis, ...]

        distribution_dict = {
            'u': lambda u_eta, u_tau: tfd.Independent(
                tfd.HalfNormal(
                    scale=u_eta*u_tau*symmetry_breaking_decay
                ), reinterpreted_batch_ndims=2
            ),
            'w': tfd.Independent(
                tfd.HalfNormal(
                    scale=10.*tf.ones(
                        (1, self.feature_dim), dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=2
            ),
            'u_eta': tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros(
                        (self.feature_dim, self.latent_dim),
                        dtype=self.dtype),
                    scale=tf.ones(
                        (self.feature_dim, self.latent_dim),
                        dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            ),
            'u_tau': tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros(
                        (1, self.latent_dim),
                        dtype=self.dtype),
                    scale=tf.ones(
                        (1, self.latent_dim),
                        dtype=self.dtype)*self.u_tau_scale
                ), reinterpreted_batch_ndims=2
            ),
            'v': tfd.Independent(
                tfd.HalfNormal(
                    scale=0.5 *
                    tf.ones((self.latent_dim, self.feature_dim),
                            dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            )}
        if self.with_s:
            distribution_dict['s'] = lambda s_eta, s_tau: tfd.Independent(
                tfd.HalfNormal(
                    scale=s_eta*s_tau
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['s_eta'] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros(
                        (2, self.feature_dim),
                        dtype=self.dtype),
                    scale=tf.ones(
                        (2, self.feature_dim),
                        dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['s_tau'] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros((1, self.feature_dim), dtype=self.dtype),
                    scale=tf.ones((1, self.feature_dim),
                                  dtype=self.dtype)*self.s_tau_scale
                ), reinterpreted_batch_ndims=2
            )

        self.bijectors['u_eta_a'] = tfb.Softplus()
        self.bijectors['u_tau_a'] = tfb.Softplus()
        if self.with_s:
            self.bijectors['s_eta_a'] = tfb.Softplus()
            self.bijectors['s_tau_a'] = tfb.Softplus()

        distribution_dict['u_eta'] = lambda u_eta_a: tfd.Independent(
            SqrtInverseGamma(
                concentration=0.5*tf.ones(
                    (self.feature_dim, self.latent_dim),
                    dtype=self.dtype
                ),
                scale=1.0/u_eta_a
            ), reinterpreted_batch_ndims=2
        )
        distribution_dict['u_eta_a'] = tfd.Independent(
            tfd.InverseGamma(
                concentration=0.5*tf.ones(
                    (self.feature_dim, self.latent_dim),
                    dtype=self.dtype
                ),
                scale=tf.ones(
                    (self.feature_dim, self.latent_dim),
                    dtype=self.dtype)
            ), reinterpreted_batch_ndims=2
        )
        distribution_dict['u_tau'] = lambda u_tau_a: tfd.Independent(
            SqrtInverseGamma(
                concentration=0.5*tf.ones(
                    (1, self.latent_dim),
                    dtype=self.dtype
                ),
                scale=1.0/u_tau_a
            ), reinterpreted_batch_ndims=2
        )
        distribution_dict['u_tau_a'] = tfd.Independent(
            tfd.InverseGamma(
                concentration=0.5*tf.ones(
                    (1, self.latent_dim), dtype=self.dtype
                ),
                scale=tf.ones(
                    (1, self.latent_dim), dtype=self.dtype
                )/self.u_tau_scale**2
            ), reinterpreted_batch_ndims=2
        )

        distribution_dict['s_eta'] = lambda s_eta_a: tfd.Independent(
            SqrtInverseGamma(
                concentration=0.5*tf.ones(
                    (2, self.feature_dim),
                    dtype=self.dtype
                ),
                scale=1.0/s_eta_a
            ), reinterpreted_batch_ndims=2
        )
        if self.with_s:
            distribution_dict['s_eta_a'] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5*tf.ones(
                        (2, self.feature_dim), dtype=self.dtype
                    ),
                    scale=tf.ones((2, self.feature_dim), dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['s_tau'] = lambda s_tau_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5*tf.ones(
                        (1, self.feature_dim), dtype=self.dtype
                    ),
                    scale=1.0/s_tau_a
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['s_tau_a'] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5*tf.ones(
                        (1, self.feature_dim), dtype=self.dtype
                    ),
                    scale=tf.ones(
                        (1, self.feature_dim),
                        dtype=self.dtype)/self.s_tau_scale**2
                ), reinterpreted_batch_ndims=2
            )
        self.joint_prior = tfd.JointDistributionNamed(
            distribution_dict)
        surrogate_dict = {
            'u': self.bijectors['u'](
                build_trainable_normal_dist(
                    -self.feature_dim/20.*tf.ones((self.feature_dim, self.latent_dim),
                                 dtype=self.dtype),
                    1e-4*tf.ones((self.feature_dim, self.latent_dim),
                                 dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            'u_eta': self.bijectors['u_eta'](
                build_trainable_InverseGamma_dist(
                    3*tf.ones(
                        (self.feature_dim, self.latent_dim), dtype=self.dtype),
                    tf.ones(
                        (self.feature_dim, self.latent_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            'u_tau': self.bijectors['u_tau'](
                build_trainable_InverseGamma_dist(
                    3*tf.ones(
                        (1, self.latent_dim),
                        dtype=self.dtype),
                    tf.ones((1, self.latent_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            'v': self.bijectors['v'](
                build_trainable_normal_dist(
                    -self.feature_dim/20.*tf.ones((self.latent_dim, self.feature_dim),
                                 dtype=self.dtype),
                    1e-4*tf.ones((self.latent_dim, self.feature_dim),
                                 dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            )
        }
        if self.with_w:
            surrogate_dict['w'] = self.bijectors['w'](
                build_trainable_normal_dist(
                    tf.ones((1, self.feature_dim), dtype=self.dtype),
                    1e-2*tf.ones((1, self.feature_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            )
        if self.with_s:
            surrogate_dict['s_eta'] = self.bijectors['s_eta'](
                build_trainable_InverseGamma_dist(
                    tf.ones((2, self.feature_dim), dtype=self.dtype),
                    tf.ones((2, self.feature_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            )
            surrogate_dict['s_tau'] = self.bijectors['s_tau'](
                build_trainable_InverseGamma_dist(
                    1*tf.ones((1, self.feature_dim), dtype=self.dtype),
                    tf.ones((1, self.feature_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            )
            surrogate_dict['s'] = self.bijectors['s'](
                build_trainable_normal_dist(
                    tf.ones(
                        (2, self.feature_dim), dtype=self.dtype)*tf.cast(
                            [[-3.], [-1.]], dtype=self.dtype),
                    1e-3*tf.ones(
                        (2, self.feature_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            )

        self.bijectors['u_eta_a'] = tfb.Softplus()
        self.bijectors['u_tau_a'] = tfb.Softplus()
        surrogate_dict['u_eta_a'] = self.bijectors['u_eta_a'](
            build_trainable_InverseGamma_dist(
                2.*tf.ones(
                    (self.feature_dim, self.latent_dim),
                    dtype=self.dtype),
                tf.ones(
                    (self.feature_dim, self.latent_dim),
                    dtype=self.dtype),
                2,
                strategy=self.strategy
            )
        )
        surrogate_dict['u_tau_a'] = self.bijectors['u_tau_a'](
            build_trainable_InverseGamma_dist(
                2.*tf.ones(
                    (1, self.latent_dim),
                    dtype=self.dtype),
                tf.ones(
                    (1, self.latent_dim),
                    dtype=self.dtype)/self.u_tau_scale**2,
                2,
                strategy=self.strategy
            )
        )
        if self.with_s:
            self.bijectors['s_eta_a'] = tfb.Softplus()
            self.bijectors['s_tau_a'] = tfb.Softplus()
            surrogate_dict['s_eta_a'] = self.bijectors['s_eta_a'](
                build_trainable_InverseGamma_dist(
                    2.*tf.ones(
                        (2, self.feature_dim), dtype=self.dtype),
                    tf.ones((2, self.feature_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            )
            surrogate_dict['s_tau_a'] = self.bijectors['s_tau_a'](
                build_trainable_InverseGamma_dist(
                    2.*tf.ones((1, self.feature_dim), dtype=self.dtype),
                    (
                        tf.ones(
                            (1, self.feature_dim),
                            dtype=self.dtype) / self.s_tau_scale**2),
                    2,
                    strategy=self.strategy
                )
            )
        self.surrogate_distribution = tfd.JointDistributionNamed(
            surrogate_dict
        )

        self.surrogate_vars = self.surrogate_distribution.variables

        self.var_list = list(surrogate_dict.keys())

        self.surrogate_sample = self.surrogate_distribution.sample(2000)
        self.set_calibration_expectations()

    def unormalized_log_prob(self, data=None, **x):
        prob_parts = self.unormalized_log_prob_parts(
            data, **x)
        return tf.add_n(
            list(prob_parts.values()))

    def unormalized_log_prob_parts(self, data=None, **params):
        """Energy function

        Keyword Arguments:
            data {} -- [description] (default: {None})

        Returns:
            [type] -- [description]
        """
        if data is None:
            #  use self.data, taking the next batch
            try:
                record = next(self.dataset_cycler)
                indices = record['indices']
                data = record['data']
            except tf.errors.OutOfRangeError:
                self.dataset_iterator = cycle(iter(self.data))
                record = next(self.dataset_iterator)
                indices = record['indices']
                data = record['data']
        else:
            # use the data as-is
            record = data
            data = record['data']
            indices = record['indices']

        prior_parts = self.joint_prior.log_prob_parts(params)
        data = tf.cast(data, self.dtype)
        weights = params['s']/tf.reduce_sum(params['s'], axis=-2)[
            ..., tf.newaxis, :]
        weights_1 = tf.expand_dims(weights[..., 0, :], -1)
        weights_2 = tf.expand_dims(weights[..., 1, :], -1)

        encoding = weights_1*params['u']
        z = tf.matmul(
            self.encoder_function(data),
            encoding)
        rv_z = tfd.Independent(
            tfd.HalfNormal(
                scale=0.25*tf.ones_like(z)
            ),
            reinterpreted_batch_ndims=2
        )
        prior_parts['z'] = rv_z.log_prob(z)
        data = tf.cast(data, self.dtype)
        L = len(weights_2.shape)
        trans = tuple(list(range(L-2)) + [L-1, L-2])
        weights_2 = tf.transpose(weights_2, trans)
        rate = self.decoder_function(
            tf.matmul(z, params['v'])) + weights_2*params['w']
        rate_shape = tf.shape(rate)[-2]
        new_shape = tf.unstack(tf.ones_like(tf.shape(rate)))
        new_shape[-2] = rate_shape
        if 'normalization' in record.keys():
            multiplier = record['normalization'][..., tf.newaxis]
            multiplier = tf.reshape(multiplier, new_shape)

            rate *= tf.cast(multiplier, self.dtype)

        rate *= self.norm_factor

        rv_x = tfd.Independent(
            tfd.Poisson(
                rate=rate
            ),
            reinterpreted_batch_ndims=0
        )
        log_likelihood = rv_x.log_prob(data)

        finite_portion = tf.where(
            tf.math.is_finite(log_likelihood), log_likelihood,
            tf.zeros_like(log_likelihood))
        min_val = tf.reduce_min(finite_portion)-1000.
        max_val = 0.
        log_likelihood = tf.clip_by_value(log_likelihood, min_val, max_val)
        log_likelihood = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            tf.ones_like(log_likelihood)*min_val
        )
        log_likelihood = tf.reduce_sum(log_likelihood, -1)
        log_likelihood = tf.reduce_sum(log_likelihood, -1)
        prior_parts['x'] = log_likelihood

        return prior_parts

    def project(self, x, threshold=1e-1):
        return tf.matmul(
            self.encoder_function(tf.cast(x, self.dtype)),
            tf.matmul(
                tf.linalg.diag(
                    self.calibrated_expectations['s'],
                    self.calibrated_expectations['u']
                )
            )
        )

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float64)])
    def encode(self, x):
        encoding = self.encoding_matrix()
        return tf.matmul(
            self.encoder_function(
                tf.cast(x, self.dtype)
                ), encoding)

    def encoding_matrix(self):
        if not self.with_s:
            return self.calibrated_expectations['u']

        weights = self.calibrated_expectations['s']/tf.reduce_sum(
            self.calibrated_expectations['s'], axis=-2)[
                ..., tf.newaxis, :]
        weights_1 = tf.expand_dims(weights[..., 0, :], -1)

        encoding = weights_1*self.calibrated_expectations['u']
        return encoding

    def decoding_matrix(self):
        return self.calibrated_expectations['v']

    def intercept_matrix(self):
        weights = self.calibrated_expectations['s']/tf.reduce_sum(
            self.calibrated_expectations['s'], axis=-2)[
                ..., tf.newaxis, :]
        weights_2 = tf.expand_dims(weights[..., 1, :], -1)
        L = len(weights_2.shape)
        trans = tuple(list(range(L-2)) + [L-1, L-2])
        weights_2 = tf.transpose(weights_2, trans)
        return self.norm_factor*weights_2*self.calibrated_expectations['w']

    def decode(self, z):
        return tf.matmul(z, self.calibrated_expectations['v'])

    @tf.function(autograph=False)
    def unormalized_log_prob_list(self, *x):
        return self.unormalized_log_prob(
            **{
                v: t for v, t in zip(self.var_list, x)
            }
        )

    def reconstitute(self, state):
        self.create_distributions()
        for j, value in enumerate(
                state['surrogate_vars']):
            self.surrogate_distribution.trainable_variables[j].assign(
                tf.cast(value, self.dtype))
        #  self.set_calibration_expectations()

    def sparsify(self):
        pass


class PoissonMatrixFactorizationNoS(PoissonMatrixFactorization):
    def __init__(self, **kwargs):
        super(PoissonMatrixFactorizationNoS, self).__init__(
            with_s=False, **kwargs)

    def log_likelihood(self, u, v, w,  data=None, *args, **kwargs):
        s = tf.ones_like(w)
        return super(PoissonMatrixFactorizationNoS, self).log_likelihood(
            s, w, u, q,  data=None, *args, **kwargs)
