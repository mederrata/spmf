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


class SparsePoissonLinearFactorization(BayesianModel):
    """Sparse (horseshoe) poisson matrix factorization
    Arguments:
        object {[type]} -- [description]
    """
    w0 = None
    bijectors = None
    var_list = []
    s_tau_scale = 1
    def forward_function(self, x): return tf.math.log(x+1.)
    def inverse_function(self, x): return tf.math.exp(x) - 1.

    def __init__(
            self, data, data_transform_fn=None, latent_dim=None,
            auxiliary_horseshoe=True,
            w_tau_scale=1., s_tau_scale=1, symmetry_breaking_decay=0.25,
            strategy=None,
            scale_rates=True,
            dtype=tf.float64, **kwargs):

        super(SparsePoissonLinearFactorization, self).__init__(
            data, data_transform_fn, strategy=strategy, dtype=dtype)
        self.dtype = dtype
        self.symmetry_breaking_decay = symmetry_breaking_decay

        record = next(iter(data))
        indices = record['indices']
        data = record['data']
        self.norm_factor = 1
        if scale_rates:
            self.norm_factor = tf.reduce_mean(data).numpy()
        if 'normalization' in record.keys():
            norm = record['normalization']
        data = tf.cast(data, self.dtype)
        self.feature_dim = data.shape[-1]
        self.scale_rates = scale_rates

        self.latent_dim = self.feature_dim if (
            latent_dim) is None else latent_dim

        self.w_tau_scale = w_tau_scale
        self.s_tau_scale = s_tau_scale

        self.w0 = tf.constant(tf.eye(
            self.feature_dim, self.latent_dim, dtype=self.dtype
        )/np.sqrt(self.feature_dim))
        self.u0 = tf.constant(tf.linalg.matrix_transpose(self.w0))
        self.z0 = tf.constant(tf.matmul(
            self.forward_function(data),
            self.w0))
        self.x0 = tf.constant(tf.matmul(
            self.z0,
            self.u0
        ))
        self.s0 = tf.constant(
            0.5*tf.ones((2, self.feature_dim), dtype=self.dtype))
        self.create_distributions(auxiliary_horseshoe=auxiliary_horseshoe)
        print(
            f"Feature dim: {self.feature_dim} -> Latent dim {self.latent_dim}")

    # @tf.function
    def log_likelihood(self, s, w, u, q,  data=None, *args, **kwargs):
        weights = s/tf.reduce_sum(s, axis=-2)[
            ..., tf.newaxis, :]
        weights_1 = tf.expand_dims(weights[..., 0, :], -1)
        weights_2 = tf.expand_dims(weights[..., 1, :], -1)

        assert isinstance(data, BatchDataset)
        encoding = weights_1*w
        log_likes = []
        for record in data:
            indices = record['indices']
            batch = record['data']
            batch = tf.cast(batch, self.dtype)
            z_batch = tf.matmul(
                self.forward_function(batch),
                encoding
            )
            L = len(weights_2.shape)
            trans = tuple(list(range(L-2)) + [L-1, L-2])
            weights_2 = tf.transpose(weights_2, trans)
            rate = self.inverse_function(tf.matmul(z_batch, u)) + weights_2*q
            if 'normalization' in record.keys():
                multiplier = record['normalization'][..., tf.newaxis]
                for _ in range(len(rate.shape)-len(multiplier.shape)):
                    multiplier = multiplier[tf.newaxis, ...]
                rate *= tf.cast(multiplier, self.dtype)
            rate *= self.norm_factor
            rv_poisson = tfd.Independent(
                tfd.Poisson(rate=rate),
                reinterpreted_batch_ndims=(len(batch.shape)-1)
            )
            log_likes += [rv_poisson.log_prob(batch)]
        combined = tf.concat(log_likes, axis=-1)
        return combined

    def create_distributions(self, auxiliary_horseshoe=True):
        self.bijectors = {
            'w': tfb.Softplus(),
            'u': tfb.Softplus(),
            'w_eta': tfb.Softplus(),
            'w_tau': tfb.Softplus(),
            's': tfb.Softplus(),
            's_eta': tfb.Softplus(),
            's_tau': tfb.Softplus(),
            'q': tfb.Softplus()
        }
        symmetry_breaking_decay = self.symmetry_breaking_decay**tf.cast(
            tf.range(self.latent_dim), self.dtype)[tf.newaxis, ...]

        distribution_dict = {
            'w': lambda w_eta, w_tau: tfd.Independent(
                tfd.HalfNormal(
                    scale=w_eta*w_tau*symmetry_breaking_decay
                ), reinterpreted_batch_ndims=2
            ),
            'q': tfd.Independent(
                tfd.HalfNormal(
                    scale=100.*tf.ones(
                        (1, self.feature_dim), dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=2
            ),
            'w_eta': tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros(
                        (self.feature_dim, self.latent_dim),
                        dtype=self.dtype),
                    scale=tf.ones(
                        (self.feature_dim, self.latent_dim),
                        dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            ),
            'w_tau': tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros(
                        (1, self.latent_dim),
                        dtype=self.dtype),
                    scale=tf.ones(
                        (1, self.latent_dim),
                        dtype=self.dtype)*self.w_tau_scale
                ), reinterpreted_batch_ndims=2
            ),
            'u': tfd.Independent(
                tfd.HalfNormal(
                    scale=tf.ones_like(self.u0, dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            ),
            's': lambda s_eta, s_tau: tfd.Independent(
                tfd.HalfNormal(
                    scale=s_eta*s_tau
                ), reinterpreted_batch_ndims=2
            ),
            's_eta': tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros(
                        (2, self.feature_dim),
                        dtype=self.dtype),
                    scale=tf.ones(
                        (2, self.feature_dim),
                        dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            ),
            's_tau': tfd.Independent(
                tfd.HalfCauchy(
                    loc=tf.zeros((1, self.feature_dim), dtype=self.dtype),
                    scale=tf.ones((1, self.feature_dim),
                                  dtype=self.dtype)*self.s_tau_scale
                ), reinterpreted_batch_ndims=2
            )
        }
        if auxiliary_horseshoe:

            self.bijectors['w_eta_a'] = tfb.Softplus()
            self.bijectors['w_tau_a'] = tfb.Softplus()

            self.bijectors['s_eta_a'] = tfb.Softplus()
            self.bijectors['s_tau_a'] = tfb.Softplus()

            distribution_dict['w_eta'] = lambda w_eta_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5*tf.ones(
                        (self.feature_dim, self.latent_dim),
                        dtype=self.dtype
                    ),
                    scale=1.0/w_eta_a
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['w_eta_a'] = tfd.Independent(
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
            distribution_dict['w_tau'] = lambda w_tau_a: tfd.Independent(
                SqrtInverseGamma(
                    concentration=0.5*tf.ones(
                        (1, self.latent_dim),
                        dtype=self.dtype
                    ),
                    scale=1.0/w_tau_a
                ), reinterpreted_batch_ndims=2
            )
            distribution_dict['w_tau_a'] = tfd.Independent(
                tfd.InverseGamma(
                    concentration=0.5*tf.ones(
                        (1, self.latent_dim), dtype=self.dtype
                    ),
                    scale=tf.ones(
                        (1, self.latent_dim), dtype=self.dtype
                    )/self.w_tau_scale**2
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
            'w': self.bijectors['w'](
                build_trainable_normal_dist(
                    -5*tf.ones((self.feature_dim, self.latent_dim),
                               dtype=self.dtype),
                    1e-3*tf.ones((self.feature_dim, self.latent_dim),
                                 dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            'w_eta': self.bijectors['w_eta'](
                build_trainable_InverseGamma_dist(
                    3*tf.ones(
                        (self.feature_dim, self.latent_dim), dtype=self.dtype),
                    tf.ones(
                        (self.feature_dim, self.latent_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            'w_tau': self.bijectors['w_tau'](
                build_trainable_InverseGamma_dist(
                    3*tf.ones(
                        (1, self.latent_dim),
                        dtype=self.dtype),
                    tf.ones((1, self.latent_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            'u': self.bijectors['u'](
                build_trainable_normal_dist(
                    tf.convert_to_tensor(self.u0, dtype=self.dtype),
                    1e-4*tf.ones_like(self.u0, dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            'q': self.bijectors['q'](
                build_trainable_normal_dist(
                    -5*tf.ones((1, self.feature_dim), dtype=self.dtype),
                    1e-4*tf.ones((1, self.feature_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            's_eta': self.bijectors['s_eta'](
                build_trainable_InverseGamma_dist(
                    3*tf.ones((2, self.feature_dim), dtype=self.dtype),
                    tf.ones((2, self.feature_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            's_tau': self.bijectors['s_tau'](
                build_trainable_InverseGamma_dist(
                    3*tf.ones((1, self.feature_dim), dtype=self.dtype),
                    tf.ones((1, self.feature_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
            's': self.bijectors['s'](
                build_trainable_normal_dist(
                    -4*tf.ones(
                        (2, self.feature_dim), dtype=self.dtype),
                    1e-3*tf.ones(
                        (2, self.feature_dim), dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            ),
        }
        if auxiliary_horseshoe:
            self.bijectors['w_eta_a'] = tfb.Softplus()
            self.bijectors['w_tau_a'] = tfb.Softplus()
            surrogate_dict['w_eta_a'] = self.bijectors['w_eta_a'](
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
            surrogate_dict['w_tau_a'] = self.bijectors['w_tau_a'](
                build_trainable_InverseGamma_dist(
                    2.*tf.ones(
                        (1, self.latent_dim),
                        dtype=self.dtype),
                    tf.ones(
                        (1, self.latent_dim),
                        dtype=self.dtype)/self.w_tau_scale**2,
                    2,
                    strategy=self.strategy
                )
            )
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
        """See if this works
        This rewrites the value of z, setting it equal to x * w
        Returns:
            [type] -- [description]
        """
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

        encoding = weights_1*params['w']
        z = tf.matmul(
            self.forward_function(data),
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
        rate = self.inverse_function(
            tf.matmul(z, params['u'])) + weights_2*params['q']
        rate_shape = tf.shape(rate)[-2]
        new_shape = tf.unstack(tf.ones_like(tf.shape(rate)))
        new_shape[-2] = rate_shape
        if 'normalization' in record.keys():
            multiplier = record['normalization'][..., tf.newaxis]
            multiplier = tf.reshape(multiplier, new_shape)

            rate *= tf.cast(multiplier, self.dtype)

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
            self.forward_function(tf.cast(x, self.dtype)), tf.matmul(tf.linalg.diag(
                self.calibrated_expectations['s'],
                self.calibrated_expectations['w']
            ))
        )

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float64)])
    def encode(self, x):
        encoding = self.encoding_matrix()
        return tf.matmul(
            self.forward_function(tf.cast(x, self.dtype)), encoding)

    def encoding_matrix(self):
        weights = self.calibrated_expectations['s']/tf.reduce_sum(
            self.calibrated_expectations['s'], axis=-2)[
                ..., tf.newaxis, :]
        weights_1 = tf.expand_dims(weights[..., 0, :], -1)

        encoding = weights_1*self.calibrated_expectations['w']
        return encoding

    def intercept_matrix(self):
        weights = self.calibrated_expectations['s']/tf.reduce_sum(
            self.calibrated_expectations['s'], axis=-2)[
                ..., tf.newaxis, :]
        weights_2 = tf.expand_dims(weights[..., 1, :], -1)
        L = len(weights_2.shape)
        trans = tuple(list(range(L-2)) + [L-1, L-2])
        weights_2 = tf.transpose(weights_2, trans)
        return weights_2*self.calibrated_expectations['q']

    def decode(self, z):
        return tf.matmul(z, self.calibrated_expectations['u'])

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
