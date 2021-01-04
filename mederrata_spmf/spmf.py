#!/usr/bin/env python3
"""Sparse probabilistic PCA using the horseshoe
See main() for usage
Note that you currently have to babysit the optimization a bit
"""

from itertools import cycle
import functools

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

    def encoder_function(self, x):
        """Encoder function (g)
        Args:
            x ([type]): [description]
        Returns:
            [type]: [description]
        """
        if self.log_transform:
            return tf.math.log(x/self.eta_i+1.)
        return x/self.eta_i

    def decoder_function(self, x):
        """Decoder function (f)
        Args:
            x ([type]): [description]
        Returns:
            [type]: [description]
        """
        if self.log_transform:
            return tf.math.exp(x*self.eta_i)-1.
        return x*self.eta_i

    def __init__(
            self, data, data_transform_fn=None, latent_dim=None,
            u_tau_scale=0.01, s_tau_scale=1., symmetry_breaking_decay=0.5,
            strategy=None, encoder_function=None, decoder_function=None,
            scale_columns=True, column_norms=None, scale_rows=True,
            with_s=True, with_w=True, log_transform=False,
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
            scale_columns {bool} -- Scale the rates by the mean of the first batch (default: {True})
            scale_row {bool} -- Scale by normalized row sums (default: {True})
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
        self.log_transform = log_transform

        if with_w:
            if not with_s:
                self.log_likelihood = functools.partial(
                    self.log_likelihood, s=1.)
                self.log_likelihood_components = functools.partial(
                    self.log_likelihood_components, s=1.)
            self.with_w = with_w
            self.with_s = with_s
        else:
            if with_s:
                print("Disabling s because w is disabled")

            self.log_likelihood = functools.partial(
                self.log_likelihood, s=1., w=0.)
            self.log_likelihood_components = functools.partial(
                self.log_likelihood_components, s=1., w=0.)
            self.with_s = False
            self.with_w = False

        record = next(iter(data))
        indices = record['indices']
        counts = record['counts']
        self.eta_i = 1.
        self.xi_u_global = 1.
        self.scale_rows = scale_rows

        if scale_columns:
            if column_norms is not None:
                self.eta_i = tf.cast(
                    column_norms, self.dtype)
            else:
                print("Looping through the entire dataset once to get some stats")

                colsums = []
                col_nonzero_Ns = []
                N = 0
                for batch in iter(data):
                    colsums += [tf.reduce_sum(batch['counts'],
                                              axis=0, keepdims=True)]
                    col_nonzero_Ns += [
                        tf.reduce_sum(
                            tf.cast(batch['counts'] > 0, tf.float32),
                            axis=0, keepdims=True)
                    ]
                    N += batch['counts'].shape[0]

                colsums = tf.add_n(colsums)
                col_nonzero_N = tf.add_n(col_nonzero_Ns)
                colmeans = colsums/N
                colmeans_nonzero = (
                    tf.cast(colsums, tf.float64) /
                    tf.cast(col_nonzero_N, tf.float64))
                rowmean = tf.reduce_sum(colmeans)
                rowmean_nonzero = tf.cast(
                    tf.reduce_sum(colmeans_nonzero), tf.float64)
                columns = colmeans.shape[-1]

                self.eta_i = tf.where(
                    colmeans_nonzero > 1,
                    colmeans_nonzero,
                    tf.ones_like(colmeans_nonzero))

            if scale_rows:
                self.xi_u_global = rowmean_nonzero

        self.feature_dim = counts.shape[-1]
        self.latent_dim = self.feature_dim if (
            latent_dim) is None else latent_dim

        self.u_tau_scale = u_tau_scale
        self.s_tau_scale = s_tau_scale

        self.create_distributions()
        print(
            f"Feature dim: {self.feature_dim} -> Latent dim {self.latent_dim}")

    def log_likelihood_components(
            self, s, u, v, w, data, *args, **kwargs):
        """Returns the log likelihood without summing along axes
        Arguments:
            s {tf.Tensor} -- Samples of s
            u {tf.Tensor} -- Samples of u
            v {tf.Tensor} -- Samples of v
            w {tf.Tensor} -- Samples of w
        Keyword Arguments:
            data {tf.Tensor} -- Count matrix (default: {None})
        Returns:
            [tf.Tensor] -- log likelihood in broadcasted shape
        """

        theta_u = self.encode(data['counts'], u, s)
        phi = self.intercept_matrix(w, s)
        B = self.decoding_matrix(v)

        theta_beta = tf.matmul(theta_u, B)
        theta_beta = self.decoder_function(theta_beta)

        rate = theta_beta + phi
        rv_poisson = tfd.Poisson(rate=rate)

        return rv_poisson.log_prob(
            tf.cast(data['counts'], self.dtype))

    # @tf.function
    def log_likelihood(
            self, s, u, v, w, data, *args, **kwargs):
        """Returns the log likelihood, summed over rows
        Arguments:
            s {tf.Tensor} -- Samples of s
            u {tf.Tensor} -- Samples of u
            v {tf.Tensor} -- Samples of v
            w {tf.Tensor} -- Samples of w
        Keyword Arguments:
            data {Dict} -- Dataset dict (default: {None})
        Returns:
            [tf.Tensor] -- log likelihood in broadcasted shape
        """
        if self.with_s:
            ll = self.log_likelihood_components(
                s=s, u=u, v=v, w=w, data=data, *args, **kwargs)
        else:
            ll = self.log_likelihood_components(
                u=u, v=v, w=w, data=data, *args, **kwargs)

        reduce_dim = len(u.shape) - 2
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
                    scale=0.25/self.latent_dim*tf.ones(
                        (self.latent_dim, self.feature_dim),
                        dtype=self.dtype)
                ), reinterpreted_batch_ndims=2
            )}
        if self.with_w:
            distribution_dict['w'] = tfd.Independent(
                tfd.HalfNormal(
                    scale=tf.ones(
                        (1, self.feature_dim), dtype=self.dtype)
                ),
                reinterpreted_batch_ndims=2
            )
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

        if self.with_s:
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
            'u': self.bijectors['u'](
                build_trainable_normal_dist(
                    -8.*tf.ones((self.feature_dim, self.latent_dim),
                                dtype=self.dtype),
                    5e-4*tf.ones((self.feature_dim, self.latent_dim),
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
                    -5*tf.ones(
                        (self.latent_dim, self.feature_dim),
                        dtype=self.dtype),
                    5e-4*tf.ones((self.latent_dim, self.feature_dim),
                                 dtype=self.dtype),
                    2,
                    strategy=self.strategy
                )
            )
        }
        if self.with_w:
            surrogate_dict['w'] = self.bijectors['w'](
                build_trainable_normal_dist(
                    0.5*tf.ones((1, self.feature_dim), dtype=self.dtype),
                    1e-3*tf.ones((1, self.feature_dim), dtype=self.dtype),
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
                                [[-2.], [-1.]], dtype=self.dtype),
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

        self.set_calibration_expectations()

    def unormalized_log_prob(self, data=None, **params):
        prob_parts = self.unormalized_log_prob_parts(
            data, **params)
        value = tf.add_n(
            list(prob_parts.values()))
        return value

    def unormalized_log_prob_parts(self, data=None, **params):
        """Energy function
        Keyword Arguments:
            data {dict} -- Should be a single batch (default: {None})
        Returns:
            tf.Tensor -- Energy of broadcasted shape
        """

        # We don't use indices so let's just get rid of them
        if data is None:
            #  use self.data, taking the next batch
            try:
                data = next(self.dataset_cycler)
            except tf.errors.OutOfRangeError:
                self.dataset_iterator = cycle(iter(self.data))
                data = next(self.dataset_iterator)

        prior_parts = self.joint_prior.log_prob_parts(params)
        log_likelihood = self.log_likelihood_components(data=data, **params)

        # For prior on theta
        if not self.with_s:
            s = None
        else:
            s = params['s']
        theta = self.encode(x=data['counts'], u=params['u'], s=s)
        rv_theta = tfd.Independent(
            tfd.HalfNormal(
                scale=tf.ones_like(theta, dtype=self.dtype)),
            reinterpreted_batch_ndims=2)

        prior_parts['z'] = rv_theta.log_prob(theta)

        finite_portion = tf.where(
            tf.math.is_finite(log_likelihood), log_likelihood,
            tf.zeros_like(log_likelihood))
        min_val = tf.reduce_min(finite_portion)-10.
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

    def encode(self, x, u=None, s=None):
        """Returns theta given x
        Args:
            x ([type]): [description]
        Returns:
            [type]: [description]
        """
        u = self.calibrated_expectations['u'] if u is None else u
        s = self.calibrated_expectations['s'] if s is None else s

        encoding = self.encoding_matrix(u, s)    # A = (\alpha_{ik})
        z = tf.matmul(
            self.encoder_function(
                tf.cast(x, self.dtype)
            ), encoding)
        if self.scale_rows:
            xi_u = tf.reduce_sum(
                tf.cast(
                    x, self.dtype), axis=-1, keepdims=True
            )/self.xi_u_global
            z *= xi_u
        return z

    def encoding_matrix(self, u=None, s=None):
        """Output A = (\alpha_{ik})

        Returns:
            tf.Tensor: batch_shape x I x K 
        """
        u = self.calibrated_expectations['u'] if u is None else u
        if not self.with_s:
            return u
        s = self.calibrated_expectations['s'] if s is None else s
        weights = s/tf.reduce_sum(
            s, axis=-2, keepdims=True)
        weights_1 = tf.expand_dims(weights[..., 0, :], -1)

        encoding = weights_1*u
        return encoding

    def decoding_matrix(self, v=None):
        """Output $B=(\beta_{ki})$

        Args:
            v (tf.Tensor): default:None

        Returns:
            [type]: [description]
        """
        v = self.calibrated_expectations['v'] if v is None else v
        return v

    def intercept_matrix(self, w=None, s=None):
        """export phi

        Args:
            w ([type], optional): [description]. Defaults to None.
            s ([type], optional): [description]. Defaults to None.

        Returns:
            tf.Tensor: batch_shape x 1 x I
        """
        if not self.with_w:
            return tf.zeros(
                (1, self.feature_dim),
                dtype=self.dtype)

        w = self.calibrated_expectations['w'] if w is None else w
        if self.with_s:
            s = self.calibrated_expectations['s'] if s is None else s
            weights = s/tf.reduce_sum(
                s, axis=-2)[
                    ..., tf.newaxis, :]
            weights_2 = tf.expand_dims(weights[..., 1, :], -1)
            L = len(weights_2.shape)
            trans = tuple(list(range(L-2)) + [L-1, L-2])
            weights_2 = tf.transpose(weights_2, trans)
        else:
            weights_2 = 1.
        return self.eta_i*weights_2*w

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
